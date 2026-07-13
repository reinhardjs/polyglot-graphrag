"""
ingest.py — Ingestion pipeline (write mode).

VRAM STRATEGY:
    Ornith 9B (port 8081) is used for graph extraction + KV profiling.
    Once ingestion finishes, Ornith is idle; Gemma takes over for retrieval.

Flow:
    1. Parse raw doc → extract entities + relationships via Ornith
    2. Generate 2-sentence KV profiles for each node/edge → Neo4j
    3. Late-chunk the full document text via Jina v3 → per-sentence vectors
    4. Store vectors in Qdrant with BinaryQuantization + metadata payloads

MAX_CONTEXT = 32768 is enforced on all LLM calls.
"""
from __future__ import annotations
import json
import re
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/data-970-plus/hf_cache")

from config import (
    ornith_client, get_qdrant, init_collections, get_neo4j,
    CHUNKS_COLLECTION, CACHE_COLLECTION, VECTOR_DIM, MAX_CONTEXT,
    ORNITH_MODEL,
)

from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────
#  1. LLM EXTRACTION (via Ornith 9B on port 8081)
# ─────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are an engineering knowledge extractor.
Given the document below, extract entities and relationships as STRICT JSON.

Return exactly this shape (no prose, no markdown fences):
<<JSON_SHAPE>>
  "entities": [
    {"id": "unique_id", "type": "Microservice|ADR|Bug|Dev|Component|Ticket", "name": "Human Label"}
  ],
  "relationships": [
    {"source": "<entity_id>", "target": "<entity_id>",
     "type": "IMPACTS|DEPENDS_ON|AUTHORED|REFERENCES|FIXES"}
  ]
<<END_SHAPE>>

Only use entity ids you defined. Document:
---
<<TEXT>>
---"""

_PROFILE_PROMPT = """Summarize the following engineering entity in exactly 2 sentences:
- What it is
- Why it matters and its key dependencies or impact

ENTITY: <<LABEL>> (<<ETYPE>>)
CONTEXT:
<<TEXT>>"""


def _read_msg(msg: dict) -> str:
    """Extract text from an OpenAI response message (handles reasoning models)."""
    c = (msg.get("content") or "").strip()
    if c:
        return c
    return (msg.get("reasoning_content") or "").strip()


def _strip_json(text: str) -> str:
    """Extract JSON object from a text (handles markdown fences + reasoning noise)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text.strip()


def llm_extract(text: str) -> dict:
    """Call Ornith to extract entities + relationships. Returns dict with
    'entities' and 'relationships' keys; empty lists on failure."""
    client = ornith_client()
    prompt = _EXTRACT_PROMPT.replace("<<TEXT>>", text[:MAX_CONTEXT])
    try:
        resp = client.chat.completions.create(
            model=ORNITH_MODEL,
            messages=[
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = _read_msg(resp.choices[0].message)
        g = json.loads(_strip_json(raw))
        if g.get("entities"):
            return g
    except Exception:
        pass
    return _extract_rulebased(text)


def llm_profile(label: str, etype: str, text: str) -> str:
    """Generate a 2-sentence KV profile via Ornith. Falls back to doc
    sentence-window extraction on failure."""
    try:
        client = ornith_client()
        prompt = _PROFILE_PROMPT.replace("<<LABEL>>", label).replace("<<ETYPE>>", etype).replace("<<TEXT>>", text[:3000])
        resp = client.chat.completions.create(
            model=ORNITH_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=200,
        )
        r = _read_msg(resp.choices[0].message)
        if r and len(r) > 20:
            return r[:600]
    except Exception:
        pass
    return _doc_profile(label, text)


# ─────────────────────────────────────────────────────────────
#  1b. DETERMINISTIC FALLBACK (when Ornith is down or reasoning-looping)
# ─────────────────────────────────────────────────────────────

_ENTITY_PATTERNS = [
    (r"\b(ADR-\d+)\b", "ADR"),
    (r"\b(BUG-\d+)\b", "BUG"),
    (r"\b(PR-\d+)\b", "TICKET"),
]
_KNOWN_COMPONENTS = ["checkout", "billing", "storefront", "payments-consumer",
                     "payment gateway", "checkout database"]


def _extract_rulebased(text: str) -> dict:
    entities = {}
    for pat, etype in _ENTITY_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            eid = m.group(1).lower()
            entities[eid] = {"id": eid, "type": etype, "name": m.group(1)}
    for name in _KNOWN_COMPONENTS:
        if name in text.lower():
            eid = name.replace(" ", "_")
            entities.setdefault(eid, {"id": eid, "type": "COMPONENT", "name": name})
    # Grab author if present
    m = re.search(r"Author:\s*(\S+)", text)
    if m:
        eid = m.group(1).lower()
        entities[eid] = {"id": eid, "type": "DEV", "name": m.group(1)}
    rels = []
    for sent in re.split(r"[.\n]", text):
        sl = sent.lower()
        found = [eid for eid, e in entities.items() if e["name"].lower() in sl]
        for rtype, kw in [("IMPACTS", "impact"), ("DEPENDS_ON", "depends on"),
                          ("AUTHORED", "authored by"), ("FIXES", "fixes"),
                          ("REFERENCES", "references")]:
            if kw in sl:
                for i in range(len(found) - 1):
                    rels.append({"source": found[i], "target": found[i + 1], "type": rtype})
                # cross-refs
                for other in re.finditer(r"\b(ADR-\d+|BUG-\d+|PR-\d+)\b", sent):
                    oeid = other.group(1).lower()
                    if oeid in entities and found:
                        rels.append({"source": found[0], "target": oeid, "type": "REFERENCES"})
                break
    return {"entities": list(entities.values()), "relationships": rels}


def _doc_profile(label: str, text: str) -> str:
    key = label.lower().replace("_", " ")
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    for i, s in enumerate(sents):
        if key in s.lower():
            lo, hi = max(0, i - 1), min(len(sents), i + 2)
            return " ".join(sents[lo:hi])[:600]
    toks = [t for t in re.findall(r"[a-z0-9]+", key) if len(t) > 2]
    for s in sents:
        if any(t in s.lower() for t in toks):
            return s[:400]
    return f"{label}"


# ─────────────────────────────────────────────────────────────
#  2. JINA LATE CHUNKING
# ─────────────────────────────────────────────────────────────

class JinaLateChunker:
    """Wraps jina-embeddings-v3 with per-sentence embedding (pragmatic late-chunking)."""

    def __init__(self, model_name: str = "/mnt/data-970-plus/hf_cache/jina-v3-flat"):
        self.model = SentenceTransformer(model_name, device="cpu",
                                          trust_remote_code=True)
        self.dim = self.model.get_sentence_embedding_dimension() or VECTOR_DIM

    def embed_doc(self, text: str):
        """Split into sentences, embed each with Jina v3 → clean chunks.

        jina-embeddings-v3's native late_chunking kwarg isn't wired through
        the SentenceTransformer wrapper in this version, so we sentence-split
        at the text level and embed each sentence independently. This avoids
        the broken token-offset mapping we saw with bge-m3.
        """
        sents = self._split_sentences(text)
        if not sents:
            return [], []
        # Encode all sentences in one batch (fast on CPU)
        embeddings = self.model.encode(
            sents, task="retrieval.passage",  # jina-optimized prefix
            normalize_embeddings=True, show_progress_bar=False,
        )
        return sents, embeddings.tolist()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        import re
        raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        return [s.strip() for s in raw if len(s.strip()) > 5]

    def embed_query(self, text: str) -> list[float]:
        v = self.model.encode([text], task="retrieval.query",
                              normalize_embeddings=True,
                              show_progress_bar=False)[0]
        return v.tolist()


# ─────────────────────────────────────────────────────────────
#  3. NEO4J WRITE (KV profiles)
# ─────────────────────────────────────────────────────────────

_VALID_TYPES = {"MICROSERVICE", "ADR", "BUG", "DEV", "COMPONENT", "TICKET"}
_VALID_RELS  = {"IMPACTS", "DEPENDS_ON", "AUTHORED", "REFERENCES", "FIXES"}


def write_graph(doc_id: str, doc_text: str, driver=None) -> int:
    """Extract entities → generate KV profiles → write to Neo4j.
    Returns node count written."""
    if driver is None:
        driver = get_neo4j()
    g = llm_extract(doc_text)
    # Profile each entity
    with driver.session() as s:
        for e in g.get("entities", []):
            eid = e.get("id") or e.get("name", "unknown")
            etype = str(e.get("type", "COMPONENT")).upper()
            if etype not in _VALID_TYPES:
                etype = "COMPONENT"
            profile = llm_profile(e.get("name", eid), etype, doc_text)
            s.run(
                "MERGE (n:Entity {id:$id}) "
                "SET n.name=$name, n.type=$type, n.profile=$profile, "
                "n.source_doc=$doc",
                id=eid, name=e.get("name", eid), type=etype,
                profile=profile, doc=doc_id,
            )
        for r in g.get("relationships", []):
            rtype = str(r.get("type", "REFERENCES")).upper()
            if rtype not in _VALID_RELS:
                rtype = "REFERENCES"
            s.run(
                f"MATCH (a:Entity {{id:$s}}), (b:Entity {{id:$t}}) "
                f"MERGE (a)-[:{rtype}]->(b)",
                s=r.get("source"), t=r.get("target"),
            )
    return len(g.get("entities", []))


# ─────────────────────────────────────────────────────────────
#  4. QDRANT WRITE (BinaryQuantized late-chunked vectors + sparse)
# ─────────────────────────────────────────────────────────────

def _lexical_sparse(text: str) -> dict[int, float]:
    """Compute a BM25-style lexical sparse vector from raw text."""
    from collections import Counter
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if not toks:
        return {}
    tf = Counter(toks)
    n = len(toks)
    vec = {}
    for term, c in tf.items():
        tid = abs(hash(term)) % (2**31)
        vec[tid] = 1.0 + (c / n)
    return vec


def write_vectors(chunker: JinaLateChunker, doc_id: str, text: str,
                  metadata: dict | None = None) -> int:
    """Late-chunk → upsert to Qdrant with BinaryQuantization + sparse.
    Returns chunk count."""
    from qdrant_client.models import PointStruct, SparseVector
    client = get_qdrant()
    init_collections(client)
    meta = metadata or {}

    chunks, vectors = chunker.embed_doc(text)
    if not chunks:
        return 0

    lexical = _lexical_sparse(text)
    points = []
    for i, (ch, vec) in enumerate(zip(chunks, vectors)):
        pid = abs(hash(f"{doc_id}:{i}")) % (2**63)
        points.append(PointStruct(
            id=pid,
            vector={
                "": vec,
                "text": SparseVector(
                    indices=list(lexical.keys()),
                    values=list(lexical.values()),
                ),
            },
            payload={"doc_id": doc_id, "chunk_idx": i, "text": ch, **meta},
        ))
    client.upsert(collection_name=CHUNKS_COLLECTION, points=points)
    return len(points)


# ─────────────────────────────────────────────────────────────
#  5. MAIN INGEST ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def ingest_doc(path: str, doc_type: str | None = None,
               author: str | None = None):
    """Ingest a single document: graph → Neo4j + late-chunk → Qdrant."""
    path = Path(path)
    doc_id = path.stem
    text = _read_file(path)
    meta = {}
    if doc_type:
        meta["doc_type"] = doc_type
    if author:
        meta["author"] = author

    # Step A: Neo4j (via Ornith on 8081)
    driver = get_neo4j()
    n_nodes = write_graph(doc_id, text, driver)
    driver.close()

    # Step B: Qdrant (Jina late-chunking)
    chunker = JinaLateChunker()
    n_chunks = write_vectors(chunker, doc_id, text, meta)

    print(f"[ingest] {doc_id}: {n_chunks} chunks → Qdrant, "
          f"{n_nodes} entities → Neo4j")


def _read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            import pypdf
            return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
        except ImportError:
            return f"[PDF support requires pypdf pip install: {path}]"
    return path.read_text("utf-8", errors="ignore")
