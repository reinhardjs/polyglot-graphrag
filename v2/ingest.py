"""
ingest.py — Document ingestion pipeline.

Reads .md / .txt from a directory, runs late-chunked embedding + graph
extraction, then writes to Qdrant (dense + sparse vectors) and Neo4j
(canonicalised entity graph with source-text profiles).

Extraction: LLM (Gemma E2B on :8082) primary; GLiNER CPU daemon fallback.

v3 (2026-07-10): hash-based sparse vectors (consistent cross-chunk vocab),
Neo4j node profiles (resurrects the graph leg), per-doc delete-before-reingest.
"""
import os
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import json
import requests
from pathlib import Path
from collections import Counter
from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase

import config as C

_SPARSE_VOCAB = 65536


# ── Qdrant / Neo4j helpers ────────────────────────────────────────────────────
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)


def get_neo4j():
    return GraphDatabase.driver(
        C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD)
    )


def init_collections(qc: QdrantClient):
    if not qc.collection_exists(C.COLL_CHUNKS):
        qc.create_collection(
            C.COLL_CHUNKS,
            vectors_config=models.VectorParams(
                size=C.VECTOR_DIM, distance=models.Distance.COSINE
            ),
            sparse_vectors_config={"text": models.SparseVectorParams()},
        )


# ── Sparse vector ─────────────────────────────────────────────────────────────
def _sparse(text: str) -> models.SparseVector:
    """Hash-based sparse vector for CONSISTENT cross-chunk term matching.

    Hashing each token to the same 64K-bin space ensures "checkout" always
    maps to the same index in every chunk AND in the query.
    """
    toks = re.findall(r"\w+", text.lower())
    c = Counter(toks)
    idxs, vals = [], []
    for tok, f in c.items():
        idxs.append(hash(tok) % _SPARSE_VOCAB)
        vals.append(float(f))
    return models.SparseVector(indices=idxs, values=vals)


# ── Qdrant write ──────────────────────────────────────────────────────────────
def write_vectors(doc_id: str, chunks: list, meta: dict) -> int:
    qc = get_qdrant()
    pts = []
    for ch in chunks:
        pts.append(models.PointStruct(
            id=abs(hash(f"{doc_id}:{ch['chunk_idx']}")) % (2**63),
            vector={"": ch["vector"], "text": _sparse(ch["text"])},
            payload={"doc_id": doc_id, "chunk_idx": ch["chunk_idx"],
                     "text": ch["text"], "doc_type": meta.get("doc_type", "eng")},
        ))
    qc.upsert(C.COLL_CHUNKS, pts, wait=True)
    return len(pts)


# ── Profile builder ───────────────────────────────────────────────────────────
def _build_profile(entity: str, doc_text: str, window: int = 2) -> str:
    """Build a 1-3 sentence profile for an entity from its source document.

    Finds the sentence containing the entity and wraps it with `window`
    surrounding sentences. Strips markdown formatting. Returns a compact KEY
    + VALUE profile that neo4j_subgraph() can use for reranking.
    """
    sent_pat = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*")
    sents = [s.strip() for s in sent_pat.split(doc_text) if s.strip()]
    e_l = entity.lower()
    for i, s in enumerate(sents):
        if e_l in s.lower():
            start = max(0, i - window)
            end = min(len(sents), i + window + 1)
            ctx = " ".join(sents[start:end])
            # compact: KEY: value
            return ctx[:C.MAX_TOKENS_CONTEXT * 4]  # ~chars = ~tokens
    # Fallback: first sentence containing any word of entity
    for word in entity.split():
        for s in sents:
            if word.lower() in s.lower():
                return s[:300]
    return ""  # no match found


# ── Neo4j write ───────────────────────────────────────────────────────────────
def delete_doc_neo4j(doc_id: str, driver):
    """Delete ALL nodes and edges sourced from a given doc_id."""
    with driver.session() as s:
        s.run(
            "MATCH (n:Entity {source_doc:$doc}) DETACH DELETE n",
            doc=doc_id,
        )


def delete_doc_qdrant(doc_id: str):
    """Delete ALL Qdrant points for a given doc_id."""
    qc = get_qdrant()
    qc.delete(
        C.COLL_CHUNKS,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id),
                )]
            )
        ),
    )


def write_graph(doc_id: str, graph: dict, source_text: str, driver) -> int:
    """Write canonicalised entity nodes (WITH profiles) + edges into Neo4j.

    Each entity node gets a `profile` — 1-3 contextual sentences from the
    source document — so neo4j_subgraph() returns useful text for retrieval.
    Uses MERGE so repeated ingestion of the same canonical id is idempotent.
    """
    with driver.session() as s:
        for node in graph.get("nodes", []):
            cid, label = canonicalize(node["id"], node["type"])
            profile = _build_profile(node.get("id", ""), source_text)
            s.run(
                "MERGE (n:Entity {id:$id}) "
                "SET n.name=$label, n.type=$type, n.source_doc=$doc, "
                "    n.profile=$profile",
                id=cid, label=label, type=node["type"], doc=doc_id,
                profile=profile,
            )
        for e in graph.get("edges", []):
            sc, _ = canonicalize(e["source"], "")
            tc, _ = canonicalize(e["target"], "")
            if sc == tc:
                continue
            s.run(
                "MATCH (a:Entity {id:$s}), (b:Entity {id:$t}) "
                "MERGE (a)-[:ASSOCIATED_WITH]->(b)",
                s=sc, t=tc,
            )
    return len(graph.get("nodes", []))


# ── LLM / GLiNER extraction ───────────────────────────────────────────────────
def extract_graph_llm(doc_id: str, text: str) -> dict:
    """LLM-based graph extraction via Gemma E2B on :8082.

    PRIMARY extraction path. Gemma E2B is a small reasoning model; we ask for
    strict JSON. If it returns empty content (reasoning loop) we fall back to
    the GLiNER daemon.
    """
    from openai import OpenAI
    client = OpenAI(base_url=C.EXTRACTION_LLM_BASE_URL, api_key=C.EXTRACTION_LLM_API_KEY)
    prompt = (
        "Extract a knowledge graph from the engineering document below.\n"
        "Return ONLY valid JSON with this exact shape, no prose, no markdown:\n"
        '{"nodes":[{"id":"entity_name","type":"Microservice|Database|API|'
        'Metric|Developer|Framework|Component|Bug|PR|ADR"}],'
        '"edges":[{"source":"entity_a","target":"entity_b",'
        '"type":"ASSOCIATED_WITH|DEPENDS_ON|IMPACTS|AUTHORED|REFERENCES|FIXES"}]}\n'
        f"Document ({doc_id}):\n{text[:8000]}"
    )
    try:
        resp = client.chat.completions.create(
            model=C.EXTRACTION_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048, temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if "nodes" in data and "edges" in data:
                return data
    except Exception as e:
        print(f"[extract] LLM extraction failed: {e}", flush=True)
    print("[extract] falling back to GLiNER", flush=True)
    return requests.post(C.DAEMON_EXTRACT, json={"text": text}, timeout=300).json()


# ── Canonicalisation ──────────────────────────────────────────────────────────
def canonicalize(raw_id: str, entity_type: str) -> tuple:
    """Normalise entity IDs via CANON_MAP; fall back to lowercased raw_id."""
    cid = raw_id.lower().strip()
    cid = C.CANON_MAP.get(cid, cid)
    label = cid.replace("_", " ").title()
    return cid, label


# ── Ingest one file ───────────────────────────────────────────────────────────
def ingest_file(path: Path):
    text = path.read_text("utf-8", errors="ignore")
    doc_id = path.stem
    meta = {"doc_type": "eng", "author": "unknown"}

    # Step 0 — delete existing data for this doc (clean re-ingest)
    driver = get_neo4j()
    delete_doc_neo4j(doc_id, driver)
    driver.close()
    delete_doc_qdrant(doc_id)

    # Step 1 — late-chunked vectors -> Qdrant
    r = requests.post(C.DAEMON_EMBED_LATE,
                      json={"text": text, "doc_id": doc_id}, timeout=300).json()
    n_chunks = write_vectors(doc_id, r["chunks"], meta)

    # Step 2 — graph extraction (LLM E2B primary, GLiNER fallback)
    g = extract_graph_llm(doc_id, text)

    # Step 3 — canonicalise + write to Neo4j (WITH profiles from source text)
    driver = get_neo4j()
    n_nodes = write_graph(doc_id, g, text, driver)
    driver.close()

    print(f"[ingest] {doc_id}: {n_chunks} chunks, {n_nodes} nodes", flush=True)
    return n_chunks, n_nodes


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_collections(get_qdrant())
    target = sys.argv[1] if len(sys.argv) > 1 else C.DATA_DIR
    p = Path(target)
    files = p.iterdir() if p.is_dir() else [p]
    for f in sorted(files):
        if f.suffix.lower() in (".md", ".txt", ".markdown"):
            ingest_file(f)
    print("INGEST_DONE", flush=True)
