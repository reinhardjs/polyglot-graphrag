"""
ingest.py — Document ingestion pipeline with vector-driven entity resolution.

Reads .md / .txt from a directory, runs late-chunked embedding + graph
extraction, then writes to Qdrant (dense + sparse vectors) and Neo4j
(vector-indexed, language-agnostic entity graph with source-text profiles).

Extraction: LLM (Gemma E2B on :8082) verbatim primary; GLiNER daemon fallback.

v4 (2026-07-10): Replaces hardcoded CANON_MAP with Jina v3 cross-lingual
vector matching in Neo4j. Entities extracted verbatim — "Basis Data",
"Database", "Base de Datos" converge via cosine similarity in the 1024-dim
embedding space. Aliases tracked on each node for audit trail.
v3 (2026-07-10): hash-based sparse vectors, Neo4j node profiles,
per-doc delete-before-reingest.
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
# ── Entity Resolution (vector-driven, language-agnostic) ──────────────────────
# Replaces the hardcoded CANON_MAP (config.py). Uses Jina v3's cross-lingual
# embedding space to find semantic equivalents stored in Neo4j's vector index.
#
# HOW IT WORKS:
#   1. E2B extracts entities VERBATIM ("Basis Data", "Database", "Base de Datos").
#   2. Each entity name is embedded via Jina v3 (/embed_query on :8000).
#   3. The vector queries Neo4j's entity_vector_idx for the closest match.
#   4. If cosine ≥ ENTITY_RESOLUTION_THRESHOLD (0.88): merge — reuse existing
#      node ID, append the new name to `aliases[]`. No new node created.
#   5. If cosine < 0.88: create a brand-new canonical entity node with its
#      name_vector stored for future cross-lingual matching.
#   6. Edge resolution maps source/target to resolved IDs from phase 1.
#
# This eliminates hardcoded translation maps entirely. "Basis Data" (ID)
# "Database" (EN), and "Base de Datos" (ES) all converge automatically because
# Jina v3 places them near each other in the 1024-dim cross-lingual vector space.


def _gen_entity_id(raw_name: str, entity_type: str) -> str:
    """Generate a stable, unique entity ID from the raw name + type."""
    return f"e_{abs(hash(f'{raw_name}|{entity_type}')) % (2**63):x}"


def _embed_name(raw_name: str) -> list:
    """Embed an entity's raw name via the resident Jina v3 daemon (:8000)."""
    try:
        r = requests.post(C.DAEMON_EMBED_QUERY,
                          json={"text": raw_name}, timeout=30)
        return r.json()["vector"]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to embed entity name '{raw_name}': {exc}"
        ) from exc


def _resolve_in_neo4j(driver, name_vector: list, raw_name: str,
                       entity_type: str) -> tuple:
    """Query Neo4j's vector index for the closest existing entity.

    Uses `db.index.vector.queryNodes()` (Neo4j 5.x native vector index).
    Returns (resolved_id, is_new, closest_score, closest_name).
    """
    with driver.session() as s:
        result = s.run(
            # Neo4j 5.x vector index query: find nearest Entity node
            "CALL db.index.vector.queryNodes($idx, $k, $vec) "
            "YIELD node, score "
            "RETURN node.id AS id, node.name AS name, node.type AS type, "
            "       score "
            "ORDER BY score DESC "
            "LIMIT 1",
            idx=C.ENTITY_VECTOR_INDEX,
            k=min(5, C.VECTOR_DIM),  # ef=5 for fast approximate
            vec=name_vector,
        )
        row = result.single()
        if row and row["score"] >= C.ENTITY_RESOLUTION_THRESHOLD:
            # CROSS-LINGUAL MERGE: Jina v3 aligned the raw name (e.g. "Basis Data")
            # near an existing entity (e.g. "Database") → same concept, merge.
            return (row["id"], False, row["score"], row["name"])
        # No match above threshold → this is a novel entity in any language
        return (_gen_entity_id(raw_name, entity_type), True, 0.0, raw_name)


def resolve_node_ids(driver, nodes: list, verbose: bool = False) -> dict:
    """Resolve ALL extracted entity names to Neo4j IDs via vector matching.

    Returns a dict mapping {raw_name: resolved_id} for use in edge creation.
    Each node's resolution is independent and could be done in parallel.
    """
    resolver = {}  # raw_name → resolved_id
    for node in nodes:
        raw_name = node["id"]            # verbatim from source text
        entity_type = node.get("type", "Unknown")
        name_vec = _embed_name(raw_name)  # Jina v3 cross-lingual embedding
        # Query Neo4j vector index for the closest existing entity
        resolved_id, is_new, score, closest = _resolve_in_neo4j(
            driver, name_vec, raw_name, entity_type,
        )
        resolver[raw_name] = resolved_id
        if verbose:
            tag = "NEW" if is_new else f"MATCH {score:.3f}→{closest}"
            print(f"[resolve] '{raw_name}' → {resolved_id} [{tag}]", flush=True)
    return resolver


# ── Neo4j write (vector-driven resolution) ────────────────────────────────────
def delete_doc_neo4j(doc_id: str, driver):
    """Remove a document's entities and edges from Neo4j safely.

    With vector-driven resolution, entities can be shared across documents
    (e.g. "Database" appears in both ADR-014 and BUG-204). This function:
      - Removes doc_id from each entity's `source_docs` list.
      - Deletes entities whose `source_docs` list becomes empty (orphaned).
      - Cleans up all edges where either endpoint was orphaned.
    """
    with driver.session() as s:
        # First: remove edges that involve nodes whose only source is this doc
        s.run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE $doc IN a.source_docs OR $doc IN b.source_docs "
            "DELETE r",
            doc=doc_id,
        )
        # Second: remove doc_id from source_docs list on all matching entities
        s.run(
            "MATCH (n:Entity) WHERE $doc IN n.source_docs "
            "SET n.source_docs = [d IN n.source_docs WHERE d <> $doc]",
            doc=doc_id,
        )
        # Third: delete entities that now have empty source_docs (orphaned)
        s.run(
            "MATCH (n:Entity) WHERE size(n.source_docs) = 0 "
            "DETACH DELETE n",
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
    """Write entity nodes + edges into Neo4j using VECTOR-DRIVEN resolution.

    Phase A — Node Resolution:
      Embed each entity name via Jina v3 → query Neo4j vector index.
      Match (≥0.88 cosine) → reuse existing node ID, append alias.
      Miss  (<0.88 cosine) → create new node with name_vector stored.

    Phase B — Node Write:
      MERGE each resolved entity with its profile, type, source_doc, and
      name_vector. Append raw names to aliases[] for audit trail.

    Phase C — Edge Write:
      Map source/target to resolved IDs from Phase A → MERGE relationships.
    """
    nodes = graph.get("nodes", [])
    if not nodes:
        return 0

    # ── Phase A: Vector-resolution of ALL entity names ─
    resolver = resolve_node_ids(driver, nodes, verbose=True)

    # ── Phase B: Write resolved nodes with profiles + aliases ─
    with driver.session() as s:
        for node in nodes:
            raw_name = node["id"]
            entity_type = node.get("type", "Unknown")
            resolved_id = resolver[raw_name]
            profile = _build_profile(raw_name, source_text)
            name_vec = _embed_name(raw_name)

            # Check if this exact name is NEW to this resolved node
            # (MERGE the node; append the alias only if novel)
            result = s.run(
                "MERGE (n:Entity {id:$id}) "
                "ON CREATE SET "
                "  n.name        = $name, "
                "  n.type        = $type, "
                "  n.name_vector = $vec, "
                "  n.aliases     = [$name], "
                "  n.source_docs = [$doc], "
                "  n.profile     = $profile "
                "ON MATCH SET "
                # Append doc_id to source_docs if not already present
                "  n.source_docs = CASE "
                "    WHEN $doc IN COALESCE(n.source_docs, []) THEN n.source_docs "
                "    ELSE COALESCE(n.source_docs, []) + $doc "
                "  END, "
                "  n.profile = $profile, "
                # Append the raw name to aliases if not already present
                "  n.aliases = CASE "
                "    WHEN $name IN COALESCE(n.aliases, []) THEN n.aliases "
                "    ELSE COALESCE(n.aliases, []) + $name "
                "  END "
                "RETURN n.name AS name, n.aliases AS aliases",
                id=resolved_id, name=raw_name, type=entity_type,
                vec=name_vec, doc=doc_id, profile=profile,
            ).single()
            if result:
                aliases = result.get("aliases", [])
                if len(aliases) > 1:
                    print(f"[write]  {raw_name} → {resolved_id} "
                          f"(aliases: {aliases})", flush=True)

    # ── Phase C: Write edges using resolved IDs ─
    edge_count = 0
    with driver.session() as s:
        for e in graph.get("edges", []):
            src_id = resolver.get(e["source"])
            tgt_id = resolver.get(e["target"])
            if not src_id or not tgt_id:
                continue  # resolution failed for one end
            if src_id == tgt_id:
                continue  # self-loop, skip
            edge_type = e.get("type", "ASSOCIATED_WITH")
            s.run(
                "MATCH (a:Entity {id:$s}), (b:Entity {id:$t}) "
                f"MERGE (a)-[:{edge_type}]->(b)",
                s=src_id, t=tgt_id,
            )
            edge_count += 1

    print(f"[write] {doc_id}: {len(nodes)} nodes resolved, "
          f"{edge_count} edges written", flush=True)
    return len(nodes)


# ── LLM / GLiNER extraction ───────────────────────────────────────────────────
def extract_graph_llm(doc_id: str, text: str) -> dict:
    """LLM-based graph extraction via the configured extraction model.

    PRIMARY extraction path. The extraction model (config: EXTRACTION_LLM)
    is prompted with config.EXTRACTION_PROMPT — change it in config.py if you
    swap the model and it needs a different prompt style.

    Entities are extracted VERBATIM — exactly as they appear in the source text.
    NO translation. Jina v3's cross-lingual vector space handles merging via
    Neo4j's vector index.
    """
    from openai import OpenAI
    client = OpenAI(base_url=C.EXTRACTION_LLM_BASE_URL,
                    api_key=C.EXTRACTION_LLM_API_KEY)
    prompt = C.EXTRACTION_PROMPT.format(doc_id=doc_id, text=text[:8000])
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

    # Step 3 — vector-resolve entities + write to Neo4j (WITH profiles, aliases)
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
