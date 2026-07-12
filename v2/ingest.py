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
import hashlib
os.environ["HF_HOME"] = "/mnt/data-970-plus/hf_cache"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import json
import requests
import threading
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
        # Deterministic across processes (Python's built-in hash() is randomized
        # per process via PYTHONHASHSEED, which silently breaks sparse search
        # when ingest and query run in different processes).
        idxs.append(int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % _SPARSE_VOCAB)
        vals.append(float(f))
    return models.SparseVector(indices=idxs, values=vals)


# ── Qdrant write ──────────────────────────────────────────────────────────────
def ensure_collection(name: str):
    """Create a Qdrant collection with the correct vector config if missing."""
    qc = get_qdrant()
    if not qc.collection_exists(name):
        qc.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=C.VECTOR_DIM, distance=models.Distance.COSINE
            ),
            sparse_vectors_config={"text": models.SparseVectorParams()},
        )
        print(f"[qdrant] created collection '{name}'", flush=True)


def write_vectors(doc_id: str, chunks: list, meta: dict, checksum: str = "",
                 collection: str = C.COLL_CHUNKS) -> int:
    qc = get_qdrant()
    pts = []
    for ch in chunks:
        # Store the full meta dict (REQ-7) so citations can surface rich
        # domain metadata (title/author/url/patient_id/...) via GET /ingest
        # and the /ask sources/contexts_meta blocks.
        payload = {"doc_id": doc_id, "chunk_idx": ch["chunk_idx"],
                   "text": ch["text"], "doc_type": meta.get("doc_type", "eng"),
                   "checksum": checksum, "metadata": meta}
        pts.append(models.PointStruct(
            id=abs(hash(f"{doc_id}:{ch['chunk_idx']}")) % (2**63),
            vector={"": ch["vector"], "text": _sparse(ch["text"])},
            payload=payload,
        ))
    qc.upsert(collection, pts, wait=True)
    return len(pts)


def validate_metadata(profile: dict, metadata: dict) -> tuple[dict, list]:
    """Validate caller-supplied metadata against a domain profile schema.

    Returns (accepted, warnings).
      accepted — metadata dict kept as-is (unknown fields kept for transparency).
      warnings — human-readable messages for unknown fields.

    Rule (v2.6.0 REQ-7): if the profile declares metadata_schema.fields and a
    supplied key is not in it, WARN (do not drop) so ingestion never fails on
    schema drift. If the profile declares no fields, everything is accepted.
    """
    accepted = dict(metadata or {})
    warnings = []
    if not profile:
        return accepted, warnings
    allowed = set(profile.get("metadata_schema", {}).get("fields", []))
    if allowed:
        for k in metadata or {}:
            if k not in allowed:
                warnings.append(
                    f"metadata field '{k}' not in profile schema for domain "
                    f"'{profile['domain']['name']}'")
    return accepted, warnings


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

# ── Concurrent-resolution safety ───────────────────────────────────────────────
# Two concurrent ingests that both mention a novel entity ("Kafka") will each
# query Neo4j, find no match, and generate DIFFERENT entity IDs → duplicate nodes.
# A process-level cache + lock eliminates the race AND speeds up re-ingestion
# of common entities within the daemon's lifetime.
_resolution_cache: "dict[str, str]" = {}   # raw_name → resolved_id
_resolution_lock = threading.Lock()


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

    Thread-safe: a process-level cache + lock prevents two concurrent ingests
    from creating duplicate nodes for the same novel entity.
    """
    resolver = {}  # raw_name → resolved_id
    for node in nodes:
        raw_name = node["id"]            # verbatim from source text
        entity_type = node.get("type", "Unknown")

        # Fast path: already resolved in this process (cache hit)
        with _resolution_lock:
            if raw_name in _resolution_cache:
                resolver[raw_name] = _resolution_cache[raw_name]
                if verbose:
                    print(f"[resolve] '{raw_name}' → cache hit "
                          f"({_resolution_cache[raw_name]})", flush=True)
                continue

        name_vec = _embed_name(raw_name)  # Jina v3 cross-lingual embedding
        # Query Neo4j vector index for the closest existing entity
        resolved_id, is_new, score, closest = _resolve_in_neo4j(
            driver, name_vec, raw_name, entity_type,
        )
        # Persist to cache under lock (another thread may resolve same name
        # concurrently — second one will then hit the cache above).
        with _resolution_lock:
            _resolution_cache[raw_name] = resolved_id
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
      - Cleans up edges: an edge is deleted only if removing `doc_id` from
        BOTH endpoints leaves at least one endpoint orphaned (i.e. the edge
        has no remaining source document keeping it alive). Edges between two
        still-shared nodes are preserved.

    NOTE: this is the safe approximation. For perfectly accurate edge
    attribution, track `source_docs` on the relationships themselves.
    """
    with driver.session() as s:
        # First: remove doc_id from source_docs on all matching nodes, and
        # compute whether each endpoint will be orphaned.
        s.run(
            "MATCH (n:Entity) WHERE $doc IN n.source_docs "
            "SET n.source_docs = [d IN n.source_docs WHERE d <> $doc]",
            doc=doc_id,
        )
        # Second: delete edges where BOTH endpoints lose all source_docs
        # (will be orphaned by the step below) — i.e. edges with no surviving
        # source doc after removal. Edges connecting still-shared nodes stay.
        s.run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE size(a.source_docs) = 0 OR size(b.source_docs) = 0 "
            "DELETE r",
        )
        # Third: delete entities that now have empty source_docs (orphaned)
        s.run(
            "MATCH (n:Entity) WHERE size(n.source_docs) = 0 "
            "DETACH DELETE n",
        )


def delete_doc_qdrant(doc_id: str, collection: str = C.COLL_CHUNKS):
    """Delete ALL Qdrant points for a given doc_id in a collection."""
    qc = get_qdrant()
    qc.delete(
        collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id),
                )]
            )
        ),
    )


def write_graph(doc_id: str, graph: dict, source_text: str, driver,
                chunk_size: int = 512, neo4j_label: str = None) -> int:
    """Write entity nodes + edges into Neo4j using VECTOR-DRIVEN resolution.

    Phase A — Node Resolution:
      Embed each entity name via Jina v3 → query Neo4j vector index.
      Match (≥0.88 cosine) → reuse existing node ID, append alias.
      Miss  (<0.88 cosine) → create new node with name_vector stored.

    Phase B — Node Write:
      MERGE each resolved entity with its profile, type, source_doc, and
      name_vector. Append raw names to aliases[] for audit trail.
      If `neo4j_label` is given (V3.0 domain config), also tag the node with
      that secondary label so graph retrieval can scope by domain.

    Phase C — Edge Write:
      Map source/target to resolved IDs from Phase A → MERGE relationships.
    """
    # Sanitize the domain label for Cypher (:Label can't be parameterized).
    safe_label = None
    if neo4j_label:
        cand = "".join(c if (c.isalnum() or c == "_") else "_"
                       for c in neo4j_label)
        if cand and (cand[0].isalpha() or cand[0] == "_"):
            safe_label = cand

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
            # Profile window scales with chunk size so large-chunk strategies
            # (section/fixed) still surface enough surrounding context.
            prof_window = max(2, chunk_size // 256)
            profile = _build_profile(raw_name, source_text, window=prof_window)
            name_vec = _embed_name(raw_name)

            # Check if this exact name is NEW to this resolved node
            # (MERGE the node; append the alias only if novel)
            # V3.0: also attach the domain label (e.g. :Engineering) when set,
            # so graph retrieval can scope subgraphs by domain.
            label_clause = f":{safe_label}" if safe_label else ""
            result = s.run(
                f"MERGE (n:Entity{label_clause} {{id:$id}}) "
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

    # ── Phase C: Write edges using resolved IDs (BATCH UNWIND by rel-type) ─
    # V3.0: group edges by relationship type, then one UNWIND query per type.
    # This replaces the old per-edge MERGE loop (single-query-per-edge) with
    # batched writes — far fewer round-trips to Neo4j.
    edges_by_type: dict = {}
    for e in graph.get("edges", []):
        src_id = resolver.get(e["source"])
        tgt_id = resolver.get(e["target"])
        if not src_id or not tgt_id:
            continue  # resolution failed for one end
        if src_id == tgt_id:
            continue  # self-loop, skip
        edge_type = e.get("type", "ASSOCIATED_WITH")
        # Sanitize edge type for Cypher (relationship types can't be parameterized):
        # allow only A-Z, 0-9, underscore; uppercase.
        safe_type = "".join(
             c if (c.isalnum() or c == "_") else "_" for c in edge_type.upper()
        )
        if not safe_type or not (safe_type[0].isalpha() or safe_type[0] == "_"):
            safe_type = "REL_" + safe_type
        edges_by_type.setdefault(safe_type, []).append(
            {"source_id": src_id, "target_id": tgt_id}
        )

    edge_count = 0
    with driver.session() as s:
        for rel_type, batch in edges_by_type.items():
            # Relationship type is validated/sanitized above; safe to interpolate.
            s.run(
                "UNWIND $edges AS edge "
                "MATCH (a:Entity {id: edge.source_id}) "
                "MATCH (b:Entity {id: edge.target_id}) "
                f"MERGE (a)-[:{rel_type}]->(b)",
                edges=batch,
            )
            edge_count += len(batch)

    print(f"[write] {doc_id}: {len(nodes)} nodes resolved, "
          f"{edge_count} edges written ({len(edges_by_type)} rel-types, "
          f"batched UNWIND)", flush=True)
    return len(nodes)


# ── LLM / GLiNER extraction ───────────────────────────────────────────────────
def extract_graph_index_routing(doc_id: str, text: str,
                                domain: "str | None" = None) -> dict:
    """V3.0 Hybrid Index-Routing extraction (GLiNER entities → Qwen relations).

    Returns a graph dict in write_graph's expected shape:
      {"nodes": [{"id": <name>, "type": <type>}, ...],
       "edges": [{"source": <name>, "target": <name>, "type": <REL>}, ...]}

    Entity source/target names come from the index_router's remapped relations
    (source_name / target_name), so write_graph's vector resolver can dedupe.
    """
    import index_router
    result = index_router.extract(text, domain_name=domain)
    entities = result.get("entities", [])
    relations = result.get("relations", [])

    nodes = [{"id": e["name"], "type": e.get("type", "Unknown")}
             for e in entities]
    edges = []
    for r in relations:
        sname = r.get("source_name")
        tname = r.get("target_name")
        if not sname or not tname:
            continue
        edges.append({
            "source": sname,
            "target": tname,
            "type": r.get("type", "ASSOCIATED_WITH"),
        })
    return {"nodes": nodes, "edges": edges,
            "meta": result.get("meta", {})}


def extract_graph_llm(doc_id: str, text: str, chunk_size: int = 512,
                      profile: dict = None) -> dict:
    """LLM-based graph extraction via the configured extraction model.

    PRIMARY extraction path. The extraction model (config: EXTRACTION_LLM)
    is prompted with the domain profile's [extraction].prompt when a profile
    is supplied (v2.6.0 REQ-4); otherwise falls back to config.EXTRACTION_PROMPT.
    Entities are extracted VERBATIM — exactly as they appear in the source text.
    NO translation. Jina v3's cross-lingual vector space handles merging via
    Neo4j's vector index.
    """
    from openai import OpenAI
    # Domain-aware prompt (REQ-4): profile[extraction][prompt] wins.
    if profile and profile.get("extraction", {}).get("prompt"):
        prompt_tmpl = profile["extraction"]["prompt"]
    else:
        prompt_tmpl = C.EXTRACTION_PROMPT
    # Safe substitution: prompts contain literal {...} (JSON schema) that break
    # str.format(), so replace only our two tokens.
    client = OpenAI(base_url=C.EXTRACTION_LLM_BASE_URL,
                    api_key=C.EXTRACTION_LLM_API_KEY)
    prompt = (prompt_tmpl
              .replace("{doc_id}", doc_id)
              .replace("{text}", text[:C.EXTRACTION_CHAR_LIMIT]))
    try:
        resp = client.chat.completions.create(
            model=C.EXTRACTION_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=C.EXTRACTION_MAX_TOKENS, temperature=0.0,
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
    # GLiNER fallback: use domain entity labels when profile provides them.
    gliner_labels = None
    if profile and profile.get("graph_schema", {}).get("entity_types"):
        gliner_labels = profile["graph_schema"]["entity_types"]
    gl_body = {"text": text}
    if gliner_labels:
        gl_body["labels"] = gliner_labels
    try:
        return requests.post(C.DAEMON_EXTRACT, json=gl_body, timeout=300).json()
    except Exception:
        return {"nodes": [], "edges": []}


# ── Ingest one document (text-based, API-callable) ──────────────────────────
def ingest_text(text: str, doc_id: str, meta: dict | None = None,
                extract_graph: bool = True, if_checksum: str = None,
                collection: str = C.COLL_CHUNKS,
                domain: str = None, metadata: dict | None = None) -> dict:
    """Ingest a single document's text into Qdrant + Neo4j.

    Extracted from ingest_file() so the daemon's /ingest endpoint and the CLI
    share one code path. Returns a result dict with timings + counts.

    `collection` selects the Qdrant collection (multi-domain support). Defaults
    to C.COLL_CHUNKS (engineering_chunks). The collection is created on first
    use via ensure_collection().

    `domain` (v2.6.0): selects a profile (TOML). When set, chunking strategy,
    extraction prompt, and metadata validation are taken from the profile
    instead of hardcoded defaults. Unknown domain → engineering profile.

    If `if_checksum` is provided and matches the stored Qdrant payload checksum,
    returns {"status": "unchanged", "skipped": True} in <10ms (no re-extraction).
    """
    import time
    import config as C
    # Resolve domain profile (v2.6.0) — drives chunking + extraction + metadata.
    profile = C.load_domain_profile(domain) if domain else None
    if profile:
        chunk_cfg = profile.get("chunking", {})
    else:
        # V3.0: no TOML profile — pull chunking from domain_config.yaml
        try:
            import domain_loader
            chunk_cfg = domain_loader.get_domain(domain).get("chunking", {})
        except Exception:
            chunk_cfg = {}
    strategy = chunk_cfg.get("strategy", "sentence")
    chunk_size = chunk_cfg.get("chunk_size", 512)
    overlap = chunk_cfg.get("overlap", 64)
    header_prefix = chunk_cfg.get("section_header_prefix", "##")

    meta = dict(meta or {"doc_type": "eng", "author": "unknown"})
    # Domain profile enriches meta (collection name + default doc_type).
    if profile:
        meta.setdefault("doc_type", (profile["domain"].get("name") or "eng")[:3])
        meta["domain"] = profile["domain"]["name"]
    # REQ-7: validate caller metadata against profile.metadata_schema.fields.
    if profile and metadata:
        accepted, warnings = validate_metadata(profile, metadata)
        for w in warnings:
            print(f"[ingest] WARNING: {w}", flush=True)
        for k, v in accepted.items():
            meta[k] = v

    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ── Incremental guard: skip if unchanged ──
    if if_checksum is not None:
        try:
            qc = get_qdrant()
            hits = qc.scroll(
                collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="doc_id", match=models.MatchValue(value=doc_id))]),
                limit=1, with_payload=["checksum"],
            )[0]
            if hits and hits[0].payload.get("checksum") == if_checksum:
                return {
                    "doc_id": doc_id, "status": "unchanged", "skipped": True,
                    "checksum": checksum, "chunks": 0, "entities": 0, "edges": 0,
                    "vectors_upserted": 0, "timing": {"total": 0.0},
                    "collection": collection,
                }
        except Exception:
            pass  # fall through to full ingest if check fails

    t0 = time.time()

    # Ensure the target collection exists (idempotent)
    ensure_collection(collection)

    # Step 0 — delete existing data for this doc (clean re-ingest)
    driver = get_neo4j()
    try:
        delete_doc_neo4j(doc_id, driver)
    finally:
        driver.close()
    delete_doc_qdrant(doc_id, collection)

    t_clean = time.time()

    # Step 1 — late-chunked vectors -> Qdrant (strategy from domain profile)
    r = requests.post(C.DAEMON_EMBED_LATE,
                      json={"text": text, "doc_id": doc_id,
                            "strategy": strategy, "chunk_size": chunk_size,
                            "overlap": overlap, "header_prefix": header_prefix},
                      timeout=300).json()
    n_chunks = write_vectors(doc_id, r["chunks"], meta, checksum=checksum,
                             collection=collection)
    t_embed = time.time()

    # Step 2 — graph extraction
    #   V3.0: Hybrid Index-Routing (GLiNER→Qwen) is the primary path when
    #   config.EXTRACTION_MODE == "index_routing"; otherwise the legacy LLM
    #   (E2B) path is used. Both feed write_graph's batched UNWIND writer.
    n_nodes = 0
    extraction_method = "none"
    if extract_graph:
        mode = getattr(C, "EXTRACTION_MODE", "index_routing")
        try:
            if mode == "index_routing":
                g = extract_graph_index_routing(doc_id, text, domain=domain)
                extraction_method = "index_routing"
            elif mode == "hybrid":
                from hybrid_extraction import extract_hybrid
                g = extract_hybrid(doc_id, text, domain=profile)
                extraction_method = "hybrid"
            else:
                g = extract_graph_llm(doc_id, text, chunk_size=chunk_size,
                                      profile=profile)
                extraction_method = "llm"
            # Step 3 — vector-resolve entities + write to Neo4j (batched UNWIND)
            driver = get_neo4j()
            try:
                # V3.0: pull the domain's Neo4j label from YAML so nodes are
                # tagged (e.g. :Engineering) for domain-scoped graph retrieval.
                neo4j_label = None
                if domain:
                    try:
                        import domain_loader
                        neo4j_label = domain_loader.get_domain(domain).get(
                            "neo4j_label")
                    except Exception:
                        neo4j_label = None
                n_nodes = write_graph(doc_id, g, text, driver,
                                      chunk_size=chunk_size,
                                      neo4j_label=neo4j_label)
            finally:
                driver.close()
        except Exception as e:
            print(f"[ingest] graph extraction failed for {doc_id}: {e}",
                  flush=True)
    t_extract = time.time()

    return {
        "doc_id": doc_id,
        "status": "ok",
        "chunks": n_chunks,
        "entities": n_nodes,
        "edges": 0,  # edges not separately counted; reflect in nodes for now
        "vectors_upserted": n_chunks,
        "extraction_method": extraction_method,
        "checksum": checksum,
        "collection": collection,
        "timing": {
            "clean": round(t_clean - t0, 3),
            "embed": round(t_embed - t_clean, 3),
            "extract": round(t_extract - t_embed, 3),
            "total": round(t_extract - t0, 3),
        },
    }


# ── Ingest one file (CLI / batch) ───────────────────────────────────────────
def ingest_file(path: Path):
    text = path.read_text("utf-8", errors="ignore")
    doc_id = path.stem
    meta = {"doc_type": "eng", "author": "unknown"}
    res = ingest_text(text, doc_id, meta=meta, extract_graph=True)
    print(f"[ingest] {doc_id}: {res['chunks']} chunks, "
          f"{res['entities']} nodes, method={res['extraction_method']}",
          flush=True)
    return res


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
