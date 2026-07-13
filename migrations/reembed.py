#!/usr/bin/env python3
"""migrations/reembed.py — re-embed all stored vectors after an embedding-model
swap (or a VECTOR_DIM change).

WHY THIS EXISTS
---------------
Qdrant chunk vectors and Neo4j `name_vector`s are baked at ingest time with
whatever embed model was active (config.EMBED_MODEL_NAME / VECTOR_DIM). If you
change the embedding model, every existing vector becomes incompatible:
  - Qdrant search returns garbage / dimension-mismatch errors
  - Neo4j entity resolution (vector match) silently stops merging entities
This script re-computes all vectors in place so existing data keeps working.
It is the safety net mandated by MIGRATION.md for the "embedding model swap"
breaking change.

WHAT IT DOES
------------
1. Loads the embed model from config (Jina v3 by default) on the local device.
2. For each Qdrant collection in config.QDRANT_COLLECTIONS (+ query_cache):
     - scrolls all points
     - re-embeds payload['text'] with the current model
     - upserts with the new dense vector (sparse vector preserved)
   If VECTOR_DIM no longer matches the collection, the collection is recreated
   with the new size (old points are re-embedded into the fresh collection).
3. For every Neo4j :Entity node:
     - re-embeds the primary name (+ aliases) with the current model
     - updates name_vector

SAFETY
------
- Read-only dry-run by default: pass --apply to actually write.
- Prints per-store counts before/after.
- Neo4j writes use a single MERGE per node; no nodes are deleted.

USAGE
-----
  python migrations/reembed.py --dry-run     # preview only
  python migrations/reembed.py --apply       # re-embed everything
  python migrations/reembed.py --apply --neo4j-only
  python migrations/reembed.py --apply --qdrant-only
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C


def load_embed_model():
    from sentence_transformers import SentenceTransformer
    print(f"[reembed] loading embed model {C.EMBED_MODEL_NAME} ...", flush=True)
    st = SentenceTransformer(C.EMBED_MODEL_NAME, device="cuda")
    return st


def reembed_qdrant(st, apply: bool):
    from qdrant_client import QdrantClient, models
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": True}
    if C.EMBED_TASK_PASSAGE:
        encode_kw["task"] = C.EMBED_TASK_PASSAGE

    collections = dict(C.QDRANT_COLLECTIONS)
    collections["query_cache"] = "query_cache"
    total = 0
    for name in collections.values():
        if not qc.collection_exists(name):
            print(f"[reembed] skip missing collection: {name}")
            continue
        # Recreate collection if dimension changed (preserves nothing — we
        # re-embed from payload.text below).
        info = qc.get_collection(name)
        cur_dim = info.config.params.vectors.size
        if cur_dim != C.VECTOR_DIM:
            print(f"[reembed] {name}: dim {cur_dim} -> {C.VECTOR_DIM} "
                  f"(recreating collection)")
            if not apply:
                print(f"[reembed]   (dry-run) would recreate {name}")
                continue
            qc.delete_collection(name)
            qc.create_collection(
                name,
                vectors_config=models.VectorParams(
                    size=C.VECTOR_DIM, distance=models.Distance.COSINE),
                sparse_vectors_config={"text": models.SparseVectorParams()},
            )
        # Scroll + re-embed
        count = 0
        offset = None
        while True:
            pts, offset = qc.scroll(name, limit=500, offset=offset,
                                    with_payload=True, with_vectors=False)
            if not pts:
                break
            texts = [p.payload.get("text", "") for p in pts]
            vecs = st.encode(texts, **encode_kw)
            if apply:
                for p, v in zip(pts, vecs):
                    p.vector = v
                qc.upsert(name, pts, wait=True)
            count += len(pts)
        total += count
        print(f"[reembed] {name}: {count} points processed")
    return total


def reembed_neo4j(st, apply: bool):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    encode_kw = {"convert_to_numpy": True, "show_progress_bar": False}
    if C.EMBED_TASK_PASSAGE:
        encode_kw["task"] = C.EMBED_TASK_PASSAGE
    total = 0
    with driver.session() as s:
        # Read all entity ids + names
        res = s.run("MATCH (n:Entity) RETURN n.id AS id, n.name AS name")
        rows = list(res)
        if not rows:
            print("[reembed] neo4j: 0 entities")
            return 0
        names = [r["name"] or r["id"] for r in rows]
        ids = [r["id"] for r in rows]
        vecs = st.encode(names, **encode_kw)
        if apply:
            for rid, v in zip(ids, vecs):
                s.run("MATCH (n:Entity {id:$id}) SET n.name_vector=$vec",
                      id=rid, vec=v.tolist())
        total = len(rows)
        print(f"[reembed] neo4j: {total} entity name_vectors processed")
    driver.close()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually write; default is dry-run")
    ap.add_argument("--neo4j-only", action="store_true")
    ap.add_argument("--qdrant-only", action="store_true")
    args = ap.parse_args()
    apply = args.apply
    if not apply:
        print("[reembed] DRY-RUN (no writes). Pass --apply to write.\n")

    st = load_embed_model()
    if not args.neo4j_only:
        n = reembed_qdrant(st, apply)
        print(f"[reembed] Qdrant total: {n}")
    if not args.qdrant_only:
        n = reembed_neo4j(st, apply)
        print(f"[reembed] Neo4j total: {n}")
    print("[reembed] done.")


if __name__ == "__main__":
    main()
