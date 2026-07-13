"""Phase 1 baseline drift benchmark (v3.1.0 dynamic-label plan, §4.3).

Ingests documents through the STATIC pipeline (dynamic labels start empty)
and measures, per document:
  - entities detected by GLiNER
  - edges written to Neo4j
  - entities DROPPED by _parse_and_validate (recorded to audit log)

Output: a markdown table + a summary drift %.

Run:
  <project-root>/venv/bin/python scripts/bench_drift.py \
      --domain journal --docs /tmp/arxiv_2401.18059.txt --repeat 1
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from neo4j import GraphDatabase
import ingest
import hybrid_extraction as H
from label_provider import get_provider, reset_provider


def _doc_edges(driver, doc_id):
    with driver.session() as s:
        return s.run(
            "MATCH (a)-[r]->(b) WHERE $doc IN a.source_docs "
            "RETURN count(r) AS c", doc=doc_id
        ).single()["c"]


def _doc_nodes(driver, doc_id):
    with driver.session() as s:
        return s.run(
            "MATCH (n) WHERE $doc IN n.source_docs RETURN count(n) AS c",
            doc=doc_id
        ).single()["c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="journal")
    ap.add_argument("--docs", nargs="+", required=True,
                    help="text/md files to ingest as separate docs")
    ap.add_argument("--mode", default="sliding_window",
                    choices=["hybrid", "sliding_window", "llm"])
    ap.add_argument("--repeat", type=int, default=1,
                    help="ingest each doc N times (to simulate many docs)")
    args = ap.parse_args()

    C.EXTRACTION_MODE = args.mode
    # Reset provider so dynamic labels start cold (pure static baseline).
    reset_provider(args.domain)

    driver = GraphDatabase.driver(C.NEO4J_URI,
                                  auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))

    rows = []
    total_entities = 0
    total_dropped = 0
    for path in args.docs:
        text = open(path).read()
        for i in range(args.repeat):
            doc_id = f"drift-{os.path.basename(path)}-{i}"
            ingest.delete_doc_neo4j(doc_id, driver)
            # Clear dropped-buffer + audit-log tail for this doc
            H._dropped_buffer.clear()
            t0 = time.time()
            r = ingest.ingest_text(text, doc_id, domain=args.domain,
                                   extract_graph=True)
            dt = time.time() - t0
            nodes = _doc_nodes(driver, doc_id)
            edges = _doc_edges(driver, doc_id)
            # Count dropped from the buffer (flushed during ingest)
            dropped = len(H._dropped_buffer.get(doc_id, {}).get("dropped", []))
            total_entities += nodes
            total_dropped += dropped
            rows.append((doc_id, nodes, edges, dropped, round(dt, 1)))
            print(f"{doc_id}: nodes={nodes} edges={edges} "
                  f"dropped={dropped} ({dt:.1f}s)")

    driver.close()

    # Print table
    print("\n=== BASELINE DRIFT (static labels only) ===")
    print(f"{'Doc':<40} {'Ent':>5} {'Edges':>6} {'Dropped':>8}")
    for doc_id, nodes, edges, dropped, dt in rows:
        print(f"{doc_id:<40} {nodes:>5} {edges:>6} {dropped:>8}")
    drift_pct = (total_dropped / total_entities * 100.0) if total_entities else 0
    print(f"\nTotal entities: {total_entities}")
    print(f"Total dropped:  {total_dropped}")
    print(f"Drift:          {drift_pct:.1f}%")
    if drift_pct >= 15:
        print("GATE: PROCEED to Phase 2 (dynamic label injection).")
    elif drift_pct <= 5:
        print("GATE: STOP — marginal value.")
    else:
        print("GATE: JUDGMENT CALL (5-15% drift).")


if __name__ == "__main__":
    main()
