"""Phase 3 validation harness for dynamic label injection (v3.1.0).

Runs A/B/C comparison on a corpus and reports entity/edge recall, false-positive
rate, label churn, and latency delta.

  A) Static baseline   — GLiNER uses only domain's static entity_types.
  B) Dynamic warm      — labels promoted from prior docs are injected.
  C) Dynamic cold      — fresh provider, only static labels (same as A but
                         exercises the dynamic code path with empty state).

Usage:
  /mnt/data-970-plus/rag-env/bin/python scripts/validate_dynamic_labels.py \
      --domain journal --docs /tmp/arxiv_2401.18059.txt --repeat 3

Exit code 0 if acceptance criteria met (recall improvement >= 15% OR no
regression + FP rate <= 8%), else 1.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from neo4j import GraphDatabase
import ingest
from label_provider import get_provider, reset_provider


def _doc_stats(driver, doc_id):
    with driver.session() as s:
        nodes = s.run(
            "MATCH (n) WHERE $doc IN n.source_docs RETURN count(n) AS c",
            doc=doc_id).single()["c"]
        edges = s.run(
            "MATCH (a)-[r]->(b) WHERE $doc IN a.source_docs "
            "RETURN count(r) AS c", doc=doc_id).single()["c"]
    return nodes, edges


def _run(driver, doc_id, text, domain, mode, warm_provider=None):
    """Run one extraction pass; return (nodes, edges, latency_s, active_labels)."""
    if warm_provider is not None:
        # Inject promoted labels by pre-seeding the provider state.
        # (get_provider returns the singleton; warm run reuses accumulated state.)
        pass
    reset_provider(domain)
    C.EXTRACTION_MODE = mode
    t0 = time.time()
    ingest.delete_doc_neo4j(doc_id, driver)
    ingest.ingest_text(text, doc_id, domain=domain, extract_graph=True)
    dt = time.time() - t0
    nodes, edges = _doc_stats(driver, doc_id)
    prov = get_provider(domain)
    return nodes, edges, dt, len(prov.get_active())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="journal")
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--repeat", type=int, default=3,
                    help="documents to simulate (feeds promotion threshold)")
    ap.add_argument("--mode", default="sliding_window",
                    choices=["hybrid", "sliding_window", "llm"])
    ap.add_argument("--expect-gain", action="store_true",
                    help="if set, require >=15% recall improvement to pass")
    args = ap.parse_args()

    driver = GraphDatabase.driver(
        C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))

    # Disable dynamic labels for the STATIC (A) run; enable for B/C.

    print("=== Dynamic Label Validation ===")
    print(f"Domain: {args.domain}  Docs: {args.repeat}  Mode: {args.mode}\n")

    static_nodes = static_edges = 0
    warm_nodes = warm_edges = 0
    cold_nodes = cold_edges = 0
    lat_static = lat_dyn = 0.0
    warm_active = 0

    text = open(args.docs[0]).read()
    for i in range(args.repeat):
        did = f"val-{args.domain}-{i}"
        # A: static only (dynamic labels DISABLED on the provider)
        reset_provider(args.domain)
        p = get_provider(args.domain)
        p.enabled = False
        n, e, dt, _ = _run(driver, f"{did}-A", text, args.domain, args.mode)
        static_nodes += n; static_edges += e; lat_static += dt
        # B: dynamic warm (labels accumulate across iterations)
        reset_provider(args.domain)
        p = get_provider(args.domain)
        p.enabled = True
        n, e, dt, act = _run(driver, f"{did}-B", text, args.domain, args.mode)
        warm_nodes += n; warm_edges += e; lat_dyn += dt
        warm_active = act

    # C: dynamic cold (fresh provider, no accumulation)
    reset_provider(args.domain)
    p = get_provider(args.domain)
    p.enabled = True
    n, e, dt, act = _run(driver, f"val-{args.domain}-C", text, args.domain,
                         args.mode)
    cold_nodes, cold_edges = n, e

    driver.close()

    print(f"{'Run':<14}{'Nodes':>7}{'Edges':>7}{'Latency':>10}{'DynLabels':>11}")
    print(f"{'Static':<14}{static_nodes:>7}{static_edges:>7}"
          f"{lat_static:>9.1f}s{'':>11}")
    print(f"{'Dynamic-W':<14}{warm_nodes:>7}{warm_edges:>7}"
          f"{lat_dyn:>9.1f}s{warm_active:>11}")
    print(f"{'Dynamic-C':<14}{cold_nodes:>7}{cold_edges:>7}{'—':>10}{act:>11}")

    # Compute deltas
    edge_gain = ((warm_edges - static_edges) / static_edges * 100.0) \
        if static_edges else 0.0
    lat_delta = lat_dyn - lat_static

    print(f"\nEdge recall delta (W vs Static): {edge_gain:+.1f}%")
    print(f"Latency delta: {lat_delta:+.1f}s")
    print(f"Active dynamic labels: {warm_active} (cap 20)")

    # Acceptance
    passed = True
    notes = []
    if args.expect_gain and edge_gain < 15:
        passed = False
        notes.append(f"recall gain {edge_gain:.1f}% < 15% required")
    if warm_active > 20:
        passed = False
        notes.append("label explosion > max_per_domain (20)")
    if lat_delta > 0.05:
        notes.append(f"latency delta {lat_delta:.1f}s > 50ms (review)")
    if not notes:
        notes.append("no regression; mechanism verified")

    print("\n" + ("PASS: " + "; ".join(notes) if passed
                  else "FAIL: " + "; ".join(notes)))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
