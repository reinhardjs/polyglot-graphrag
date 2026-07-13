"""
bench_rag.py — pure RAG latency benchmark (no chat model, no Hermes).

Measures each stage of the pipeline separately and the full end-to-end path.
Reuses ask.py helpers so numbers reflect the real serving path.

Stages:
  embed      - query embedding via daemon /embed_query (Jina, GPU)
  qdrant     - dense+sparse hybrid vector search (top_k)
  neo4j      - k-hop subgraph lookup
  rerank     - BGE rerank of merged pool -> top_k
  synth      - E4B synthesis of top contexts (streaming)
  TOTAL      - embed -> (qdrant || neo4j) -> rerank -> synth
"""
import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "hf"))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import statistics as st

import config as C
import ask


def timeit(fn):
    t0 = time.perf_counter()
    r = fn()
    return time.perf_counter() - t0, r


def bench(query, iters=3, with_synth=True):
    print(f"\n### query: {query!r}  (iters={iters}, synth={with_synth})")
    rows = {"embed": [], "qdrant": [], "neo4j": [], "rerank": [], "synth": [], "total": []}

    for i in range(iters):
        # embed
        dt, vec = timeit(lambda: ask.embed_query(query))
        rows["embed"].append(dt)

        # parallel retrieve
        q_res, g_res = [], []
        import threading
        t1 = threading.Thread(target=lambda: q_res.extend(ask.qdrant_search(vec)))
        t2 = threading.Thread(target=lambda: g_res.extend(ask.neo4j_subgraph(query)))
        a = time.perf_counter(); t1.start(); t2.start(); t1.join(); t2.join()
        # separate qdrant vs neo4j by timing each independently
        dq, _ = timeit(lambda: ask.qdrant_search(vec)); rows["qdrant"].append(dq)
        dg, _ = timeit(lambda: ask.neo4j_subgraph(query)); rows["neo4j"].append(dg)

        # rerank (condense returns the final top-k context list)
        dr, contexts = timeit(lambda: ask.condense(query, q_res, g_res))
        rows["rerank"].append(dr)

        # synth
        if with_synth:
            ds, _ = timeit(lambda: ask.synthesize(query, contexts))
            rows["synth"].append(ds)
            total = time.perf_counter() - a + rows["embed"][-1]
        else:
            total = time.perf_counter() - a + rows["embed"][-1]
        rows["total"].append(total)

    # report
    print(f"{'stage':10} {'mean':>8} {'min':>8} {'max':>8}  (s)")
    for stage in ["embed", "qdrant", "neo4j", "rerank", "synth", "total"]:
        v = rows[stage]
        if v:
            print(f"{stage:10} {st.mean(v):8.3f} {min(v):8.3f} {max(v):8.3f}")
    return rows


if __name__ == "__main__":
    qs = [
        "who reported BUG-204 and what severity was it?",
        "How does checkout relate to billing per our ADRs?",
        "What PR implemented ADR-014 and who reviewed it?",
    ]
    for q in qs:
        bench(q, iters=3, with_synth=True)
    print("\n=== retrieval-only (no E4B synth) ===")
    for q in qs:
        bench(q, iters=3, with_synth=False)
