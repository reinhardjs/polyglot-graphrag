#!/usr/bin/env python3
"""
bench_synth_compare.py — Compare synthesize:false (retrieval-only) vs
synthesize:true (LLM answer) on the REAL ingested corpus,
using whichever synthesis backend the LIVE daemon is configured with.

Run it twice to compare backends:
  # with E4B (default, :8084):
  python scripts/bench_synth_compare.py
  # then restart daemon with E2B synthesis and re-run:
  SYNTHESIS_LLM_BASE_URL=http://localhost:8082/v1 \
  SYNTHESIS_LLM_MODEL=gemma-4-E2B_q4_0-it.gguf python serve_gpu.py
  python scripts/bench_synth_compare.py --label e2b

Reports p50/p95 for both modes + which synthesis backend is live, and asserts
the <3s synthesis target (TARGET_SYNTH_P95_MS).
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://localhost:8000/ask"
HEALTH = "http://localhost:8000/health"
TARGET_SYNTH_P95_MS = 3000.0

# Realistic enterprise queries (derived from corpus structure, no confidential text)
QUERIES = [
    "how to handle a production incident",
    "cassandra repair procedure",
    "deployment rollback steps",
    "on-call escalation path",
    "postmortem template structure",
    "debugging distributed system",
    "data consistency verification",
    "security hardening guideline",
    "capacity planning framework",
    "redis overload investigation",
]


def call(query, synthesize, top_k=5):
    body = json.dumps({"query": query, "domain": "enterprise",
                       "synthesize": synthesize, "top_k": top_k,
                       "skip_cache": True}).encode()
    req = urllib.request.Request(BASE, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
            return d, time.time() - t0, None
    except Exception as e:
        return None, time.time() - t0, str(e)


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def live_backend():
    try:
        d = json.loads(urllib.request.urlopen(HEALTH, timeout=5).read())
        b = d.get("backends", {})
        return "e4b" if b.get("e4b") == "ok" else "?"
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="backend label for reporting")
    ap.add_argument("--n", type=int, default=1,
                    help="rounds per query (default 1 = realistic single-user "
                         "load; higher -n hammers the shared E2B extraction+"
                         "synthesis backend and shows contention, not steady-state)")
    args = ap.parse_args()

    label = args.label or live_backend()
    print("=" * 64)
    print(f"SYNTHESIS COMPARE BENCHMARK — backend={label.upper()} (ingested corpus)")
    print("=" * 64)

    # Warmup (discard first call: model/conn setup)
    call(QUERIES[0], False)
    call(QUERIES[0], True)

    false_lat, true_lat, true_errs, true_lens = [], [], [], []
    for rnd in range(args.n):
        for q in QUERIES:
            d, dt, e = call(q, False)
            if not e and d:
                false_lat.append(dt * 1000)
            d, dt, e = call(q, True)
            if e or not d:
                true_errs.append((q, e))
            else:
                true_lat.append(dt * 1000)
                true_lens.append(len(d.get("answer", "") or ""))

    f_p95 = pct(false_lat, 95)
    t_p95 = pct(true_lat, 95)
    t_p50 = pct(true_lat, 50)
    avg_len = statistics.mean(true_lens) if true_lens else 0

    print(f"\n  synthesize:false (retrieval-only):")
    print(f"    p50={pct(false_lat,50):.0f}ms  p95={f_p95:.0f}ms  (n={len(false_lat)})")
    print(f"\n  synthesize:true (LLM answer, backend={label.upper()}):")
    print(f"    p50={t_p50:.0f}ms  p95={t_p95:.0f}ms  (n={len(true_lat)})")
    print(f"    avg answer length: {avg_len:.0f} chars")
    if true_errs:
        print(f"    ERRORS: {len(true_errs)} {true_errs[:2]}")

    synth_ok = t_p95 < TARGET_SYNTH_P95_MS and not true_errs
    print(f"\n  TARGET synthesis p95 < {TARGET_SYNTH_P95_MS/1000:.0f}s: "
          f"{'PASS' if synth_ok else 'FAIL'} (measured {t_p95/1000:.1f}s)")
    print(f"  Speedup synthesize:true vs false: {f_p95/t_p95:.1f}x slower"
          if t_p95 > 0 else "")
    print(f"\n  BACKEND={label.upper()}  synth_p95={t_p95/1000:.1f}s  "
          f"retrieval_p95={f_p95:.0f}ms  errors={len(true_errs)}")

    sys.exit(0 if synth_ok else 1)


if __name__ == "__main__":
    main()
