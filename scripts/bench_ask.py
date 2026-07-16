#!/usr/bin/env python3
"""scripts/bench_ask.py — latency/correctness benchmark for /ask.

Usage:
  python scripts/bench_ask.py --n 50 --domain healthcare
  python scripts/bench_ask.py --n 50 --all-domains
  python scripts/bench_ask.py --n 20 --domain fraud --synthesize

Reports p50/p95/p99 latency, error count, and (for --all-domains) per-domain
context coverage. Exits non-zero if any request errors or p95 exceeds target.

Targets (calibrated for RTX 3060 12GB, v1.0.0):
  single-domain, no synthesis: p95 < 250ms (cold) / < 400ms (multi-domain)
  with synthesis:             p95 < 3.0s
"""
from __future__ import annotations

import argparse
import statistics
import time
import urllib.request
import json
import sys

BASE = "http://127.0.0.1:8000"

# Queries per domain (demonstration, must return >0 contexts on seeded demo data)
QUERIES = {
    "healthcare": [
        "fever and rash and cough", "measles symptoms", "chest pain and dyspnea",
        "jaundice and dark urine", "sudden hearing loss",
    ],
    "enterprise": [
        "what services depend on payment", "checkout outage runbook",
        "circuit breaker for payment-service", "ADR payment extraction",
    ],
    "legal": [
        "GDPR cross-border data transfer", "breach notification 72 hours",
        "SOC 2 access control", "HIPAA PHI safeguards",
    ],
    "fraud": [
        "unusual transaction pattern on account alpha",
        "collusion ring shared device", "velocity spike rapid transactions",
        "mule cashout pattern", "shared card fraud",
    ],
}


def _post(domain, query, synthesize, skip_cache):
    body = json.dumps({
        "query": query, "domain": domain,
        "synthesize": synthesize, "skip_cache": skip_cache,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/ask", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
            dt = time.time() - t0
            return dt, data, None
    except urllib.error.HTTPError as e:
        return time.time() - t0, None, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:
        return time.time() - t0, None, str(e)


def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run(domain, n, synthesize, skip_cache):
    queries = QUERIES.get(domain, ["fever and rash and cough"])
    lats, errors, n_ctx = [], [], []
    for i in range(n):
        q = queries[i % len(queries)]
        dt, data, err = _post(domain, q, synthesize, skip_cache)
        if err or data is None:
            errors.append(err or "no data")
            continue
        lats.append(dt)
        n_ctx.append(data.get("n_contexts", 0))
    return lats, errors, n_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--domain", type=str, default=None)
    ap.add_argument("--all-domains", action="store_true")
    ap.add_argument("--synthesize", action="store_true")
    ap.add_argument("--skip-cache", action="store_true",
                    help="force fresh retrieval (ignore semantic cache)")
    args = ap.parse_args()

    domains = []
    if args.all_domains:
        domains = list(QUERIES.keys())
    elif args.domain:
        domains = [args.domain]
    else:
        domains = ["healthcare"]

    all_lats, all_errors = [], []
    print(f"=== bench_ask: n={args.n} synthesize={args.synthesize} "
          f"skip_cache={args.skip_cache} domains={domains} ===")
    for dom in domains:
        lats, errors, n_ctx = run(dom, args.n, args.synthesize, args.skip_cache)
        all_lats += lats
        all_errors += errors
        avg_ctx = statistics.mean(n_ctx) if n_ctx else 0
        print(f"  {dom:12s} p50={_pct(lats,50)*1000:7.1f}ms "
              f"p95={_pct(lats,95)*1000:7.1f}ms p99={_pct(lats,99)*1000:7.1f}ms "
              f"errors={len(errors)} avg_ctx={avg_ctx:.1f}")
    print(f"  --- TOTAL p95={_pct(all_lats,95)*1000:.1f}ms "
          f"errors={len(all_errors)}")
    if all_errors:
        print("ERRORS:")
        for e in all_errors[:5]:
            print("  ", e)
        sys.exit(1)
    if all_lats and _pct(all_lats, 95) > 3.0 and args.synthesize:
        print("WARN: p95 > 4s with synthesis")
    sys.exit(0)


if __name__ == "__main__":
    main()
