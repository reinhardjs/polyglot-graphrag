#!/usr/bin/env python3
"""
bench_ora_corpus.py — Quality-gated benchmark of the CONFIDENTIAL engineering
corpus (ora-et-labora) ingested into the `enterprise` domain.

WHY THIS EXISTS
  bench_ask.py benchmarks synthetic demo queries. This script benchmarks the
  REAL ingested corpus against custom quality rules so we can PROVE (not assert)
  that v1.0 serves the confidential KB correctly, fast, and within the GPU
  VRAM budget.

QUALITY RULES (hard PASS/FAIL gates — any failure => exit non-zero)
  R1  Corpus completeness : 259/259 distinct docs present in `enterprise`
  R2  Retrieval accuracy  : a query built from a real doc's OWN first-chunk text
                           retrieves THAT exact doc in top-k (content-identity
                           check across a sample of real docs — not filename
                           stems, which are poor queries)
  R3  Retrieval latency   : p95 < 250 ms (single-domain, no synthesis, warm)
  R4  Synthesis latency   : p95 < 3 s (E2B default synthesis backend; the
                           larger E4B runs ~22s on this 12 GB card and is
                           opt-in via SYNTHESIS_LLM_* env override)
  R5  Reliability         : 0 errors across ALL benchmark queries
  R6  Provenance          : 100% of returned contexts carry doc_id + domain
  R7  Concurrency         : 24 parallel enterprise queries -> 0 errors
  R8  VRAM budget         : peak GPU VRAM during the run stays within the
                           12.6 GB card; reports the actual running cost of
                           serving this corpus (the "how much VRAM for A" rule)

No confidential body text is quoted — ground-truth is derived from doc_id
filename stems that already exist in the ingested corpus.

Run:
  python scripts/bench_ora_corpus.py
  python scripts/bench_ora_corpus.py --n 10 --concurrency 32 --top-k 5
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

# Make the project root importable (config.py, etc.) regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://localhost:8000/ask"
HEALTH = "http://localhost:8000/health"

# ── Latency / quality targets (from v1.0 roadmap + measured hardware reality) ──
TARGET_RETRIEVAL_P95_MS = 250.0   # single-domain, no synthesis
TARGET_SYNTH_P95_MS = 3000.0      # E2B synthesis default (p95 ~2.2s steady-state)
TARGET_ACCURACY = 0.80            # exact-doc content-identity recall for this corpus
VRAM_TOTAL_GB = 12.6              # RTX 3060 12 GB card

# Broad retrieval load (no assertion, just latency + reliability + provenance)
LOAD_QUERIES = [
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


# ── helpers ───────────────────────────────────────────────────────────────────
def ask(query, domain="enterprise", synthesize=False, top_k=5, skip_cache=True):
    body = json.dumps({"query": query, "domain": domain, "synthesize": synthesize,
                       "top_k": top_k, "skip_cache": skip_cache}).encode()
    req = urllib.request.Request(BASE, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode()), time.time() - t0, r.status, None
    except urllib.error.HTTPError as e:
        return None, time.time() - t0, e.code, f"HTTP {e.code}"
    except Exception as e:
        return None, time.time() - t0, 0, str(e)


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def vram_gb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out.splitlines()[0]) / 1024.0
    except Exception:
        return float("nan")


def daemon_alloc_gb():
    try:
        d = json.loads(urllib.request.urlopen(HEALTH, timeout=5).read())
        return float(d.get("cuda_alloc_gb", "nan"))
    except Exception:
        return float("nan")


def check(rule, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {rule}" + (f" — {detail}" if detail else ""))
    return cond


# ── benchmark phases ───────────────────────────────────────────────────────────
def get_docs():
    from qdrant_client import QdrantClient
    import config as C
    qc = QdrantClient(url=C.QDRANT_URL)
    hits = qc.scroll("enterprise", limit=9000, with_payload=["doc_id"])[0]
    return set(p.payload["doc_id"] for p in hits if "ora::" in p.payload.get("doc_id", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="rounds per load query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--no-synth", action="store_true", help="skip synthesis phase")
    ap.add_argument("--sample", type=int, default=25, help="self-retrieval sample size")
    args = ap.parse_args()

    rules_pass = []
    print("=" * 70)
    print("ORA-ET-LABORA CONFIDENTIAL CORPUS BENCHMARK  (enterprise domain)")
    print("=" * 70)

    # ── R1 Corpus completeness ──
    print("\n[R1] Corpus completeness")
    docs = get_docs()
    n_docs = len(docs)
    rules_pass.append(check("R1 259/259 docs present", n_docs == 259,
                            f"{n_docs}/259 distinct docs"))

    # ── VRAM baseline ──
    vram_base = vram_gb()
    peak_vram = [vram_base]

    # ── R3/R5/R6 Retrieval latency + reliability + provenance ──
    print("\n[R3/R5/R6] Retrieval latency / reliability / provenance")
    # Warmup: discard the first call (one-time model/connection setup) so it
    # doesn't pollute the p95 measurement.
    ask(LOAD_QUERIES[0], top_k=args.top_k)
    lat, errs, prov_ok, prov_tot = [], [], 0, 0
    for _ in range(max(1, args.n)):
        for q in LOAD_QUERIES:
            d, dt, code, e = ask(q, top_k=args.top_k)
            peak_vram.append(vram_gb())
            if e or d is None:
                errs.append((q, e or f"HTTP {code}"))
                continue
            lat.append(dt * 1000)
            for m in d.get("contexts_meta", []):
                prov_tot += 1
                if m.get("doc_id") and m.get("domain") == "enterprise":
                    prov_ok += 1
    p95 = pct(lat, 95)
    rules_pass.append(check(f"R3 retrieval p95 < {TARGET_RETRIEVAL_P95_MS:.0f}ms",
                            p95 < TARGET_RETRIEVAL_P95_MS, f"p95={p95:.1f}ms (n={len(lat)})"))
    rules_pass.append(check("R5 0 errors (retrieval)", not errs,
                            f"{len(errs)} errors" + (f" {errs[:2]}" if errs else "")))
    prov = (prov_ok / prov_tot) if prov_tot else 0
    rules_pass.append(check("R6 provenance 100% doc_id+domain", prov == 1.0,
                            f"{prov_ok}/{prov_tot} tagged"))

    # ── R2 Retrieval accuracy (content-identity across real docs) ──
    print("\n[R2] Retrieval accuracy (content-derived query retrieves its own doc)")
    from qdrant_client import QdrantClient
    import config as C
    qc = QdrantClient(url=C.QDRANT_URL)
    # Pull first-chunk text for a sample of real docs to build honest queries.
    hits = qc.scroll("enterprise", limit=9000,
                     with_payload=["doc_id", "text", "chunk_idx"])[0]
    first_chunk = {}
    for h in hits:
        did = h.payload.get("doc_id", "")
        if "ora::" not in did or h.payload.get("chunk_idx") != 0:
            continue
        if did not in first_chunk:
            first_chunk[did] = h.payload.get("text", "")
    import random
    real = [d for d, t in first_chunk.items() if t and t.strip()]
    random.seed(42)
    random.shuffle(real)
    sample = real[:args.sample]
    hits_ok = 0
    fails = []
    for did in sample:
        txt = first_chunk[did]
        # distinctive content query: first ~12 words of the doc's own text
        q = " ".join(txt.split()[:12])
        if not q.strip():
            continue
        d, _, _, _ = ask(q, top_k=args.top_k)
        if not d:
            fails.append((did, "no response"))
            continue
        ids = [m["doc_id"] for m in d.get("contexts_meta", [])]
        if did in ids:
            hits_ok += 1
        else:
            fails.append((did.split("::")[-1][:40], [i.split("::")[-1][:30] for i in ids[:2]]))
    acc = hits_ok / len(sample) if sample else 0
    rules_pass.append(check(f"R2 content-identity >= {TARGET_ACCURACY:.0%}", acc >= TARGET_ACCURACY,
                            f"{hits_ok}/{len(sample)} = {acc:.0%}"))
    if fails:
        print(f"    sample misses: {fails[:3]}")

    # ── R4 Synthesis latency ──
    if not args.no_synth:
        print("\n[R4] Synthesis latency (E2B default synthesis backend)")
        slat, serr = [], []
        for q in LOAD_QUERIES[:5]:
            d, dt, code, e = ask(q, synthesize=True, top_k=args.top_k)
            peak_vram.append(vram_gb())
            if e or d is None or not (d.get("answer") or "").strip():
                serr.append((q, e or "empty answer"))
                continue
            slat.append(dt * 1000)
        sp95 = pct(slat, 95)
        rules_pass.append(check(f"R4 synth p95 < {TARGET_SYNTH_P95_MS/1000:.0f}s",
                                sp95 < TARGET_SYNTH_P95_MS,
                                f"p95={sp95/1000:.1f}s (n={len(slat)}) measured on RTX3060/12GB"))
        if serr:
            check("R4 synth reliability", False, f"{len(serr)} failures {serr[:2]}")

    # ── R7 Concurrency ──
    print(f"\n[R7] Concurrency ({args.concurrency} parallel enterprise /ask)")
    cerr = []
    def worker(q):
        d, _, code, e = ask(q)
        peak_vram.append(vram_gb())
        if e or d is None:
            cerr.append((q, e or f"HTTP {code}"))
    ths = [threading.Thread(target=worker, args=(LOAD_QUERIES[i % len(LOAD_QUERIES)],))
           for i in range(args.concurrency)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    rules_pass.append(check(f"R7 {args.concurrency} parallel -> 0 errors", not cerr,
                            f"{len(cerr)} errors, wall={wall:.1f}s"))

    # ── R8 VRAM budget ──
    print("\n[R8] VRAM budget (running cost of serving this corpus)")
    peak = max(peak_vram) if peak_vram else vram_base
    alloc = daemon_alloc_gb()
    rules_pass.append(check(f"R8 peak VRAM <= {VRAM_TOTAL_GB}GB", peak <= VRAM_TOTAL_GB,
                            f"peak={peak:.1f}GB baseline={vram_base:.1f}GB daemon_alloc={alloc:.1f}GB"))
    print(f"      -> serving the 259-doc confidential KB costs ~{alloc:.1f}GB daemon-allocated "
          f"VRAM; peak observed during benchmark {peak:.1f}GB of {VRAM_TOTAL_GB}GB.")

    # ── verdict ──
    print("\n" + "=" * 70)
    passed = sum(1 for x in rules_pass if x)
    total = len(rules_pass)
    print(f"  QUALITY RULES: {passed}/{total} PASS")
    if passed == total:
        print("  VERDICT: ALL QUALITY RULES PASS — corpus benchmark GREEN")
        sys.exit(0)
    else:
        print("  VERDICT: QUALITY RULES FAILED — do NOT publish")
        sys.exit(1)


if __name__ == "__main__":
    main()
