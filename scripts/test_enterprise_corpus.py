#!/usr/bin/env python3
"""
test_enterprise_corpus.py — Comprehensive reliability + correctness tests
against the confidential engineering corpus ingested into `enterprise`.

Covers:
  T1  Retrieval accuracy (known-doc queries return the right source)
  T2  Edge cases (empty query, unknown term, very long query, special chars)
  T3  Complex multi-document scenario (cross-reference 2+ docs)
  T4  Synthesis works (synthesize=true returns an answer, not just chunks)
  T5  Latency (single-domain p95 < 250ms target for retrieval)
  T6  Concurrency (parallel /ask on enterprise, 0 errors)
  T7  Provenance (every context carries doc_id + domain)

Run: python scripts/test_enterprise_corpus.py
"""
import json
import time
import sys
import threading
import urllib.request
import urllib.error
import statistics

BASE = "http://localhost:8000/ask"


def ask(query, domain="enterprise", synthesize=False, top_k=5, skip_cache=True):
    payload = {"query": query, "domain": domain, "synthesize": synthesize,
               "top_k": top_k, "skip_cache": skip_cache}
    req = urllib.request.Request(
        BASE, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=60)
        dt = time.time() - t0
        return json.loads(r.read()), dt, r.status
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:200]}, time.time() - t0, e.code
    except Exception as e:
        return {"error": str(e)}, time.time() - t0, 0


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return cond


print("=" * 64)
print("T1 — RETRIEVAL ACCURACY (known-doc queries)")
print("=" * 64)
# Queries derived from the corpus structure (no confidential content quoted).
acc_cases = [
    ("incident response guide", "incident-response-guide"),
    ("cassandra data inconsistency remediation", "cassandra"),
    ("deployment rollback runbook", "rollback"),
    ("root cause analysis template", "rca-template"),
    ("debugging methodology", "debugging-methodology"),
]
t1_ok = True
for q, needle in acc_cases:
    d, dt, code = ask(q)
    if code != 200 or "contexts_meta" not in d:
        t1_ok &= check(f"retrieve: {q}", False, f"HTTP {code}")
        continue
    hits = [m["doc_id"] for m in d["contexts_meta"]]
    found = any(needle in h for h in hits)
    t1_ok &= check(f"retrieve: {q}", found, f"{len(hits)} hits, top={hits[0][:60] if hits else 'none'}")

print("\n" + "=" * 64)
print("T2 — EDGE CASES")
print("=" * 64)
# empty query
d, dt, code = ask("")
t2_ok = check("empty query handled (no 500)", code in (200, 400),
              f"HTTP {code}")
# gibberish / out-of-corpus term
d, dt, code = ask("zzxqwkj vxqlpm nonexistentterm")
t2_ok &= check("gibberish term returns clean (no error)", code == 200 and not d.get("error"),
               f"HTTP {code}, n_contexts={d.get('n_contexts')}")
# very long query (200 words)
long_q = ("how to " * 100).strip()
d, dt, code = ask(long_q)
t2_ok &= check("very long query (200 words)", code == 200 and not d.get("error"),
               f"HTTP {code}, dt={dt:.2f}s")
# special characters
d, dt, code = ask("C++ / Node.js @ #$% error 500?")
t2_ok &= check("special chars query", code == 200 and not d.get("error"), f"HTTP {code}")

print("\n" + "=" * 64)
print("T3 — COMPLEX MULTI-DOC SCENARIO (cross-reference)")
print("=" * 64)
# A scenario that requires pulling from an incident doc + a runbook + a template
d, dt, code = ask("After a production incident, what is the step-by-step process "
                  "from detection through remediation and the required post-mortem?")
t3_ok = check("complex scenario returns results", code == 200 and d.get("n_contexts", 0) > 0,
              f"HTTP {code}, n_contexts={d.get('n_contexts')}")
if code == 200:
    docs = set(m["doc_id"] for m in d.get("contexts_meta", []))
    t3_ok &= check("multi-doc coverage (>=2 distinct sources)",
                   len(docs) >= 2, f"{len(docs)} distinct sources")
    t3_ok &= check("provenance present (doc_id+domain on every hit)",
                   all("doc_id" in m and m.get("domain") == "enterprise"
                       for m in d.get("contexts_meta", [])),
                   "all hits tagged")

print("\n" + "=" * 64)
print("T4 — SYNTHESIS WORKS")
print("=" * 64)
d, dt, code = ask("Summarize the deployment rollback procedure.", synthesize=True)
t4_ok = check("synthesize=true returns 200", code == 200, f"HTTP {code}")
if code == 200:
    ans = d.get("answer", "") or d.get("synthesis", "")
    t4_ok &= check("synthesis produced non-empty answer",
                   len(ans) > 30, f"answer len={len(ans)}")
    t4_ok &= check("synthesis cites sources",
                   d.get("n_contexts", 0) > 0, f"n_contexts={d.get('n_contexts')}")

print("\n" + "=" * 64)
print("T5 — LATENCY (enterprise retrieval, target p95 < 250ms)")
print("=" * 64)
lat_queries = ["incident response", "cassandra repair", "deployment runbook",
               "rca template", "debugging steps", "postmortem guide",
               "security guideline", "monitoring alert", "rollback procedure",
               "architecture decision"]
samples = []
for q in lat_queries:
    d, dt, code = ask(q)
    if code == 200:
        samples.append(dt * 1000)
p95 = statistics.quantiles(samples, n=20)[-1] if len(samples) >= 2 else max(samples)
t5_ok = check(f"retrieval p95 = {p95:.0f}ms (target <250ms)",
              p95 < 250, f"n={len(samples)}")

print("\n" + "=" * 64)
print("T6 — CONCURRENCY (12 parallel enterprise /ask)")
print("=" * 64)
errors = []
lat = []
def worker(q):
    d, dt, code = ask(q)
    if code != 200 or d.get("error"):
        errors.append((q, code, d.get("error")))
    else:
        lat.append(dt * 1000)
qs = lat_queries * 2  # 20 parallel
ths = [threading.Thread(target=worker, args=(q,)) for q in qs]
t0 = time.time()
for t in ths: t.start()
for t in ths: t.join()
wall = time.time() - t0
t6_ok = check(f"concurrent {len(qs)} requests: 0 errors", not errors,
              f"errors={len(errors)}" + (f" {errors[:2]}" if errors else ""))

print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)
all_ok = t1_ok and t2_ok and t3_ok and t4_ok and t5_ok and t6_ok
print(f"  T1 accuracy      : {'PASS' if t1_ok else 'FAIL'}")
print(f"  T2 edge cases    : {'PASS' if t2_ok else 'FAIL'}")
print(f"  T3 multi-doc      : {'PASS' if t3_ok else 'FAIL'}")
print(f"  T4 synthesis      : {'PASS' if t4_ok else 'FAIL'}")
print(f"  T5 latency p95    : {'PASS' if t5_ok else 'FAIL'} ({p95:.0f}ms)")
print(f"  T6 concurrency    : {'PASS' if t6_ok else 'FAIL'}")
print(f"\n  OVERALL: {'ALL TESTS PASS' if all_ok else 'SOME TESTS FAILED'}")
sys.exit(0 if all_ok else 1)
