#!/usr/bin/env python3
"""WS2 (E2B Server Tuning) benchmark harness.

Measures the RAW extraction latency via ingest.extract_graph_llm (verbatim
entity names) and compares raw edges to the GOLD set for exact-match
precision/recall. This isolates server-flag impact (latency + VRAM) from the
constant Jina vector-resolution/write step.

Also captures VRAM via nvidia-smi and the llama-server throughput line
('tg = X t/s', 'pp = Y t/s') from the server log for context.

Usage:
    python bench_ws2.py --label B --short /tmp/prove_extraction.md \
        [--long /tmp/long_test_doc.md] [--runs 2] [--log /tmp/e2b-tune.log]

Appends a row per run to /tmp/ws2_results.jsonl.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ingest
import config as C
from openai import OpenAI

GOLD = {
    ("PR-482", "FIXES", "BUG-204"),
    ("PR-482", "AUTHORED", "bob"),
    ("alice", "REVIEWED", "PR-482"),
    ("checkout-publisher", "DEPENDS_ON", "checkout database"),
    ("ADR-014", "REFERENCES", "auth-service"),
}


def get_profile():
    return C.load_domain_profile("engineering")


def raw_extract(text, profile):
    """Call E2B directly with robust JSON extraction (mirrors ingest path but
    never falls back to GLiNER on trailing-prose parse errors — those are
    model/prompt noise, not flag effects)."""
    prompt_tmpl = (profile.get("extraction", {}).get("prompt")
                   if profile else None) or C.EXTRACTION_PROMPT
    prompt = (prompt_tmpl.replace("{doc_id}", "ws2")
              .replace("{text}", text[:C.EXTRACTION_CHAR_LIMIT]))
    client = OpenAI(base_url=C.EXTRACTION_LLM_BASE_URL, api_key=C.EXTRACTION_LLM_API_KEY)
    resp = client.chat.completions.create(
        model=C.EXTRACTION_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=C.EXTRACTION_MAX_TOKENS, temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    # Find the outermost { ... } object, robust to trailing prose.
    start = content.find("{")
    if start == -1:
        return {"nodes": [], "edges": []}
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        end = len(content)
    try:
        data = json.loads(content[start:end])
    except Exception:
        return {"nodes": [], "edges": []}
    if "nodes" in data and "edges" in data:
        return data
    return {"nodes": [], "edges": []}


def norm_edge(e):
    s = (e.get("source") or e.get("source_name") or "").strip()
    t = (e.get("target") or e.get("target_name") or "").strip()
    r = (e.get("type") or e.get("relation_type") or "").strip()
    return (s.lower(), r.upper(), t.lower())


def quality(edges):
    got = {norm_edge(e) for e in edges}
    gold = {(s.lower(), r.upper(), t.lower()) for (s, r, t) in GOLD}
    tp = got & gold
    precision = len(tp) / len(got) if got else 0.0
    recall = len(tp) / len(gold) if gold else 0.0
    return precision, recall, sorted(tp)


def vram_used():
    out = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader").read()
    try:
        return int(out.strip().split()[0])
    except Exception:
        return -1


def parse_throughput(logpath):
    if not logpath or not os.path.exists(logpath):
        return {}
    txt = open(logpath).read()
    out = {}
    for key, pat in (("tg", r"tg\s*=\s*([\d.]+)"),
                     ("pp", r"pp\s*=\s*([\d.]+)"),
                     ("tps", r"([\d.]+)\s*tokens/s")):
        m = re.findall(pat, txt)
        if m:
            out[key] = m[-1]
    return out


def bench_one(label, text, doc_id, run_idx, profile, logpath):
    t0 = time.time()
    g = raw_extract(text, profile)
    wall = (time.time() - t0) * 1000.0
    nodes = g.get("nodes", [])
    edges = g.get("edges", [])
    prec, rec, tp = quality(edges)
    edge_types = sorted({norm_edge(e)[1] for e in edges})
    return {
        "label": label,
        "doc_id": doc_id,
        "run": run_idx,
        "nodes": len(nodes),
        "edges": len(edges),
        "edge_types": edge_types,
        "extract_ms": round(wall, 1),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "gold_hits": tp,
        "vram_mib": vram_used(),
        "throughput": parse_throughput(logpath),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--long")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--log", default="/tmp/e2b-tune.log")
    ap.add_argument("--out", default="/tmp/ws2_results.jsonl")
    a = ap.parse_args()

    profile = get_profile()
    short_text = open(a.short).read()
    long_text = open(a.long).read() if a.long else None

    rows = []
    for i in range(a.runs):
        doc_id = f"ws2-{a.label.lower()}-short-{i}"
        rows.append(bench_one(a.label, short_text, doc_id, i, profile, a.log))
    if long_text:
        rows.append(bench_one(a.label, long_text, f"ws2-{a.label.lower()}-long-0",
                              0, profile, a.log))

    with open(a.out, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(rows, indent=2))
    last_short = rows[a.runs - 1]
    tp = last_short["gold_hits"]
    print(
        f"SUMMARY label={a.label} warm_extract_ms={last_short['extract_ms']} "
        f"edges={last_short['edges']} types={last_short['edge_types']} "
        f"prec={last_short['precision']} recall={last_short['recall']} "
        f"gold_hits={tp} vram={last_short['vram_mib']}MiB "
        f"tp={last_short['throughput']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
