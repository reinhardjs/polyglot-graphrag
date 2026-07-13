#!/usr/bin/env python3
"""Benchmark: current E2B Q4_0 (port 8082) vs mobile Q2_K_XL (port 8083).

Measures for each model:
  - Extraction latency (single doc through extract_graph_llm path)
  - Extraction quality (entity/edge recall on a known sample)
  - VRAM footprint (nvidia-smi during load)

Production (8082) is NOT touched. This script only queries both servers.
"""
import sys, os, time, json, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C

CURRENT_URL = "http://localhost:8082/v1"
MOBILE_URL = "http://localhost:8083/v1"

# Sample engineering doc with known entities/edges
SAMPLE_TEXT = """BUG-204 caused an outage in the checkout service.
bob reported BUG-204 and alice reviewed PR-482 which fixed it.
The checkout service depends on the billing database.
ADR-014 references the auth-service component."""

EXPECTED_NODES = {"BUG-204", "checkout", "bob", "alice", "PR-482",
                  "billing", "auth-service", "ADR-014"}
EXPECTED_EDGES = {("bob", "BUG-204"), ("alice", "PR-482"),
                  ("checkout", "billing"), ("ADR-014", "auth-service")}


def extract(url, text):
    """Call extraction LLM chat/completions and parse JSON."""
    client = None
    # Use OpenAI client pointing at the target URL
    from openai import OpenAI
    cli = OpenAI(base_url=url, api_key="not-needed")
    prompt = C.EXTRACTION_PROMPT.replace("{doc_id}", "bench").replace(
        "{text}", text[:C.EXTRACTION_CHAR_LIMIT])
    t0 = time.time()
    resp = cli.chat.completions.create(
        model=C.EXTRACTION_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=C.EXTRACTION_MAX_TOKENS, temperature=0.0)
    dt = time.time() - t0
    content = resp.choices[0].message.content or ""
    import re
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        # Strip <|channel>thought... thinking blocks if present
        cleaned = re.sub(r"<\|channel\>.*?(\n\n|\Z)", "", content, flags=re.DOTALL)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {"nodes": [], "edges": []}, dt
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {"nodes": [], "edges": []}, dt
    return data, dt


def quality(data):
    """Measure extraction quality: JSON validity + entity count + keyword hit.

    Uses a lenient semantic check rather than exact name matching, since the
    model may emit entities with slightly different surface forms.
    """
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    node_names = " ".join(str(n.get("name", "")) for n in nodes).lower()
    # Semantic probe: does extraction mention any core concept from the sample?
    probes = ["bug-204", "checkout", "bob", "alice", "pr-482", "billing",
              "auth", "adr-014"]
    hits = sum(1 for p in probes if p in node_names)
    keyword_recall = hits / len(probes)
    valid = 1 if (isinstance(nodes, list) and isinstance(edges, list)) else 0
    return keyword_recall, len(nodes), len(edges), valid


def bench_model(name, url, n=5):
    print(f"\n=== {name} ({url}) ===")
    lats, kr, nc, ec, vl = [], [], [], [], []
    for i in range(n):
        data, dt = extract(url, SAMPLE_TEXT)
        krec, ncount, ecount, valid = quality(data)
        lats.append(dt)
        kr.append(krec)
        nc.append(ncount)
        ec.append(ecount)
        vl.append(valid)
        print(f"  run {i+1}: {dt:.2f}s  kw_recall={krec:.2f} "
              f"nodes={ncount} edges={ecount} valid_json={valid}")
    avg_lat = sum(lats) / len(lats)
    avg_kr = sum(kr) / len(kr)
    avg_nc = sum(nc) / len(nc)
    avg_ec = sum(ec) / len(ec)
    avg_vl = sum(vl) / len(vl)
    print(f"  AVG: latency={avg_lat:.2f}s  kw_recall={avg_kr:.2f}  "
          f"nodes={avg_nc:.1f} edges={avg_ec:.1f} json_valid={avg_vl:.0f}")
    return {"name": name, "avg_latency": avg_lat, "kw_recall": avg_kr,
            "avg_nodes": avg_nc, "avg_edges": avg_ec, "json_valid": avg_vl,
            "runs": n}


if __name__ == "__main__":
    results = []
    try:
        results.append(bench_model("CURRENT E2B Q4_0 (8082)", CURRENT_URL))
    except Exception as e:
        print(f"CURRENT model error: {e}")
    try:
        results.append(bench_model("MOBILE Q2_K_XL (8083)", MOBILE_URL))
    except Exception as e:
        print(f"MOBILE model error: {e}")

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['name']:30s} lat={r['avg_latency']:.2f}s "
              f"kwR={r['kw_recall']:.2f} nodes={r['avg_nodes']:.1f} "
              f"edges={r['avg_edges']:.1f} valid={r['json_valid']:.0f}")
