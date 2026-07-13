"""End-to-end tests for Phase 2 (pruning) + Phase 3 (CRAG) via the live daemon.

Hits /ask on :8000. Skips automatically if the daemon is unreachable so the
unit suite stays green on machines without a running stack.

    cd /mnt/data-970-plus/rag-system
    /mnt/data-970-plus/rag-env/bin/python -m pytest tests/test_e2e_neuro_symbolic.py -q
"""
import sys, os, time, requests
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://127.0.0.1:8000"


def _daemon_up():
    try:
        return requests.get(f"{BASE}/health", timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _daemon_up(),
                                reason="rag-gpu-daemon not running on :8000")


# ── Phase 3: CRAG adaptive routing ────────────────────────────────────────────
def test_crag_factual_routes_to_graph():
    r = requests.post(f"{BASE}/ask", json={
        "query": "who reported BUG-204?", "domain": "engineering",
        "crag": True, "synthesize": False}, timeout=120)
    assert r.status_code == 200
    d = r.json()
    assert d["source"] == "crag"
    assert d["crag_route"] == "graph"        # ID lookup → graph route
    assert "crag_trace" in d
    assert d["crag_trace"][0] == "route:graph"


def test_crag_analytical_routes_to_hybrid():
    r = requests.post(f"{BASE}/ask", json={
        "query": "why did checkout stop calling the payment gateway synchronously?",
        "domain": "engineering", "crag": True, "synthesize": False}, timeout=120)
    assert r.status_code == 200
    d = r.json()
    assert d["crag_route"] == "hybrid"
    assert d["crag_grade"] in ("CORRECT", "AMBIGUOUS", "INCORRECT")


def test_crag_trace_has_evaluate_step():
    r = requests.post(f"{BASE}/ask", json={
        "query": "what depends on the payment gateway?",
        "domain": "engineering", "crag": True, "synthesize": False}, timeout=120)
    assert r.status_code == 200
    trace = r.json()["crag_trace"]
    assert any(s.startswith("evaluate:") for s in trace)


def test_crag_response_has_sources_and_contexts_meta():
    r = requests.post(f"{BASE}/ask", json={
        "query": "who reported BUG-204?", "domain": "engineering",
        "crag": True, "synthesize": False}, timeout=120)
    assert r.status_code == 200
    d = r.json()
    # Shape consistency with standard /ask (v2.6.0 contract)
    assert "sources" in d, "CRAG response missing 'sources'"
    assert "contexts_meta" in d, "CRAG response missing 'contexts_meta'"
    assert isinstance(d["sources"], dict)
    assert isinstance(d["contexts_meta"], list)


# ── Phase 2: pruning keeps graph results bounded ──────────────────────────────
def test_graph_results_bounded_by_prune_top_n():
    # A broad graph query should not return an unbounded flood of contexts.
    r = requests.post(f"{BASE}/ask", json={
        "query": "payment gateway", "domain": "engineering",
        "synthesize": False, "skip_cache": True}, timeout=120)
    assert r.status_code == 200
    d = r.json()
    # RERANK_TOP_K caps final contexts; the point is retrieval succeeded and is
    # bounded (no explosion). Contexts should be a small, finite list.
    assert 0 <= d["n_contexts"] <= 10


def test_standard_ask_still_works():
    # Regression: non-CRAG path unaffected by the new features.
    r = requests.post(f"{BASE}/ask", json={
        "query": "who reported BUG-204?", "domain": "engineering",
        "synthesize": False, "skip_cache": True}, timeout=120)
    assert r.status_code == 200
    assert "contexts" in r.json()
