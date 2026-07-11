"""Unit tests for Phase 3 crag_pipeline.py (no daemon; injected fake deps)."""
import crag_pipeline as CR


# ── Router ────────────────────────────────────────────────────────────────────
def test_route_id_lookup_is_graph():
    assert CR.route("who reported BUG-204?") == "graph"
    assert CR.route("status of PR-482") == "graph"
    assert CR.route("what is ADR-014") == "graph"


def test_route_analytical_is_hybrid():
    assert CR.route("why did the checkout service fail?") == "hybrid"
    assert CR.route("how do payments and billing interact over time?") == "hybrid"
    assert CR.route("compare the two caching strategies") == "hybrid"


def test_route_long_question_is_hybrid():
    q = "what are all the downstream services that depend on the gateway component"
    assert CR.route(q) == "hybrid"


# ── Evaluator ─────────────────────────────────────────────────────────────────
def test_evaluate_empty_is_incorrect():
    assert CR.evaluate_context("checkout timeout", []) == "INCORRECT"


def test_evaluate_strong_overlap_is_correct():
    g = CR.evaluate_context("checkout timeout gateway",
                            ["the checkout timeout was caused by the payment gateway"])
    assert g == "CORRECT"


def test_evaluate_no_overlap_is_incorrect():
    g = CR.evaluate_context("quantum entanglement theory",
                            ["the checkout timeout was caused by the gateway"])
    assert g == "INCORRECT"


# ── Query rewrite ─────────────────────────────────────────────────────────────
def test_rewrite_strips_stopwords_when_llm_off():
    # LLM daemon likely unreachable in unit env → heuristic path
    out = CR.rewrite_query("what is the cause of the checkout timeout?")
    assert "checkout" in out and "timeout" in out
    assert "what" not in out.split()


# ── Full FSM (fake deps) ──────────────────────────────────────────────────────
def _deps(retrieve_fn):
    return CR.CragDeps(
        embed=lambda q: [0.1] * 4,
        retrieve=retrieve_fn,
        condense=lambda q, qr, gr: [r["text"] for r in (qr + gr)],
        synth=lambda q, ctx, prof: "SYNTH:" + " | ".join(ctx),
        use_llm_router=False,
    )


def test_fsm_happy_path_no_correction():
    def retrieve(vec, q, colls, strat):
        return ([{"text": "checkout timeout gateway latency root cause", "doc_id": "d"}], [])
    out = CR.run_crag("checkout timeout gateway latency", deps=_deps(retrieve))
    assert out["corrected"] is False
    assert out["grade"] == "CORRECT"
    assert "evaluate:CORRECT" in out["path"]
    assert out["answer"].startswith("SYNTH:")


def test_fsm_corrective_path_recovers():
    calls = {"n": 0}

    def retrieve(vec, q, colls, strat):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([{"text": "unrelated banana content", "doc_id": "d1"}], [])
        return ([{"text": "checkout timeout gateway latency confirmed", "doc_id": "d2"}], [])

    out = CR.run_crag("checkout timeout gateway latency", deps=_deps(retrieve))
    assert out["corrected"] is True
    assert calls["n"] == 2
    assert out["rewritten_query"]
    assert "rewrite" in out["path"]


def test_fsm_graph_route_ignores_vector_pool():
    def retrieve(vec, q, colls, strat):
        # vector pool present but should be ignored for graph route grading
        return ([{"text": "noise", "doc_id": "x"}],
                [{"text": "BUG-204 reported by Alice high severity", "doc_id": "y"}])
    out = CR.run_crag("who reported BUG-204?", deps=_deps(retrieve))
    assert out["route"] == "graph"
    # graph context should drive the answer, not the vector noise
    assert "Alice" in out["answer"]


def test_fsm_synthesize_false_returns_no_answer():
    def retrieve(vec, q, colls, strat):
        return ([{"text": "checkout timeout gateway latency", "doc_id": "d"}], [])
    out = CR.run_crag("checkout timeout gateway latency",
                      synthesize=False, deps=_deps(retrieve))
    assert out["answer"] is None
