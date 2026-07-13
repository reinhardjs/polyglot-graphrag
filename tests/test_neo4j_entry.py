"""Unit tests for neo4j_subgraph entry strategies (v2.6.0 REQ-6).

Neo4j + embed are mocked; we assert the entry-selection logic picks the
expected node under keyword / vector / hybrid strategies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ask as A


class _FakeRow(dict):
    pass


def _make_rows():
    # Three entities: bug-204 (technical, keyword-hits), angina (NL, vector),
    # checkout (neutral).
    return [
        {"id": "bug-204", "name": "BUG-204", "prof": "BUG-204 info", "source_doc": "d1"},
        {"id": "angina", "name": "angina", "prof": "angina info", "source_doc": "d2"},
        {"id": "checkout", "name": "checkout", "prof": "checkout info", "source_doc": "d3"},
    ]


def _patch(driver_rows, embed_sim):
    """Patch GraphDatabase.driver + ask.embed_query."""
    import neo4j
    orig_driver = neo4j.GraphDatabase.driver

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def run(self, *a, **k):
            return type("R", (), {"data": lambda *aa, **kk: driver_rows})()

    class _Driver:
        def __init__(self, *a, **k): pass
        def session(self, *a, **k): return _Sess()
        def close(self): pass
    neo4j.GraphDatabase.driver = lambda *a, **k: _Driver()

    orig_embed = A.embed_query
    A.embed_query = lambda text: embed_sim.get(text, [0.0] * 4)
    return orig_driver, orig_embed


def test_keyword_strategy_picks_overlap_node():
    rows = _make_rows()
    # embed returns tiny vectors (vector path shouldn't win for keyword)
    sim = {r["name"]: [0.1] * 4 for r in rows}
    import neo4j
    o1, o2 = _patch(rows, sim)
    try:
        out = A.neo4j_subgraph("bug-204 cascade", entry_strategy="keyword")
    finally:
        neo4j.GraphDatabase.driver = o1
        A.embed_query = o2
    # entry should be bug-204 (overlap with 'bug','204')
    assert any(r["doc_id"] == "d1" for r in out), out


def test_vector_strategy_picks_cosine_node():
    rows = _make_rows()
    # make 'angina' vector match the query embedding best
    sim = {"angina": [1.0, 0, 0, 0], "bug-204": [0, 1, 0, 0], "checkout": [0, 0, 1, 0]}
    import neo4j
    o1, o2 = _patch(rows, sim)
    try:
        out = A.neo4j_subgraph("chest pain treatment", entry_strategy="vector")
    finally:
        neo4j.GraphDatabase.driver = o1
        A.embed_query = o2
    # vector entry should be angina (highest cosine to query vec)
    assert any(r["doc_id"] == "d2" for r in out), out


def test_hybrid_falls_back_to_vector_when_no_keyword():
    rows = _make_rows()
    sim = {"angina": [1.0, 0, 0, 0], "bug-204": [0, 1, 0, 0], "checkout": [0, 0, 1, 0]}
    import neo4j
    o1, o2 = _patch(rows, sim)
    try:
        # NL query with no token overlap -> hybrid uses vector entry (angina)
        out = A.neo4j_subgraph("chest pain", entry_strategy="hybrid")
    finally:
        neo4j.GraphDatabase.driver = o1
        A.embed_query = o2
    assert any(r["doc_id"] == "d2" for r in out), out


def test_keyword_auto_fallback_to_vector():
    rows = _make_rows()
    sim = {"angina": [1.0, 0, 0, 0], "bug-204": [0, 1, 0, 0], "checkout": [0, 0, 1, 0]}
    import neo4j
    o1, o2 = _patch(rows, sim)
    try:
        # default keyword with zero overlap -> auto vector fallback -> angina
        out = A.neo4j_subgraph("chest pain", entry_strategy="keyword")
    finally:
        neo4j.GraphDatabase.driver = o1
        A.embed_query = o2
    assert any(r["doc_id"] == "d2" for r in out), out
