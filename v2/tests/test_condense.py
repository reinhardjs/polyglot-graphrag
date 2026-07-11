"""Unit tests for ask.condense() with dict records (v2.6.0 citation work).

condense() must accept dict records {text,doc_id,...} (from qdrant_search /
neo4j_subgraph) and dedupe by (doc_id, text). The old dict.fromkeys() raised
TypeError on unhashable dicts — this guards against regression.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
import ask as A


def test_condense_dedupes_dict_records():
    q = [{"text": "alpha", "doc_id": "d1", "doc_type": "eng", "chunk_idx": 0},
         {"text": "beta", "doc_id": "d2", "doc_type": "eng", "chunk_idx": 1}]
    g = [{"text": "alpha", "doc_id": "d1", "doc_type": "eng", "chunk_idx": 0}]  # dup
    # monkeypatch rerank to return identity ranking
    orig_post = __import__("requests").post

    class _R:
        def json(self):
            return {"ranked": [[0, 0.9], [1, 0.8]]}
    __import__("requests").post = lambda *a, **k: _R()
    try:
        out = A.condense("q", q, g)
    finally:
        __import__("requests").post = orig_post
    assert out == ["alpha", "beta"], out  # dup removed, order kept


def test_condense_empty():
    assert A.condense("q", [], []) == []
