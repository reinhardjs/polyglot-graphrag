"""Unit tests for Phase 2 graph_prune.py (no daemon, no Neo4j)."""
import graph_prune as GP


# Topology: entry 'a' central hub; b-c-d form a cluster; e,f isolated leaves.
EDGES = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("c", "d")]
RECS = [{"id": x, "prof": f"profile {x}"} for x in ["a", "b", "c", "d", "e", "f"]]


def test_entry_node_always_first_and_kept():
    keep = GP.rank_node_ids("a", ["a", "b", "c", "d", "e", "f"], EDGES,
                            strategy="degree", top_n=3)
    assert keep[0] == "a"
    assert len(keep) == 3


def test_degree_prefers_central_nodes():
    # c has degree 3 (a,b,d), d has degree 2 — both beat isolated e,f
    keep = GP.rank_node_ids("a", ["a", "b", "c", "d", "e", "f"], EDGES,
                            strategy="degree", top_n=4)
    assert "c" in keep
    assert "e" not in keep and "f" not in keep


def test_strategy_none_returns_all():
    keep = GP.rank_node_ids("a", ["a", "b", "c", "d", "e", "f"], EDGES,
                            strategy="none", top_n=2)
    assert len(keep) == 6


def test_pagerank_falls_back_gracefully():
    # pagerank should run (networkx installed) or fall back to degree
    keep = GP.rank_node_ids("a", ["a", "b", "c", "d", "e", "f"], EDGES,
                            strategy="pagerank", top_n=3)
    assert keep[0] == "a"
    assert len(keep) == 3


def test_prune_records_drops_low_centrality():
    pruned = GP.prune_records("a", RECS, EDGES, strategy="degree", top_n=3)
    ids = [r["id"] for r in pruned]
    assert ids[0] == "a"
    assert len(pruned) == 3
    assert "e" not in ids and "f" not in ids


def test_prune_records_preserves_ranking_order():
    pruned = GP.prune_records("a", RECS, EDGES, strategy="degree", top_n=4)
    ids = [r["id"] for r in pruned]
    # entry first, then descending centrality
    assert ids[0] == "a"


def test_empty_inputs_safe():
    assert GP.rank_node_ids("", [], [], "degree", 5) == []
    assert GP.prune_records("", [], [], "degree", 5) == []


def test_top_n_zero_returns_all():
    keep = GP.rank_node_ids("a", ["a", "b", "c"], EDGES, "degree", 0)
    assert set(keep) == {"a", "b", "c"}
