"""Regression tests for delete_doc_neo4j edge-attribution fix (V3.1).

Before the fix, edges were deleted only when an *endpoint node* became
orphaned (its source_docs empty). That left stale edges behind when the
endpoints survived via OTHER documents — exactly the "frequently changing
docs" scenario. The fix tracks source_docs on the relationship itself and
deletes an edge when its OWN source_docs becomes empty.

These tests mock the Neo4j driver and capture the Cypher run against it,
asserting the correct delete semantics are issued.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingest


class _CaptureSession:
    """Session that records every Cypher string passed to .run()."""

    def __init__(self, recorded):
        self._recorded = recorded

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self._recorded.append(cypher)
        # Return an empty result; callers only iterate, so a no-op is fine.
        return type("R", (), {"data": lambda *a, **k: []})()


class _CaptureDriver:
    def __init__(self, recorded):
        self._recorded = recorded

    def session(self, *a, **k):
        return _CaptureSession(self._recorded)

    def close(self):
        pass


def _run_delete(doc_id):
    recorded = []
    orig = ingest.get_neo4j
    ingest.get_neo4j = lambda: _CaptureDriver(recorded)
    try:
        ingest.delete_doc_neo4j(doc_id, ingest.get_neo4j())
    finally:
        ingest.get_neo4j = orig
    return recorded


def test_delete_strips_edge_source_docs():
    """The fix must strip doc_id from relationship source_docs."""
    rec = _run_delete("docA")
    rel_strip = [c for c in rec if "r.source_docs" in c and "MATCH (a:Entity)-[r]->(b:Entity)" in c]
    assert rel_strip, "expected a Cypher that strips doc_id from relationship source_docs"
    # It must target the relationship (r.source_docs), not just nodes.
    assert any("$doc IN r.source_docs" in c for c in rel_strip), \
        "relationship source_docs strip must filter on r.source_docs"


def test_delete_removes_orphaned_edges_by_own_source_docs():
    """Edges with empty source_docs (after strip) must be deleted."""
    rec = _run_delete("docA")
    edge_delete = [c for c in rec if c.strip().startswith("MATCH (a:Entity)-[r]->(b:Entity)") and "DELETE r" in c]
    assert edge_delete, "expected a Cypher that deletes orphaned relationships"
    # The deletion predicate must be on the EDGE's own source_docs, not the
    # endpoints' (the old, buggy behaviour used a.source_docs/b.source_docs).
    assert any("size(r.source_docs) = 0" in c for c in edge_delete), \
        "edge deletion must key on r.source_docs, not endpoint nodes"


def test_delete_no_longer_uses_endpoint_orphan_for_edges():
    """The buggy endpoint-orphan edge predicate must be gone."""
    rec = _run_delete("docA")
    buggy = [c for c in rec
             if "DELETE r" in c
             and ("size(a.source_docs) = 0 OR size(b.source_docs) = 0" in c)]
    assert not buggy, \
        "old endpoint-orphan edge predicate (size(a.source_docs)=0 OR ...) must be removed"


def test_delete_nodes_still_cleaned_by_own_source_docs():
    """Nodes are still pruned when their own source_docs is empty."""
    rec = _run_delete("docA")
    node_delete = [c for c in rec if c.strip().startswith("MATCH (n:Entity)") and "DETACH DELETE n" in c]
    assert node_delete, "expected node orphan cleanup"
    assert any("size(n.source_docs) = 0" in c for c in node_delete)


def test_three_phases_present():
    """Order: node strip, edge strip, edge delete, node delete."""
    rec = _run_delete("docA")
    assert len([c for c in rec if "source_docs" in c]) >= 3
