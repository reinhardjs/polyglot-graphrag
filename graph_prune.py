"""Phase 2 — Subgraph Pruning (Context Window Protection).

Problem
-------
A k-hop Neo4j traversal from a popular entity (e.g. a shared "Database" node)
can return dozens/hundreds of neighbours — a "neighbourhood explosion" that
floods the LLM context window and dilutes relevance.

Solution
--------
After fetching the k-hop neighbourhood *with its relationships*, build an
in-memory graph and rank nodes by structural centrality, then keep only the
Top-N. This is domain-agnostic: it operates on whatever edges exist
(`ASSOCIATED_WITH`, `AFFECTS`, …) with no hardcoded types.

Strategies (config.GRAPH_PRUNE_STRATEGY):
  "degree"   — degree centrality within the retrieved subgraph (fast, no deps).
               A node linked to more neighbours in the neighbourhood is more
               structurally central to the query's topic.
  "pagerank" — networkx PageRank if available; falls back to degree otherwise.
  "none"     — no pruning (return all, legacy behaviour).

The entry node is ALWAYS kept (rank 0) — it is the query's anchor.

Public API
----------
    rank_node_ids(entry_id, nodes, edges, strategy, top_n) -> list[str]
    prune_records(entry_id, records, edges, strategy, top_n) -> list[dict]
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def _degree_scores(node_ids: List[str],
                   edges: List[Tuple[str, str]]) -> Dict[str, int]:
    """Degree centrality: count incident edges per node within the subgraph."""
    deg: Dict[str, int] = defaultdict(int)
    node_set = set(node_ids)
    for a, b in edges:
        if a in node_set:
            deg[a] += 1
        if b in node_set:
            deg[b] += 1
    # ensure isolated nodes appear with degree 0
    for n in node_ids:
        deg.setdefault(n, 0)
    return dict(deg)


def _pagerank_scores(node_ids: List[str],
                     edges: List[Tuple[str, str]]) -> Dict[str, float]:
    """PageRank via networkx; falls back to degree if networkx is absent."""
    try:
        import networkx as nx
    except Exception:
        deg = _degree_scores(node_ids, edges)
        return {k: float(v) for k, v in deg.items()}
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for a, b in edges:
        if a in node_ids and b in node_ids:
            g.add_edge(a, b)
    if g.number_of_nodes() == 0:
        return {}
    try:
        return dict(nx.pagerank(g))
    except Exception:
        deg = _degree_scores(node_ids, edges)
        return {k: float(v) for k, v in deg.items()}


def rank_node_ids(entry_id: str,
                  node_ids: List[str],
                  edges: List[Tuple[str, str]],
                  strategy: str = "degree",
                  top_n: int = 10) -> List[str]:
    """Return the Top-N most central node ids (entry node always first).

    Ties are broken deterministically by node id so results are reproducible.
    """
    # De-dup while preserving input order.
    uniq = list(dict.fromkeys([n for n in node_ids if n]))
    if entry_id and entry_id not in uniq:
        uniq.insert(0, entry_id)

    if strategy == "none" or top_n <= 0:
        return uniq

    if strategy == "pagerank":
        scores = _pagerank_scores(uniq, edges)
    else:  # degree (default)
        scores = {k: float(v) for k, v in _degree_scores(uniq, edges).items()}

    # Entry node gets an infinite score so it is never pruned.
    def _key(nid: str):
        base = float("inf") if nid == entry_id else scores.get(nid, 0.0)
        return (base, nid)  # id as deterministic tiebreaker

    ranked = sorted(uniq, key=_key, reverse=True)
    return ranked[:top_n]


def prune_records(entry_id: str,
                  records: List[dict],
                  edges: List[Tuple[str, str]],
                  strategy: str = "degree",
                  top_n: int = 10,
                  id_key: str = "id") -> List[dict]:
    """Prune a list of node records to the Top-N central ones.

    `records` are dicts that each carry an `id_key`. Records whose id is not in
    the Top-N ranked set are dropped. Order follows the centrality ranking, with
    the entry node first.
    """
    node_ids: List[str] = [r[id_key] for r in records if r.get(id_key)]
    keep = rank_node_ids(entry_id, node_ids, edges, strategy, top_n)
    keep_order = {nid: i for i, nid in enumerate(keep)}
    kept = [r for r in records if r.get(id_key) in keep_order]
    kept.sort(key=lambda r: keep_order[r[id_key]])
    return kept
