"""domains/fraud/retrieve.py — Fraud detection retrieval (graph pattern + dense).

Fraud answers two kinds of queries:
  1. Graph pattern queries — "unusual transaction pattern on account X" — via
     Neo4j: find accounts with shared cards/devices, rapid fund movement,
     amount outliers. Returns ranked risk signals.
  2. Dense semantic — fraud narrative matching via the `fraud` Qdrant companion
     (seeded synthetic fraud-case writeups).

For v1.0 the primary path is the graph retriever; `fraud` companion is the
dense prose. This file implements the graph retriever. It reuses the resident
Neo4j driver singleton pattern and degrades gracefully if Neo4j is unavailable.
"""
from __future__ import annotations

from typing import Optional

import config as C
from neo4j import GraphDatabase

_DRIVER = None
_DRIVER_LOCK = None


def _driver():
    global _DRIVER, _DRIVER_LOCK
    if _DRIVER_LOCK is None:
        import threading
        _DRIVER_LOCK = threading.Lock()
    with _DRIVER_LOCK:
        if _DRIVER is None:
            _DRIVER = GraphDatabase.driver(
                C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    return _DRIVER


# Fraud-signal Cypher: given a query, pull accounts mentioned (by id) and
# surface their risk context. Kept simple + bounded so it always returns fast.
def _graph_fraud_signals(query: str, top_k: int = 5) -> list:
    ql = query.lower()
    # crude account id extraction: look for "account <id>" or "acct <id>"
    import re
    acct = None
    m = re.search(r"(?:account|acct)\s+([a-z0-9_]+)", ql)
    if m:
        acct = m.group(1)

    out = []
    try:
        with _driver().session() as s:
            if acct:
                # Co-use ring detection: an account shares a DEVICE or CARD with
                # peers. The graph models this as
                #   (Account)-[:SHARED_DEVICE]->(Device)<-[:SHARED_DEVICE]-(Account)
                # so we traverse THROUGH the Device/Card node to find peer accounts.
                rec = s.run(
                    "MATCH (a:Account {id:$acct})-[r:SHARED_DEVICE|SHARED_CARD]->"
                    "(hub) <-[:SHARED_DEVICE|SHARED_CARD]- (peer:Account) "
                    "WHERE peer.id <> $acct "
                    "RETURN DISTINCT peer.id AS peer, "
                    "type(r) AS rel LIMIT $k",
                    acct=acct, k=top_k,
                ).data()
                for row in rec:
                    kind = row["rel"].split("_")[-1].lower()
                    out.append({
                        "text": f"[FRAUD SIGNAL] Account {acct} shares "
                                f"{kind} with account {row['peer']} "
                                f"(possible collusion ring).",
                        "doc_id": f"fraud:{acct}:{row['peer']}",
                        "doc_type": "fraud_graph",
                        "chunk_idx": -1,
                        "concept_id": None,
                    })
            # Always surface known fraud-pattern narratives from the Qdrant
            # companion (handled by federated via the `fraud` companion). Here
            # we also match any Account whose id appears in the query against
            # its known transaction count as a velocity hint.
            if acct:
                cnt = s.run(
                    "MATCH (a:Account {id:$acct})-[r:SENT]->(t:Transaction) "
                    "RETURN count(t) AS c", acct=acct,
                ).single()
                if cnt and cnt["c"] and cnt["c"] >= 10:
                    out.append({
                        "text": f"[FRAUD SIGNAL] Account {acct} shows velocity "
                                f"spike: {cnt['c']} outbound transactions "
                                f"(possible rapid-cycling fraud).",
                        "doc_id": f"fraud:velocity:{acct}",
                        "doc_type": "fraud_graph",
                        "chunk_idx": -1,
                        "concept_id": None,
                    })
    except Exception as e:
        print(f"[fraud/retrieve] graph query failed: {e}", flush=True)
    return out


def retrieve(query: str, top_k: int = 5, vec: Optional[list] = None) -> list:
    """Fraud retrieval: graph patterns first; dense companion handled by federated."""
    return _graph_fraud_signals(query, top_k=top_k)
