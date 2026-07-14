"""domains/fraud/ingest.py — Fraud CSV/JSON ingestion into Neo4j graph.

Builds a transaction graph: (:Account)-[:SENT]->(:Transaction)-[:TO]->(:Account)
plus co-use edges (SHARED_DEVICE / SHARED_CARD) used by the retriever's ring
detection. Companion `fraud` Qdrant collection holds narrative case writeups
(seeded by seed.py).

Idempotent: MERGE on account id / txn id. Safe to re-run.
"""
from __future__ import annotations

import csv
import json
import os
import config as C
from neo4j import GraphDatabase

_DRIVER = None
_LOCK = None


def _driver():
    global _DRIVER, _LOCK
    if _LOCK is None:
        import threading
        _LOCK = threading.Lock()
    with _LOCK:
        if _DRIVER is None:
            _DRIVER = GraphDatabase.driver(
                C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
    return _DRIVER


def ingest(path: str, **kw) -> int:
    """Ingest a CSV/JSON of transactions. Columns: txn_id, from, to, amount,
    device, card, ts (optional). Returns count of transactions loaded."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        with open(path) as f:
            rows = json.load(f)
    count = 0
    with _driver().session() as s:
        for r in rows:
            txn_id = r.get("txn_id") or r.get("id")
            frm = r.get("from") or r.get("src")
            to = r.get("to") or r.get("dst")
            amt = float(r.get("amount", 0) or 0)
            dev = r.get("device")
            card = r.get("card")
            s.run(
                "MERGE (a:Account {id:$frm}) MERGE (b:Account {id:$to}) "
                "MERGE (t:Transaction {id:$txn}) "
                "SET t.amount=$amt "
                "MERGE (a)-[:SENT]->(t)-[:TO]->(b)",
                frm=frm, to=to, txn=txn_id, amt=amt,
            )
            if dev:
                s.run("MERGE (d:Device {id:$dev}) "
                      "MERGE (a:Account {id:$frm})-[:SHARED_DEVICE]->(d)",
                      dev=dev, frm=frm)
            if card:
                s.run("MERGE (c:Card {id:$card}) "
                      "MERGE (a:Account {id:$frm})-[:SHARED_CARD]->(c)",
                      card=card, frm=frm)
            count += 1
    return count
