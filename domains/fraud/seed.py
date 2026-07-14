"""domains/fraud/seed.py — synthetic fraud data + narrative companion seed.

Generates 500 synthetic transactions (deterministic seed) with 5 embedded
known fraud patterns, loads them into Neo4j via ingest.py, and seeds 5 fraud-
case narrative writeups into the `fraud` Qdrant collection.

Known patterns (for VERIFY):
  P1 ring: accounts alpha/beta/gamma share device D1 (collusion ring)
  P2 velocity: account rapid-fire 20 txns in window
  P3 amount outlier: account sends 99999 (round-number outlier)
  P4 shared card: accounts zeta/eta share card K9
  P5 mule: account mule forwards 100% of incoming to cashout
"""
from __future__ import annotations

import os
import random
import csv
import requests
import config as C
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue

COLLECTION = "fraud"
CSV_PATH = os.path.join(os.path.dirname(__file__), "seed_transactions.csv")
DAEMON_EMBED = C.DAEMON_EMBED_QUERY

PATTERNS = [
    ("ring_collusion", "Collusion ring: multiple accounts share one device and "
     "rapidly move funds between each other to launder through peer accounts."),
    ("velocity_spike", "Velocity spike: an account initiates an abnormal burst of "
     "transactions in a short window — classic bust-out / rapid-cycling fraud."),
    ("amount_outlier", "Amount outlier: a single transaction far above the account's "
     "normal range (e.g. a round 99999) indicates account takeover or smurfing."),
    ("shared_card", "Shared card: distinct accounts reusing one card number is a "
     "strong indicator of card-sharing fraud or synthetic identity reuse."),
    ("mule_cashout", "Mule cashout: an account receives funds then forwards ~100% "
     "outward to a cashout account with no retained balance (money mule)."),
]


def _gen_csv() -> str:
    if os.path.exists(CSV_PATH):
        return CSV_PATH
    rng = random.Random(42)
    rows = []
    n = 500
    # base benign traffic
    for i in range(n - 30):
        frm = f"acct_{rng.randint(1, 80):03d}"
        to = f"acct_{rng.randint(1, 80):03d}"
        while to == frm:
            to = f"acct_{rng.randint(1, 80):03d}"
        rows.append({
            "txn_id": f"t{i}", "from": frm, "to": to,
            "amount": round(rng.uniform(5, 500), 2),
            "device": f"dev_{rng.randint(1, 40):02d}",
            "card": f"card_{rng.randint(1, 30):02d}",
        })
    # P1 ring: alpha/beta/gamma share D1
    for i, a in enumerate(["alpha", "beta", "gamma"]):
        rows.append({"txn_id": f"t_ring_{i}", "from": a, "to": "gamma",
                     "amount": 120.0, "device": "D1", "card": "K1"})
    # P2 velocity: rapid 20 txns from 'rapid'
    for i in range(20):
        rows.append({"txn_id": f"t_vel_{i}", "from": "rapid", "to": f"acct_{i%80:03d}",
                     "amount": round(rng.uniform(10, 90), 2),
                     "device": f"dev_{rng.randint(1,40):02d}",
                     "card": f"card_{rng.randint(1,30):02d}"})
    # P3 outlier: single 99999 from 'whale'
    rows.append({"txn_id": "t_outlier", "from": "whale", "to": "offshore",
                 "amount": 99999.0, "device": "D7", "card": "K7"})
    # P4 shared card: zeta/eta share K9
    rows.append({"txn_id": "t_card1", "from": "zeta", "to": "merchant_x",
                 "amount": 60.0, "device": "D3", "card": "K9"})
    rows.append({"txn_id": "t_card2", "from": "eta", "to": "merchant_y",
                 "amount": 75.0, "device": "D4", "card": "K9"})
    # P5 mule: mule receives then forwards all to cashout
    rows.append({"txn_id": "t_mule_in", "from": "victim", "to": "mule",
                 "amount": 2000.0, "device": "D5", "card": "K5"})
    rows.append({"txn_id": "t_mule_out", "from": "mule", "to": "cashout",
                 "amount": 1990.0, "device": "D5", "card": "K5"})
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["txn_id", "from", "to", "amount",
                                           "device", "card"])
        w.writeheader()
        w.writerows(rows)
    return CSV_PATH


def _embed(text: str) -> list:
    resp = requests.post(DAEMON_EMBED, json={"text": text}, timeout=60)
    resp.raise_for_status()
    return resp.json()["vector"]


def seed() -> str:
    from domains.fraud import ingest as fraud_ingest
    csvp = _gen_csv()
    n = fraud_ingest.ingest(csvp)
    # Qdrant narrative companion
    qc = QdrantClient(url=C.QDRANT_URL, prefer_grpc=False)
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=C.VECTOR_DIM, distance=Distance.COSINE),
        )
    existing = set()
    for name, _ in PATTERNS:
        hits = qc.scroll(
            COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(
                key="doc_id", match=MatchValue(value=f"fraudcase:{name}"))]),
            limit=1, with_payload=False,
        )[0]
        if hits:
            existing.add(name)
    new = [(nm, d) for nm, d in PATTERNS if nm not in existing]
    if new:
        pts = []
        for nm, d in new:
            vec = _embed(d)
            pts.append(PointStruct(
                id=abs(hash(f"fraudcase:{nm}")) % (2 ** 63),
                vector=vec,
                payload={"doc_id": f"fraudcase:{nm}", "doc_type": "fraud_case",
                         "chunk_idx": 0, "text": d, "metadata": {"pattern": nm}},
            ))
        qc.upsert(COLLECTION, pts, wait=True)
        case_msg = f"seeded {len(new)} fraud cases"
    else:
        case_msg = "fraud cases already exist"
    return f"fraud: loaded {n} txns, {case_msg}"
