# Fraud Detection — Quick Start

This domain detects fraud via a transaction GRAPH (Neo4j) plus a narrative
companion (Qdrant). The demo seed loads 500 synthetic transactions embedding
5 known fraud patterns: collusion ring (shared device), velocity spike,
amount outlier, shared card, and mule cashout.

## 1. Start the daemon + Neo4j (if not running)

    cd /mnt/data-970-plus/rag-system
    ./venv/bin/python serve_gpu.py > logs/daemon_gpu.log 2>&1 &

## 2. Seed demo data (idempotent, loads graph + narrative cases)

    ./venv/bin/python -c "import domains.fraud.seed as s; print(s.seed())"

## 3. Ask a question — graph ring detection

    curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
      -d '{"query":"unusual transaction pattern on account alpha","domain":"fraud","synthesize":false}' \
      | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_contexts'],'signals'); [print(c[:120]) for c in d['contexts']]"
    # Returns the alpha<->beta/gamma shared-device / shared-card ring.

## 4. Ingest your own transactions (CSV or JSON)

    # CSV columns: txn_id,from,to,amount,device,card
    curl -s -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
      -d '{"text":"txn t1 from acct_x to acct_y amount 50.0 device D1 card K1","doc_id":"fraud-batch","doc_type":"fraud","domain":"fraud"}'
    # Or run the seed CSV directly:
    #   ./venv/bin/python -c "from domains.fraud.ingest import ingest; print(ingest('domains/fraud/seed_transactions.csv'))"

## 5. Known patterns in the seed

  P1 ring_collusion   : alpha/beta/gamma share device D1
  P2 velocity_spike   : account 'rapid' fires 20 txns
  P3 amount_outlier   : account 'whale' sends 99999
  P4 shared_card      : zeta/eta share card K9
  P5 mule_cashout     : 'mule' forwards 100% of incoming to 'cashout'
