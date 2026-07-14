# Legal & Compliance — Quick Start

This domain answers questions about regulations and compliance: GDPR,
SOC 2, HIPAA, contract clauses. Demo content is synthetic and clearly
labelled — not legal advice.

## 1. Start the daemon (if not running)

    cd /mnt/data-970-plus/rag-system
    ./venv/bin/python serve_gpu.py > logs/daemon_gpu.log 2>&1 &

## 2. Seed demo data (idempotent, runs once)

    ./venv/bin/python -c "import domains.legal.seed as s; print(s.seed())"

## 3. Ask a question (single domain)

    curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
      -d '{"query":"GDPR obligations for cross-border data transfers","domain":"legal","synthesize":false}' \
      | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_contexts'],'contexts'); [print(c[:120]) for c in d['contexts']]"

## 4. Ingest your own policy text

    curl -s -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
      -d '{"text":"All vendor contracts require a Data Processing Agreement before PHI is shared.","doc_id":"policy-dpa","doc_type":"legal","domain":"legal"}'

## 5. Cross-domain (combine with enterprise)

    curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
      -d '{"query":"compliance for payment service","domain":["enterprise","legal"],"synthesize":false}'
