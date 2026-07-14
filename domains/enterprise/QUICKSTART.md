# Enterprise KM — Quick Start

This domain answers questions about your internal engineering knowledge:
ADRs, runbooks, postmortems, RFCs, service dependencies.

## 1. Start the daemon (if not running)

    cd /mnt/data-970-plus/rag-system
    ./venv/bin/python serve_gpu.py > logs/daemon_gpu.log 2>&1 &

## 2. Seed demo data (idempotent, runs once)

    ./venv/bin/python -c "import domains.enterprise.seed as s; print(s.seed())"

## 3. Ask a question (single domain)

    curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
      -d '{"query":"what services depend on payment","domain":"enterprise","synthesize":false}' \
      | python -c "import sys,json; d=json.load(sys.stdin); print(d['n_contexts'],'contexts'); [print(c[:120]) for c in d['contexts']]"

## 4. Ingest your own docs

    curl -s -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
      -d '{"text":"Our checkout service calls payment-service via a circuit breaker (PR-482).","doc_id":"my-doc-1","doc_type":"enterprise","domain":"enterprise"}'

## 5. Cross-domain (combine with legal)

    curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
      -d '{"query":"compliance for payment service","domain":["enterprise","legal"],"synthesize":false}'
