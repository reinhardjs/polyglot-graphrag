# Healthcare (SNOMED) — Quick Start

This is the default clinical domain. It answers symptom-based differential
diagnosis questions by combining the SNOMED CT terminology graph (512k
concepts) with a free-text clinical-prose companion corpus. `healthcare` is
an alias for `snomed`; both resolve to the same hybrid retriever.

## 1. Start the daemon (if not running)

    cd /mnt/data-970-plus/rag-system
    ./venv/bin/python serve_gpu.py > logs/daemon_gpu.log 2>&1 &

## 2. Ask a clinical question

    curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
      -d '{"query":"fever and rash and cough","domain":"healthcare","synthesize":false}' \
      | python -c "import sys,json; d=json.load(sys.stdin); print('signals:', sorted(set(m['signal'] for m in d['contexts_meta']))); [print(c[:120]) for c in d['contexts']]"

## 3. Ranked differential with confidence (synthesis on)

    curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
      -d '{"query":"fever and rash and cough","domain":"healthcare","synthesize":true,"mode":"differential"}' \
      | python -c "import sys,json; d=json.load(sys.stdin); [print(x['rank'], x['candidate'], x['confidence'], 'dual='+str(x['dual_signal'])) for x in d['diagnoses']]"

## 4. Cross-domain (combine with the other 3 domains)

    curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
      -d '{"query":"data breach and compliance","domain":"all","synthesize":false}'
