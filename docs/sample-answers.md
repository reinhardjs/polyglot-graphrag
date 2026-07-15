# Sample Queries & Answers

This document demonstrates the system producing grounded, relevant answers from the auto-seeded enterprise self-docs corpus. Use the curl commands below to verify the live system yourself.

> **Prerequisite**: the daemon must be running (`serve_gpu.py` or `serve_cpu.py`) and the enterprise corpus must be seeded (happens automatically on first boot).

---

## Enterprise domain queries

### 1. BGE reranker role

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the role of the BGE reranker in the retrieval pipeline?","domain":"enterprise","synthesize":true,"skip_cache":true}' \
  | python3 -m json.tool
```

> **Answer**: Two-stage: bi-encoder (Jina-v3) retrieves ~20-50 candidates, then cross-encoder (BGE Reranker v2-m3) re-scores final top_k candidates [2].
>
> **Stats**: `n_contexts=5`, `degraded=False`, source citations present.

### 2. Cache handling

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does the system handle cache for repeated queries?","domain":"enterprise","synthesize":true,"skip_cache":true}' \
  | python3 -m json.tool
```

> **Answer**: Checks semantic cache before retrieval pipeline. If cosine similarity >0.95, cached answer is returned (~0.13s, 0 E4B tokens) [2].
>
> **Stats**: `n_contexts=5`, `degraded=False`.

### 3. Storage backends

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What backends are used for vector storage and graph storage?","domain":"enterprise","synthesize":true,"skip_cache":true}' \
  | python3 -m json.tool
```

> **Answer**: Qdrant (:6333) for vector collections; Neo4j (:7687) for entities/edges [1, 4].
>
> **Stats**: `n_contexts=5`, `degraded=False`.

### 4. Synthesis backend

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Explain the synthesis step and what LLM backend it uses.","domain":"enterprise","synthesize":true,"skip_cache":true}' \
  | python3 -m json.tool
```

> **Answer**: Synthesis runs on E2B (:8082), tuned to stay under 3s on RTX 3060 12GB [1].
>
> **Stats**: `n_contexts=5`, `degraded=False`.

### 5. Configured domains

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What domains are configured in the system?","domain":"enterprise","synthesize":true,"skip_cache":true}' \
  | python3 -m json.tool
```

> **Answer**: `domain_config.yaml` is the source of truth. Domain profiles control collection routing, chunking, prompts, graph labels. Default is `enterprise` if omitted [2, 3].
>
> **Stats**: `n_contexts=5`, `degraded=False`.

---

## Edge cases

### Honest abstention (out-of-domain question)

The system does **not** hallucinate. When asked something outside its corpus, it returns a brief abstention:

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the capital of France?","domain":"enterprise","synthesize":true,"skip_cache":true}' \
  | python3 -m json.tool
```

> **Answer**: `insufficient information`
>
> **Stats**: `n_contexts=5`, `degraded=False`. No fabricated geography answer.

### Cross-domain query

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What backends does the system use for storage and retrieval?","domain":"all","synthesize":true,"skip_cache":true}' \
  | python3 -m json.tool
```

> **Answer**: Qdrant (domain-specific collections) + Neo4j (graph entities/edges) [4].
>
> **Stats**: `n_contexts=5`, `path=cross_domain`, `degraded=False`.

### Non-existent domain

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"test","domain":"nonexistent","synthesize":false}'
```

> **Response**: `{"error":"unknown_domain","available":["enterprise","snomed","clinical_prose","legal","fraud","healthcare"],"degraded":true,"notice":"unknown domain 'nonexistent'; available: enterprise, snomed, clinical_prose, legal, fraud, healthcare"}`
>
> Graceful 200 response with error field — never a 500.

---

## Running the release gate

The full 14-check release gate verifies answer quality, retrieval latency, concurrency, and doc consistency:

```bash
cd /path/to/rag-system
./venv/bin/python scripts/release-gate.py
```

Expected output: **14/14 checks passed | ALL SYSTEMS GO**

---

## Automated relevance audit

To run a multi-question relevance scan similar to what's documented here:

```bash
python3 << 'EOF'
import json, urllib.request

questions = [
    ("enterprise", "What is the role of the BGE reranker?"),
    ("enterprise", "How does cache work?"),
    ("enterprise", "What are the storage backends?"),
    ("enterprise", "What LLM does synthesis use?"),
    ("enterprise", "What domains are configured?"),
    ("enterprise", "What is the capital of France?"),
    ("all", "What backends does the system use?"),
]

for domain, q in questions:
    body = json.dumps({"query": q, "domain": domain,
                       "synthesize": True, "skip_cache": True}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        "http://localhost:8000/ask", data=body,
        headers={"Content-Type": "application/json"}, method="POST"))
    d = json.loads(r.read())
    ans = d.get("answer", "").strip()
    nc = d.get("n_contexts", 0)
    deg = d.get("degraded", None)
    print(f"[{domain}] n={nc} deg={deg}  {q[:50]}...")
    print(f"  {ans[:150]}")
    print()
EOF
```
