# New User Walkthrough — polyglot-graphrag v1.0.6

> **Audience:** You want the guided tutorial — concepts explained, your confidential corpus in `external/`, quality benchmark run.
> - For 5-minute copy-paste → [QUICKSTART.md](../QUICKSTART.md)
> - For every knob, systemd, limitations → [RUN.md](../RUN.md)

**One linear path from "fresh clone" to a working GraphRAG query with synthesis, then to your own confidential corpus, then to the quality benchmark.**

> **Prerequisites**: Linux (Ubuntu 22.04+ tested), Python 3.11, NVIDIA GPU with 12 GB VRAM (RTX 3060 used here), Docker + docker compose v2, ~25 GB free disk.
>
> **Clone & enter**:
> ```bash
> git clone https://github.com/reinhardjs/polyglot-graphrag.git
> cd polyglot-graphrag
> ```
> This guide assumes your CWD is the repo root (`/mnt/data-970-plus/rag-system` in examples).

---

## 0. One-time Setup (≈2 min)

### 0.1 Create venv + install pinned deps
```bash
bash run.sh setup
# or manually:
# python3.11 -m venv venv
# ./venv/bin/pip install -r requirements.txt
```
`run.sh setup` is idempotent — safe to re-run.

### 0.2 Download the ONE required model (E2B GGUF)
The system needs **one GGUF**: `gemma-4-E2B-it-QAT-Q4_0.gguf` (serves BOTH extraction + synthesis).

**Option A — via Hugging Face CLI** (recommended):
```bash
huggingface-cli download lmstudio-community/gemma-4-E2B-it-QAT-GGUF \
  --include "gemma-4-E2B-it-QAT-Q4_0.gguf" \
  --local-dir ./models
```

**Option B — manual**: Download from https://huggingface.co/lmstudio-community/gemma-4-E2B-it-QAT-GGUF/tree/main and place the `.gguf` file in `./models/`.

> **Note**: Only E2B is required. E4B is retired (gave ~22s answers for no quality gain; E2B is ~2.2s p95).

### 0.3 Run the doctor — it tells you exactly what's missing
```bash
bash run.sh doctor
```
If anything shows `[ACTION]`, run the suggested command and re-run `doctor` until it says **"Setup OK"**.

---

## 1. Start the Stack

### GPU path (RTX 3060 12GB recommended) — ≈30s
The stack has **three layers**:

| Layer | What | How to start | Port |
|-------|------|--------------|------|
| **Docker stores** | Neo4j (graph) + Qdrant (vectors) | `docker compose up -d` | 7474/7687, 6333 |
| **GPU LLM** | E2B llama.cpp server (extraction + synthesis) | `bash run.sh serve` (auto-starts with daemon) | 8082 |
| **Daemon** | FastAPI `serve_gpu.py` (your `/ask`, `/ingest`, `/health`) | `bash run.sh serve` | 8000 |

```bash
# 1a) Start Neo4j + Qdrant (background, persistent data in ./data/)
docker compose up -d

# 1b) Wait for them healthy
curl -s -o /dev/null -w "%{http_code}" http://localhost:6333   # → 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:7474   # → 200

# 1c) Start E2B + daemon (runs doctor first, then launches both)
bash run.sh serve

# 1d) Confirm everything green
bash run.sh health
# Expected: all backends ok, VRAM ~8.5 GB under load, daemon pid shown
```

> **If health shows `degraded` or a backend `down`**: restart the daemon: `bash run.sh stop && bash run.sh serve`. The daemon recreates missing Qdrant collections (`query_cache`, `entity_vector_idx`) on startup.

### CPU-only path (no GPU required) — ≈60s first run
**Two options for synthesis:**
1. **Local E2B on CPU** — if you have `./models/gemma-4-E2B-it-QAT-Q4_0.gguf`, runs via llama.cpp with `gpu_layers=0`
2. **Remote/OpenRouter** — set `SYNTHESIS_LLM_BASE_URL` to any OpenAI-compatible endpoint (e.g., OpenRouter)

Embedding (Jina v3) + Reranker (BGE) run locally on CPU (fp32) in both cases.

```bash
# 1a) Start Neo4j + Qdrant (same as GPU path)
docker compose up -d

# 1b) Wait for them healthy
curl -s -o /dev/null -w "%{http_code}" http://localhost:6333   # → 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:7474   # → 200

# 1c) Start CPU daemon
#     Set CPU env vars FIRST:
export CUDA_VISIBLE_DEVICES=""  # forces CPU
export CONDA_NO_PLUGINS=1       # required to avoid subprocess SIGTERM issues

# Option 1: Local E2B on CPU (requires GGUF in ./models/)
# bash run.sh serve --cpu   # auto-starts E2B with gpu_layers=0 if GGUF exists

# Option 2: Remote synthesis (OpenRouter or other)
# Set the endpoint first:
# export SYNTHESIS_LLM_BASE_URL="https://openrouter.ai/api/v1"
# export SYNTHESIS_LLM_API_KEY="your-key"
# export SYNTHESIS_LLM_MODEL="nvidia/nemotron-3-ultra-550b-a55b:free"
# Then:
bash run.sh serve --cpu --no-llm

# 1d) Confirm everything green
bash run.sh health
# Expected: all backends ok, daemon pid shown (no VRAM on CPU)
```

> **Performance note**: CPU retrieval is ~23× slower than GPU (3.8s vs 0.16s p95) because Jina embed + BGE rerank run in fp32 on CPU. However, synthesis latency depends on your backend — local E2B on CPU is ~10-15s; remote (OpenRouter) varies. Full pipeline: ~7-15s CPU vs ~6.5s GPU. See [CPU vs GPU benchmark](../benchmarks/cpu-vs-gpu-benchmark.md) for details.

---

## 2. Try It Immediately — Zero Ingest Required (enterprise self-docs)

On first startup, the **`enterprise`** domain (the default) **auto-seeds itself with this project's own `docs/` tree**. You can ask questions about the system immediately:

```bash
# Full pipeline (retrieval + synthesis) — default domain = enterprise
bash run.sh ask "how does hybrid retrieval fuse Qdrant and Neo4j"

# Same as raw curl:
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does hybrid retrieval fuse Qdrant and Neo4j","synthesize":true}'

# Retrieval ONLY (no LLM answer, fast, works even if E2B is down)
bash run.sh retrieve "how does hybrid retrieval work"
# or:
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does hybrid retrieval work","synthesize":false}'
```

**What you should see** (`synthesize:true`):
```json
{
  "query": "how does hybrid retrieval fuse Qdrant and Neo4j",
  "source": "enterprise",
  "qdrant_hits": 12,
  "graph_hits": 8,
  "n_contexts": 10,
  "answer": "The hybrid retrieval pipeline fuses Qdrant vector search with Neo4j graph traversal..."
}
```

**What you should see** (`synthesize:false`):
```json
{
  "query": "how does hybrid retrieval work",
  "source": "enterprise",
  "qdrant_hits": 12,
  "graph_hits": 8,
  "n_contexts": 10,
  "contexts": [ {"text": "...", "score": 0.87, "doc_id": "..."}, ... ]
}
```

> `synthesize:false` returns the raw retrieved contexts (vector + graph) — use for debugging retrieval quality. `synthesize:true` sends those contexts to E2B and streams back a grounded answer.

---

## 3. Ingest YOUR Corpus (the general pattern)

The system is **domain-agnostic**. To add your own docs:

### 3.1 Put your markdown files under `external/`
```
external/
  my-project/
    docs/
      architecture/
        overview.md
      api/
        endpoints.md
```
> **Why `external/`?** It's `.gitignore`d — your confidential docs never enter the repo.

### 3.2 Ingest into a domain (here: `enterprise` works, but you can add a domain)
```bash
# Ingest all .md under external/my-project/docs into the 'enterprise' domain
./venv/bin/python scripts/ingest_corpus_docs.py \
  --docs external/my-project/docs \
  --domain enterprise \
  --source my-project \
  --workers 2 \
  --extract-graph        # <-- REQUIRED: populates Neo4j, not just vectors
```
- `--extract-graph` **must be present** (or `--no-extract-graph` omitted) to get entities + relations into Neo4j.
- Idempotent: re-running with the same `--source` + doc paths **replaces** those docs cleanly.
- Output ends with `submitted: N, done: N, failed: 0`.

### 3.3 Verify BOTH stores populated
```bash
./venv/bin/python -c "
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import config as C
qc = QdrantClient(url=C.QDRANT_URL)
d  = GraphDatabase.driver(C.NEO4J_URI, auth=(C.NEO4J_USER, C.NEO4J_PASSWORD))
qdrant_n = qc.count('enterprise')
neo4j_e  = d.session().run('MATCH (n:Entity) RETURN count(n) AS c').single()['c']
neo4j_r  = d.session().run('MATCH ()-[x]->() RETURN count(x) AS c').single()['c']
d.close()
print(f'qdrant: {qdrant_n}  neo4j_entities: {neo4j_e}  neo4j_relations: {neo4j_r}')
assert qdrant_n > 0 and neo4j_e > 0 and neo4j_r > 0
"
```

---

## 4. Your Confidential Corpus — The ora-et-labora Example

This repo **already ships** a realistic confidential engineering corpus at:
```
external/ora-et-labora/          # your private repo clone (gitignored)
external/ora-et-labora-golden.json  # corpus-matched golden QA set (gitignored)
```

### 4.1 Register the `oraetlabora` domain (one-time, already in `domain_config.yaml`)
```yaml
# domain_config.yaml — already present
oraetlabora:
  collection: oraetlabora
  neo4j_label: OraEtLabora
  entity_types: [Product, Service, Component, Module, API, ADR, Bug, Guideline, Team, Config]
  relation_types: [DOCUMENTS, OWNS, DEPENDS_ON, CAUSED_BY, REFERENCES, IMPLEMENTS, DESCRIBES, CONFIGURES]
```

### 4.2 Ingest it (graph extraction ON)
```bash
./venv/bin/python scripts/ingest_corpus_docs.py \
  --docs external/ora-et-labora \
  --domain oraetlabora \
  --source ora-et-labora \
  --workers 3 \
  --extract-graph
```
Expect: ~hundreds of docs, thousands of entities, thousands of relations. Takes 5–15 min depending on corpus size.

### 4.3 Ask questions against your corpus
```bash
# Synthesis ON (full answer)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does the transaction service handle retries","domain":"oraetlabora","synthesize":true}'

# Synthesis OFF (raw retrieval contexts)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does the transaction service handle retries","domain":"oraetlabora","synthesize":false}'
```

### 4.4 Verify it hits BOTH stores
```bash
./venv/bin/python -c "
import urllib.request, json
q = {'query':'how does the transaction service handle retries','domain':'oraetlabora','synthesize':False,'skip_cache':True,'top_k':5}
r = urllib.request.Request('http://localhost:8000/ask', data=json.dumps(q).encode(), headers={'Content-Type':'application/json'}, method='POST')
d = json.loads(urllib.request.urlopen(r, timeout=60).read())
print('qdrant_hits:', d.get('qdrant_hits'), 'graph_hits:', d.get('graph_hits'), 'n_contexts:', d.get('n_contexts'), 'degraded:', d.get('degraded'))
assert d.get('qdrant_hits',0)>0 and d.get('graph_hits',0)>0 and d.get('n_contexts',0)>0 and not d.get('degraded')
print('HYBRID OK')
"
```

---

## 5. Quality Benchmark — Prove It's Good

The repo has a **corpus-matched golden set** for the ora-et-labora corpus (`external/ora-et-labora-golden.json`). The release gate runs quality checks against it.

### 5.1 Run the full release gate (15 checks)
```bash
./venv/bin/python scripts/release-gate.py
```
**Expected (v1.0.6 on RTX 3060 12 GB):**
```
15/15 checks passed  |  ALL SYSTEMS GO
```
Checks include:
- Health, metrics, `/ask`, unknown domain, cross-domain, synthesis
- Bench retrieval p95 < 1100 ms (all domains)
- Config integrity, code claims, concurrency (12 parallel)
- **Answer quality (enterprise self-docs golden)** — faithfulness ≥ 0.85
- **Answer quality (ora-et-labora foreign corpus)** — faithfulness ≥ 0.85
- Doc/code consistency audit
- Synthesis benchmark p95 < 3.0 s, 0 errors

### 5.2 Run the synthesis/quality benchmark standalone
```bash
./venv/bin/python scripts/bench_synth_compare.py
```
**Output you want:**
```
TARGET: PASS (p95=2.3s, threshold=3.0s, 0 errors)
```

### 5.3 Manual quality spot-check (faithfulness)
```bash
# Uses the external golden set against oraetlabora domain
./venv/bin/python scripts/evaluate_pipeline.py \
  --golden external/ora-et-labora-golden.json \
  --domain oraetlabora \
  --live \
  --workers 3
```
Key metrics reported: `faithfulness`, `context_precision`, `context_recall`, `answer_relevance`. The gate uses **ragas semantic faithfulness** (≥ 0.85) as the authoritative grounding signal.

---

## 6. Both Modes — synthesis:true vs synthesis:false

| Mode | Use case | Latency | Requires E2B |
|------|----------|---------|--------------|
| `synthesize:true` | User-facing answer, grounded in your docs | ~2–3 s p95 | YES (E2B on :8082) |
| `synthesize:false` | Debug retrieval, build your own prompt, feed to another LLM | ~500 ms | NO |

**Same query, both modes:**
```bash
# With synthesis
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is the retry policy for the payment gateway","domain":"oraetlabora","synthesize":true}' | jq '.answer'

# Without synthesis (inspect contexts)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is the retry policy for the payment gateway","domain":"oraetlabora","synthesize":false}' | jq '.contexts[] | {doc_id, score, text: .text[:120]}'
```

---

## 7. Stop / Restart Cleanly

```bash
# Stop GPU daemon + E2B (keeps Docker stores + data)
bash run.sh stop

# Stop Docker stores (Neo4j + Qdrant) — DATA PRESERVED in ./data/
docker compose down

# Full wipe (fresh clone state) — removes ./data/
docker compose down -v
rm -rf data .cache logs __pycache__ .run
```

---

## 8. Common Issues (Quick Fixes)

| Symptom | Fix |
|---------|-----|
| `/health` shows `qdrant: down` or `neo4j: down` | `bash run.sh stop && bash run.sh serve` — daemon recreates missing collections |
| `docker compose up -d` ports already in use | `docker compose down` first, or stop conflicting containers |
| `doctor` says E2B model missing | Put `gemma-4-E2B-it-QAT-Q4_0.gguf` in `./models/` (or set `E2B_MODEL=/path/to.gguf`) |
| Synthesis latency > 3 s | Ensure you're on the **CUDA** llama.cpp build (not Vulkan). `run.sh` auto-detects CUDA first. |
| Graph hits = 0 | Ingest with `--extract-graph` (not `--no-extract-graph`). Check Neo4j label matches `domain_config.yaml`. |
| `module not found` in daemon | `bash run.sh setup` to sync venv, then `bash run.sh serve` (uses project venv). |

---

## 9. One-Page Cheat Sheet (Copy-Paste)

```bash
# 0. Setup (once)
bash run.sh setup
# download gemma-4-E2B-it-QAT-Q4_0.gguf → ./models/
bash run.sh doctor   # repeat until "Setup OK"

# 1. Start stack
docker compose up -d
bash run.sh serve
bash run.sh health   # wait for all green

# 2. Instant query (enterprise self-docs)
bash run.sh ask "how does hybrid retrieval work"
bash run.sh retrieve "what is the ingestion pipeline"

# 3. Your confidential corpus → external/my-docs/
./venv/bin/python scripts/ingest_corpus_docs.py \
  --docs external/my-docs --domain enterprise --source my-docs --workers 2 --extract-graph

# 4. Query your corpus
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"your question","domain":"enterprise","synthesize":true}'

# 5. Quality gate
./venv/bin/python scripts/release-gate.py    # want 14/14 PASS
./venv/bin/python scripts/bench_synth_compare.py  # want TARGET: PASS, errors=0

# 6. Stop
bash run.sh stop
docker compose down
```

---

## 10. Where to Look Next

| File | Purpose |
|------|---------|
| `RUN.md` | Full operational reference (ingest details, domain config, systemd, monitoring) |
| `QUICKSTART.md` | Condensed 5-min path (this guide expands it) |
| `domain_config.yaml` | Add/remove domains, set entity/relation types, Qdrant collections, Neo4j labels |
| `scripts/ingest_corpus_docs.py` | `--help` shows all ingest options (force, no-graph, workers, source) |
| `scripts/release-gate.py` | The 14 checks, thresholds, how to recalibrate honestly |
| `goal_100pct_functional.md` | Machine-readable checklist the CI/automation uses |

---

**You're now at "100% Functional":** ingest → both stores populated → hybrid retrieval works → synthesis works → release gate green → quality benchmark passes. Add your own corpus under `external/`, register a domain, and repeat.