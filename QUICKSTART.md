# QUICKSTART — ask in 5 minutes

> **Audience:** You want the absolute shortest path to a working query. No explanations.
> - For concepts + your corpus + benchmark → [Getting Started](docs/guides/getting-started.md)
> - For every knob, systemd, limitations → [RUN.md](RUN.md)

This is the shortest path from zero to a working query. The system ships with
four example domains — **`snomed`** (clinical terminology graph), **`enterprise`**
(auto-seeded with this project's own docs on first boot), **`legal`**, and
**`fraud`** — so you can ask questions immediately. Full detail in [RUN.md](RUN.md).

## 0. One-time: get the model file + check setup

### GPU path (RTX 3060 12GB recommended)
Download ONE GGUF into `<project-root>/models/` (it's NOT in the repo):
- `gemma-4-E2B-it-QAT-Q4_0.gguf` ← extraction + answer writing (HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`)

> Only E2B is needed — it serves BOTH entity/edge extraction (classifies
> relations; GLiNER detects entities) AND answer synthesis.
> The larger E4B (:8084) is retired (gave ~22s answers for no
> quality gain on the 12 GB card; E2B is ~2.2s p95).

### CPU-only path (no GPU required)
**Two options for synthesis:**
1. **Local E2B on CPU** — if you have `./models/gemma-4-E2B-it-QAT-Q4_0.gguf`, runs via llama.cpp with `gpu_layers=0`
2. **Remote/OpenRouter** — set `SYNTHESIS_LLM_BASE_URL` to any OpenAI-compatible endpoint (e.g., OpenRouter)

Embedding (Jina v3) + Reranker (BGE) run locally on CPU (fp32) in both cases.

```bash
# Set CPU mode before any command:
export CUDA_VISIBLE_DEVICES=""  # forces CPU
export CONDA_NO_PLUGINS=1       # required to avoid subprocess SIGTERM issues
```

Everything else (Neo4j, Qdrant, Jina embed, BGE rerank) is
pulled or installed automatically. You need **Python 3.11**. Create the venv:
```bash
bash run.sh setup      # creates venv + installs pinned requirements
# (or manually: python3.11 -m venv venv && pip install -r requirements.txt)
```

Then run the setup checker — it tells you exactly what is missing:
```bash
bash run.sh doctor
```
If `doctor` reports `[MISSING]` items, fix them (it prints the exact command),
then re-run. When it says "Setup OK", continue.

```bash
cd polyglot-graphrag   # or: cd <project-root>

# 1) Supporting stores (Neo4j + Qdrant) — Docker, runs in background.
docker compose up -d

# 2a) GPU path: GPU models (E2B on :8082 — extraction + synthesis) + the daemon.
#     Requires the GGUF model file in ./models/
bash run.sh serve

# 2b) CPU-only path: Export CPU env vars FIRST:
export CUDA_VISIBLE_DEVICES=""  # forces CPU
export CONDA_NO_PLUGINS=1       # required

# Option 1: E2B on OpenRouter (no GGUF needed)
bash run.sh serve --cpu --no-llm
# (or: CUDA_VISIBLE_DEVICES="" CONDA_NO_PLUGINS=1 ./venv/bin/python serve_cpu.py &)

# Option 2: Local E2B on CPU (requires GGUF in ./models/)
# bash run.sh serve --cpu   # auto-starts E2B with gpu_layers=0 if GGUF exists

# 3) Confirm everything is up (daemon + stores + VRAM/CPU).
bash run.sh health
```

When `run.sh health` shows green, the system is ready. Stop later with
`bash run.sh stop` (daemon + E2B) and `docker compose down` (stores).

## 2. Ask immediately (enterprise self-docs are auto-seeded)

You don't have to ingest anything to try it. On first startup the **`enterprise`**
domain (the default) auto-seeds itself with this project's *own documentation*
(the `docs/` tree, tagged `self-docs`) — so you can immediately ask
questions about how the GraphRAG system itself works, with no external corpus:

```bash
# Full pipeline on the system's own docs (default domain = enterprise)
bash run.sh ask "how does hybrid retrieval fuse Qdrant and Neo4j"
# equivalent raw curl (omit domain to use default enterprise):
curl -s -X POST 127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"how does hybrid retrieval fuse Qdrant and Neo4j","synthesize":true}'

# Retrieval only (no LLM answer — fast, works even if E2B is off)
bash run.sh retrieve "how does hybrid retrieval work"
```

> The SNOMED clinical graph (`snomed`) and its `clinical_prose` companion are
> **NOT** auto-loaded. They are populated on demand — see §3 to ingest them
> (or any corpus of your own) before asking clinical questions.

(Idempotent: enterprise only seeds when its collection is empty, and skips docs
that already exist. Point it at your own corpus anytime with
`ingest_corpus_docs.py`.)

## 3. Add your own corpus (optional)

The project is domain-agnostic: add a new corpus by copying an existing plugin
and registering it — no daemon code change.
```bash
cp -r domains/clinical_prose domains/my_corpus   # or domains/snomed for a graph corpus
# edit my_corpus/ingest.py + my_corpus/retrieve.py, then add a block in
# domain_config.yaml (set collection / neo4j_label / companions as needed)
curl -X POST localhost:8000/admin/reload         # hot-reload config
curl -X POST localhost:8000/ingest_domain -H 'Content-Type: application/json' -d '{"domain":"my_corpus"}'
```

## 4. Ask (retrieve + synthesize)

```bash
# Retrieval + synthesis (E2B on :8082 does both):
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"chest pain and shortness of breath","domain":"snomed","synthesize":true}'

# Retrieval only (no E4B needed):
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"chest pain","synthesize":false}'
```

Response contract: `{query, source, path, n_contexts, contexts[],
contexts_meta[] (incl. signal + dual_signal), answer?}`.

---

> **Want to add your own 2nd corpus (companion)?** Copy `domains/clinical_prose/`
> and list it under a domain's `companions:` in `domain_config.yaml`. No daemon
> code change. See [docs/domains/README.md](docs/domains/README.md).
