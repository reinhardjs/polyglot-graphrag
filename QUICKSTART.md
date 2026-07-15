# QUICKSTART — ask the SNOMED clinical graph (5 minutes)

This is the shortest path from zero to a working query. The system ships with
the **`snomed`** domain (a SNOMED CT terminology graph in Neo4j) plus its
**`clinical_prose`** companion (Wikipedia medicine articles in Qdrant) already
loaded — so you can ask differential-diagnosis questions immediately. Full
detail in [RUN.md](RUN.md).

## 0. One-time: get the model files

Download two GGUFs into `<project-root>/models/` (they're NOT in the repo):
- `gemma-4-E2B-it-QAT-Q4_0.gguf` ← extraction + answer writing (HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`)

> Only E2B is needed — it serves BOTH entity/edge extraction AND answer
> synthesis. The larger E4B (:8084) is retired (gave ~22s answers for no
> quality gain on the 12 GB card; E2B is ~2.2s p95).

Everything else (Neo4j, Qdrant, Jina embed, BGE rerank, llama.cpp server) is
pulled or installed automatically. You do need a Python venv:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # sliding-window tokenization
```

## 1. Start the stack (boot once, then you can ask)

```bash
cd /mnt/data-970-plus/rag-system

# 1) Supporting stores (Neo4j + Qdrant) — Docker, runs in background.
docker compose up -d

# 2) GPU models (E2B on :8082 — extraction + synthesis) + the daemon.
bash run.sh serve

# 3) Confirm everything is up (daemon + E2B + VRAM).
bash run.sh health
```

When `run.sh health` shows green, the system is ready. Stop later with
`bash run.sh stop` (daemon + E2B) and `docker compose down` (stores).

## 2. Ask immediately (snomed + clinical_prose are pre-loaded)

You don't have to ingest anything to try it. The `snomed` graph and the
`clinical_prose` companion are already populated:
```bash
# Full pipeline: SNOMED term-match + clinical_prose semantic + E2B writes the answer
bash run.sh ask "chest pain and shortness of breath"

# Retrieval only (no LLM answer — fast, works even if E2B is off)
bash run.sh retrieve "fever and rash with joint pain"
```

Raw curl equivalents (omit `domain` to use the `default` alias → `snomed`):
```bash
curl -s -X POST 127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"chest pain and shortness of breath","domain":"snomed","synthesize":true}'

curl -s -X POST 127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"fever and rash","synthesize":false}'
```

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
