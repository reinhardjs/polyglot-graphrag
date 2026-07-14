# QUICKSTART — ingest your doc and ask questions (5 minutes)

This is the shortest path from zero to a working query. Full detail in
[RUN.md](RUN.md); sync-daemon detail in [SYNC.md](SYNC.md).

## 0. One-time: get the model files

Download two GGUFs into `<project-root>/models/` (they're NOT in the repo):
- `gemma-4-E2B-it-QAT-Q4_0.gguf` ← extraction (HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`)
- `gemma-4-E4B-it-QAT-Q4_0.gguf` ← answer writing (HuggingFace `lmstudio-community/gemma-4-E4B-it-QAT-GGUF`)

Everything else (Neo4j, Qdrant, Jina embed, BGE rerank, llama.cpp server) is
pulled or installed automatically. You do need a Python venv:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # sliding-window tokenization
```

## 1. Start the stack (boot once, then you can ask)

Run from the project root. The demo companion corpora (`engineering_docs`,
`example_companion`) are **auto-seeded on first daemon start**. For a FULL
demo that also seeds the **primary engineering graph corpus** (so graph
questions like "who reported BUG-204?" work out-of-the-box), add `--demo` —
it ingests a bundled sample set (`demo/engineering/`) into `engineering_chunks`
+ the Neo4j `Engineering` graph, with zero external data.

```bash
cd /mnt/data-970-plus/rag-system

# 1) Supporting stores (Neo4j + Qdrant) — Docker, runs in background.
docker compose up -d

# 2) GPU models + the daemon. Add --demo to also seed the primary graph corpus.
bash run.sh serve            # companion corpora auto-seed
# OR for the FULL demo (primary graph + companion), start the daemon with --demo:
#   ./venv/bin/python serve_gpu.py --demo
# (run.sh serve does not pass --demo; use the line above for the full demo.)

# 3) Confirm everything is up (daemon + both LLMs + VRAM).
bash run.sh health
```

> Note: `run.sh serve` starts the daemon without `--demo`. To get the full
> primary-graph demo, run `./venv/bin/python serve_gpu.py --demo` directly
> (the LLMs are already up from `bash run.sh llms`, or just start everything
> with `bash run.sh llms` then the `--demo` daemon). The `--demo` seed runs in
> the background after the daemon is healthy; watch the daemon log for
> "[gpu-daemon] demo seed complete".

When `run.sh health` shows green, the system is ready. Stop later with
`bash run.sh stop` (daemon + LLMs) and `docker compose down` (stores).

## 2. Ask immediately (demo corpus is auto-seeded on startup)

You don't have to ingest anything to try it. The `engineering` primary corpus
already has real data and the `engineering_docs` companion (this repo's `docs/`)
is seeded on startup:
```bash
# Full pipeline: retrieve + E4B writes the answer
bash run.sh ask "who reported BUG-204?"

# Retrieval only (no LLM answer — fast, works even if E4B is off)
bash run.sh retrieve "what is the total VRAM budget on the RTX 3060"
```

Raw curl equivalents:
```bash
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"who reported BUG-204?","domain":"engineering","synthesize":true}'

curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"what is the total VRAM budget on the RTX 3060","domain":"engineering","synthesize":false}'
```

## 3. Ingest your real engineering doc (optional)

```bash
mkdir -p mydocs/eng
cp /path/to/your-doc.md mydocs/eng/
./venv/bin/python sync_docs.py mydocs          # POSTs each file to /ingest
```

- `doc_id` = relative path (`eng/your-doc.md`); `domain` inferred from folder.
- PDFs work too — drop `*.pdf` in and re-run; `pdftotext` extracts them.
- Re-run after edits → only changed files re-ingest. Deleted files → removed
  from both stores. State: `mydocs/.sync_state.json`.

Verify it landed:
```bash
docker exec rag-system-neo4j-1 cypher-shell -a bolt://localhost:7687 -u neo4j -p ragpassword123 \
  "MATCH (n:Entity) WHERE 'eng/your-doc.md' IN n.source_docs RETURN n.name, n.type LIMIT 20;"
```

## 4. Ask (retrieve + synthesize)

```bash
# Retrieval + synthesis (needs E4B on :8084):
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"who reported BUG-204?","domain":"engineering","synthesize":true}'

# Retrieval only (no E4B needed):
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"who reported BUG-204?","synthesize":false}'
```

Response contract: `{query, source, path, n_contexts, contexts[],
contexts_meta[] (incl. signal + dual_signal), answer?}`.

---

> **Want to add your own 2nd corpus (companion)?** Copy
> `domains/example_companion/` and list it under a domain's `companions:` in
> `domain_config.yaml`. No daemon code change. See [docs/domains/README.md](docs/domains/README.md).
