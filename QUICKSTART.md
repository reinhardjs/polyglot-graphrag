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

## 1. Start the stack

```bash
cd <project-root>
docker compose up -d          # Neo4j + Qdrant
bash run.sh serve             # E2B (:8082) + E4B (:8084) + daemon (:8000)
bash run.sh health            # confirm all 3 up + VRAM
```

## 2. Ask immediately (a demo corpus is auto-seeded on startup)

You don't have to ingest anything to try it — the `example_companion` corpus is
seeded automatically, and `engineering` already has real data:
```bash
bash run.sh ask "what is the total VRAM budget on the RTX 3060 and which models share it"
# or raw:
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"who reported BUG-204?","domain":"engineering"}'
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
