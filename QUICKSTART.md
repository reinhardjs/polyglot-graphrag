# QUICKSTART — ingest your doc and ask questions (5 minutes)

This is the shortest path from zero to a working GraphRAG query. Full detail
in [RUN.md](RUN.md); sync-daemon detail in [SYNC.md](SYNC.md).

## 0. One-time: get the model files

Download two GGUFs into `<project-root>/models/` (they're NOT in the repo):
- `gemma-4-E2B-it-QAT-Q4_0.gguf` ← extraction (HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`)
- `gemma-4-E4B-it-QAT-Q4_0.gguf` ← answer writing (HuggingFace `lmstudio-community/gemma-4-E4B-it-QAT-GGUF`)

Everything else (Neo4j, Qdrant, Jina embed, BGE rerank, llama.cpp server)
is pulled or installed automatically. **You do need a Python venv:**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 1. Start the stack

```bash
cd <project-root>
docker compose up -d          # Neo4j + Qdrant
bash run.sh serve             # E2B (:8082) + E4B (:8084) + daemon (:8000)
bash run.sh health            # confirm all 3 up + VRAM
```

## 2. Ingest your real engineering doc

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

## 3. Ask (retrieve + synthesize)

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"your question here","domain":"engineering","synthesize":true}' \
  | ./venv/bin/python -m json.tool
```

- `synthesize:true` → E4B writes the answer (needs E4B up).
- `synthesize:false` → retrieval only (contexts, no LLM answer).

Response has `answer`, `graph_hits` (Neo4j), `qdrant_hits` (vectors),
`contexts` (the passages used).

## 4. Keep it in sync

```bash
./venv/bin/python sync_docs.py mydocs --watch   # auto-sync on every file save
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `bash run.sh health` shows a service DOWN | `bash run.sh serve` again; wait ~20s for Jina load |
| `synthesize:true` returns 500 | E4B not up → `bash run.sh serve` (starts E4B) or use `synthesize:false` |
| 0 entities from a doc | doc may be too short/empty; check `logs/daemon_gpu.log` |
| ingest seems slow (~minutes) | normal for `sliding_window` extraction on long docs; one-time cost, retrieval is fast |

That's the whole flow. See [RUN.md](RUN.md) for internals and [SYNC.md](SYNC.md)
for the file-sync daemon options.
