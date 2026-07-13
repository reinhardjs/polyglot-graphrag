# RUN.md — Run GraphRAG (all-in-one, from zero)

This guide assumes **you have never heard of Gemma, llama.cpp, or E2B** and just
want to get the whole system running. It explains every moving part in plain
terms, what you need, and the limits you must know before it will work.

---

## 0. What this system is (30-second version)

GraphRAG ingests your documents (markdown/text) and builds **two** searchable
stores at once:

- **Qdrant** — a *vector* store. Each chunk of text becomes a number-array
  (embedding) so we can find "similar text" by meaning, not keywords.
- **Neo4j** — a *graph* store. The system extracts *entities* (BUG-204, bob,
  checkout service…) and *relationships* (bob → reported → BUG-204) into a
  knowledge graph.

When you ask a question (`/ask`), it searches BOTH stores, fuses the results,
and (optionally) asks a language model to write a natural-language answer.

```
        ┌─────────────── INGEST ───────────────┐
 text ──► chunk ──► embed (Jina) ──► Qdrant      │
        │                  │                     │
        │            extract graph (Gemma E2B) ──┤──► Neo4j
        └───────────────────────────────────────┘

        ┌─────────────── ASK ───────────────────┐
 query ─► embed ─► Qdrant (vector) + Neo4j (graph) ─► rerank (BGE) ─► answer
                                                    (Gemma E4B, optional)
        └────────────────────────────────────────┘
```

---

## 1. Requirements (read before you start)

| Requirement | Why | Minimum |
|-------------|-----|---------|
| **NVIDIA GPU** | Embeddings + both LLMs run on CUDA | RTX 3060 12 GB (tested) |
| **~12 GB VRAM free** | E2B (~2.3 GB) + daemon models (Jina/MiniLM/BGE ~2.5 GB) | 8 GB *will* stutter |
| **Docker + docker compose** | Runs Neo4j + Qdrant | any recent version |
| **Python 3.11 venv** | Runs the FastAPI daemon + ingest | 3.11 |
| **~25 GB disk** | models (E2B 3.3 GB, E4B 5.1 GB) + docker images + data | 30 GB |
- **Linux** | llama.cpp CUDA build | Ubuntu 22.04 (tested) |

### Install Python dependencies

```bash
cd <project-root>
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # fastapi, torch(cu121), gliner, spacy, …
python -m spacy download en_core_web_sm  # sliding_window sentence tokenization
```

`requirements.txt` pins PyTorch to the CUDA 12.1 wheel (via
`--extra-index-url https://download.pytorch.org/whl/cu121`); on a CPU-only
host drop that line and use `torch==2.3.1`. `gliner`, `spacy`, `watchdog`
(for `--watch`), `fastapi`/`uvicorn` are all required at runtime — they were
previously undocumented.

**You must download two model files yourself** (they are not in the repo —
too large, and license/distribution constraints):

1. **Gemma 4 E2B** (extraction) — `gemma-4-E2B-it-QAT-Q4_0.gguf`
   - Source: HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`
   - Place at: `<project-root>/models/gemma-4-E2B-it-QAT-Q4_0.gguf`
   - **Required launch flag:** `--reasoning off`. Gemma-4 routes answers to
     `reasoning_content` (the `<|channel>thought` token) by default, leaving
     `message.content` EMPTY — extraction would get nothing. `run.sh` sets this
     automatically.
2. **Gemma 4 E4B** (answer synthesis, *optional*) — `gemma-4-E4B-it-QAT-Q4_0.gguf`
   - Source: HuggingFace `lmstudio-community/gemma-4-E4B-it-QAT-GGUF`
   - Place at: `<project-root>/models/gemma-4-E4B-it-QAT-Q4_0.gguf`

> **What is "Gemma E2B"?** Gemma 4 is Google's open model family. **E2B** =
> the "Extraction 2-Billion-parameter" variant — small/fast, used here to pull
> entities & relationships out of text. **E4B** = 4-Billion variant, used to
> *write* the final answer. Both run locally via **llama.cpp** (a C++
> engine that serves GGUF-format models over an OpenAI-compatible HTTP API).
> Nothing leaves your machine.

> **Limitation — synthesis needs E4B.** If E4B is NOT running, `/ask` with
> `synthesize:true` returns a 500 error. Use `synthesize:false` for
> retrieval-only (contexts only, no LLM-written answer). This is by design, not
> a bug — see "Known limitations" at the bottom.

---

## 2. One-command-ish startup

```bash
cd <project-root>

# 1) Supporting stores (Neo4j + Qdrant) — Docker
docker compose up -d

# 2) GPU LLMs (Gemma E2B on :8082, E4B on :8084) + the FastAPI pipeline daemon
#    in ONE command. This is the recommended path — it launches llama-server
#    directly (no systemd/sudo needed) and waits for each service to be ready.
bash run.sh serve

# 3) Health check (all three must be up)
bash run.sh health
```

> The `run.sh` helper launches E2B with `--ctx-size 32768` (required so the
> parallel sliding-window extractor doesn't overflow the shared KV cache — see
> §3) and exports `SW_EXTRACT_WORKERS=4`. No sudo, no systemd units needed.

Handy `run.sh` subcommands:
```bash
bash run.sh serve      # start LLMs + daemon (idempotent)
bash run.sh health     # check everything (LLMs + daemon + VRAM)
bash run.sh ask "who reported BUG-204?"   # full pipeline (retrieve + synthesize)
bash run.sh retrieve "who reported BUG-204?"  # retrieval only (no synthesis)
bash run.sh stop       # stop daemon + LLMs
```

> **Models:** place the two GGUF files in `<project-root>/models/` (see §1).
> `run.sh` resolves them there automatically. The legacy systemd units point at
> `/home/reinhard/.lmstudio/...` and are NOT used in the portable setup.

---

## 3. Ingest your data (the flow)

There are two ways to ingest. **For real documents you'll edit over time, use
`sync_docs.py`** (Option A) — it tracks file→`doc_id` and keeps Neo4j + Qdrant
consistent with your files via SHA256 change detection (like git). For a
one-off, use the raw API (Option B).

### Option A — `sync_docs.py` (recommended for real docs)

```bash
cd <project-root>

# 1) Put your docs in a folder. Sub-path becomes the doc_id; eng/ → engineering.
mkdir -p mydocs/eng
cp /path/to/your-real-doc.md mydocs/eng/

# 2) Sync the folder → POST /ingest for each new/changed file
./venv/bin/python sync_docs.py mydocs

# 3) (Optional) Watch the folder and auto-sync on every save:
./venv/bin/python sync_docs.py mydocs --watch
```

- `doc_id` = relative path (`eng/your-real-doc.md`); `domain` inferred from the
  path (`eng/`→engineering, `journal/` or `*.pdf`→journal, `legal/`→legal).
- Re-running skips unchanged files; edits are re-ingested; deleted files are
  removed from BOTH stores. State lives in `mydocs/.sync_state.json`.
- **PDFs are auto-extracted** with `pdftotext` — just drop them in the folder.
- See `SYNC.md` for full options (`.syncignore`, `--config`, `--force`, dry-run).

### Option B — raw `/ingest` API (one-off)

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"BUG-204 caused an outage in checkout. bob reported it.",
       "doc_id":"my-doc-1","domain":"engineering"}'

# Poll until done (ingest is async):
curl http://localhost:8000/ingest/status/<task_id>
```

**PDF one-off** (extract text first, then send it):
```bash
pdftotext my-paper.pdf /tmp/my-paper.txt
TID=$(curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"text\":$(python3 -c "import json;print(json.dumps(open('/tmp/my-paper.txt').read()))"),\
       \"doc_id\":\"my-paper\",\"domain\":\"journal\"}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['task_id'])")
curl http://localhost:8000/ingest/status/$TID
```

> **Verify it landed** (replace the doc_id with yours):
> ```bash
> docker exec rag-system-neo4j-1 cypher-shell -a bolt://localhost:7687 \
>   -u neo4j -p ragpassword123 \
>   "MATCH (n:Entity) WHERE 'eng/your-real-doc.md' IN n.source_docs \
>    RETURN n.name, n.type LIMIT 20;"
> curl -s "http://localhost:6333/collections/engineering_chunks/points/count" \
>   -H "Content-Type: application/json" \
>   -d '{"filter":{"must":[{"key":"doc_id","match":{"value":"eng/your-real-doc.md"}}]}}'
> ```

**What happens inside (so you can debug):**

1. **Chunk** — text is split (late-chunking strategy per domain).
2. **Embed** — each chunk → 1024-d vector via **Jina v3** (multilingual) on GPU.
3. **Store vectors** — upserted into the domain's Qdrant collection
   (`engineering_chunks`, `journal_chunks`, …), tagged with `doc_id`.
4. **Extract graph** — default mode is **`sliding_window`**: the doc is split
   into windows, each window's entities are found (GLiNER) and relationships
   extracted (Gemma E2B) **in parallel** (`SW_EXTRACT_WORKERS=4`). This yields
   far more entities than the old single-pass mode (~130 entities on the RAPTOR
   paper vs ~22 before the fix).
5. **Resolve + write graph** — each entity name is embedded with Jina and
   matched against existing Neo4j nodes (≥0.88 cosine → reuse; else create).
   Edges carry `source_docs` so deletes are exact (no stale edges). See
   MIGRATION.md for the forward-compatibility contract.

> Re-running ingest on the same `doc_id` **replaces** that doc's data cleanly
> (old chunks + that doc's nodes/edges removed, new written). Shared entities
> across docs are merged, never duplicated.

---

## 4. Ask / retrieve (the flow)

```bash
# Retrieval + synthesis (needs E4B on :8084):
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","domain":"engineering","synthesize":true}'

# Retrieval only (no E4B needed):
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"who reported BUG-204?","synthesize":false}'
```

**What happens inside:**

1. **Embed query** → Jina v3 vector.
2. **Qdrant** — top-K similar chunks from the domain collection.
3. **Neo4j** — expand the entity subgraph around matched concepts (graph
   retrieval adds relational context vectors can't).
4. **Rerank** — BGE reranker scores fused candidates by relevance.
5. **Synthesize** (if `synthesize:true`) — top contexts → **Gemma E4B** writes
   the answer. If E4B is down → 500 (use `synthesize:false`).

Response contract: `{query, source, path, qdrant_hits, graph_hits,
n_contexts, contexts[], answer?}`.

---

## 5. Known limitations (must read)

- **E4B synthesis needs E4B running.** `synthesize:true` 500s unless E4B
  (`:8084`) is up. Use `synthesize:false` for retrieval-only. `bash run.sh
  serve` starts E4B by default, so this is only an issue if you skipped it.
- **VRAM is tight on 12 GB.** E2B (~2.3 GB) + daemon models (Jina/MiniLM/BGE
  ~2.5 GB) + E4B (~3.6 GB) ≈ 11 GB. All three fit on a 12 GB card but leave
  little headroom; avoid concurrent heavy ingest + many parallel queries. To
  free VRAM, run E4B only when needed or offload the BGE reranker to CPU.
- **Single-doc `/ingest` is async** — always poll `/ingest/status/{task_id}`.
  `sync_docs.py` handles polling internally.
- **Cross-lingual** matching uses Jina v3 alignment; very different scripts may
  not merge into one node.
- **Edges may be 0 for some documents.** Relation extraction relies on the LLM
  emitting typed relationships; dense narrative papers can yield entities but
  `edges:0`. Graph retrieval still works on the entity nodes. This is a quality
  characteristic, not a failure.
- **PDFs are auto-extracted** by `sync_docs.py` (needs `pdftotext` installed);
  the raw API consumes text, so extract first for one-offs.
- **Keep docs consistent with `sync_docs.py`.** Editing a file on disk WITHOUT
  re-syncing leaves Neo4j/Qdrant stale. Re-run `sync_docs.py <folder>` (or use
  `--watch`) after any change. The daemon does not watch the filesystem itself.

---

## 6. Stop / restart

```bash
bash run.sh stop                 # daemon + LLMs
docker compose down              # Neo4j + Qdrant (DATA IS PRESERVED on host)
```

Data in `<project-root>/data/neo4j` and `<project-root>/data/qdrant`
is **never** touched by stop/restart. Only `docker compose down -v` (with `-v`)
would delete it.
