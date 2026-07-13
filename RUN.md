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
| **Linux** | systemd units + llama.cpp CUDA build | Ubuntu 22.04 (tested) |

**You must download two model files yourself** (they are not in the repo — too
large, and license/distribution constraints):

1. **Gemma 4 E2B** (extraction) — `gemma-4-E2B-it-QAT-Q4_0.gguf`
   - Source: HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`
   - Place at: `/mnt/data-970-plus/models/gemma-4-E2B_q4_0-it.gguf`
2. **Gemma 4 E4B** (answer synthesis, *optional*) — `gemma-4-E4B-it-QAT-Q4_0.gguf`
   - Source: HuggingFace `lmstudio-community/gemma-4-E4B-it-QAT-GGUF`
   - Place at: `/home/reinhard/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-QAT-GGUF/`

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
cd /mnt/data-970-plus/rag-system

# 1) Supporting stores (Neo4j + Qdrant) — Docker
docker compose up -d

# 2) GPU LLMs (Gemma E2B on :8082, E4B on :8084) — systemd
sudo systemctl start gemma-4-e2b.service
sudo systemctl start gemma-4-e4b.service        # optional, for synthesis

# 3) The FastAPI pipeline daemon (embed/rerank/GLiNER + /ask + /ingest)
sudo systemctl start rag-gpu-daemon

# 4) Wait for health
sleep 20
curl -s http://localhost:8000/health            # → {"status":"ok",...}
curl -s http://localhost:8082/v1/models >/dev/null && echo "E2B up"
```

That's the all-in-one. The `run.sh` helper wraps steps 2–3 with readiness waits:

```bash
bash run.sh serve      # start LLMs + daemon (idempotent)
bash run.sh health     # check everything
bash run.sh ingest sample_data   # load the bundled sample docs
bash run.sh ask "who reported BUG-204?"   # full pipeline
bash run.sh stop       # stop daemon + LLMs
```

---

## 3. Ingest your data (the flow)

```bash
# Single document via the HTTP API (non-blocking — poll the status):
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"BUG-204 caused an outage in checkout. bob reported it.",
       "doc_id":"my-doc-1","domain":"engineering"}'

# Poll until done:
curl http://localhost:8000/ingest/status/<task_id_from_above>

# Or bulk-ingest a folder of .md/.txt:
/mnt/data-970-plus/rag-env/bin/python ingest.py /path/to/docs --domain engineering
```

> **PDFs are NOT ingested directly.** The pipeline consumes plain text / markdown
> / txt. To ingest a PDF, extract its text first (system `pdftotext` works), then
> send the text via the API:
> ```bash
> # 1) PDF → text
> pdftotext my-paper.pdf /tmp/my-paper.txt
> # 2) Ingest the extracted text (journal domain example)
> TID=$(curl -s -X POST http://localhost:8000/ingest \
>   -H "Content-Type: application/json" \
>   -d "{\"text\":$(python3 -c "import json;print(json.dumps(open('/tmp/my-paper.txt').read()))"),\
>        \"doc_id\":\"my-paper\",\"domain\":\"journal\"}" \
>   | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)
> # 3) Poll until done
> curl http://localhost:8000/ingest/status/$TID
> ```
> Verified walkthrough: the ICLR 2024 RAPTOR paper (`archive/legacy-v1/data/2401.18059v1.pdf`)
> extracted to 1612 lines → 1882 chunks + 22 entities into `journal_chunks`, and
> `POST /ask` (both `synthesize:false` and `synthesize:true`) returned answers.

**What happens inside (so you can debug):**

1. **Chunk** — text is split (late-chunking strategy per domain: `engineering`
   & `medical` use sentence chunking; others may use fixed/section).
2. **Embed** — each chunk → 1024-d vector via **Jina v3** (multilingual) on GPU.
3. **Store vectors** — upserted into the domain's Qdrant collection
   (`engineering_chunks`, `legal_chunks`, …). A `doc_id` tag lets us delete later.
4. **Extract graph** — the chunk (or whole doc) is sent to **Gemma E2B** over
   `:8082`; it returns JSON `{nodes:[…], edges:[…]}`.
5. **Resolve + write graph** — each entity name is **embedded with Jina** and
   matched against existing Neo4j nodes (≥0.88 cosine → reuse, append alias;
   else create). This *vector-resolution* is what makes re-ingestion safe (see
   MIGRATION.md). Edges are MERGEd between resolved IDs. Domain label
   (`:Engineering`) + optional `:Discovered` tag applied.

> Re-running ingest on the same `doc_id` **merges**, not duplicates — entities
> accumulate aliases/source_docs, never overwrite your graph.

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

- **E4B synthesis is optional but not started by default** in this environment.
  `synthesize:true` 500s unless `gemma-4-e4b.service` is running.
- **Model paths are machine-specific.** `gemma-4-e2b.service` points at
  `/home/reinhard/.lmstudio/...`; the live server here actually uses
  `/mnt/data-970-plus/models/...`. Edit the `.service` `ExecStart -m` path to
  match your download location.
- **VRAM is tight on 12 GB.** E2B + daemon models ≈ 5 GB. E4B adds ~3 GB — if
  you start all three on a 12 GB card, expect OOM. Run E4B only when you need
  synthesis, or lower `--n-gpu-layers`.
- **Single-doc `/ingest` is async** — always poll `/ingest/status/{task_id}`.
- **Cross-lingual** matching uses Jina v3 alignment; very different scripts may
  not merge into one node.
- **Edges may be 0 for some documents.** Relation extraction relies on the LLM
  emitting typed relationships; dense narrative papers (e.g. some journal
  articles) can yield entities but `edges:0`. Graph retrieval (`graph_hits`)
  still works on the entity nodes — it just has fewer relationship paths. This is
  a quality characteristic, not a failure; the doc is fully searchable via vectors.
- **PDFs need pre-extraction** (see §3) — the pipeline ingests text, not PDF bytes.

---

## 6. Stop / restart

```bash
bash run.sh stop                 # daemon + LLMs
docker compose down              # Neo4j + Qdrant (DATA IS PRESERVED on host)
sudo systemctl stop rag-gpu-daemon
```

Data in `/mnt/data-970-plus/neo4j_data` and `/mnt/data-970-plus/qdrant_storage`
is **never** touched by stop/restart. Only `docker compose down -v` (with `-v`)
would delete it.
