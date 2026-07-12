# polyglot-graphrag v3.0.0 — E2B Full-Doc Extraction Stack

**Status:** Deployed (2026-07-12)
**Extraction:** Gemma-4-E2B Q4_0 (llm mode, full-doc single-pass)
**Infra:** Domain-agnostic YAML config, batch UNWIND Neo4j, hybrid retrieval
**Quality:** 89% edge precision (live benchmark vs gold), 28 edges/doc typical

---

## 1. Architecture

```
                         ┌─────────────────────────────────────────┐
                         │         rag-gpu-daemon (:8000)           │
                         │  ┌───────────────────────────────────┐  │
                         │  │ Jina v3 (embed) + bge-reranker    │  │
                         │  │ GLiNER (entity detection, GPU)    │  │
                         │  │ POST /extract_entities            │  │
                         │  └───────────────────────────────────┘  │
                         └─────────────────────────────────────────┘
                                             │
                         ┌───────────────────┴─────────────────────┐
                         │                                          │
                         ▼                                          ▼
                ┌─────────────────┐                      ┌─────────────────────┐
                │ Qdrant (vectors)│                      │ Neo4j (graph)       │
                │ embed_late →    │                      │ batch UNWIND →      │
                │ hybrid retrieval│                      │ typed edges         │
                └─────────────────┘                      └─────────────────────┘
                         │                                          │
                         └───────────────────┬──────────────────────┘
                                             │
                                             ▼
                         ┌─────────────────────────────────────────┐
                         │      Gemma-4-E2B (:8082, llama.cpp)      │
                         │  EXTRACTION_MODE="llm"                   │
                         │  Full-doc single-pass → {nodes, edges}   │
                         │  89% edge precision, ~12s/doc             │
                         └─────────────────────────────────────────┘
```

**Extraction:** Gemma-4-E2B Q4_0 does full-document single-pass JSON extraction.
The old `extract_graph_llm()` path is the primary; index-routing (Qwen 1.5B
batch classification) is preserved as a fallback via `EXTRACTION_MODE` config.

---

## 2. Index-Routing JSON Schema

### Input to Qwen (`build_route_request`)
```json
{
  "text": "<source document text>",
  "entities": [{"index": 0, "name": "PR-482", "type": "PR"}, ...],
  "pairs": [{"source": 0, "target": 1}, ...],
  "relation_types": ["FIXES", "AUTHORED", "DEPENDS_ON", ...]
}
```

### Output from Qwen (`query_routes`)
```json
{
  "relations": [
    {"source": 0, "target": 1, "type": "FIXES", "confidence": 0.9}
  ]
}
```
Source/target are **entity indices** into the entities list (not names), so
there is zero ambiguity in mapping back. Type is constrained to the allowed
`relation_types` vocabulary (enforced server-side in parsing).

---

## 3. Domain Configuration (YAML)

All schema lives in `v2/domain_config.yaml` — consumed by
`v2/domain_loader.py` at runtime. No TOML, no hardcoded logic.

```yaml
engineering:
  collection: engineering_chunks
  neo4j_label: Engineering
  entity_types: [Microservice, Database, API, Metric, Developer, ...]
  relation_types: [FIXES, AUTHORED, REFERENCES, IMPACTS, DEPENDS_ON, REVIEWED]
  pair_strategy: all_pairs          # or: same_sentence, window
  min_confidence: 0.6
  chunking:
    strategy: sentence
    chunk_size: 512
    overlap: 64
```

Domains shipped: `engineering`, `legal`, `medical`, `accounting`, `hospitality`.

**Runtime API:**
- `GET /admin/domains` → list all domains + schemas
- `POST /admin/reload` → hot-reload YAML without daemon restart
- `GET /admin/domains` reads from `domain_loader.get_domain(name)`

---

## 4. Server Contract (rag-gpu-daemon :8000)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/extract_entities` | POST | GLiNER entity detection → spans |
| `/ingest` | POST | Full ingest (text + `domain` param) → 202 + task_id |
| `/ingest/status/{task_id}` | GET | Poll extraction result |
| `/admin/domains` | GET | List domain schemas |
| `/admin/reload` | POST | Hot-reload domain_config.yaml |
| `/health` | GET | Daemon health (CUDA GB) |
| `/models` | GET | Loaded model identities |

**Ingest request:**
```json
{"text": "...", "doc_id": "doc-1", "domain": "engineering", "extract_graph": true}
```

---

## 5. Systemd Services

| Unit | Port | Role | Boot |
|------|------|------|------|
| `rag-gpu-daemon.service` | 8000 | GLiNER + Jina embed + bge-reranker | **enabled** |
| `qwen-index-router.service` | 8082 | Gemma-4-E2B Q4_0 llm extraction | disabled (manual) |
| `gemma-4-e4b.service` | 8084 | gemma-4-E4B synthesis (archived) | disabled |

**VRAM budget (measured):** E2B Q4_0 ≈ 2.3 GB + Jina v3 ≈ 3.9 GB + GLiNER ≈
1.4 GB ≈ **7.6 GB** total. Peak pipeline ~8 GB / 12 GB.

**Start the stack:**
```bash
sudo systemctl start rag-gpu-daemon      # GLiNER + Jina
sudo systemctl start qwen-index-router   # E2B extraction
```

---

## 6. Batch UNWIND Ingestion

`write_graph()` in `ingest.py` groups resolved edges by relationship type and
issues **one `UNWIND` query per type** instead of one `MERGE` per edge:

```cypher
UNWIND $edges AS edge
MATCH (a:Entity {id: edge.source_id})
MATCH (b:Entity {id: edge.target_id})
MERGE (a)-[:REL_TYPE]->(b)
```

Relationship types are validated/sanitized before interpolation (only
`[A-Z0-9_]`, uppercased, must start with letter/`_`). Self-loops and
out-of-range indices are dropped during parsing.

---

## 7. Pair Batching

The 1.5B Qwen degrades when asked to classify many pairs in one prompt.
`query_routes()` chunks the candidate pairs into batches of `batch_size=6` and
merges the classified relations (dedup by `(src, tgt, type)`). This keeps
per-call context small and output under `max_tokens`, avoiding truncation.

---

## 8. Tests

`v2/test_index_router.py` (pytest):
- **15 pure-logic tests** (no GPU): pairing strategies, truncated-JSON
  salvage, vocabulary enforcement, confidence gate, index bounds, self-loop
  rejection, dedup, YAML schema integrity.
- **3 live integration tests** (gated by `IR_LIVE=1`): real GLiNER + Qwen
  extraction against `:8000` and `:8082`.

```bash
cd /mnt/data-970-plus/rag-system/v2
python -m pytest test_index_router.py -q            # logic only
IR_LIVE=1 python -m pytest test_index_router.py -q  # + live servers
```

---

## 9. Known Limitations / Quality Notes

- **Qwen Q4_K_M is a 1.5B model.** Relation classification is *functional* but
  not perfect — it occasionally over-predicts (e.g. assigns AUTHORED to pairs
  that only have a weaker relation). For higher precision, swap
  `EXTRACTION_LLM_MODEL` in `config.py` to a larger model (Qwen2.5-7B,
  Phi-3-mini) — the pipeline is model-agnostic.
- **Q8_0 quantization is BROKEN** on this llama.cpp build (2.23.1): produces
  garbage output. Use Q4_K_M (or Q5_K_M). See Phase 1 notes.
- **KV-cache quantization (`--cache-type-k q4_0`) corrupts long outputs** on
  small models — use f16 KV cache (default in the systemd unit).
- GLiNER entity spans are character-offset based; entity *types* come from the
  domain's `entity_types` vocabulary passed as `labels`.

---

## 11. Full Benchmark Reference

All measured performance data lives in **[BENCHMARKS.md](BENCHMARKS.md)**. Key
headlines:

| Dimension | Value | Source |
|-----------|-------|--------|
| Extraction precision (E2B llm) | **89%** | BENCHMARKS §1 |
| Extraction precision (Qwen index_routing) | 20% | BENCHMARKS §1 |
| E2B extraction latency | 10.5s/doc | BENCHMARKS §1 |
| Hybrid retrieval latency | 127ms | BENCHMARKS §3 |
| VRAM (idle stack) | 5.8 GB / 12 GB | BENCHMARKS §4 |
| E2B generation speed | 129 tok/s | BENCHMARKS §2 |

**Next phase plan:** See **[NEXT_PHASE_PLAN.md](NEXT_PHASE_PLAN.md)** for
GLiNER+E2B hybrid mode, E2B server tuning, and higher Qwen intelligence tests.

### Retrieval Stack (extraction-independent)

| Model | Role | In retrieval? | VRAM |
|-------|------|---------------|------|
| **Jina v3** (jina-embeddings-v3, 1024d, fp16) | Query + passage embed | ✅ core | ~3.9 GB |
| **bge-reranker-v2-m3** (fp16) | Rerank fused candidates | ✅ | shared w/ Jina |
| **Neo4j** (DB) | Graph traversal | ✅ | — |
| **Qdrant** (DB) | Vector store | ✅ | — |
| GLiNER / E2B / E4B | extraction / synthesis | ❌ | — |

**Latency (median of 5, warm, post-fix):**
- Query embed (Jina v3 GPU): **58 ms**
- Rerank 20 docs (bge-reranker): **1 ms** (CPU pool)
- Full `/ask` HYBRID retrieve (embed + qdrant + graph + rerank): **127 ms**

**Two graph-retrieval bugs found & fixed during perf review:**
1. v3 ingestion wrote nodes as `:Entity` only (missing domain label
   `:Engineering`) → `neo4j_subgraph(label="Engineering")` matched nothing.
   Fixed: `write_graph` now tags nodes with the domain `neo4j_label` from
   `domain_config.yaml`.
2. Cypher syntax error in 2-hop traversal (`WITH collect(DISTINCT m) + n AS ns`
   — aggregation without grouping). Fixed: proper `WITH ... AS ms, n AS entry`
   then `WITH ms + entry AS ns`.

After fixes, true hybrid retrieval is confirmed (graph_hits > 0 in `/ask`).

---

## 12. Quick Start

```bash
# 1. Start services
sudo systemctl start rag-gpu-daemon      # GLiNER + Jina + reranker (:8000)
sudo systemctl start qwen-index-router   # E2B extraction (:8082)
# (or start E2B directly if sudo unavailable — see BENCHMARKS §7)

# 2. Ingest a document (engineering domain)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"PR-482 authored by bob fixes BUG-204.","doc_id":"demo-1","domain":"engineering"}'

# 3. Ask a question (hybrid retrieval)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What does PR-482 fix?","domain":"engineering"}'

# 4. Query the graph directly
# MATCH (a:Entity:Engineering)-[r]->(b:Entity:Engineering) RETURN a.name, type(r), b.name

# 5. Full demo pipeline
python -c "
import ingest
text = open('/tmp/prove_extraction.md').read()
res = ingest.ingest_text(text, 'demo-full', domain='engineering', extract_graph=True)
print(f'Extracted {res[\"entities\"]} entities via {res[\"extraction_method\"]}')
"
```

---

## 13. Known Limitations

- **E2B context capped at 4096** — documents >4K tokens may truncate edges.
  Fix: raise `-c 8192` in systemd unit (+512 MB VRAM). Tracked in
  [NEXT_PHASE_PLAN.md](NEXT_PHASE_PLAN.md) Workstream 2.
- **Single ASSOCIATED_WITH fallback** (1 of 9 edges). Prompt tuning target.
- **Systemd unit naming** — `qwen-index-router.service` actually runs Gemma-4-E2B
  (unit name preserved for backward compatibility). Rename is cosmetic.
- **E2B duplicate edges** — E2B emits duplicates of same relation which MERGE
  deduplicates in Neo4j. No correctness impact, minor latency cost.
- **Qwen index_routing mode preserved as fallback** — switch
  `EXTRACTION_MODE="index_routing"` for fast/low-quality extraction if E2B
  is unavailable.

---

## 14. Document Index

| Document | Purpose |
|----------|---------|
| [`ARCHITECTURE_V3.0.0.md`](ARCHITECTURE_V3.0.0.md) | System architecture (this file) |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Comprehensive performance data |
| [`NEXT_PHASE_PLAN.md`](NEXT_PHASE_PLAN.md) | Execution plan for hy3: GLiNER+E2B hybrid, tuning, higher Qwen |
| `V3_IMPLEMENTATION_BRIEF.md` | Original design brief (historical) |
| `domain_config.yaml` | Domain vocabulary definitions |
| `config.py` | Runtime configuration |

---

*Architecture decision: E2B full-doc llm extraction (89% precision, 10.5s)
chosen over Qwen index-routing (20% precision, 1.2s) after live benchmarking.
Domain-agnostic YAML config, batch UNWIND Neo4j ingestion, and hybrid
vector+graph retrieval form the operational shell. See BENCHMARKS.md for all
evidence.*
