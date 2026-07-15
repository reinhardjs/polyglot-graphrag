# V3.0 Hybrid Index-Routing Stack — Implementation Brief

**Target Agent:** hy3
**Author:** Lead Architect
**Date:** 2026-07-12
**Repo:** `<project-root>/`

---

## OBJECTIVE

Refactor the v2 GraphRAG pipeline into **v3.0 Hybrid Index-Routing Stack** — GLiNER for entity detection, Qwen2.5-1.5B for indexed relation classification, YAML-driven domain config, and batch-UNWIND Neo4j ingestion. This is a breaking refactor: stop E2B, E4B, Phi-3, and NuExtract; remove GLiREL; rebuild around the new architecture.

---

## ARCHITECTURE CONSTRAINTS (non-negotiable)

1. **GPU Daemon (:8000):** GLiNER-medium-v2.5 only. Jina embeddings move to CPU (serve_cpu.py). No GLiREL. No spaCy. No Phi-3. No NuExtract.
2. **LLM Engine (:8082):** llama.cpp + `Qwen2.5-1.5B-Instruct-Q8_0.gguf`. Flash Attention. Context 2048. This **replaces** the current E2B Gemma-4 service.
3. **Extraction Logic:** GLiNER extracts entities + character spans → map to indices → Qwen classifies relations for entity pairs using **Index-Routing JSON schema** (below).
4. **Ingestion:** Batch Neo4j via UNWIND — one Cypher statement *per relationship type*, not per edge. Nodes merged first, then edges batched.
5. **Domain Config:** Single `domain_config.yaml` replacing all `domains/*.toml`. No hardcoded logic — everything derived from YAML.
6. **Idempotency:** All ingestion supports re-running without duplication (MERGE, not CREATE).

---

## INDEX-ROUTING JSON SCHEMA

This is the core innovation. The LLM never sees entity extraction — it only classifies relationships between already-known entities.

**Input to Qwen (the "route request"):**
```json
{
  "text": "PR-482 authored by bob fixes BUG-204. See ADR-014 for context.\nalice reviewed the checkout-publisher migration...",
  "entities": [
    {"index": 0, "name": "PR-482",       "type": "PR"},
    {"index": 1, "name": "bob",          "type": "Developer"},
    {"index": 2, "name": "BUG-204",      "type": "Bug"},
    {"index": 3, "name": "ADR-014",      "type": "ADR"},
    {"index": 4, "name": "alice",        "type": "Developer"},
    {"index": 5, "name": "checkout-publisher", "type": "Microservice"}
  ],
  "pairs": [
    {"source": 0, "target": 1},
    {"source": 0, "target": 2},
    {"source": 0, "target": 3},
    {"source": 1, "target": 2},
    {"source": 4, "target": 0},
    {"source": 5, "target": 2}
  ],
  "relation_types": ["FIXES", "AUTHORED", "REFERENCES", "IMPACTS", "DEPENDS_ON", "REVIEWED"]
}
```

**Output from Qwen (the "route response"):**
```json
{
  "relations": [
    {"source": 1, "target": 0, "type": "AUTHORED", "confidence": 0.94},
    {"source": 0, "target": 2, "type": "FIXES",    "confidence": 0.97},
    {"source": 0, "target": 3, "type": "REFERENCES","confidence": 0.82},
    {"source": 4, "target": 0, "type": "REVIEWED",  "confidence": 0.88},
    {"source": 5, "target": 2, "type": "FIXES",     "confidence": 0.91}
  ],
  "no_relation": [3, 5]
}
```

**Rules:**
- `pairs` is all-against-all (n×(n-1)/2) OR smart filtering (same-sentence pairs only).
- `no_relation` lists pair indices where no relationship exists.
- `confidence` threshold at 0.6 — below that, treat as no-relation.
- Qwen prompt MUST include the schema as a JSON example. No free-form text extraction.

---

## VRAM BUDGET (Target: 12 GB total)

```
GLiNER (daemon)  : ~0.5 GB
Qwen 1.5B Q8_0   : ~1.5 GB
Qwen KV cache    : ~1.0 GB (2048 ctx, q4_0 cache)
Daemon misc      : ~2.3 GB (Jina on CPU + GLiNER overhead)
─────────────────────────────────
Total            : ~5.3 GB  (well within 12 GB)
```

---

## EXECUTION PHASES

### Phase 0: Clean Slate

1. Stop all current services:
   ```bash
   sudo systemctl stop gemma-4-e2b.service gemma-4-e4b.service rag-gpu-daemon.service
   pkill -f "llama-server.*8088"; pkill -f "llama-server.*8090"
   ```
2. Remove E2B/E4B/Phi-3/NuExtract systemd units (keep rag-gpu-daemon — we'll reconfigure it).
3. Purge GLiREL from HF cache:
   ```bash
   rm -rf <hf-cache>/models--jackboyla--glirel-large-v0/
   ```
4. Delete corrupted Qwen GGUF:
   ```bash
   rm -f <data-root>/models/Qwen2.5-1.5B-Instruct-Q4_0.gguf
   ```
5. Verify VRAM baseline: `nvidia-smi` should show <500MB used.

**Gate:** VRAM < 500MB before proceeding.

---

### Phase 1: Qwen2.5-1.5B-Q8_0 Download + Server

1. Download the Q8_0 GGUF. Use `huggingface-cli` (with resume support — curl failed 3x in testing):
   ```bash
   <legacy-venv>/bin/pip install -q huggingface-hub[cli] 2>/dev/null
   <legacy-venv>/bin/huggingface-cli download \
     Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf \
     --local-dir <data-root>/models/ --local-dir-use-symlinks False
   ```
   Expected: ~1.5GB, sha256 verifiable.

2. Create systemd unit `qwen-index-router.service`:
   ```ini
   [Unit]
   Description=Qwen 1.5B Index-Routing LLM (:8082)
   After=network.target

   [Service]
   Type=forking
   Restart=no
   User=reinhard
   ExecStart=/home/reinhard/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1/llama-server \
       -m <data-root>/models/qwen2.5-1.5b-instruct-q8_0.gguf \
       -c 2048 -t 12 -tb 12 --gpu-layers 999 \
       --flash-attn \
       --cache-type-k q4_0 --cache-type-v q4_0 \
       --parallel 4 --kv-unified \
       --host 0.0.0.0 --port 8082 --no-warmup
   ExecStartPost=/bin/bash -c 'for i in $(seq 1 60); do curl -s --max-time 2 http://localhost:8082/v1/models >/dev/null && exit 0; sleep 1; done; exit 1'

   [Install]
   WantedBy=multi-user.target
   ```

3. Start and verify:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start qwen-index-router.service
   curl -s http://localhost:8082/v1/models | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
   ```

**Gate:** VRAM increase ~1.5GB, Qwen responds on :8082 with valid model listing.

---

### Phase 2: Rewrite GPU Daemon (GLiNER Only)

1. Edit `v2/serve_gpu.py`:
   - Remove Jina embedding model loading (move to CPU).
   - Keep GLiNER lazy-load on first `/extract_entities` call.
   - New endpoint: `POST /extract_entities` — accepts `{"text":"...", "labels":["Bug","PR",...]}` → returns `{"entities":[{"name":"...","type":"...","start":0,"end":6},...]}`.
   - Remove `/extract_graph` (old GLiNER+co-occurrence endpoint — no longer needed).
   - Keep `/embed` health check but note it's now CPU-routed (redirect to :8001).
   - Keep `/health` endpoint.

2. Update systemd unit `rag-gpu-daemon.service` to reflect new role: "GPU Daemon — GLiNER only".

3. Start and test:
   ```bash
   sudo systemctl start rag-gpu-daemon.service
   curl -s -X POST http://localhost:8000/extract_entities \
     -H "Content-Type: application/json" \
     -d '{"text":"PR-482 fixes BUG-204 authored by bob","labels":["Bug","PR","Developer"]}'
   ```

**Gate:** Response contains >0 entities with correct `start`/`end` character offsets and `type`.

---

### Phase 3: YAML Domain Configuration

1. Create `v2/domain_config.yaml`:
   ```yaml
   domains:
     engineering:
       collection: engineering_chunks
       neo4j_label: Engineering
       entity_types: [Microservice, Database, API, Metric, Developer, Framework, Component, Bug, PR, ADR]
       relation_types: [FIXES, AUTHORED, REFERENCES, IMPACTS, DEPENDS_ON, REVIEWED]
       pair_strategy: all_pairs
       min_confidence: 0.6
       chunking:
         strategy: sentence
         chunk_size: 384
         overlap: 0
       metadata_schema:
         fields: [author, doc_type]
     legal:
       collection: legal_chunks
       neo4j_label: Legal
       entity_types: [Statute, Case, Regulation, Court, Party, Citation, Clause, Jurisdiction, Agency]
       relation_types: [CITES, OVERRULES, AMENDS, INTERPRETS, APPLIES_TO, GOVERNS, REFERENCES, SUPERSEDES]
       pair_strategy: same_sentence
       min_confidence: 0.6
       chunking:
         strategy: paragraph
         chunk_size: 512
         overlap: 50
       metadata_schema:
         fields: [jurisdiction, court, date]
     medical:
       collection: medical_chunks
       neo4j_label: Medical
       entity_types: [Symptom, Diagnosis, Disease, Medication, Procedure, LabTest, Anatomy, Patient, Doctor]
       relation_types: [TREATS, CAUSES, INDICATES, CONTRAINDICATES, PRESCRIBED_FOR, DOSAGE, ADMINISTERED_BY]
       pair_strategy: all_pairs
       min_confidence: 0.65
       chunking:
         strategy: sentence
         chunk_size: 384
         overlap: 0
       metadata_schema:
         fields: [patient_id, report_date]
     accounting:
       collection: accounting_chunks
       neo4j_label: Accounting
       entity_types: [Account, Transaction, Ledger, TaxCode, Invoice, Vendor, Customer, Payment, JournalEntry]
       relation_types: [DEBITS, CREDITS, POSTS_TO, REFERENCES, PAYS, OWES, APPLIES_TO, RECONCILES]
       pair_strategy: same_sentence
       min_confidence: 0.6
       chunking:
         strategy: paragraph
         chunk_size: 512
         overlap: 50
       metadata_schema:
         fields: [fiscal_year, entity_name]
     hospitality:
       collection: hospitality_chunks
       neo4j_label: Hospitality
       entity_types: [Recipe, Ingredient, Station, Dish, Supplier, Equipment, Procedure, Menu, Staff]
       relation_types: [CONTAINS, PREPARED_BY, SOURCED_FROM, REQUIRES, SERVED_WITH, REPLACES, PAIRS_WITH]
       pair_strategy: all_pairs
       min_confidence: 0.55
       chunking:
         strategy: sentence
         chunk_size: 384
         overlap: 0
       metadata_schema:
         fields: [venue, meal_period]
   ```

2. Create `v2/domain_loader.py`:
   - Load YAML → return domain dict by name.
   - Validate: all required keys present, relation_types ⊂ canonical set.
   - Cache in memory (reload on SIGHUP or `/admin/reload` endpoint).
   - Default domain: `engineering`.

3. **Do NOT delete the old TOML files** (per non-destructive preference). Move them to `v2/domains/.archive/` with a README explaining the migration.

**Gate:** `python -c "from domain_loader import get_domain; d=get_domain('engineering'); assert d['relation_types'] == ['FIXES','AUTHORED','REFERENCES','IMPACTS','DEPENDS_ON','REVIEWED']"` passes.

---

### Phase 4: Index-Routing Extraction Pipeline

This is the core of the refactor. Create `v2/index_router.py` with three functions:

1. **`build_route_request(text, entities, relation_types, pair_strategy)`**
   - Build entity pairs: if `"all_pairs"`, generate n×(n-1)/2 pairs. If `"same_sentence"`, only pair entities whose spans overlap in the same sentence.
   - Return the Index-Routing JSON payload (as defined in the schema above).

2. **`query_routes(payload, qwen_url="http://localhost:8082/v1/chat/completions")`**
   - Send the payload to Qwen.
   - Prompt structure:
     ```
     You are a relationship classifier. Given a text and a list of entity pairs,
     determine which pairs have a relationship and what type. Use ONLY these types:
     {relation_types}. Return ONLY valid JSON — no prose, no markdown.

     Text:
     {payload.text}

     Entity pairs to classify:
     {formatted_pairs}

     Output format:
     {"relations":[{"source":0,"target":1,"type":"FIXES","confidence":0.9}],"no_relation":[2]}
     ```
   - Parse response, validate against schema, filter by `min_confidence`.

3. **`extract(text, domain_name, qwen_url)`**
   - Orchestrator: call GLiNER → build route request → call Qwen → return `{"entities":[...], "relations":[...]}`.
   - GLiNER labels come from `domain.entity_types`.

**Gate:** Run against `/tmp/prove_extraction.md` (the same test doc used for E2B/Phi-3/NuExtract comparison):
```python
result = extract(text, "engineering", "http://localhost:8082")
# Must produce >=3 typed relations (FIXES, AUTHORED, REFERENCES)
# Latency target: GLiNER ~0.2s + Qwen ~2s = ~2.2s total
```

---

### Phase 5: Batch UNWIND Neo4j Ingestion

Rewrite `v2/ingest.py` `_write_to_neo4j()`:

1. **Node batch (single query):**
   ```cypher
   UNWIND $nodes AS node
   MERGE (n:Entity:Engineering {id: node.id})
   SET n.name = node.name, n.type = node.type,
       n.source_doc = node.source_doc, n.source_docs = node.source_docs,
       n.name_vector = node.name_vector
   ```

2. **Edge batch — ONE query per relationship type:**
   ```python
   edges_by_type = {}
   for rel in relations:
       edges_by_type.setdefault(rel["type"], []).append(rel)

   for rel_type, batch in edges_by_type.items():
       cypher = f"""
       UNWIND $edges AS edge
       MATCH (a:Entity {{id: edge.source_id}})
       MATCH (b:Entity {{id: edge.target_id}})
       MERGE (a)-[:{rel_type}]->(b)
       """
       tx.run(cypher, edges=batch)
   ```

3. **Remove** the old per-edge MERGE loop. Replace with this UNWIND pattern.

4. **Update `ingest_text()`** to accept `domain` param, load domain config via `domain_loader`, use `index_router.extract()` instead of the old E2B endpoint.

5. **Validate** `source_id` / `target_id` use stable entity IDs (the `_gen_entity_id` function already exists — reuse it).

**Gate:** Ingest the test doc, then verify in Neo4j:
```cypher
MATCH (a)-[r]->(b) WHERE a.name CONTAINS 'PR-482' RETURN a.name, type(r), b.name
```
Should return FIXES, AUTHORED, REFERENCES edges (not ASSOCIATED_WITH for these).

---

### Phase 6: Server Contract Update

1. **`POST /ingest`** in `serve_gpu.py`:
   - Add `domain` parameter (default: `engineering`).
   - Load domain config → extract with `index_router.extract()` → batch-write to Neo4j.
   - Return `{"status":"ok", "nodes":N, "edges":M, "domain":"engineering"}`.

2. **`POST /ask`** — unchanged for this phase (retrieval pipeline stays).
   - But verify it still works: query endpoint → receives response with sources.

3. **New endpoint: `GET /admin/domains`** — list available domains from `domain_config.yaml`.

4. **New endpoint: `POST /admin/reload`** — reload `domain_config.yaml` without restart.

**Gate:** `curl -X POST http://localhost:8000/admin/domains` returns list of 5 domains.

---

### Phase 7: Tests

Create `v2/tests/test_index_router.py` with:
1. `test_build_route_request_all_pairs` — verify pair count = n×(n-1)/2
2. `test_query_routes_mock` — mock Qwen response, verify parsing
3. `test_extract_end_to_end` — live test against Qwen with prove_extraction.md
4. `test_domain_loader_valid` — all 5 domains load without errors
5. `test_domain_loader_missing_key` — raises on malformed config
6. `test_unwind_ingestion` — verify batch UNWIND produces correct edge types in Neo4j
7. `test_idempotent_ingestion` — double-ingest same doc, no duplicate edges

Run: `pytest tests/test_index_router.py -v`
**Gate:** All tests pass.

---

### Phase 8: Systemd Finalization + Cleanup

1. Enable all services for boot-only:
   ```bash
   sudo systemctl enable rag-gpu-daemon.service qwen-index-router.service
   ```
2. Start E4B only if needed for `/ask` synthesis (check if `serve_gpu.py`'s `/ask` endpoint uses it — if yes, keep it; if no, leave it down).
3. Verify full pipeline round-trip:
   ```bash
   # Ingest a test doc
   curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"text":"...", "doc_id":"test-v3", "domain":"engineering"}'
   # Query for it
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query":"what fixes BUG-204?", "domain":"engineering"}'
   ```
4. Update `docs/architecture/neuro-symbolic-v3.0.0.md` with the Index-Routing flow diagram (ASCII art showing GLiNER → entity indices → Qwen → batch UNWIND → Neo4j).

**Gate:** Full round-trip (ingest → query) produces a correct answer with source citations.

---

## DELIVERABLES (report back after each phase)

For each phase, hy3 must report:
1. **Code diff** (files changed, lines added/removed)
2. **Gate test output** (copy-paste the verification command + its output)
3. **VRAM snapshot** (`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader`)
4. **Latency measurement** (Phase 4: GLiNER time + Qwen time; Phase 8: full round-trip)

---

## ERROR CHECKLIST (what NOT to do)

- ❌ Do NOT reinstall GLiREL or spaCy.
- ❌ Do NOT start Phi-3 or NuExtract servers (they're not in the architecture).
- ❌ Do NOT use `sed -i` on any `.yaml` or `.py` file — use `patch` tool or write-then-cp.
- ❌ Do NOT call `.half()` on CPU models.
- ❌ Do NOT create new TOML files — all config goes in `domain_config.yaml`.
- ❌ Do NOT write single-edge MERGE queries — use UNWIND batches.
- ❌ Do NOT hardcode the Qwen prompt — it must load `relation_types` from `domain_config.yaml` at runtime.
- ❌ Do NOT delete old TOML files — move them to `.archive/`.
- ❌ Do NOT use `curl` for the Qwen GGUF download — use `huggingface-cli` (supports resume).
- ❌ Do NOT skip a phase gate — sequential execution only.

---

## APPENDIX A: Current State Before Refactor

| Component | Current | Target |
|-----------|---------|--------|
| GPU Daemon (:8000) | Jina + GLiNER lazy (3.9GB) | GLiNER only |
| LLM Engine (:8082) | E2B Gemma-4 (2.6GB) | Qwen 1.5B Q8_0 |
| Phi-3 (:8088) | Running | Shut down |
| NuExtract (:8090) | Running | Shut down |
| E4B (:8084) | Running (4.3GB) | Shut down |
| GLiREL | Cache (1.4GB) | Purge |
| Domain Config | individual TOMLs | Single YAML |
| Ingestion | Per-edge MERGE | Batch UNWIND |

## APPENDIX B: Proven Extraction Benchmarks (same test doc)

| Model | Time | VRAM | Edges | Quality |
|-------|------|------|-------|---------|
| E2B (Gemma 4) | 10.2s | 2.6GB | 5/5 correct | Best |
| Phi-3-mini (Q4) | 2.9s | 1.5GB | 5 (2 reversed) | Good |
| NuExtract (Q4_K_M) | 29.8s | 3.0GB | 4/4 correct | Best directionality |
| GLiNER only | 0.2s | 0.5GB | 30 (ASSOCIATED_WITH) | No semantics |
| GLiNER + GLiREL | 208s+ | 1.5GB | 90 garbage | Rejected |

**Target for Index-Routing (Phase 4):** ~2.2s total, >=3 typed edges, all correct.

## APPENDIX C: Test Document

Available at `/tmp/prove_extraction.md` — use this for all extraction benchmarks to ensure comparable results.
