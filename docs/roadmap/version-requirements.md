# GraphRAG v2 — Version Requirements

**Created:** 2026-07-10, updated 2026-07-11
**Current:** v2.6.0 ✅ SHIPPED (domain profile system: chunking, extraction/synthesis prompts, hybrid entry, metadata schema, citation traceability)
**Next:** v3.0.0 (agent learning system — see agent-learning-system.md)

> [!NOTE]
> **v2.5.0 shipped 2026-07-11.** All requirements below are implemented,
> smoke-tested, and committed (`4d68638` → `449aebc`). serve_cpu.py has full
> endpoint parity. All P0 review findings resolved. See CHANGELOG.md [2.5.0].

> [!NOTE]
> **v2.6.0 shipped 2026-07-11.** REQ-2.6.0-1 through REQ-2.6.0-8 implemented,
> unit + e2e tested (33 pytest tests), and committed (`9afc35e` → `7b545bf`).
> Domain profiles in `v3/domains/*.toml`; canonical schema in
> `docs/domains/README.md`. serve_cpu.py retains full endpoint parity.

> [!IMPORTANT]
> **Review amendments (2026-07-11):** All requirements updated with findings
> from full codebase review. See `> [!WARNING]` and `> [!CAUTION]` blocks
> throughout for specific changes. Summary:
> - **P0 bug fix:** `_sparse()` uses non-deterministic `hash()` across processes
> - **P0 design:** `POST /ingest` must use `BackgroundTasks` (non-blocking)
> - **P0 safety:** Entity resolution cache + lock for concurrent ingests
> - **P0 fix:** `delete_doc_neo4j` over-deletes edges on shared nodes
> - **P1:** Chunking target corrected to `serve_gpu.py`/`serve_cpu.py` (not `ask.py`)
> - **P1:** `serve_cpu.py` parity for all new endpoints
> - **P1:** Synthesis prompt consolidation (duplicated in 3 files)
> - **P1:** Cache invalidation on re-ingest
> - **P2:** Cross-domain query support (`collection` as list)
> - **P2:** `_build_profile()` chunking-strategy awareness

---

## v2.5.0 — Single-Doc Ingest + Multi-Domain Collections

**Goal:** Turn the RAG from batch-only into API-driven, with namespace-isolated
domains and checksum-based incremental updates. No more `bash run.sh ingest`.

**Epoch:** ~6 hours total (revised from ~4h 44min after review)
**Depends on:** v2.4.1 (done)

### PRE-REQUISITE: `_sparse()` Hash Determinism Fix

> [!CAUTION]
> Python's `hash()` is randomized across processes (`PYTHONHASHSEED`). The
> `_sparse()` function in both `ingest.py` and `ask.py` uses `hash(tok) % 65536`.
> If ingest and query run in different Python processes, the same token maps to
> different sparse indices, breaking sparse search. **Fix before v2.5.0.**
>
> Replace `hash(tok)` with `int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], 'little') % 65536`
> in both `ingest.py:_sparse()` and `ask.py:_sparse()`. Effort: ~10 min.

---

### REQ-2.5.0-1: `/ingest` Endpoint (POST)

**What:** Single-document ingestion via HTTP API. The daemon handles the full
write path — chunk, embed, extract graph, write Qdrant + Neo4j — from one call.

**API contract:**

```
POST /ingest
Content-Type: application/json

{
  "text": string,           // raw document text (markdown or plain)
  "doc_id": string,         // unique identifier (e.g. "bug-204")
  "collection": string,     // Qdrant collection name (default: "engineering_chunks")
  "doc_type": string,       // domain type (e.g. "Bug", "ADR", "Law", "Diagnosis")
  "author": string,         // optional author attribution
  "extract_graph": bool,    // run LLM extraction (default: true)
  "if_checksum": string     // optional SHA-256 — skip if unchanged
}

→ 200 {
  "doc_id": "bug-204",
  "status": "ok",
  "chunks": 43,
  "entities": 7,
  "edges": 11,
  "vectors_upserted": 43,
  "extraction_method": "hybrid",   // "hybrid" (primary, 100% precision) | "sliding_window" | "llm" | "gliner"
  "timing": {
    "embed": 0.42,
    "extract": 3.18,
    "write_qdrant": 0.11,
    "write_neo4j": 0.09,
    "total": 3.80
  }
}

→ 304 { "status": "unchanged", "skipped": true }  // if_checksum matches
→ 400 { "detail": "text is required" }
→ 502 { "detail": "Neo4j unavailable" }
→ 503 { "detail": "GLiNER still loading" }
```

**Implementation tasks (T-2.5.0-1):**

> [!CAUTION]
> `serve_gpu.py` runs with `workers=1` (GPU memory safety). A synchronous
> handler will block ALL `/ask` queries for ~12s during E2B extraction.
> The handler MUST use `BackgroundTasks` and return `202 Accepted`.

| Step | Description | Effort |
|------|-------------|--------|
| 1a | Extract `ingest_text()` from `ingest.py` CLI into reusable function | 20 min |
| 1b | Extract `delete_doc()` into reusable function (wraps Qdrant + Neo4j delete) | 10 min |
| 1c | Extract `list_docs()` — queries Qdrant for doc_id summary | 10 min |
| 1d | Add `IngestReq` Pydantic model to `serve_gpu.py` | 5 min |
| 1e | Wire `POST /ingest` handler **with BackgroundTasks** — returns 202 + task_id | 20 min |
| 1f | Add `GET /ingest/status/{task_id}` for polling ingest completion | 10 min |
| 1g | Error handling: E2B timeout → GLiNER fallback; GLiNER loading → 503 | 10 min |
| 1h | `if_checksum` logic: compare SHA-256 with stored Qdrant payload | 10 min |
| 1i | Response timing dict (embed, extract, write_qdrant, write_neo4j, total) | 10 min |
| 1j | **Add entity resolution cache + lock** in `ingest.py` (prevent race conditions) | 15 min |
| 1k | **Fix `delete_doc_neo4j` edge scoping** (only delete edges from orphaned nodes) | 15 min |
| 1l | **Cache invalidation:** flush `query_cache` entries referencing re-ingested doc | 10 min |
| 1m | **Mirror `/ingest` endpoints in `serve_cpu.py`** for CPU fallback parity | 20 min |
| 1n | Smoke test: ingest test doc, verify retrieval, delete, verify gone | 15 min |

**Acceptance criteria:**
- [ ] `curl POST /ingest` creates doc; `curl POST /ask` retrieves it
- [ ] Same checksum → 304 skipped (no extraction, <10ms response)
- [ ] Changed text → re-extracts, new entities merged correctly
- [ ] E2B down → falls back to GLiNER (heuristic edges only)
- [ ] Empty text → 400
- [ ] Qdrant/Neo4j down → 502

---

### REQ-2.5.0-2: `/ingest` Endpoint (DELETE)

```
DELETE /ingest/{doc_id}?collection=engineering_chunks

→ 200 {
  "doc_id": "bug-204",
  "status": "deleted",
  "vectors_deleted": 43,
  "nodes_cleaned": 7,
  "edges_deleted": 11
}
→ 404 { "detail": "doc_id not found" }
```

**Tasks (T-2.5.0-2):**

| Step | Description | Effort |
|------|-------------|--------|
| 2a | Wire `DELETE /ingest/{doc_id}` handler | 10 min |
| 2b | Delete Qdrant points by doc_id payload filter | 5 min |
| 2c | Remove doc_id from Neo4j entity source_docs; delete orphan nodes | 10 min |
| 2d | Smoke test: delete + verify retrieval returns 0 contexts | 5 min |

---

### REQ-2.5.0-3: `/ingest` Endpoint (GET / Collections List)

```
GET /ingest

→ 200 {
  "documents": [
    {"doc_id": "bug-204", "doc_type": "Bug", "collection": "engineering_chunks", 
     "chunks": 43, "entities": 7, "checksum": "abc123..."},
    {"doc_id": "adr-014", "doc_type": "ADR", "collection": "engineering_chunks",
     "chunks": 68, "entities": 12, "checksum": "def456..."}
  ]
}

GET /collections

→ 200 {
  "collections": {
    "engineering_chunks": {"docs": 5, "chunks": 215, "entities": 34},
    "legal_chunks":       {"docs": 0, "chunks": 0,   "entities": 0},
    "medical_chunks":     {"docs": 0, "chunks": 0,   "entities": 0}
  }
}
```

**Tasks (T-2.5.0-3):**

| Step | Description | Effort |
|------|-------------|--------|
| 3a | Wire `GET /ingest` — scrolls Qdrant payloads for doc_id summary | 10 min |
| 3b | Wire `GET /collections` — queries each collection for point count | 10 min |
| 3c | Add `checksum` to Qdrant payload on ingest (sha256 of text) | 5 min |

---

### REQ-2.5.0-4: Multi-Domain Qdrant Collections

**What:** Each domain gets its own Qdrant collection. Query routes to the right
one based on request param.

**Tasks (T-2.5.0-4):**

| Step | Description | Effort |
|------|-------------|--------|
| 4a | Add `QDRANT_COLLECTIONS` dict to config.py | 5 min |
| 4b | Add `QDRANT_COLLECTION_DEFAULT = "engineering_chunks"` | 2 min |
| 4c | Add `collection` field to `AskReq` (optional, defaults from config) | 5 min |
| 4d | Pass `collection` to `rag.qdrant_search()` in /ask handler | 10 min |
| 4e | Create collections on first use (Qdrant auto-creates but needs correct vector config) | 15 min |
| 4f | Backward compat: if `collection` not in request, use default | 0 min (built in) |

**Acceptance criteria:**
- [ ] `POST /ask {"query":"...", "collection":"legal_chunks"}` queries only legal
- [ ] `POST /ask {"query":"..."}` (no collection) queries engineering (backward compat)
- [ ] `POST /ask {"query":"...", "collection":["eng...","legal..."]}` cross-domain federated search
- [ ] Different collections return different results for same query
- [ ] `GET /collections` shows all domains
- [ ] All new endpoints mirrored in `serve_cpu.py`

---

### REQ-2.5.0-5: Checksum-Based Incremental Update

**What:** Re-ingesting an unchanged document costs 0ms. SHA-256 comparison
before extraction.

**Tasks (T-2.5.0-5):**

| Step | Description | Effort |
|------|-------------|--------|
| 5a | Store `sha256(text)` in Qdrant payload on initial ingest | 5 min |
| 5b | `GET /ingest/{doc_id}/checksum` — returns stored checksum | 5 min |
| 5c | `POST /ingest` with `if_checksum` field — compare before extraction | 10 min (overlap with 1g) |
| 5d | `POST /ingest/revalidate` — batch compare: `[{doc_id, checksum}, ...]` → returns only changed | 20 min |
| 5e | Update `ingest.py` CLI to support checksum comparison | 10 min |

**Acceptance criteria:**
- [ ] Ingest same text twice → second call returns 304 in <10ms
- [ ] Ingest changed text → re-extracts, updates entities
- [ ] `/revalidate` batch: 10 docs, 2 changed → returns only 2

---

### REQ-2.5.0-6: Documentation & Migration

| Step | Description | Effort |
|------|-------------|--------|
| 6a | Update `docs/architecture.md` with `/ingest` endpoint | 10 min |
| 6b | Update `docs/development.md` with single-doc ingest examples | 10 min |
| 6c | Update `CHANGELOG.md` with v2.5.0 entries | 5 min |
| 6d | Mark `docs/ingest-endpoint-plan.md` as implemented | 2 min |
| 6e | Update `v3/README.md` with API reference | 10 min |

---

### v2.5.0 Summary

| REQ | Description | Tasks | Est. Effort |
|-----|-------------|-------|-------------|
| pre | `_sparse()` hash fix | 1 step | 10min |
| 1 | POST /ingest (BackgroundTasks) | 14 steps | 2h 30min |
| 2 | DELETE /ingest | 4 steps | 30min |
| 3 | GET /ingest + /collections | 3 steps | 25min |
| 4 | Multi-domain collections | 6 steps | 45min |
| 5 | Checksum incremental | 5 steps | 50min |
| 6 | Documentation | 5 steps | 37min |
| **TOTAL** | | | **~5h 47min** |

---

## v2.6.0 — Domain Profile System + General-Purpose RAG

> [!NOTE]
> **SHIPPED 2026-07-11.** All REQ-2.6.0-1 … REQ-2.6.0-8 are implemented and
> tested. Profiles in `v3/domains/*.toml`; canonical schema + field reference in
> `docs/domains/README.md`. Commit range `9afc35e` → `7b545bf`.

**Goal:** Eliminate ALL domain-specific hardcoded assumptions. A `.toml` profile
file per domain controls chunking, prompts, graph schema, metadata, and entry
strategy. The same codebase ingests and queries engineering bugs, medical records,
legal documents, and accounting ledgers — only the profile changes.

**Epoch:** ~6 hours total
**Depends on:** v2.5.0 (multi-domain collections + /ingest endpoint)

---

### REQ-2.6.0-1: Domain Profile System (.toml)

**What:** Replace 11 hardcoded assumptions with a TOML profile per domain.
Engineering.toml captures current behavior as defaults. Medical.toml, legal.toml,
accounting.toml, hospitality.toml are new.

**Profile schema:**

```toml
[domain]
name = "medical"                           # domain KEY — must match filename (medical.toml)
collection = "medical_chunks"              # Qdrant collection name
neo4j_label = "Medical"                    # :Entity:Medical label in Neo4j
description = "Medical knowledge base — symptoms, diagnoses, medications"

[chunking]
strategy = "section"                       # sentence | paragraph | section | fixed
chunk_size = 512                           # max tokens per chunk
overlap = 64                               # token overlap between chunks
section_header_prefix = "##"              # for section strategy

[graph_schema]
entity_types = [
    "Symptom", "Diagnosis", "Disease", "Medication",
    "Procedure", "LabTest", "Anatomy", "Patient", "Doctor"
]
edge_types = [
    "TREATS", "CAUSES", "INDICATES", "CONTRAINDICATES",
    "DOSAGE", "PERFORMS", "PRESCRIBES", "AFFECTS",
    "ASSOCIATED_WITH"
]

[entity_resolution]
threshold = 0.88                           # cosine similarity for merger
aliases_enabled = true

[prompts]
extraction = """
Extract a medical knowledge graph from the clinical document below.
...
"""

synthesis = """
You are a medical information assistant...
...
"""

[metadata_schema]
fields = ["patient_id", "encounter_date", "department", "doctor", "document_type"]

[neo4j_entry]
strategy = "hybrid"                        # keyword | vector | hybrid
max_k_hops = 2                             # traversal depth from entry node
```

**Tasks (T-2.6.0-1):**

| Step | Description | Effort |
|------|-------------|--------|
| 1a | Create `v3/domains/` directory | 2 min |
| 1b | Define TOML schema with validation (all required fields) | 20 min |
| 1c | Create `engineering.toml` — captures ALL current behavior as defaults | 25 min |
| 1d | Create `medical.toml` — full medical schema (symptoms, diagnoses, medications) | 20 min |
| 1e | Create `legal.toml` — full legal schema (statutes, cases, citations) | 20 min |
| 1f | Create `accounting.toml` — (accounts, transactions, ledgers, tax codes) | 15 min |
| 1g | Create `hospitality.toml` — (SOPs, recipes, inventory, service) | 15 min |
| 1h | `load_domain_profile(name: str) -> dict` in config.py | 15 min |
| 1i | Profile cache (parse TOML once, cache in dict) | 10 min |
| 1j | Fallback: unknown domain → load `engineering.toml` as default | 5 min |

**Acceptance criteria:**
- [ ] `load_domain_profile("medical")` returns dict with all sections
- [ ] `load_domain_profile("nonexistent")` returns engineering defaults
- [ ] All 5 TOML files pass schema validation
- [ ] Changing a TOML file + restart daemon → new behavior takes effect

---

### REQ-2.6.0-2: Domain-Aware /ask (Routing)

**What:** `/ask` reads the domain profile and uses it to route Qdrant search,
select chunking strategy, use domain synthesis prompt, and optionally filter
Neo4j by domain label.

**API contract change:**

```json
POST /ask
{
  "query": "what treats hypertension?",
  "domain": "medical",           // NEW — selects profile
  "collection": "medical_chunks", // if omitted, derived from domain profile
  "synthesize": true
}
```

**Tasks (T-2.6.0-2):**

| Step | Description | Effort |
|------|-------------|--------|
| 2a | Add `domain` field to `AskReq` (optional, default: "engineering") | 5 min |
| 2b | Load profile in `/ask` handler; derive collection from profile if not explicit | 10 min |
| 2c | Pass profile to `rag.synthesize()` — inject domain synthesis prompt | 10 min |
| 2d | Pass domain label to `rag.neo4j_subgraph()` — optional filter `MATCH (n:Entity:Medical)` | 15 min |
| 2e | Backward compat: no domain field → engineering profile | 0 min |

**Acceptance criteria:**
- [ ] `{"query":"...","domain":"medical"}` queries medical_chunks, uses medical synthesis prompt
- [ ] `{"query":"..."}` (no domain) queries engineering, uses engineering prompt
- [ ] Medical query returns medical-domain contexts only
- [ ] Synthesis prompt includes "medical information assistant" role

---

### REQ-2.6.0-3: Pluggable Chunking Strategies

**What:** Replace hardcoded sentence-based chunking with 4 strategies,
selected per domain profile.

> [!WARNING]
> **Corrected target.** The chunking logic lives in `serve_gpu.py:embed_late`
> (line 174) and `serve_cpu.py:embed_late` (line 121), NOT in `ask.py`.
> There is no `late_chunk()` function in `ask.py`. Both daemon files must
> be refactored.

```python
CHUNK_STRATEGIES = {
    "sentence": _chunk_sentence,     # current: split on .!? → group by token limit
    "paragraph": _chunk_paragraph,   # split on \n\n → group by token limit
    "section": _chunk_section,       # split on ## headers → each section is a chunk
    "fixed": _chunk_fixed,           # sliding window: fixed size + overlap
}
```

**Tasks (T-2.6.0-3):**

| Step | Description | Effort |
|------|-------------|--------|
| 3a | Refactor `serve_gpu.py:embed_late()` to accept strategy param | 10 min |
| 3b | Refactor `serve_cpu.py:embed_late()` to accept strategy param | 10 min |
| 3c | Implement `_chunk_sentence()` — extract current logic | 10 min |
| 3d | Implement `_chunk_paragraph()` — split on double newline | 15 min |
| 3e | Implement `_chunk_section()` — split on markdown headers, respect hierarchy | 20 min |
| 3f | Implement `_chunk_fixed()` — sliding window with overlap | 15 min |
| 3g | Strategy dispatch from profile `[chunking].strategy` | 10 min |
| 3h | **Make `_build_profile()` chunking-strategy-aware** (not hardcoded 2-sentence window) | 15 min |
| 3i | Unit test each strategy with sample texts | 15 min |

**Acceptance criteria:**
- [ ] Sentence: "The cat sat. The dog ran." → 2 chunks
- [ ] Paragraph: 3 paragraphs → 3 chunks (or fused by token limit)
- [ ] Section: 4 h2 sections → 4 chunks, each with section header preserved
- [ ] Fixed: 1000-char text, size=500, overlap=100 → 3 chunks
- [ ] Profile `strategy = "section"` → section-chunking used for that domain

---

### REQ-2.6.0-4: Domain-Aware Extraction Prompt

**What:** E2B extraction prompt comes from the domain profile, not a single
hardcoded string. Medical extraction asks for symptoms/medications; legal
extraction asks for statutes/cases.

**Tasks (T-2.6.0-4):**

| Step | Description | Effort |
|------|-------------|--------|
| 4a | Deprecate `config.EXTRACTION_PROMPT` — move to `engineering.toml` | 10 min |
| 4b | `ingest_text()` accepts `domain` param, loads profile, uses profile prompt | 10 min |
| 4c | `/ingest` endpoint: `domain` field → loads profile → uses profile prompt | 10 min |
| 4d | GLiNER fallback: edge type from `profile.graph_schema.edge_types[0]` | 5 min |
| 4e | `POST /ingest {"text":"...","domain":"medical"}` tests with medical prompt | 10 min |

**Acceptance criteria:**
- [ ] Engineering: extraction prompt mentions "Microservice, Database, API"
- [ ] Medical: extraction prompt mentions "Symptom, Diagnosis, Medication"
- [ ] No more `config.EXTRACTION_PROMPT` — all in engineering.toml

---

### REQ-2.6.0-5: Domain-Aware Synthesis Prompt

**What:** E4B answer generation includes domain-specific role and constraints.
Medical: "Be cautious, do not diagnose." Legal: "Cite statutes, no legal advice."

> [!WARNING]
> **Prompt consolidation required.** The synthesis prompt is currently duplicated
> in 3 files: `ask.py:synthesize()`, `serve_gpu.py:/ask` (via `rag.synthesize()`),
> and `retrieve_json.py:_synthesize_nonstream()`. Extract into a shared
> `prompts.py` module before making domain-aware.

**Tasks (T-2.6.0-5):**

| Step | Description | Effort |
|------|-------------|--------|
| 5a | **Create `prompts.py`** with `build_synthesis_prompt(query, contexts, profile)` | 15 min |
| 5b | Refactor `ask.py:synthesize()` to use `prompts.py` | 10 min |
| 5c | Refactor `retrieve_json.py:_synthesize_nonstream()` to use `prompts.py` | 10 min |
| 5d | `/ask` handler passes loaded profile to prompt builder | 5 min |
| 5e | Medical: cautious role, limit to context, no diagnosis | 0 min (TOML) |
| 5f | Legal: cite sources, no legal advice disclaimer | 0 min (TOML) |
| 5g | Test: same query, different domains → different synthesis tone | 10 min |

**Acceptance criteria:**
- [ ] Medical synthesis says "based on the clinical documents provided"
- [ ] Legal synthesis says "based on the statutes and cases cited"
- [ ] Engineering synthesis unchanged

---

### REQ-2.6.0-6: Hybrid Neo4j Entry Strategy

**What:** Current keyword-overlap Neo4j entry works for `bug-204` (technical IDs)
but fails for natural-language queries (`"chest pain treatment"`). Hybrid mode
runs both keyword and vector, picks best.

**Tasks (T-2.6.0-6):**

| Step | Description | Effort |
|------|-------------|--------|
| 6a | Refactor `ask.neo4j_subgraph()` to accept `entry_strategy` param | 10 min |
| 6b | Implement hybrid: run keyword AND vector in parallel (threads) | 15 min |
| 6c | Pick best entry: keyword if overlap > 2, else vector | 10 min |
| 6d | Configurable via `profile.neo4j_entry.strategy` | 5 min |
| 6e | Benchmark: engineering (keyword works) vs medical (vector needed) | 10 min |

**Acceptance criteria:**
- [ ] `bug-204` query → keyword entry (fast, 1ms)
- [ ] `chest pain treatment` query → vector entry (accurate, 50ms)
- [ ] Hybrid mode: both run in parallel, best result used

---

### REQ-2.6.0-7: Domain-Aware Metadata Schema

**What:** Qdrant payload includes domain-specific metadata fields. Medical docs
carry `patient_id`, `encounter_date`. Legal docs carry `jurisdiction`, `court`.

**Tasks (T-2.6.0-7):**

| Step | Description | Effort |
|------|-------------|--------|
| 7a | `ingest_text()` accepts `metadata: dict` param | 10 min |
| 7b | Validate metadata against `profile.metadata_schema.fields` | 10 min |
| 7c | Store validated metadata in Qdrant payload | 5 min |
| 7d | `/ingest` endpoint: `metadata` JSON field in request body | 10 min |
| 7e | `GET /ingest` returns metadata per doc | 5 min |

**Acceptance criteria:**
- [ ] `POST /ingest {"text":"...","domain":"medical","metadata":{"patient_id":"P123"}}` stores patient_id
- [ ] Unknown metadata field → warning, not error
- [ ] Engineering docs carry `author`, `doc_type` (current behavior)

---

### REQ-2.6.0-8: Documentation & Migration

| Step | Description | Effort |
|------|-------------|--------|
| 8a | Update `docs/architecture.md` with domain profile system | 15 min |
| 8b | Update `docs/development.md` with TOML profile creation guide | 15 min |
| 8c | Update `docs/general-purpose-rag-plan.md` mark as implemented | 5 min |
| 8d | Update `docs/multi-domain-plan.md` mark as implemented | 5 min |
| 8e | Create `v3/domains/README.md` — how to create a new domain | 15 min |
| 8f | Update `CHANGELOG.md` with v2.6.0 entries | 10 min |
| 8g | Migration guide: old config → engineering.toml | 15 min |

---

### v2.6.0 Summary

| REQ | Description | Tasks | Est. Effort |
|-----|-------------|-------|-------------|
| 1 | Domain profile system | 10 steps | 2h 27min |
| 2 | Domain-aware /ask routing | 5 steps | 40min |
| 3 | Pluggable chunking + profile-aware _build_profile | 9 steps | 2h 00min |
| 4 | Domain-aware extraction prompt | 5 steps | 45min |
| 5 | Domain-aware synthesis prompt + consolidation | 7 steps | 50min |
| 6 | Hybrid Neo4j entry | 5 steps | 50min |
| 7 | Domain metadata schema | 5 steps | 40min |
| 8 | Documentation | 7 steps | 1h 20min |
| **TOTAL** | | | **~9h 32min** |

---

## Full Roadmap Summary

```
v2.4.1
  └── Dual reranker (GPU/CPU), pool cap, modular config, fp16 fix,
      comprehensive benchmarks, all docs synced

PRE-REQ — _sparse() hash fix                    ✅ DONE
  └── Replaced hash(tok) with hashlib.md5 in ingest.py + ask.py

v2.5.0 ★ SHIPPED (2026-07-11)                    ✅ DONE
  └── POST/DELETE/GET /ingest (BackgroundTasks, non-blocking),
      checksum incremental updates, entity resolution cache,
      namespace collections, cross-domain queries ("all"/list),
      serve_cpu.py parity, delete_doc_neo4j edge-scoping fix,
      /ask nested-list crash fix
      commits: 4d68638 (REQ-1) abf1bba (REQ-2) b2d4bb0 (REQ-3)
               183e256 (REQ-4) 449aebc (cpu parity)

v2.6.0 ← NEXT — General-Purpose RAG        (~10h)
  └── Domain profiles (.toml), pluggable chunking (both daemons),
      chunking-aware _build_profile, domain-aware prompts,
      prompt consolidation (prompts.py), hybrid Neo4j entry,
      medical + legal + accounting sample profiles
```

## Resource Impact Across Versions

| Metric | v2.4.1 | v2.5.0 | v2.6.0 |
|--------|--------|--------|--------|
| VRAM (GPU) | 10.4 GB | 10.4 GB | 10.4 GB |
| RAM (CPU daemon) | 5.2 GB | 5.3 GB | 5.3 GB |
| Qdrant storage | ~5 MB | ~5 MB × N domains | ~5 MB × N domains |
| Neo4j storage | ~2 MB | ~2 MB × N domains | ~2 MB × N domains |
| New files | 0 | 0 | 6 (5 .toml + 1 README) |
| New deps | 0 | 0 | `tomli` (stdlib `tomllib` in 3.11+) |
| API breaking changes | 0 | 0 | 0 (all additive) |
| Config migration needed | No | No | Yes (constants → engineering.toml) |
