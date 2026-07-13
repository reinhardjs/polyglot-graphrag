# Follow-up Plan: Incremental / Differential Ingestion & Retrieval Hardening

## Context

After PR #3, the system correctly handles document lifecycle (create / edit /
delete) via `sync_docs.py` + `source_docs` attribution on both nodes and
relationships. Re-ingestion is **whole-document replace**: any change reprocesses
the entire file through E2B extraction (~165s for a paper-sized doc).

This plan covers the next hardening steps that were discussed but NOT done in
PR #3. They are independent and can be picked in any order.

---

## F1 — True differential (section-level) update instead of full re-ingest

**Problem today:** `sync_docs.py` sends the FULL file text on every change,
even a one-line edit. The server's `ingest_text` does delete-doc-then-rewrite.
For large, frequently-edited docs this is wasteful (full E2B re-extract).

**Goal:** Only re-process the changed *sections*, merging deltas into the
existing graph/chunks instead of wiping the doc.

**Design sketch:**
- `sync_docs.py` already tracks per-file SHA256. Extend state to track
  per-**section** (or per-chunk-group) checksums (e.g. split file by Markdown
  headings `##`, hash each section body).
- On change: diff old vs new section checksums → set of CHANGED/ADDED/REMOVED
  sections.
- New server endpoint `PATCH /ingest/{doc_id}` that:
  - For ADDED/CHANGED sections: extract graph for just that section text,
    `MERGE` into existing nodes/edges (already supported by `write_graph`
    vector resolution), upsert only the affected Qdrant chunks.
  - For REMOVED sections: call `delete_section(doc_id, section_id)` which
    strips that section's `source_docs` from nodes/edges (reuses the
    `source_docs` machinery from PR #3) and deletes its Qdrant chunks.
- Requires a stable `section_id` (e.g. slug of heading + index) stored in
  `source_docs` alongside doc_id.

**Risk:** section boundaries must be deterministic; heading renames become
delete+add (acceptable). Chunk→section mapping must be stored in Qdrant payload.

**Acceptance:** editing one section of a 50-section doc re-extracts ~1 section
(<10s), not the whole doc. No stale entities from removed sections.

---

## F2 — Edge-level precision/recall benchmarking

**Problem today:** extraction quality is measured only by *volume* (entity/edge
counts). We never measured *correctness* — are the 34 RAPTOR edges right?

**Goal:** A repeatable eval that scores extraction precision/recall against a
small hand-labeled gold set.

**Design sketch:**
- Create `tests/gold/` with 2-3 docs + labeled expected (entity, type, edge)
  tuples.
- Script `eval_extraction.py`: ingest each, extract actual graph, compute
  precision/recall/F1 at entity and edge level (exact + cosine-soft match).
- Compare modes: `sliding_window` vs `llm` vs `index_routing` on the same docs.
- Report table; gate future extractor swaps on not regressing F1.

**Acceptance:** we can state extraction precision numerically, not "it looks
richer."

---

## F3 — EXTRACTION_MAX_TOKENS tuning (quality-neutral speed)

**Problem today:** `EXTRACTION_MAX_TOKENS=4096` but extraction JSON rarely needs
4K output. Decoding time dominates the 165s.

**Goal:** Lower to ~1024 and re-time; confirm entity/edge counts don't drop.

**Design sketch:** env override already exists (`EXTRACTION_MAX_TOKENS`).
Run RAPTOR at 1024 vs 4096, compare counts + timing. If equal, make 1024 default.

**Acceptance:** ~20-30% faster ingest, same graph quality.

---

## F4 — CRAG routing tuning

**Problem today:** `CRAG_USE_LLM_ROUTER=False` (route decided by keyword/fact
heuristic, no LLM confirm/rewrite).

**Goal:** Evaluate whether enabling LLM router improves answer grounding
(precision) at acceptable latency cost.

**Design sketch:** `eval_pipeline.py` (exists) already scores faithfulness.
A/B: `CRAG_USE_LLM_ROUTER` True vs False on the `tests/gold` question set.
Measure faithfulness + latency. Decide default.

**Acceptance:** data-backed default for CRAG router.

---

## F5 — Concurrency / VRAM headroom under load

**Problem today:** at 91% VRAM, a concurrent ingest + /ask could spike. We never
tested concurrent load.

**Goal:** Confirm the 1.1 GB headroom holds, or document the limit.

**Design sketch:** script that fires a long ingest (background) while issuing
N parallel /ask calls; assert no 500/OOM; measure p95 latency. If it fails,
options: lower E2B ctx, offload BGE reranker to CPU, or document "don't ingest
and query concurrently on 12GB."

**Acceptance:** known, documented concurrency envelope.

---

## Priority

1. **F3** (trivial, safe, immediate speed win) — do first.
2. **F2** (establishes quality baseline everything else depends on).
3. **F1** (biggest UX win for frequently-changing large docs).
4. **F4**, **F5** (tuning/validation, lower urgency).

## Non-goals

- No change to the `source_docs` ownership model (it works).
- No new vector DB / graph DB.
- Keep `sync_docs.py` as the sync client; F1 adds a server endpoint it calls.
