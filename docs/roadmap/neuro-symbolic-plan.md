# Neuro-Symbolic GraphRAG Upgrade Plan

**Status:** in progress — Phase 1 shipped (2026-07-11)
**Target:** v2.7.0
**Created:** 2026-07-11

This plan upgrades `polyglot-graphrag` from a vector+graph RAG into a
**Neuro-Symbolic** system: symbolic pre-filtering and graph pruning guard the
neural retrieval/synthesis steps, improving accuracy and protecting the context
window. Every phase is **domain-agnostic** — behavior is driven by the existing
`v2/domains/*.toml` profile system + the live Neo4j graph, never hardcoded
terms.

## Design invariant (read before implementing)

The plan as originally written assumed a Neo4j schema of
`(:Alias)-[:RESOLVES_TO]->(:Concept {domain:})`. **That schema does not exist
in this deployment.** Our graph stores entities as plain `(:Entity)` nodes with
`id, name, type, source_doc, source_docs, profile, name_vector, aliases[]`.
Domain routing happens at the Qdrant *collection* level (not Neo4j labels).
Phase 1 was adapted to this real schema — see the schema notes in each phase.

---

## Phase 1 — Domain-Agnostic Query Modulation (Symbolic Pre-Filtering) ✅ SHIPPED

**Goal:** expand domain slang/abbreviations before the query is embedded, so
Jina v3 grounds ambiguous terms correctly (e.g. `N/V` → `nausea`).

**Deliverable:** `v2/query_modulator.py`
- `build_alias_map(domain, driver)` — reads `:Entity` nodes that carry an
  `aliases` list (and name→type pairs) from Neo4j. Builds
  `{lower_alias: canonical_name}`. Longest alias wins on token overlap.
- `modulate_query(query, domain, driver)` — word-boundary rewrites the raw
  query, appending `(Canonical Term)` expansions. Original text preserved.
- `QueryModulator.moderate(...)` — daemon wrapper with a per-domain alias cache
  (call `QueryModulator.refresh(domain)` after ingest to pick up new synonyms).

**Wiring:** both `serve_gpu.py` and `serve_cpu.py` now call
`QueryModulator.moderate(req.query, domain=...)` inside `/ask` and `/embed_query`
*before* embedding. `/embed_query` returns `text_used` for transparency.

**Schema note:** alias map is global (nodes aren't tagged per-domain), which is
correct — `N/V` means nausea in every domain. Domain isolation still holds at
retrieval (collection level).

**Tests:** `tests/test_query_modulator.py` (6 unit tests, mocked driver).

**Verified live:** `embed_query(domain=medical, "patient has N/V")` →
`"patient has N/V (nausea)"` before embedding.

---

## Phase 2 — Subgraph Pruning (Context Window Protection) — NEXT

**Goal:** prevent Neo4j "neighborhood explosion" from overflowing the LLM
context window.

**Action:** update `ask.neo4j_subgraph()` to prune the k-hop neighborhood to the
Top-N most structurally/causally relevant nodes. Options: degree-centrality,
PageRank (via Neo4j GDS or in-memory Python), or path-weighting.

**Deliverable:** `neo4j_subgraph()` returns Top 10 ranked nodes/relationships
for the active domain instead of the full neighborhood.

**Schema note:** edges today are mostly `ASSOCIATED_WITH` (typed edges are rare).
Pruning must work on the existing `ASSOCIATED_WITH`/`AFFECTS` edge set, not
assume rich edge types.

**Acceptance:** a high-degree hub node (e.g. a common component) no longer floods
the context; retrieval latency + context size bounded.

---

## Phase 3 — Corrective RAG (CRAG) & Adaptive Routing

**Goal:** route queries to save compute + evaluate context quality.

**Action:** `v2/crag_pipeline.py` — a state machine (LangGraph optional; a plain
Python FSM is fine and keeps deps light):
1. **Router** — lightweight classifier: strict factual lookup → Neo4j only;
   complex analysis → Qdrant + Neo4j hybrid.
2. **Evaluator** — score retrieved context (CORRECT / INCORRECT / AMBIGUOUS).
3. **Fallback** — on INCORRECT, rewrite query + secondary search.

**Deliverable:** `crag_pipeline.py` demonstrating the FSM, wired as an optional
`/ask` mode (`crag: true`).

**Schema note:** router can reuse the existing `path` field (`hybrid`/`qdrant`/
`graph`) already returned by `/ask`.

---

## Phase 4 — Evaluation Harness (Ragas)

**Goal:** rigorous, domain-segmented accuracy measurement.

**Action:** `v2/evaluate_pipeline.py` using `ragas` (`pip install ragas datasets`).
Accepts a domain-specific golden JSON (queries, ground-truth, generated answers)
and computes `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`,
`ContextRecall`.

**Deliverable:** `evaluate_pipeline.py` + a `sample_data/golden/{domain}.json`
template per domain.

---

## Open questions

- Should Phase 2 pruning also write back pruned "importance" scores to Neo4j as
  node properties (enables caching)? Defer until Phase 2 lands.
- Phase 3 router: train a tiny classifier or use an LLM-as-judge prompt? Start
  with an E4B prompt (already available on `:8084`) to avoid new deps.
