# Neuro-Symbolic GraphRAG Upgrade Plan

**Status:** COMPLETE — all 4 phases shipped (2026-07-11)
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

## Phase 2 — Subgraph Pruning (Context Window Protection) ✅ SHIPPED

**Goal:** prevent Neo4j "neighborhood explosion" from overflowing the LLM
context window.

**Deliverable:** `v2/graph_prune.py`
- `rank_node_ids(entry_id, node_ids, edges, strategy, top_n)` — ranks
  neighbourhood nodes by centrality; entry node always kept first.
- `prune_records(entry_id, records, edges, strategy, top_n)` — prunes node
  record dicts to the Top-N.
- Strategies: `degree` (default, no deps), `pagerank` (networkx, falls back to
  degree), `none` (legacy).

**Wiring:** `ask.neo4j_subgraph()` now fetches the k-hop neighbourhood **plus its
relationships**, builds the edge list, and prunes to `config.GRAPH_PRUNE_TOP_N`
(default 10) via `config.GRAPH_PRUNE_STRATEGY`. Pruning is wrapped in try/except
so it can never break retrieval.

**Config:** `GRAPH_PRUNE_TOP_N=10`, `GRAPH_PRUNE_STRATEGY="degree"`.

**Schema note:** works on the existing `ASSOCIATED_WITH`/`AFFECTS` edges — no
assumption of rich typed edges.

**Tests:** `tests/test_graph_prune.py` (8 unit tests). Verified: entry node always
first, central nodes beat isolated leaves, `none` disables pruning.

---

## Phase 3 — Corrective RAG (CRAG) & Adaptive Routing ✅ SHIPPED

**Goal:** route queries to save compute + evaluate context quality with a
corrective fallback.

**Deliverable:** `v2/crag_pipeline.py` — a dependency-free Python state machine:
1. **Router** (`route()`) — ID/short-factual → `graph`; analytical → `hybrid`.
   Optional E4B confirmation via `config.CRAG_USE_LLM_ROUTER`.
2. **Evaluator** (`evaluate_context()`) — grades context CORRECT / AMBIGUOUS /
   INCORRECT by lexical overlap.
3. **Fallback** (`rewrite_query()` + 2nd pass) — on weak grades, rewrite (E4B or
   heuristic) and run a wider hybrid retrieval, then synthesize.

`CragDeps` injects embed/retrieve/condense/synth so the FSM is unit-testable
without daemons. Degrades gracefully to pure heuristics if E4B is down.

**Wiring:** `/ask` in both daemons accepts `crag: true` → delegates to
`run_crag()` and returns `source:"crag"`, `path` (route), `crag_grade`,
`crag_corrected`, `crag_rewritten_query`, `crag_trace`.

**Verified live:** factual `who reported BUG-204?` → graph route, 0 hits →
INCORRECT → rewrite → recovered 5 contexts. Analytical query → hybrid, CORRECT
first pass.

**Tests:** `tests/test_crag_pipeline.py` (12 unit) + `tests/test_e2e_neuro_symbolic.py`
(live routing/trace assertions).

---

## Phase 4 — Evaluation Harness (Ragas) ✅ SHIPPED

**Goal:** rigorous, domain-segmented accuracy measurement.

**Deliverable:** `v2/evaluate_pipeline.py` — computes Faithfulness,
AnswerRelevancy, ContextPrecision, ContextRecall over a golden JSON dataset.
Two backends, auto-selected:
- **ragas** — real ragas framework wired to the LOCAL E4B LLM + Jina embeddings
  (used when `ragas`+`datasets` installed AND `EVAL_USE_RAGAS=1`).
- **local** — built-in metric implementations using the local Jina embed daemon
  (cosine) + lexical coverage. Zero cloud, zero heavy deps. Default.

`--live` fills answer+contexts by calling the running daemon `/ask` (supports
`--domain` and `--crag`).

**Golden templates:** `v2/sample_data/golden/{engineering,medical}.json`.

**Verified live:** engineering set → faithfulness 0.90, context_precision 1.0,
answer_relevancy 0.75, context_recall 0.57 (local backend).

**Tests:** `tests/test_evaluate_pipeline.py` (9 unit tests for metric math).

**Note:** ragas is intentionally optional (it pulls in langchain). Install with
`pip install ragas datasets langchain-openai` and set `EVAL_USE_RAGAS=1` to use
the ragas backend; otherwise the local backend runs with no extra deps.

---

## Test runner

`v2/run_tests.sh` — one entry point for everything:
- `bash run_tests.sh unit` — fast, no daemon (51 tests)
- `bash run_tests.sh e2e` — live daemon tests
- `bash run_tests.sh eval` — Phase 4 eval smoke on golden datasets
- `bash run_tests.sh phase N` — one phase (1|2|3|4)
- `bash run_tests.sh all` — full pytest suite (75 tests) + eval smoke

---

## Open questions (resolved)

- **Phase 2 write-back of importance scores:** deferred — degree/pagerank is
  computed per-query in-memory; cheap enough (<1k nodes) that caching isn't
  worth the write complexity yet.
- **Phase 3 router (classifier vs LLM):** shipped heuristic-first with an
  optional E4B confirmation flag (`CRAG_USE_LLM_ROUTER`, default off for speed).
