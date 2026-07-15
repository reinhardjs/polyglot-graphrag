# Changelog

All notable changes to this project are documented here. This project
adheres to [Semantic Versioning](https://semver.org). v1.0.0 is the first
stable release; see `VERSIONING.md` for the exact rules.

The format is based on [Keep a Changelog](https://keepachangelog.com).

> **Current release: `v1.0.0`** (git tag `v1.0.0`). The internal
> increments `1.0.1`–`1.0.3` below were pre-release refinements that were
> folded into the `v1.0.0` tag rather than shipped as separate versions.
> The substantive changes from those increments — **E4B retired (E2B
> serves extraction + synthesis)**, **BGE reranker moved to GPU**,
> **answer-quality release gate (12/12 checks)**, and the **git-ignored
> `golden/` question drop-zone** — are all part of the shipped `v1.0.0`.
> See the `[1.0.1]`–`[1.0.3]` entries for detail.

## [1.0.3] — 2026-07-15 (E4B retired, E2B-only + systemd)

### Changed
- **E4B (:8084) retired.** Synthesis now runs on E2B (:8082), which serves
  BOTH extraction AND synthesis. E4B gave ~22s p95 for no quality gain on the
  12 GB card; E2B is ~2.2s. The `gemma-4-e4b.service` system unit is left
  `disabled` and is not started. (Deeper answers remain available opt-in via
  `SYNTHESIS_LLM_*` env pointing at a manually-started E4B.)
- **E2B is now managed by a user-level systemd unit**
  (`~/.config/systemd/user/gemma-4-e2b.service`) — no sudo needed. Gives
  auto-start-on-login, crash-restart (`Restart=on-failure`), and journald logs.
  For headless reboot survival: `sudo loginctl enable-linger reinhard`.
- `run.sh` no longer starts E4B; it starts E2B only as a fallback if the
  systemd unit isn't already running (idempotent).

### Docs
- RUN.md: new "§2b Model server setup (systemd, no sudo)" section; startup
  steps reflect E2B-only + user-systemd.
- docs/architecture/{architecture,full-architecture}.md + model-startup.md +
  neuro-symbolic-v2.7.0.md + CONTRIBUTING.md: E4B marked RETIRED, E2B
  shown as extraction+synthesis, VRAM totals corrected (~8.5 GB under load).
- config.py header + tests/test_e2e_chunking.py comment updated.


## [1.0.2] — 2026-07-15 (synthesis latency: 22s → <3s)

Performance fix folded into the `v1.0.0` tag (option C). Backward-compatible.

### Fixed
- **Synthesis latency 22.6s → ~2.2s p95.** Previously a `synthesize:true`
  enterprise query took ~22s. Root causes found and fixed:
  - **IPv6 resolution bug:** synthesis backends were reached via `localhost`,
    which resolves to IPv6 `::1` first while the llama-servers bind IPv4
    `127.0.0.1` — adding a multi-second connection retry per call. Now
    `127.0.0.1` in `config.py` defaults.
  - **CPU reranker on synthesis path:** the BGE reranker (`RERANK_DEVICE=cpu`)
    was running even when synthesizing (~9.6s for a 10-candidate pool) for no
    benefit — the LLM reads all contexts and synthesizes them itself. Now
    skipped when `synthesize=true`.
  - **Prompt bloat:** full (ungapped) chunks were fed to synthesis. Added
    `MAX_SYNTH_CONTEXT_CHARS` (350), `MAX_SYNTH_CONTEXTS` (3), and
    `SYNTH_MAX_TOKENS_OUT` (400) to bound prefill + generation.
- **E2B is now the default synthesis backend** (was E4B). E2B on :8082 gives
  p95 ~2.2s; E4B (:8084) gives deeper answers but ~22s on this 12 GB card and
  is opt-in via `SYNTHESIS_LLM_*` env override.

### Added
- `scripts/bench_synth_compare.py`: compares `synthesize:false` (retrieval-only)
  vs `synthesize:true` latency on the ingested corpus; asserts the <3s
  synthesis target. `bench_corpus.py` R4 target tightened to 3s.

## [1.0.1] — 2026-07-15 (stability + bulk-ingest hardening)

Patch release folded into the `v1.0.0` tag (option C: the v1.0.0 tag now points
at this commit). All changes are backward-compatible fixes.

### Fixed
- **Concurrency 500s (CUDA OOM):** the Jina embedding model was shared on the
GPU with no serialization. Under parallel `/ask` (or parallel ingest) the
`.encode()` calls collided and OOM'd. Added `_jina_lock` around every embed
call (query, doc-batch, `/embed_late`, `/embed_query`).
- **Bulk-ingest reliability:** the ingest background task now holds a global
`_ingest_lock` so only ONE ingest pipeline (delete → embed → graph) runs at a
time — eliminates intermittent CUDA OOM from overlapping ingest tasks.
- **Large-doc ingestion OOM:** `/embed_late` embedded ALL chunks in one giant
tensor (4042 chunks for a 220 KB doc → OOM). Now batched (64 chunks/batch)
with bounded peak VRAM.
- **Self-HTTP loopback removed:** `ingest_text` no longer POSTs to the daemon's
own `/embed_late` endpoint (which starved the daemon's own threadpool under
large-doc load). It embeds in-process via the shared `embed_late.py` helper.
- **Reranker offloaded to CPU (`RERANK_DEVICE="cpu"`):** frees ~2.5 GB GPU VRAM
so the Jina embedder coexists with the E2B/E4B GGUF backends on a 12 GB card
without OOM. Reranking 5 candidates on CPU is <50 ms.

### Added
- `scripts/ingest_corpus_docs.py`: bounded-concurrency, idempotent (if_checksum),
retry-on-transient-error bulk ingester for the confidential engineering KB.
- `scripts/test_enterprise_corpus.py`: 6-group reliability suite (accuracy,
edge cases, multi-doc, synthesis, latency p95, concurrency) — all PASS.
- `scripts/release-gate.py` concurrency regression check (was 10 → 11 checks).

## Versioning rules (summary)

- Series: `1.x.x` (stable). v1.0.0 was tagged 2026-07-15.
- `1.MINOR.PATCH`:
- `MINOR` bumps on a **breaking config / API contract change**.
- `PATCH` bumps for backward-compatible fixes/additions and new domains
  added behind existing contracts.
- Tags are created with `git tag -a vX.Y.Z -m "..."` and pushed with
`git push origin --tags`. No automated release tooling is used yet.
## [1.0.0] — 2026-07-15 (enterprise MVP)

First stable, domain-agnostic, enterprise-grade release. Four operational
primary domains (healthcare/SNOMED, enterprise KM, legal & compliance, fraud
detection) behind one federated `/ask` API.

### Added
- **Phase 1 — Core Performance** (single-domain cold p95 < 250ms):
  - P1.1 Redundant embedding eliminated: `vec` forwarded through
    `get_retriever()` so companions reuse the resident daemon embed.
  - P1.2 Merged keyword Cypher (`_idf_for_keyword` + `_concepts_for_keyword`
    → single query returning `cnt` + `ids`).
  - P1.3 Removed redundant `_edge_boost`; only `_symptom_to_disorder` remains
    (`EXPAND_BOOST=2.0`).
  - P1.4 Connection pooling: module-level Neo4j + Qdrant singletons.
  - P1.5 Semantic cache enabled for ALL queries (was synthesize-only).
  - P1.6 `_names_batch` non-English fallback via
    `ORDER BY CASE WHEN d.languageCode='en' THEN 0 ELSE 1 END`.
  - P1.7 SNOMED IDF precomputed into an in-memory lookup (`_IDF_CACHE`, built
    at daemon startup) — eliminates per-keyword Cypher.
  - P1.8 Reranker skipped when `synthesize=false` and pool ≤ 10 (raw scores).
- **Phase 2 — Multi-Domain Expansion**:
  - `domains/enterprise/` (dense prose + seeded ADRs/runbooks).
  - `domains/legal/` (dense prose + seeded GDPR/SOC2/HIPAA).
  - `domains/fraud/` (Neo4j transaction graph + Qdrant narrative companion,
    500 synthetic txns with 5 known fraud patterns).
  - `healthcare` public alias → SNOMED hybrid retriever.
  - Cross-domain `domain=all` and `domain=["a","b"]` fan-out with `_domain`
    tagging.
- **Phase 3 — Enterprise Hardening**:
  - P3.12 Graceful degradation: per-domain retriever failures return
    `degraded:true` with `failed_domains`; unknown domain → 200 with
    `error`+`available`; E4B-down → empty answer, no crash.
  - P3.13 Extended `/health`: per-domain status, Neo4j/Qdrant/E4B backend
    connectivity, VRAM.
  - P3.14 Observability: `/metrics` (Prometheus text, no external dep),
    `request_id` on every `/ask`, `scripts/bench_ask.py`.
- **Phase 4 — Release**:
  - `--demo` flag seeds all 4 domains idempotently (non-fatal per domain).
  - 4 per-domain QUICKSTARTs + root README/QUICKSTART/CHANGELOG/ARCHITECTURE.

### Changed
- `domain_config.yaml` now declares `healthcare`, `enterprise`, `legal`,
  `fraud` as primary domains; `default` → `snomed`.
- `get_retriever` resolves domain aliases (e.g. `healthcare` → `snomed`
  custom retriever) instead of falling back to the slow default graph path.

### Removed
- Legacy `engineering` corpus (purged; SNOMED + clinical_prose retained).

### Fixed
- **Cache poisoning**: degraded/empty synthesis results are no longer cached
  (would otherwise poison every repeat of that query).
- **Differential cache round-trip**: `diagnoses` is now stored in AND returned
  from the cache (cached `mode=differential` reads returned 0 diagnoses before).
- **`synthesize=false` cache regression**: the cache-worthiness guard was
  `answer != ""`, which blocked all retrieval-only calls (they legitimately
  have an empty answer) from caching. Now only skips when synthesis was
  *requested but failed*.
- **Cache variant-scoping**: cache entries are now keyed by a `scope` covering
  every output-affecting flag (`synthesize`, `domain`, `mode`, `min_confidence`,
  `top_k`). Previously keyed on query text alone, so a `synthesize=false` entry
  (answer="") was wrongly served to a later `synthesize=true` request, and a
  `domain=snomed` result could leak into `domain=all`.
- **Synthesis latency (healthcare)**: tightened `prompts/clinical_dx.md` to a
  concise ≤5-candidate ranked differential (≤250 words). Healthcare synthesis
  p95 dropped 10.97s → 2.44s (answer still a valid ranked differential).

### Verified baseline (2026-07-15, RTX 3060 12GB; `skip_cache=true`, n=20/domain)
- Retrieval (`synthesize=false`): healthcare p95 241ms, enterprise 74ms,
  legal 74ms, fraud 62ms; all-domain TOTAL p95 **89ms** (target <400ms). 0 errors.
- Synthesis (`synthesize=true`): healthcare p95 2.44s, enterprise 1.93s,
  legal 1.47s, fraud 1.01s — all under the 3s target. 0 errors.
- Cross-domain (`domain=all`, n=20): p50 298ms, p95 301ms (target <400ms). 0 errors.
- Cache hit (variant-scoped): ~68ms end-to-end (query embed runs before the
  cache lookup; the cache saves retrieval+rerank+synthesis, not the embed).

## [0.1.0] — 2026-07-14 (baseline)

Initial public *experimental* baseline after resetting all prior `v1.x`/`v3.x`
tags (the project had not reached a stable 1.0).

### Added
- Domain-agnostic **Federated retrieval** via `federated.py` (Factory pattern):
  `federated.assemble(domain)` builds the retriever set from `domain_config.yaml`
  with zero hardcoded domain branches.
- **Companion corpora**: any domain can attach extra Qdrant collections
  (`companions:`) searched in parallel and tagged with `_signal`; candidates
  confirmed by ≥2 signals get the dual-evidence rerank boost.
- **Public `default` domain alias** → resolves to the `snomed` strategy
  without renaming the `snomed:` key.
- **Zero-setup demo**: `config.SEED_ON_STARTUP` auto-ingests the
  `example_companion` corpus on daemon boot; unqualified `/ask` (no `domain`)
  hits the configured `default_domain`.
- **Real engineering companion** `engineering_docs`: ingests the repo `docs/`
  tree into a 2nd Qdrant collection.
- Externalized synthesis prompts (`prompts/<kind>.md`).
- `Context._signal` exposed in the `/ask` response so callers see which
  domain/companion each context came from.

### Fixed
- Qdrant requires **unique ascending sparse indices**; `ask._sparse` (and the
  domain helpers that mirrored it) emitted duplicate indices on repetitive
  docs → `422 must be unique` on upsert. Now aggregated per index.

### Known limitations (experimental)
- `synthesize:true` uses the E2B model (`:8082`, which also performs
  extraction). Use `synthesize:false` for retrieval-only if you want the
  fastest path. (The larger E4B model is an optional opt-in for deeper
  answers — see RUN.md.)
- Several domains in `domain_config.yaml` were **schema stubs** —
  configured but not yet ingested/seeded. As of v1.0.1 the active domains are
  `snomed` (graph-only) + `clinical_prose` companion, `enterprise` (the
  ingested engineering knowledge base), `legal`, and `fraud`. Treat the
  removed stubs as templates to copy.
- VRAM is tight on 12 GB; the reranker runs on CPU (`RERANK_DEVICE="cpu"`) to
  leave headroom for Jina + the E2B/E4B GGUF backends. Avoid concurrent heavy
  ingest + many parallel queries.
- P3.14 Prometheus `/metrics` endpoint (no extra deps) emits request counts,
  latencies, and degraded flags for scraping.

