# Changelog

All notable changes to this project are documented here. This project
adheres to [Semantic Versioning](https://semver.org) **within the 0.x.x
experimental series** — see `VERSIONING.md` for the exact rules (they are
relaxed vs. strict SemVer during 0.x).

The format is based on [Keep a Changelog](https://keepachangelog.com).

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
- `synthesize:true` requires the Gemma-4 E4B model (:8084) running; otherwise
  use `synthesize:false` for retrieval-only.
- Several domains in `domain_config.yaml` were **schema stubs** —
  configured but not yet ingested/seeded. After the engineering corpus was
  purged, the only configured domain is **`snomed`** (graph-only) with its
  `clinical_prose` companion. Treat the removed stubs as templates to copy.
- VRAM is tight on 12 GB; avoid concurrent heavy ingest + many parallel queries.

---

## Versioning rules (summary)

- Series: `0.x.x` (experimental). We are NOT at 1.0.
- `0.MINOR.PATCH`:
  - `MINOR` bumps when there is a **breaking config / API contract change**
    (e.g. renaming a domain key, changing an endpoint's required field).
  - `PATCH` bumps for backward-compatible fixes/additions and new domains
    added behind existing contracts.
- Tags are created with `git tag -a vX.Y.Z -m "..."` and pushed with
  `git push origin --tags`. No automated release tooling is used yet.
