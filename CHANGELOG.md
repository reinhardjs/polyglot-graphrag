# Changelog

All notable changes to this project are documented here. This project
adheres to [Semantic Versioning](https://semver.org) **within the 0.x.x
experimental series** — see `VERSIONING.md` for the exact rules (they are
relaxed vs. strict SemVer during 0.x).

The format is based on [Keep a Changelog](https://keepachangelog.com).

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
