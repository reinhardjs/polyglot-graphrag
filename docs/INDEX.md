# docs/ — Documentation Index

This folder holds the deeper reference material. Start at the **root** docs
first:

- [README.md](../README.md) — try-first entry point
- [RUN.md](../RUN.md) — run from zero
- [QUICKSTART.md](../QUICKSTART.md) — 5-min ingest→ask
- [ARCHITECTURE.md](../ARCHITECTURE.md) — system design
- [VERSIONING.md](../VERSIONING.md) / [CHANGELOG.md](../CHANGELOG.md) — versions

## Subfolders

### `architecture/`
Design deep-dives (neuro-symbolic pipeline, full architecture, Hermes
integration). Some describe historical versions — read for context, not as the
current contract.

### `benchmarks/`
- `BENCHMARKS.md` — extraction + retrieval benchmarks.
- `cpu-vs-gpu-benchmark.md`, `full-benchmark-history.md` — historical runs.
- `benchmark_report.md` — lighter-E2B model experiment.
- `SNOMED_BENCHMARK.md` — why SNOMED is loaded directly (not via LLM ingest).

### `roadmap/`
Planning docs (multi-domain, neuro-symbolic, agent-learning, version
requirements, ingest-endpoint, general-purpose RAG). These are **roadmaps**,
not shipped features — status may lag the code.

### `guides/`
- `model-startup.md` — starting the model servers.
- `development.md` — dev setup.
- `model-switching.md` — how to swap an embedding/LLM model in `config.py`.

### `API.md` — **authoritative endpoint reference**
All 22 daemon routes (retrieval, ingestion, status, admin, differentials) with
request/response shapes and curl examples. Start here for the API contract.

### `domains/`
- `README.md` — **how to add a domain or companion** (the key how-to).

## Top-level reference docs (root)
- `SYNC.md` — `sync_docs.py` file→store sync.
- `MIGRATION.md` — evolve the data model without wiping stores.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.

> **Accuracy note:** docs in `architecture/`, `roadmap/`, and the benchmark
> history describe work done at various points. The authoritative current
> behavior is `ARCHITECTURE.md` + `domain_config.yaml` + the live `/config`
> and `/domains` endpoints. When a roadmap doc disagrees with the code, the
> code wins.
