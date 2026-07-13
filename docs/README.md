# Polyglot GraphRAG — Documentation Index

> Design & planning docs. The **live codebase is `../v3/`** (v3.1.x
> neuro-symbolic line). These docs are historical design references — feature
> versions noted (v2.5.0, v2.6.0, v2.7.0) describe the evolution that led to
> the current `v3/` architecture. Directory paths in older docs may read `v2/`
> but now map to `v3/`.

**Last updated:** 2026-07-13 (restructured: `v2/` → `v3/`, legacy archived)

---

## Structure

```
docs/
├── architecture/          System design & integrations
│   ├── architecture.md        Overview: GPU layout, VRAM budget, data flow
│   ├── full-architecture.md   Deep-dive: every component, every config knob
│   └── hermes-integration.md  Hermes rag_query plugin setup & usage
│
├── benchmarks/            Performance measurements
│   ├── cpu-vs-gpu-benchmark.md   CPU vs GPU latency comparison (serve_cpu vs serve_gpu)
│   └── full-benchmark-history.md Full benchmark history across versions
│
├── guides/                Development & operations
│   └── development.md         Environment setup, prerequisites, workflows
│
├── domains/               Domain profile reference (v2.6.0)
│   └── README.md             TOML profile schema + how to add a domain
│
└── roadmap/               Planning & future versions
    ├── version-requirements.md     ★ Master spec: v2.5.0 + v2.6.0 detailed requirements
    ├── ingest-endpoint-plan.md     /ingest endpoint + frequent-update strategy
    ├── multi-domain-plan.md        Namespace collections, multi-domain queries
    ├── general-purpose-rag-plan.md 11 hardcoded assumptions + domain profiles
    └── agent-learning-system.md    v3.0.0 — self-improving knowledge loop (future)
```

---

## Quick Links

### Architecture
- [Overview](architecture/architecture.md) — GPU layout, model table, data flow diagram
- [Full Architecture](architecture/full-architecture.md) — complete technical deep-dive
- [Hermes Integration](architecture/hermes-integration.md) — rag_query plugin for Hermes agents

### Benchmarks
- [CPU vs GPU](benchmarks/cpu-vs-gpu-benchmark.md) — latency comparison across serve modes
- [Benchmark History](benchmarks/full-benchmark-history.md) — per-stage timing across all versions

### Guides
- [Development](guides/development.md) — prerequisites, environment, common workflows

### Domain Profiles
- [Domain Profiles](domains/README.md) — TOML schema, available domains, how to add one

### Roadmap
- [**Version Requirements**](roadmap/version-requirements.md) — master spec (start here)
- [Ingest Endpoint Plan](roadmap/ingest-endpoint-plan.md) — `/ingest` API + incremental updates
- [Multi-Domain Plan](roadmap/multi-domain-plan.md) — namespace collections + cross-domain queries
- [General-Purpose RAG Plan](roadmap/general-purpose-rag-plan.md) — domain profiles + pluggable chunking
- [Agent Learning System](roadmap/agent-learning-system.md) — v3.0.0 future vision

---

## Current Version

**v2.6.0** — domain profile system: pluggable chunking (sentence/paragraph/section/fixed), domain-aware extraction + synthesis prompts, hybrid Neo4j entry strategy (keyword/vector/hybrid), domain metadata schema, and citation traceability (every `/ask` response carries `contexts_numbered`, `contexts_meta`, and a `sources` bibliography mapping `[n]` → source doc). Multi-domain collections + per-domain Neo4j labels. All models on GPU (RTX 3060, 12 GB); CPU fallback at full endpoint parity. See [version-requirements.md](roadmap/version-requirements.md) for the full roadmap.
