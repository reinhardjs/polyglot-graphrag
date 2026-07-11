# GraphRAG v2 — Documentation Index

**Last updated:** 2026-07-11

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

### Roadmap
- [**Version Requirements**](roadmap/version-requirements.md) — master spec (start here)
- [Ingest Endpoint Plan](roadmap/ingest-endpoint-plan.md) — `/ingest` API + incremental updates
- [Multi-Domain Plan](roadmap/multi-domain-plan.md) — namespace collections + cross-domain queries
- [General-Purpose RAG Plan](roadmap/general-purpose-rag-plan.md) — domain profiles + pluggable chunking
- [Agent Learning System](roadmap/agent-learning-system.md) — v3.0.0 future vision

---

## Current Version

**v2.5.0** — single-doc ingest API (`POST /ingest` non-blocking) + multi-domain collections.
All models on GPU (RTX 3060, 12 GB); CPU fallback at full endpoint parity. See [version-requirements.md](roadmap/version-requirements.md) for the full roadmap.
