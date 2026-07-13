# Polyglot GraphRAG

Neuro-symbolic, 100% local GraphRAG with domain-agnostic extraction.

## Active codebase: `v3/`

All current, production code lives in **[`v3/`](v3/)** — the v3.1.x
neuro-symbolic line (GLiNER + E2B extraction, dynamic-label injection,
LLM-fallback NER). Start there.

```
rag-system/
├── v3/                 # ← ACTIVE: current neuro-symbolic pipeline (v3.1.5)
│   ├── README.md         Quick start, endpoints, benchmarks
│   ├── ARCHITECTURE.md   Live system design
│   ├── config.py         All ports/URIs/model names
│   ├── domain_config.yaml  Domain schemas (entity/relation types, chunking)
│   ├── hybrid_extraction.py  GLiNER+E2B hybrid (primary path)
│   ├── sliding_window.py     Long-doc sentence-chunk extraction
│   ├── label_provider.py     Dynamic-label injection (L3 persistence)
│   ├── ingest.py          Vectors → Qdrant, graph → Neo4j
│   ├── serve_gpu.py       GPU daemon (:8000)
│   ├── plans/             Architecture proposals (incl. dynamic-label-injection.md)
│   ├── tests/             Test suite
│   ├── scripts/           Benchmarks / validation
│   └── labels/ logs/      Runtime state (gitignored)
├── docs/               # Project design docs (architecture, roadmap, guides)
└── archive/legacy-v1/  # Dead v1-era code + superseded docs (NOT used)
```

## Quick start (from `v3/`)

```bash
cd v3
sudo systemctl start neo4j qdrant rag-gpu-daemon
# E2B extraction model on :8082, E4B synthesis on :8084 (see v3/run.sh)
python ingest.py sample_data
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"who reported BUG-204?","synthesize":false}'
```

Full details, endpoints, and benchmarks: **[`v3/README.md`](v3/README.md)**.

## What is `archive/legacy-v1/`?

Legacy v1-era code (`config_v1.py`, `ingest_v1.py`, `main_v1.py`,
`router_v1.py`, `retrieve_v1.py`, `run_v1.sh`, `docker-compose_v1.yml`,
`config_v1.yaml`) and superseded docs (`ARCHITECTURE_v1.md`,
`SYSTEM_CONTEXT_v1.md`, `EXTERNAL_ACCESS_v1.md`, `V3_IMPLEMENTATION_BRIEF.md`,
`CHANGELOG_v1.md`, `README_v1.md`). These are **not used by the running
system** (which runs `v3/serve_gpu.py` with
`WorkingDirectory=/mnt/data-970-plus/rag-system/v3`). Kept for history only.

## Versioning

- Current: **v3.1.5** (see `v3/README.md` + git tags `v3.1.x`).
- Changelog for the current line lives in `v3/README.md` "Changelog" section.
- Pre-v3 history is in `archive/legacy-v1/CHANGELOG_v1.md`.
