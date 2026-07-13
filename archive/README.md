# Archive: Legacy v1

This directory contains **dead v1-era code and superseded documentation** that
is **NOT used by the running system**.

The live pipeline is `../v3/` (v3.1.x neuro-symbolic line). The production
GPU daemon (`rag-gpu-daemon.service`) runs `../v3/serve_gpu.py` with
`WorkingDirectory=/mnt/data-970-plus/rag-system/v3`.

## Contents

### Code (legacy, unused)
- `config_v1.py` — old config module (superseded by `../v3/config.py`)
- `ingest_v1.py` — old ingest (superseded by `../v3/ingest.py`)
- `main_v1.py` — old CLI entrypoint (superseded by `../v3/ask.py` + `run.sh`)
- `router_v1.py` — old router (unused)
- `retrieve_v1.py` — old retrieval (unused)
- `run_v1.sh` — old run script (superseded by `../v3/run.sh`)
- `config_v1.yaml` — old flat config (superseded by `../v3/domain_config.yaml`)
- `requirements_v1.txt` — old deps
- `docker-compose_v1.yml` — old compose (Qdrant/Neo4j services; still usable
  standalone but unmaintained)

### Docs (superseded)
- `ARCHITECTURE_v1.md` — marked LEGACY v1
- `SYSTEM_CONTEXT_v1.md` — marked LEGACY v1
- `EXTERNAL_ACCESS_v1.md` — marked LEGACY v1
- `V3_IMPLEMENTATION_BRIEF.md` — a *different* (unbuilt) v3.0 rewrite brief;
  the actually-built line is the neuro-symbolic dynamic-label architecture in
  `../v3/plans/dynamic-label-injection.md`
- `CHANGELOG_v1.md` — changelog up to v2.7.0 (pre-neuro-symbolic)
- `README_v1.md` — old root README

These are retained for historical reference only. Do not import or run them.
