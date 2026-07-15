# polyglot-graphrag

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![version](https://img.shields.io/badge/version-1.0.0-blue)
![status](https://img.shields.io/badge/status-stable-brightgreen)

**100% local, domain-agnostic Knowledge Discovery & Reasoning Platform** (graph + vector retrieval) that runs
entirely on your own GPU. No API keys, no cloud. You bring documents; it builds
a Neo4j knowledge graph + a Qdrant vector store, then answers questions by
fusing both.

> **Status: STABLE (`1.0.0`).** Enterprise MVP. 4-domain federated retrieval,
> verified baseline (retrieval p95=89ms, synthesis <3s). Backwards-compatible
> API. See [VERSIONING.md](VERSIONING.md), [CHANGELOG.md](CHANGELOG.md), and
> [the roadmap](plans/v1.0-enterprise-mvp-roadmap.md).

---

## Try it in 5 minutes (zero knowledge required)

You need an NVIDIA GPU (RTX 3060 12 GB tested), Docker, and Python 3.11.

```bash
# 1) Clone + venv
git clone https://github.com/reinhardjs/polyglot-graphrag.git
cd polyglot-graphrag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2) Start the two stores (Neo4j + Qdrant)
docker compose up -d

# 3) Start the models + daemon. This needs TWO GGUF files you download
#    yourself (see below). If you already have them in ./models, just run:
bash run.sh serve        # Gemma E2B (:8082, extraction+synthesis) + daemon (:8000)
bash run.sh health       # confirm both are up
```

**You must download one model file** (not in the repo — too large / license):
- `gemma-4-E2B-it-QAT-Q4_0.gguf` → extraction + answer synthesis (HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`)

Put it in `<project-root>/models/`. `run.sh` finds it automatically.

> **Optional deeper answers:** the larger `gemma-4-E4B-it-QAT-Q4_0.gguf`
> (HuggingFace `lmstudio-community/gemma-4-E4B-it-QAT-GGUF`) may be
> placed in `models/` and enabled at runtime via the `SYNTHESIS_LLM_*`
> environment variables for longer, deeper answers (~22s p95 on the
> 12 GB card). It is **not required** — E2B handles synthesis by default
> at ~2.2s p95.

**Ask a question immediately** (a demo corpus is auto-seeded on startup):

```bash
curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is the total VRAM budget on the RTX 3060 and which models share it"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('answer','')[:500])"
```

If you got an answer, you're done — the system works. Now explore.

---

## What just happened (30-second mental model)

```
INGEST:  doc.md ──► chunk ──► embed (Jina) ──► Qdrant (vectors)
                      │
                      └──► extract entities/relations (Gemma E2B) ──► Neo4j (graph)

ASK:     query ─► embed ─► Qdrant (similar text) + Neo4j (related entities)
                     ─► rerank (BGE) ─► (E2B) writes the answer
```

Two stores, searched together:
- **Qdrant** — vector store (find similar *text* by meaning).
- **Neo4j** — graph store (find *related entities*: "bob reported BUG-204").

A **companion** is a second Qdrant collection attached to a domain (e.g. the
repo's `docs/` tree) so the system searches both corpora and boosts answers
confirmed by more than one — see [docs/domains/README.md](docs/domains/README.md).

---

## Explore (next steps, in order)

1. **Ask with the SNOMED clinical graph** — `domain:"snomed"` (or omit `domain`
   to hit the `default` alias → snomed). It runs SNOMED term-matching + the
   `clinical_prose` semantic companion in parallel for differential diagnosis.
   See [QUICKSTART.md](QUICKSTART.md).
2. **Ingest your own docs into a NEW domain** — copy `domains/clinical_prose/`
   (or `domains/snomed/`) and edit `domain_config.yaml`. No daemon code change
   needed. See [docs/domains/README.md](docs/domains/README.md).
3. **Understand the architecture** — [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Documentation map (so you're not lost)

| You want to… | Read |
|---|---|
| Get it running from zero | [RUN.md](RUN.md) |
| The 5-min ingest→ask path | [QUICKSTART.md](QUICKSTART.md) |
| Understand the design | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **All API endpoints (request/response + curl)** | [docs/API.md](docs/API.md) |
| Add a domain / companion | [docs/domains/README.md](docs/domains/README.md) |
| How versioning works | [VERSIONING.md](VERSIONING.md) |
| What changed / release notes | [CHANGELOG.md](CHANGELOG.md) |
| Keep doc↔store in sync | [SYNC.md](SYNC.md) |
| Evolve the data model safely | [MIGRATION.md](MIGRATION.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

> **Configured domains.** The deployment ships with **7 domains**:
> `snomed` (graph-only terminology), `clinical_prose` (medicine prose companion),
> `enterprise` (engineering knowledge base — the confidential corpus),
> `legal`, `fraud` (graph + vector), plus `default` and `healthcare` (both alias
> to `snomed`). `domain="all"` fans out to every concrete domain. To add a new
> corpus, copy an existing `domains/<name>/` plugin and register it in
> `domain_config.yaml` — the federated factory picks it up with no daemon-code
> changes.
>


## Requirements

- **NVIDIA GPU** (~8–12 GB VRAM; RTX 3060 12 GB tested). CPU fallback exists
  (`serve_cpu.py`) but is slow.
- **Docker + docker compose** (Neo4j + Qdrant).
- **Python 3.11** venv.
- Two GGUF model files (above).

Full detail, including VRAM budgets and known limitations, in [RUN.md §1](RUN.md).
