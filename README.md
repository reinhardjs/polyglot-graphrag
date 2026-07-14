# polyglot-graphrag

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![version](https://img.shields.io/badge/version-0.1.0-blue)
![status](https://img.shields.io/badge/status-experimental-orange)

**100% local, domain-agnostic GraphRAG** (graph + vector retrieval) that runs
entirely on your own GPU. No API keys, no cloud. You bring documents; it builds
a Neo4j knowledge graph + a Qdrant vector store, then answers questions by
fusing both.

> **Status: EXPERIMENTAL (`0.1.0`).** We are **not** at 1.0. See
> [VERSIONING.md](VERSIONING.md) and [CHANGELOG.md](CHANGELOG.md). The
> config/API contract may still change.

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
bash run.sh serve        # Gemma E2B (:8082) + E4B (:8084) + daemon (:8000)
bash run.sh health       # confirm all three are up
```

**You must download two model files** (not in the repo — too large / license):
- `gemma-4-E2B-it-QAT-Q4_0.gguf` → extraction  (HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`)
- `gemma-4-E4B-it-QAT-Q4_0.gguf` → answer writing (HuggingFace `lmstudio-community/gemma-4-E4B-it-QAT-GGUF`)

Put both in `<project-root>/models/`. `run.sh` finds them automatically.

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

ASK:     query ──► embed ──► Qdrant (similar text) + Neo4j (related entities)
                     ──► rerank (BGE) ──► (optional) Gemma E4B writes the answer
```

Two stores, searched together:
- **Qdrant** — vector store (find similar *text* by meaning).
- **Neo4j** — graph store (find *related entities*: "bob reported BUG-204").

A **companion** is a second Qdrant collection attached to a domain (e.g. the
repo's `docs/` tree) so the system searches both corpora and boosts answers
confirmed by more than one — see [docs/domains/README.md](docs/domains/README.md).

---

## Explore (next steps, in order)

1. **Ingest your own docs** — `./venv/bin/python sync_docs.py mydocs`
   (needs `mydocs/eng/your-doc.md`). See [RUN.md](RUN.md) for the full flow
   (and PDF support, watch mode, verification).
2. **Ask with the graph** — `domain:"engineering"` uses the knowledge graph;
   omit `domain` to hit the default. See [QUICKSTART.md](QUICKSTART.md).
3. **Add a domain or companion** — copy `domains/example_companion/` and edit
   `domain_config.yaml`. No daemon code change needed. See
   [docs/domains/README.md](docs/domains/README.md).
4. **Understand the architecture** — [ARCHITECTURE.md](ARCHITECTURE.md).

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

> **Heads-up on domains.** Only `engineering`, `snomed`, and their companions
> (`clinical_prose`, `example_companion`, `engineering_docs`) hold **real
> ingested data**. `legal`, `medical`, `accounting`, `hospitality`, `journal`
> are **schema stubs** in `domain_config.yaml` — configured but not yet seeded.
> Treat them as copy-paste templates, not working corpora.

---

## Requirements

- **NVIDIA GPU** (~8–12 GB VRAM; RTX 3060 12 GB tested). CPU fallback exists
  (`serve_cpu.py`) but is slow.
- **Docker + docker compose** (Neo4j + Qdrant).
- **Python 3.11** venv.
- Two GGUF model files (above).

Full detail, including VRAM budgets and known limitations, in [RUN.md §1](RUN.md).
