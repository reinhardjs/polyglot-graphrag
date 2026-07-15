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
> verified baseline (retrieval p95=89ms, synthesis <4s). Backwards-compatible
> API. See [VERSIONING.md](VERSIONING.md), [CHANGELOG.md](CHANGELOG.md), and
> [the roadmap](plans/v1.0-enterprise-mvp-roadmap.md).

---

## Try it in 5 minutes (zero knowledge required)

You need an NVIDIA GPU (RTX 3060 12 GB tested), Docker, and **Python 3.11**
(`run.sh` auto-creates the venv with `python3.11` if present).

```bash
# 1) Clone + venv
git clone https://github.com/reinhardjs/polyglot-graphrag.git
cd polyglot-graphrag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2) Check your setup — prints exactly what is missing, if anything:
bash run.sh doctor

# 3) Start the two stores (Neo4j + Qdrant)
docker compose up -d

# 4) Start the model + daemon. Needs ONE GGUF (E2B) — `doctor` tells you
#    where to put it (or set E2B_MODEL=/path). If it's already in ./models:
bash run.sh serve        # Gemma E2B (:8082, extraction+synthesis) + daemon (:8000)
bash run.sh health       # confirm both are up
```

On first startup the **`enterprise`** domain auto-seeds itself with this
project's own `docs/` (tagged `self-docs`) — so you can ask questions about
the system immediately, e.g. `bash run.sh ask "how does hybrid retrieval
work" --domain enterprise`, before you've added any of your own data.

**One model file is required** (not in the repo — too large / license):
- `gemma-4-E2B-it-QAT-Q4_0.gguf` → extraction + answer synthesis (HuggingFace `lmstudio-community/gemma-4-E2B-it-QAT-GGUF`)

Put it in `<project-root>/models/`. `run.sh` finds it automatically.

> **Retired:** the larger `gemma-4-E4B-it-QAT-Q4_0.gguf` was retired — it
> gave ~22s answers for no measurable quality gain on the 12 GB card, so
> E2B now serves BOTH extraction and synthesis at ~2.2s p95. E4B is no
> longer started (its systemd unit is disabled).

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

## Quality gate (before you ship)

`scripts/release-gate.py` runs 14 checks (health, retrieval-latency p95 < 400ms,
synthesis non-empty, synthesis-latency p95 < 4s, answer quality, concurrency,
doc/code consistency). The answer-quality check
enforces **faithfulness ≥ 0.85** AND **context_precision ≥ 0.50** against the
golden set:

```bash
bash run.sh doctor && bash run.sh serve
./venv/bin/python scripts/release-gate.py        # 14/14 ALL SYSTEMS GO
```

By default answer quality scores with a fast **local lexical proxy** (no
dependencies, no API keys, works offline). For stricter *semantic*
faithfulness, install the optional deps and opt into ragas:

```bash
./venv/bin/pip install ragas datasets langchain-openai
EVAL_USE_RAGAS=1 ./venv/bin/python scripts/release-gate.py
```

**Why ragas is NOT the default hard gate.** RAGAS semantic faithfulness
requires calling an external LLM (typically GPT-4 via an OpenAI API key).
That directly violates the project's intended "100% local, no API keys"
principle, so it cannot be the mandatory gate. It also needs heavy deps
(`ragas` + `datasets` + `langchain-openai`) that a fresh clone won't have,
and its per-run latency (~3-5 min with real ragas) is prohibitive for
quick iteration. The gate therefore auto-selects ragas only when both
`EVAL_USE_RAGAS=1` AND ragas is installed; otherwise it uses the local
lexical proxy — so a fresh clone stays green without those extras.

---

## Explore (next steps, in order)

1. **Ask with the `enterprise` domain (default)** — omit `domain` or pass
   `domain:"enterprise"`. On first boot it auto-seeds itself with this
   project's own `docs/` (tagged `self-docs`), so you can ask how the
   system works immediately. See [QUICKSTART.md](QUICKSTART.md).
2. **Add the SNOMED clinical graph + companion** — `snomed` (terminology graph)
   and `clinical_prose` are NOT auto-loaded; ingest them when you want
   clinical/differential answers (see [QUICKSTART.md §3](QUICKSTART.md)).
3. **Ingest your own docs into a NEW domain** — copy `domains/clinical_prose/`
   (or `domains/snomed/`) and edit `domain_config.yaml`. No daemon code change
   needed. See [docs/domains/README.md](docs/domains/README.md).
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

> **Configured domains.** The deployment ships with **4 concrete domains** plus
> 1 alias: `enterprise` (default — auto-seeded with this project's own
> `docs/`, non-confidential self-docs/ADR corpus), `snomed` (clinical
> terminology graph), `clinical_prose` (medicine prose companion),
> `legal`, `fraud` (graph + vector), and `healthcare` (alias → `snomed`).
> `domain="all"` fans out to every concrete domain. To add a new
> corpus, copy an existing `domains/<name>/` plugin and register it in
> `domain_config.yaml` — the federated factory picks it up with no daemon-code
> changes.
>


## Requirements

- **NVIDIA GPU** (~8–12 GB VRAM; RTX 3060 12 GB tested). CPU fallback exists
  (`serve_cpu.py`) but is slow.
- **Docker + docker compose** (Neo4j + Qdrant).
- **Python 3.11** venv.
- **One GGUF model file** — `gemma-4-E2B-it-QAT-Q4_0.gguf` (extraction + synthesis). Optional: add more domains' models as needed.

Full detail, including VRAM budgets and known limitations, in [RUN.md §1](RUN.md).
