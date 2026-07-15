# domains/ — per-domain ingest + retrieve plugins

`domains/` is where every knowledge domain (and every **companion** corpus)
keeps its custom ingest/retrieve logic. The rest of the system never hardcodes
a domain's internals — retrieval composition is driven by `domain_config.yaml`
via the **federated factory** (`federated.py`), not `if domain == "snomed"`.

```python
from domains import get_retriever, get_ingestor
retrieve = get_retriever("engineering")   # primary corpus (uses default unless overridden)
ctxs = retrieve("who reported BUG-204?", top_k=5)
ingest  = get_ingestor("engineering_docs") # companion (custom ingest.py)
```

## What is a "corpus"?

A **corpus** = one self-contained body of searchable knowledge: a Qdrant
collection of embedded text chunks, plus (optionally) a Neo4j graph of entities
& relationships extracted from those chunks. You query a corpus by embedding
your question and finding the nearest chunks (vector search), and/or walking
its graph (entity questions like "who reported BUG-204?").

There are two kinds of corpus per domain:

| Kind | Role | Example (engineering) | How it's searched |
|------|------|------------------------|-------------------|
| **Primary corpus** | The domain's *own* real documents (ADRs, bugs, PRs, wiki). The main source of truth. | `engineering_chunks` (+ `Engineering` graph) | Vector search + graph walk |
| **Secondary / companion corpus** | An *extra* collection attached to a domain to fill a gap the primary can't. Declared via `companions:` in `domain_config.yaml`. | `engineering_docs` (this repo's `docs/`) | Vector search (tagged `_signal: engineering_docs`) |

**Why both?** They answer different question types and are searched in parallel,
then fused. The primary's graph is blind to prose/design knowledge (e.g. "what
is the total VRAM budget on the RTX 3060?" — only the companion's
`architecture.md` chunk knows). The companion is NOT a duplicate of the primary;
it's a different collection with different text, cut differently, for a
different recall job.

## Layout

```
domains/
  __init__.py            # registry: get_retriever / get_ingestor (with default fallback)
  _default.py            # default retriever + ingestor (the PRIMARY path; single source of truth)
  engineering/           # PRIMARY domain (wrapped in domains/ for uniformity)
    retrieve.py          # delegates to the default retriever (Qdrant + Neo4j graph)
                          # (no ingest.py -> uses default ingest; see below)
  engineering_docs/      # companion (SECONDARY corpus) of engineering
    ingest.py            # custom loader: this repo's docs/ -> Qdrant
    retrieve.py          # dense + sparse vector search over the prose
  clinical_prose/        # companion of snomed
    ingest.py
    retrieve.py
  example_companion/     # template/standalone companion (copy to start a new one)
    ingest.py
    README.md
  snomed/                # a real domain retrieved by term-match (graph-only, no Qdrant)
    retrieve.py
federated.py             # config-driven multi-retriever assembly (the Factory)
```

## Custom plugin is OPTIONAL — the default fallback

You do NOT need a `domains/<name>/` folder for a domain to work. The factory
resolves a retriever/ingestor in this order:

1. **Custom** `domains/<name>/retrieve.py` (or `ingest.py`) if the folder exists.
2. **Default** (`domains/_default.py`) otherwise.

So:
* The **primary** corpus of `enterprise` works with **no custom code** — it
  uses the default retriever (Qdrant `collection` + Neo4j `neo4j_label`) and the
  default ingestor (`ingest.py::ingest_text`, the same machinery `sync_docs.py`
  uses). `domains/enterprise/retrieve.py` exists only to make the primary's
  home in `domains/` explicit; it just points at the default.
* A **companion** (e.g. `enterprise_docs`) normally DOES need a custom
  `ingest.py` (different source, different chunking) — that's the whole point of
  a plugin. But if a companion's data can be ingested by the generic path, it can
  omit `ingest.py` and use the default too.

**Rule of thumb:** *primary = default unless overridden; companion = custom
ingest, default retrieve unless overridden.*

## Contract

A domain folder MAY contain:

* `retrieve.py` → `retrieve(query, top_k=5) -> list[dict]` (each dict:
  `{"text", "doc_id", "doc_type", "chunk_idx"}`). Optional
  `retrieve_temporal(presentation, top_k)` for graduated presentations.
  If absent, the default retriever is used.
* `ingest.py` → `ingest(source=None, **kw) -> int` that loads data into the
  domain's Qdrant collection / Neo4j subgraph. If absent, the default ingestor
  is used.

The Qdrant collection name and Neo4j label come from `domain_config.yaml`
(`collection` / `neo4j_label`) — **never hardcoded** in the code.

## Adding a new domain (no core changes)

1. Add a block to `domain_config.yaml` under `domains:` with `collection`,
   `neo4j_label`, `chunking`, `companions:` (optional secondary corpora), etc.
2. (Optional) `mkdir domains/<name>` and add `retrieve.py` / `ingest.py` for
   custom behavior. Skip this entirely to use the default (primary) path.
3. Ingest: primary via `sync_docs.py` (or `POST /ingest`); companion via
   `POST /ingest_domain {"domain":"<name>"}`.

`/ask`, the reranker, and synthesis pick it up automatically.

## Registry semantics

* `get_retriever(name)` → `domains/<name>/retrieve.py` if present, else the
  default retriever (searches `collection` + `neo4j_label` from config).
* `get_ingestor(name)` → `domains/<name>/ingest.py` if present, else the
  default ingestor (`ingest.py::ingest_text`, the `sync_docs.py` path).
* `list_domains()` → domains with a `retrieve.py` on disk (runnable set).
* `supported_domains()` → domains declared in `domain_config.yaml`.
