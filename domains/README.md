# domains/ — per-domain ingest + retrieve plugins

Every knowledge domain owns its OWN ingest and retrieve logic here. The rest
of the system never hardcodes a domain's internals. Retrieval composition
is driven by `domain_config.yaml` via the **federated factory**
(`federated.py`), not `if domain == "snomed"` in the daemon.

```python
from domains import get_retriever
retrieve = get_retriever("snomed")          # free-text query
ctxs = retrieve("fever and rash", top_k=5)
# graduated presentation:
retrieve_t = get_retriever("snomed", temporal=True)
ctxs = retrieve_t({"day1": ["fever"], "day2": ["rash"]}, top_k=5)
```

## Architecture: Strategy + Mediator + Factory

* **Strategy** — each `domains/<name>/retrieve.py` is a retrieval strategy
  (`retrieve(query, top_k)`, optional `retrieve_temporal`). Swappable
  per domain, no shared base class.
* **Factory** — `federated.assemble(req, vec, rag)` reads the domain's
  config and assembles the strategy set: the primary retriever (unless
  `retrieval: dense_prose`), its `companions:` list, AND the generic
  Qdrant + Neo4j parallel path (for domains with a `collection`).
  It tags every record with `_signal = <domain name>` so the fusion
  step can compute dual-evidence generically.
* **Mediator** — `/ask` (serve_gpu.py) orchestrates cache → factory →
  fuse/rerank → (optional) differential persist → synthesize. It contains
  NO domain-specific `if` branches; all domain knowledge lives in config.

## Layout

```
domains/
  __init__.py            # registry: get_retriever(name, temporal=False)
  snomed/
    retrieve.py          # term-match + IDF + temporal (the diagnosis retriever)
  clinical_prose/
    ingest.py            # Wikipedia medicine -> Qdrant (semantic corpus)
    retrieve.py          # dense + sparse vector search over the prose
federated.py             # config-driven multi-retriever assembly (the Factory)
```

## Contract

Each domain folder is a **package** (directory with code). To be runnable it
MUST contain `retrieve.py` exposing:

* `retrieve(query: str, top_k: int = 5) -> list[dict]`
  each dict: `{"text", "doc_id", "doc_type", "chunk_idx"}`
* optionally `retrieve_temporal(presentation: dict, top_k=5) -> list[dict]`
  for graduated / day-by-day presentations.

Optionally `ingest.py` exposing `ingest(source=None, **kw)` that loads data
into that domain's Qdrant collection / Neo4j subgraph.

The domain's Qdrant collection name and Neo4j label are declared in
`domain_config.yaml` (`collection` / `neo4j_label`) — **never hardcoded**
in the retrieve/ingest code.

## Adding a new domain (no core changes)

1. `mkdir domains/<name>`
2. write `retrieve.py` with `retrieve(query, top_k)` (+ optional
   `retrieve_temporal`)
3. write `ingest.py` with `ingest(...)` (if it needs data loaded)
4. add a block to `domain_config.yaml` under `domains:` with `collection`,
   `neo4j_label`, `retrieval`, `companions:` (optional companion corpora),
   `synthesis:` (optional `{kind: ...}` to select a synthesis prompt),
   `entity_types`, `relation_types`, `pair_strategy`, `min_confidence`,
   `chunking`, `metadata_schema`
5. (optional) trigger ingest via `POST /ingest_domain {"domain":"<name>"}`

That's it — `/ask`, the reranker, and synthesis pick it up automatically.

## SNOMED + clinical_prose (hybrid diagnosis)

`snomed` alone can't describe disease presentations (concept names only). The
`clinical_prose` domain supplies free-text disease descriptions (Wikipedia
medicine articles) embedded into Qdrant. When `/ask` runs with
`domain=snomed`, the federated factory calls BOTH retrievers (declared via
the `companions: [clinical_prose]` list in the `snomed` config block) and
merges their contexts before rerank + synthesis, so a query like
"fever, cough, rash" matches the Measles *article* (semantic) even though
the SNOMED "Measles (disorder)" name contains none of those words
(term-match would miss it). Every record is tagged `_signal` so the
dual-evidence boost is computed generically — works for ANY domain with
≥2 signals, not just snomed+prose.

## Registry semantics

* `get_retriever(name)` imports `domains/<name>/retrieve.py` on first use and
  caches the module.
* If the module defines `retrieve_temporal`, `get_retriever(name, temporal=True)`
  returns a callable that dispatches to it.
* `list_domains()` returns domains with a `retrieve.py` on disk (runnable set).
* `supported_domains()` returns domains declared in `domain_config.yaml`.
