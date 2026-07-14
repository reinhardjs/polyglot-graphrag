# domains/example_companion/ — TEMPLATE: a 2nd semantic corpus

This folder is a complete, self-contained example of a **companion corpus**:
a second dense-prose collection you attach to ANY primary domain via that
domain's `companions:` list in `domain_config.yaml`. It exists to teach the
pattern (copy it, rename it, point it at your own data). It is ALSO wired
live as a companion of `engineering`, so querying `domain=engineering`
exercises the generic ≥2-signal dual-evidence path on a NON-clinical domain.

## The pattern in 3 steps

1. **Write `retrieve.py`** — expose `retrieve(query, top_k=5) -> list[dict]`
   with the standard `{"text","doc_id","doc_type","chunk_idx"}` shape, plus an
   optional `retrieve_temporal(presentation, top_k)`. Embed via the resident
   GPU daemon (`config.DAEMON_EMBED_QUERY`) so you never load a 2nd model.
   Prefix each record's text with a `[[SIGNAL] ...]` tag (e.g.
   `[EXAMPLE COMPANION] `) — the fusion step uses `_signal` to compute
   cross-signal boosts, and the prefix makes the signal visible in context.

2. **Write `ingest.py`** — expose `ingest(source=None) -> int` that loads
   your docs, batch-embeds via `config.DAEMON_EMBED_LATE`, and upserts
   dense+sparse points into a Qdrant collection. Declare the collection name
   in `config.QDRANT_COLLECTIONS` AND in the domain's `domain_config.yaml`
   `collection:` field (they must match).

3. **Register + wire it** in `domain_config.yaml`:
   - add a block under `domains:` for the companion itself
     (`retrieval: dense_prose`, `entity_types: []`, a `collection:`).
   - add its name to the PRIMARY domain's `companions:` list.

That's it. The **federated factory** (`federated.py`) runs the companion in
parallel with the primary, tags records with `_signal`, and the dual-evidence
boost fires automatically — NO daemon code changes.

## Files
- `retrieve.py` — semantic retriever over `example_companion` (Strategy impl).
- `ingest.py`  — sample ingest (2 docs) so the retriever has something to find.

## Try it

```bash
# 1) Ingest the sample docs into Qdrant (collection: example_companion)
curl -X POST localhost:8000/ingest_domain \
     -H 'Content-Type: application/json' \
     -d '{"domain":"example_companion"}'

# 2) Engineering now has 2 signals: engineering_chunks + example_companion.
#    The dual-evidence boost can corroborate a candidate across both.
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"query":"companion corpus dual signal","domain":"engineering","skip_cache":true}'

# 3) Inspect signals returned for the query
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"query":"example","domain":"engineering","skip_cache":true}' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('path',d['path']);\
print({m['_signal'] for m in d['contexts_meta'] if '_signal' in m})"
```

## Adding YOUR OWN companion (real use)
```bash
cp -r domains/example_companion domains/<my_corpus>
# edit retrieve.py (COLLECTION, prefix), inges.py (your source), then:
#   config.QDRANT_COLLECTIONS['<my_corpus>'] = '<my_corpus>'
#   domain_config.yaml: add `domains.<my_corpus>` block + list it under
#   the primary domain's `companions:`
```
