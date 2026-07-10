# Hermes Integration — GraphRAG Plugin

The `rag_query` Hermes tool gives any Hermes agent access to the engineering
knowledge base by calling the GraphRAG daemon's `/ask` endpoint over HTTP.

## Quick start

```bash
# Enable the plugin (if not already enabled)
hermes plugins enable rag

# Use in chat — Hermes auto-invokes rag_query for knowledge-base questions
hermes
> who reported BUG-204 and what severity was it?
```

No configuration needed if the daemon runs on `http://127.0.0.1:8000`.
For remote agents, set the `RAG_DAEMON_URL` environment variable:

```bash
export RAG_DAEMON_URL=http://192.168.1.50:8000
hermes
```

## Architecture

```
┌──────────────┐   HTTP POST /ask    ┌─────────────────────┐
│  Hermes      │ ───────────────────>│  serve_gpu.py :8000 │
│  rag plugin  │ <───────────────────│  (GPU daemon)        │
│  (tools.py)  │   JSON response     │                     │
└──────────────┘                     │  Jina embed         │
                                     │  Qdrant + Neo4j     │
                                     │  BGE rerank         │
                                     │  E4B synth          │
                                     └─────────────────────┘
```

The plugin makes ONE HTTP call per query. No rag-env Python needed on the
agent side — the daemon runs the full pipeline server-side.

## Plugin files

| File | Purpose |
|------|---------|
| `plugin.yaml` | Registration: provides `rag_query` tool, version, dependency on `requests` |
| `schemas.py` | Tool schema: description, parameters (query, synthesize). Hermes reads this to decide when to auto-invoke |
| `tools.py` | HTTP handler: `POST RAG_DAEMON_URL/ask`, returns JSON string. `check_requirements()` pings `/health` |
| `__init__.py` | `ctx.register_tool(rag_query_handler, check_fn=check_requirements)` |

Location: `~/.hermes/plugins/rag/`

## Tool behavior

- **When Hermes auto-invokes**: The schema description signals the tool covers engineering KB questions. Hermes targets `rag_query` for queries about bugs, ADRs, PRs, components, and developers.
- **`synthesize` parameter**: `false` returns contexts only (~0.2s). Hermes reasons over the raw passages and drafts its own answer. `true` adds E4B synthesis (~6s total) — the answer comes from the KB directly.
- **`check_requirements()`**: Pings `GET /health` on the daemon. If unreachable, the tool is disabled (Hermes won't offer it).
- **Output format**: JSON string — `{"query": "...", "n_contexts": 5, "contexts": [...], "answer": ""}`. Hermes parses this and uses the contexts to answer.

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `RAG_DAEMON_URL` | `http://127.0.0.1:8000` | Daemon base URL. Set for remote agents. |

## Verified queries

All produce correct, sourced answers via `rag_query`:

| Query | Answer |
|-------|--------|
| who reported BUG-204 and what severity? | bob, SEV-2 [1] |
| who reviewed PR-482? | carol [1] |
| what PR implemented ADR-014? | PR-482, reviewed by carol [1,3] |
| how does checkout relate to billing per our ADRs? | billing depends on checkout; ADR-014 event-driven migration |
| basis data apa yang digunakan ADR-021? | PostgreSQL [1,3] (multilingual, ID→EN via CANON_MAP) |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `rag_query` not offered | Daemon down or plugin disabled | `curl http://127.0.0.1:8000/health`; `hermes plugins enable rag` |
| Returns HTTP error | Daemon unreachable | Check `RAG_DAEMON_URL`, verify daemon process |
| Returns empty contexts | KB not ingested | `bash run.sh ingest sample_data` |
| Slow response | E4B synthesis (6s) | Set `synthesize:false` for fast retrieval (~0.2s) |
