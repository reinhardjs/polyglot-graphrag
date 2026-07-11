# Polyglot GraphRAG vs GreyCat — Comparison

GreyCat publishes a comparison at
[greycat.io/compare.html](https://greycat.io/compare.html) framing the typical
"polyglot data/RAG stack" (Qdrant + Neo4j + Elasticsearch + embedding server +
reranker + orchestration + cache + API/MCP) as **~8 services** that should
collapse into their **1 binary**. Our project *is* one of those polyglot stacks,
so this is a direct apples-to-apples comparison.

Below is an honest read — what GreyCat gets right, where our stack wins, and
where the comparison page oversimplifies.

---

## What GreyCat claims

| Dimension | Polyglot stack (us) | GreyCat |
|-----------|---------------------|---------|
| Components | 8 separate services | 1 ~4.6 MB binary |
| Data models | One store per shape | Unified time-series + graph + vector + full-text |
| Temporal queries | Bolted on per store | Native (time is first-class) |
| Hybrid search | 2 engines + reranker | One index (BM25 + vector + RRF) |
| AI / MCP | Bolt-on services | Built-in on-device embeddings + MCP |
| Footprint | Multiple clusters | Single binary |
| Ops | Keep N stores in sync | One store, one endpoint |
| Sovereignty | Managed APIs may off-site data | Fully self-hosted, EU (Luxembourg) |
| Cost | Multiple licenses + cloud | ~8× cheaper |

---

## Where GreyCat is right

1. **Operational simplicity is real.** We run Qdrant (`:6333`) + Neo4j (`:7687`)
   in Docker plus 3 systemd services (E2B `:8082`, E4B `:8084`, daemon `:8000`).
   That is genuinely 5+ moving parts to provision, secure, and keep alive. A
   single binary with one import, one endpoint, one transaction is objectively
   easier to operate.
2. **Cross-system queries are a real tax.** Our `/ask` does parallel Qdrant +
   Neo4j calls and fuses them in Python. A multi-hop graph walk that also needs
   vector similarity means two network round-trips and a join in app code.
   GreyCat does this inside one engine and one transaction.
3. **Temporal data.** We have no native time dimension. If you need
   "what did this entity look like on date X", you'd model it manually. GreyCat
   has it built in.
4. **Footprint.** A 4.6 MB binary vs our Docker + model weights (~10 GB VRAM,
   several GB on disk) is not a contest on the "ship a single artifact" axis.

---

## Where our stack wins

1. **Model freedom (the big one).** GreyCat's pitch is "on-device embeddings +
   built-in reranker" — but you get *their* models. Our stack is
   **model-agnostic by design**: `EMBED_MODEL_NAME`, `RERANK_MODEL_NAME`,
   `EXTRACTION_LLM_MODEL`, `SYNTHESIS_LLM_MODEL` are config flags. Swap Jina v3
   for `bge-m3`, E2B for Granite, E4B for Llama — change a line, no recompile.
   GreyCat's unified engine is also a locked-in engine.
2. **Best-in-class components.** We use Jina v3 (1024-d, cross-lingual),
   BGE v2-m3 (strong multilingual reranker), and Gemma 4 E2B/E4B (QAT Q4_0).
   These are SOTA open-weight models you can audit, quantize, and run offline.
   GreyCat's built-in embeddings are not separately tunable per task.
3. **Domain profiles.** Our entire v2.6.0 is built around
   `v2/domains/*.toml` — chunking, extraction prompt, synthesis prompt, graph
   schema, metadata, and entry strategy per domain, hot-swappable without code
   changes. GreyCat's "one query model" doesn't expose this level of
   per-domain behavioral configuration out of the box.
4. **Graph extraction quality.** We extract a typed knowledge graph via LLM
   (E2B) with GLiNER fallback, then resolve entities cross-lingually in Neo4j's
   vector index. The graph is a first-class artifact you can query with Cypher,
   visualize, and traverse. GreyCat's graph is typed nodes + dot-notation — good,
   but our extraction pipeline is purpose-built for messy real-world documents
   (ADRs, bug reports, clinical notes).
5. **Transparency / no black box.** Every component is open-source or
   open-weight. You can read the embedding model's card, the reranker's paper,
   the LLM's weights. GreyCat is a commercial binary (free tier, but closed).
6. **Ecosystem familiarity.** Qdrant + Neo4j + FastAPI is a stack thousands of
   engineers already know. Hiring, debugging, and extending use skills your team
   already has.

---

## Where the comparison page oversimplifies

- **"8 services" is a strawman for our case.** We run 2 Docker containers +
  3 systemd units — and 2 of those (the LLMs) are only because we chose
  open-weight models. You could run our stack on a single process if you used
  smaller in-process models. The "8" includes things like a separate cache and
  MCP server we don't run as independent clusters — our cache is in-Qdrant and
  our API is one FastAPI app.
- **"Keep everything in sync" assumes you're running managed services.** Ours
  is fully local: Qdrant and Neo4j are on the same host, same Docker network.
  There is no cross-cloud sync tax. The fan-out in `/ask` is two localhost
  calls — microseconds, not cross-region latency.
- **"~8× cheaper" is undefined.** If you already own the GPU (we run on a
  single RTX 3060, ~$300 used), our marginal cost is $0. GreyCat's "free binary"
  still needs hardware to run on. The cost claim only bites if you were paying
  for Pinecone + Neo4j Aura + a managed embedding API — which we never did.
- **Temporal is presented as a gap we can't close.** True natively, but you can
  model time as a node property / relationship in Neo4j today. It's not free,
  but it's not impossible either.

---

## Bottom line

| If you need… | Choose |
|--------------|--------|
| One artifact to ship, temporal data native, minimal ops | **GreyCat** |
| Model freedom, per-domain behavior, open-weight SOTA, auditability | **Polyglot GraphRAG** |
| A team that already knows Qdrant/Neo4j/FastAPI | **Polyglot GraphRAG** |
| Zero infrastructure, single binary, EU-hosted | **GreyCat** |
| Multi-domain RAG with typed graph extraction + citations | **Polyglot GraphRAG** |

Neither is "wrong." GreyCat optimizes for **operational simplicity and unified
data model**. Our stack optimizes for **model freedom, domain specialization,
and transparency**. Pick based on which constraint is tighter for your use case.

For a self-hosted, GPU-owned, multi-domain knowledge base where you want to
swap models and tune per-domain extraction without vendor lock-in — our stack is
the better fit. For a quick-to-deploy, temporally-aware, single-binary backend
where you don't care which embedding model runs — GreyCat is compelling.
