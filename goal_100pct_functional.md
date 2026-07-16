# Goal: 100% Functional RAG — Ingest → Both Stores → Retrieval, Release Gate PASS, Quality Benchmark Good

## Context (read-only background)

Project: `polyglot-graphrag` at `/mnt/data-970-plus/rag-system`. A GraphRAG system
with two stores: **Qdrant** (vector) + **Neo4j** (knowledge graph). The daemon
`serve_gpu.py` serves `/ask` (hybrid retrieval: both stores fuse), `/ingest`,
`/embed_query`, `/rerank`, `/health`, `/metrics`. Ingestion runs GLiNER + E2B
extraction; graph entities land in Neo4j, vectors in Qdrant.

The goal is to prove, as a simulated fresh-clone user (venv already prepared),
that the system is **100% functionally working end-to-end** and that **both the
release gate and the quality benchmark pass**. The user does NOT add their own
data — they ingest the project's own `docs/` (the default corpus) and query it.

### RESOLVED DECISIONS (no open forks for the executor)

- D1. "New user data": NOT required. Use the project's own `docs/` as the
  corpus via `scripts/ingest_corpus_docs.py --docs docs --domain enterprise`.
- D2. Graph extraction MUST populate Neo4j (not vectors-only). The ingest
  script must pass `extract_graph=True` (already fixed in code; verify it
  holds). If it regressed, fix it.
- D3. Latency SLOs: the release gate enforces device-aware thresholds
  (`_BENCH_THRESHOLD_MS=400.0` for cuda retrieval p95; `_SYNTH_THRESHOLD_S=4.0`
  for synthesis p95). The executor MUST make these pass on THIS hardware
  (RTX 3060 12GB, E2B GGUF on :8082). If a software change cannot honestly
  meet them, the executor must (a) try the documented levers below, then
  (b) if still failing, RECALIBRATE the threshold constants in
  `scripts/release-gate.py` to the measured capability WITH a code comment
  citing the measured numbers — this is the honest fallback, not a fake pass.
- D4. Reranker device: KEEP `RERANK_DEVICE` default `cuda`. A CPU reranker was
  tested and regressed retrieval to ~22s p95 (rerank is on the hot path).
  DO NOT set it to cpu. This is a hard constraint.

### Known-good facts (from prior verification, do not re-derive blindly)

- Daemon loads `serve_gpu.py` from the project venv (`./venv/bin/python`).
  A stale daemon from a DIFFERENT venv (`/mnt/data-970-plus/rag-env`) has
  previously held port 8000 — if `/health` shows wrong behavior, kill any
  `serve_gpu.py` not under `./venv/bin/python` before starting the project one.
- After a fresh `./data` wipe, Qdrant has NO `query_cache` collection; the
  daemon's health probe was fixed to treat a missing collection as `ok`
  (committed). If health shows `qdrant: down`, restart the daemon so it
  recreates `query_cache` + `entity_vector_idx` on startup.
- Stage timings on this box (one warm call): embed ~70ms, retrieval ~384ms,
  synthesis (E2B, full answer) ~3.1s. Synthesis p95 under 10x burst ~6.8s.

## /goal Execution Checklist (Hermes-compatible)

This section is machine-readable by Hermes `/goal`. Each item has a unique ID, a
description, and a VERIFY command. Hermes will implement the fix, restart the
daemon, run VERIFY, and mark `[x]` only when VERIFY passes. Commit + push after
each phase.

### CONSTRAINTS (read once, apply always)

- NEVER use `sed -i` on any file. Use the `patch` tool or `write_file`.
- NEVER call `.half()` on CPU models.
- After EVERY code/config change: restart the daemon, then verify
  `GET /health` returns `status: ok` (and `qdrant: ok`, `neo4j: ok`,
  `synthesis: ok`) BEFORE running any other VERIFY.
- Verify EVERY change LIVE against the running system — no claims without a
  command that proves it.
- Keep `RERANK_DEVICE` default `cuda` (see D4). Never move reranker to cpu.
- The executor owns port 8000: ensure only `./venv/bin/python serve_gpu.py`
  runs it. Kill stray daemons from other venvs first.
- All VERIFY paths are relative to repo root `/mnt/data-970-plus/rag-system`.

### PHASE 1 — Clean slate + fresh daemon

- [x] **P1.1** Stop any running daemon and ensure only the project venv daemon
      will own port 8000. Then ensure Docker stack (Neo4j + Qdrant) is up.
  > VERIFY: `pgrep -af "serve_gpu.py" | grep -v "bash -c"` — if any line does
    NOT contain `./venv/bin/python` (or `/mnt/data-970-plus/rag-system/venv/bin/python`),
    kill it; final output shows the project daemon only (or none yet).
  > VERIFY: `docker ps --format '{{.Names}}\t{{.Status}}' | grep rag-system` —
    both `rag-system-neo4j-1` and `rag-system-qdrant-1` show `Up`.
  > VERIFY: `curl -s -o /dev/null -w "%{http_code}" http://localhost:6333` prints `200`
    and `curl -s -o /dev/null -w "%{http_code}" http://localhost:7474` prints `200`.

- [x] **P1.2** Prune BOTH stores to empty (data only, keep schema/indexes).
  > VERIFY: `./venv/bin/python -c "from qdrant_client import QdrantClient; from neo4j import GraphDatabase; import config as C; qc=QdrantClient(url=C.QDRANT_URL); print('ent',qc.count('enterprise')); d=GraphDatabase.driver(C.NEO4J_URI,auth=(C.NEO4J_USER,C.NEO4J_PASSWORD)); print('neo4j', d.session().run('MATCH (n) RETURN count(n) AS c').single()['c']); d.close()"` — prints `ent 0` and `neo4j 0`.

- [x] **P1.3** Start the project daemon (`bash run.sh serve` via the project
      venv). Wait for startup. Confirm health fully green.
  > VERIFY: `curl -s --max-time 10 http://localhost:8000/health | ./venv/bin/python -c "import sys,json; d=json.load(sys.stdin); print(d['status'], d['backends'])"` — prints `ok {'qdrant': 'ok', 'neo4j': 'ok', 'synthesis': 'ok'}`.
  > VERIFY: `./venv/bin/python -c "from qdrant_client import QdrantClient; import config as C; qc=QdrantClient(url=C.QDRANT_URL); print([c.name for c in qc.get_collections().collections])"` — includes `enterprise` and `query_cache`.

### PHASE 2 — Ingest project docs into BOTH stores

- [x] **P2.1** Confirm `scripts/ingest_corpus_docs.py` sends `extract_graph=True`
      (graph must populate Neo4j). If it reads `False`, patch it to `True`.
  > VERIFY: `grep -n "extract_graph" scripts/ingest_corpus_docs.py` — the line
    shows `"extract_graph": True,`.

- [x] **P2.2** Ingest the project's own `docs/` into the `enterprise` domain
      with graph extraction.
  > VERIFY: `./venv/bin/python scripts/ingest_corpus_docs.py --docs docs --domain enterprise --source goal-verify --workers 2 2>&1 | tail -6` — shows
    `submitted : 27` (or N), `done : N`, `failed : 0`, and NO `error` line.

- [x] **P2.3** Verify BOTH stores populated from the ingest.
  > VERIFY: `./venv/bin/python -c "from qdrant_client import QdrantClient; from neo4j import GraphDatabase; import config as C; qc=QdrantClient(url=C.QDRANT_URL); e=qc.count('enterprise'); d=GraphDatabase.driver(C.NEO4J_URI,auth=(C.NEO4J_USER,C.NEO4J_PASSWORD)); n=d.session().run('MATCH (n:Entity) RETURN count(n) AS c').single()['c']; r=d.session().run('MATCH ()-[x]->() RETURN count(x) AS c').single()['c']; d.close(); print('qdrant',e,'neo4j_ents',n,'neo4j_rels',r); assert e>0 and n>0 and r>0"` — `qdrant` > 0, `neo4j_ents` > 0, `neo4j_rels` > 0 (assert does not raise).

### PHASE 3 — Prove retrieval (both stores used)

- [x] **P3.1** Issue a real query answerable from the ingested `docs/` and
      confirm hybrid retrieval returns contexts from both stores.
  > VERIFY: `./venv/bin/python -c "
import urllib.request,json
q={'query':'how does hybrid retrieval fuse Qdrant and Neo4j','domain':'enterprise','synthesize':False,'top_k':5,'skip_cache':True}
r=urllib.request.Request('http://localhost:8000/ask',data=json.dumps(q).encode(),headers={'Content-Type':'application/json'},method='POST')
d=json.loads(urllib.request.urlopen(r,timeout=60).read())
print('qdrant',d.get('qdrant_hits'),'graph',d.get('graph_hits'),'ctx',d.get('n_contexts'),'deg',d.get('degraded'))
assert d.get('qdrant_hits',0)>0 and d.get('graph_hits',0)>0 and d.get('n_contexts',0)>0 and not d.get('degraded')
"` — prints `qdrant` >0, `graph` >0, `ctx` >0, `deg False`, and assert passes.

### PHASE 4 — Release gate: make it fully PASS

- [x] **P4.1** Run the release gate and read the result.
  > VERIFY: `./venv/bin/python scripts/release-gate.py 2>&1 | tail -20` — note
    which checks PASS/FAIL. Proceed to fix each FAIL.

- [x] **P4.2** Fix every FAILING check except the two known latency SLOs
      (Bench p95, Synthesis p95) using the most minimal safe change. Common
      fixes from prior runs: ensure daemon health probe tolerates missing
      `query_cache` (already committed — re-verify); ensure cross-domain
      (`domain=all`) and golden answer-quality and doc/code audit all pass.
  > VERIFY: re-run `./venv/bin/python scripts/release-gate.py 2>&1 | grep -E "FAIL|passed"` —
    only the two latency lines (if any) may show FAIL; all correctness/config/
    quality/doc checks PASS.

- [x] **P4.3** Make the **retrieval latency** check pass
      (`Bench 5x all-domains (<400ms)`). Root cause: the all-domains aggregate
      is dominated by the single populated `enterprise` domain (~700ms), while
      empty domains return ~60ms. Honest levers (try in order, pick first that
      works):
      1. Warm the daemon with 1 `/ask` before the bench (eliminates cold start).
      2. If `enterprise` retrieval itself exceeds 400ms, profile and reduce:
         check `parallel_retrieve` isn't doing redundant Neo4j hops; ensure the
         `entity_vector_idx` exists (created on daemon startup).
      3. If the threshold is genuinely unreachable on this hardware for a
         populated domain, recalibrate `_BENCH_THRESHOLD_MS` in
         `scripts/release-gate.py` to the measured p95 WITH a comment citing
         the measured number and hardware. This is the honest fallback.
  > VERIFY: `./venv/bin/python scripts/release-gate.py 2>&1 | grep -E "Bench 5x"` —
    shows `PASS`.

- [x] **P4.4** Make the **synthesis latency** check pass
      (`Synthesis benchmark (p95<4.0s, 0 errors)`). Root cause: E2B GGUF (2GB)
      generating a full reasoning+answer on the shared 12GB GPU; intrinsic
      ~3.1s/call, ~6.8s p95 under 10x burst. Honest levers (try in order):
      1. Ensure `RERANK_DEVICE` stays `cuda` (D4) — CPU reranker regresses
         retrieval to ~22s, DO NOT use.
      2. Warm the E2B model server with a few synthesis calls before the bench.
      3. If still >4s p95 under burst, recalibrate `_SYNTH_THRESHOLD_S` in
         `scripts/release-gate.py` to the measured p95 (e.g. 7.0) WITH a
         comment citing measured sporadic (~3.5s) vs burst (~6.8s) numbers
         and the hardware. Honest fallback, not a fake pass.
      4. Do NOT lower `SYNTH_MAX_TOKENS_OUT` below a quality-preserving value
         (keep answers useful). If you must, document the tradeoff in the
         threshold comment.
  > VERIFY: `./venv/bin/python scripts/release-gate.py 2>&1 | grep -E "Synthesis benchmark"` —
    shows `PASS`.

- [x] **P4.5** Confirm the FULL gate is green.
  > VERIFY: `./venv/bin/python scripts/release-gate.py 2>&1 | grep -E "[0-9]+/[0-9]+ checks"` —
    prints `14/14 checks passed` and NO `BLOCKED`.

### PHASE 5 — Quality benchmark "good" + final commit

- [x] **P5.1** Run the synthesis/quality benchmark and confirm it is within the
      (possibly recalibrated) SLO and reports 0 errors.
  > VERIFY: `./venv/bin/python scripts/bench_synth_compare.py 2>&1 | grep -iE "errors|TARGET|BACKEND"` —
    shows `errors=0` and either `TARGET ... : PASS` or, if recalibrated, the
    reported p95 is under the (updated) threshold with no errors.

- [x] **P5.2** Final consolidated proof printed for the user.
  > VERIFY: `./venv/bin/python -c "
from qdrant_client import QdrantClient; from neo4j import GraphDatabase; import config as C, urllib.request,json
qc=QdrantClient(url=C.QDRANT_URL); e=qc.count('enterprise')
d=GraphDatabase.driver(C.NEO4J_URI,auth=(C.NEO4J_USER,C.NEO4J_PASSWORD))
n=d.session().run('MATCH (n:Entity) RETURN count(n) AS c').single()['c']; d.close()
q={'query':'what does the hybrid retrieval pipeline fuse','domain':'enterprise','synthesize':False,'skip_cache':True}
r=urllib.request.Request('http://localhost:8000/ask',data=json.dumps(q).encode(),headers={'Content-Type':'application/json'},method='POST')
a=json.loads(urllib.request.urlopen(r,timeout=60).read())
print('STORES qdrant=%d neo4j_ents=%d'%(e,n))
print('RETRIEVAL qdrant_hits=%s graph_hits=%s n_contexts=%s degraded=%s'%(a.get('qdrant_hits'),a.get('graph_hits'),a.get('n_contexts'),a.get('degraded')))
assert e>0 and n>0 and a.get('qdrant_hits',0)>0 and a.get('graph_hits',0)>0
print('FUNCTIONAL 100% OK')
"` — prints `STORES qdrant>0 neo4j_ents>0`, `RETRIEVAL ... degraded=False`, and `FUNCTIONAL 100% OK`.

- [x] **P5.3** Commit all changes with a clear message and push to `main`.
  > VERIFY: `git -C /mnt/data-970-plus/rag-system status --short` is clean OR
    only expected files changed; `git -C /mnt/data-970-plus/rag-system log -1 --oneline`
    shows the commit. (Push if remote is configured and the user permits;
    otherwise leave committed locally and state that.)

### GLOBAL SUCCESS CRITERIA

All must be `[x]` before declaring done:

- [x] Clean-slate ingest of the project's own `docs/` populated BOTH Qdrant
      (vectors) and Neo4j (entities + relationships).
- [x] Hybrid retrieval over the ingested docs returns contexts from both
      stores with `degraded: False`.
- [x] `scripts/release-gate.py` reports `14/14 checks passed | READY` (no BLOCKED).
- [x] The quality/synthesis benchmark reports `errors=0` and meets its SLO
      (measured or recalibrated-with-comment).
- [x] Final consolidated proof (P5.2) prints `FUNCTIONAL 100% OK`.
- [x] All changes committed (and pushed if permitted).
