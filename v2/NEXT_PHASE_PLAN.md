# NEXT PHASE PLAN — GLiNER + E2B Hybrid & Model Intelligence Tuning

**Author:** Hermes Agent (via deepseek-v4-pro)  
**Date:** 2026-07-12  
**Handoff target:** hy3:free (OpenRouter) — self-contained, no parent context needed  
**Repo:** `/mnt/data-970-plus/rag-system/v2/`

---

## EXECUTIVE SUMMARY

Current state: E2B full-doc llm extraction delivers 89% edge precision at 10.5s/doc on Gemma-4-E2B Q4_0. The v3 infrastructure (domain YAML, batch UNWIND Neo4j, hybrid vector+graph retrieval) is proven and stable.

This plan covers three parallel workstreams:
1. **GLiNER + E2B hybrid extraction** (GLiNER entity spans + E2B relation reasoning)
2. **E2B server tuning** (llama.cpp flag optimization for the 3060 12GB)
3. **Higher Qwen intelligence** (test Qwen2.5-3B and 7B locally as faster alternatives)

---

## WORKSTREAM 1: GLiNER + E2B HYBRID (primary deliverable)

### Why
- E2B full-doc extraction is accurate (89%) but slower (10.5s) — it detects entities AND relations in one pass
- GLiNER entity detection is fast (~50ms) and provides character-level spans + types
- Combining both: GLiNER gives precise entity spans/types, E2B only classifies relations between them
- Expected: similar or better precision than full-doc E2B, potentially faster (E2B doesn't decide "what is an entity")

### Design
```
document → GLiNER (:8000) → entities [{name, type, start, end}]
         → E2B (:8082) receives: "Here are entities: [...]. Document: {...}. List all relationships."
         → {relations: [{source, target, type, confidence}]}
         → batch UNWIND Neo4j
```

### Implementation Steps

#### 1. Add new EXTRACTION_MODE: "hybrid"
- `config.py`: add `EXTRACTION_MODE = "hybrid"` option
- This mode uses GLiNER for entities + E2B for relations (single full-doc prompt, not batch pairs)

#### 2. Create `hybrid_extraction.py` (new file)
```python
def extract_hybrid(text: str, domain: str) -> dict:
    """
    1. Call GLiNER /extract_entities → entities
    2. Build prompt: entity list + document + relation vocabulary
    3. Call E2B on :8082 with prompt → parse JSON relations
    4. Return {entities: [...], relations: [...]}
    """
```

**Critical difference from index_routing:** 
- NOT batched pairs (that's what broke E2B before)
- Single full-doc prompt: "Here are the entities GLiNER found: [...]\n Document: {...}\n List all relationships using only: [vocab]. Output JSON."
- E2B gets the full document context (where it excels)
- E2B uses GLiNER's entity list as a *hint*, not a constraint

#### 3. Prompt Design (key to success)
```
You are extracting relationships from an engineering document.

ENTITIES (pre-detected, use these exact names):
- PR-482 (entity_type: PullRequest)
- bob (entity_type: Person)
- BUG-204 (entity_type: Bug)
- checkout-publisher (entity_type: Service)
- checkout database (entity_type: Database)
- ADR-014 (entity_type: ADR)
- auth-service (entity_type: Service)

RELATION TYPES allowed: AUTHORED, FIXES, DEPENDS_ON, IMPACTS, REFERENCES, REVIEWED

DOCUMENT:
{text}

For each pair of entities that have a relationship in the text, output one relation.
Return ONLY compact JSON (no prose, no markdown):
{"relations": [
  {"source": "PR-482", "target": "bob", "type": "AUTHORED", "confidence": 0.95}
]}
```

#### 4. Integration into ingest.py
- Add `"hybrid"` branch to `ingest_text()` alongside `"llm"` and `"index_routing"`
- Reuse existing `write_graph()` (batch UNWIND already works)
- Same document parsing + Neo4j write path

#### 5. Benchmark against existing modes
- Run same `/tmp/prove_extraction.md` through hybrid mode
- Compare: edge count, precision vs gold, latency, VRAM
- Target: ≥ E2B llm precision (89%+) with < full-doc latency

### Expected Outcome
- **Precision:** 85-95% (E2B's relation reasoning + GLiNER's entity precision)
- **Latency:** ~6-8s (E2B prompt is smaller: no entity detection, just relations)
- **VRAM:** unchanged (same models, same ports)

### Risk: E2B "Thinking Process" Leakage
- Even with `--reasoning-format none`, E2B may emit prose before JSON
- Mitigation: use the existing `_salvage_objects()` parser which survives leading prose
- Fallback: if E2B refuses compact output in this prompt, keep full-doc llm mode

---

## WORKSTREAM 2: E2B LLAMA.CPP SERVER TUNING

### Current Flags
```
-c 4096 -t 12 -tb 12 --gpu-layers 999 --flash-attn on
--reasoning-format none --parallel 2 --kv-unified
--host 0.0.0.0 --port 8082 --no-warmup
```

### Issues Identified
1. **`--gpu-layers 999`** — brute force. Some layers might be better on CPU for VRAM/pipeline balance. Actual effective layers unknown.
2. **No `-b` / `-ub` tuning** — batch-size 2048 (default) is for training. For single-extraction workload, smaller batch = less VRAM = room for Jina cache.
3. **`--no-warmup`** — first extraction pays warmup cost. Remove for production.
4. **No `--cache-type-k` / `--cache-type-v`** — default q8_0 KV cache. For E2B's 4096 ctx, f16 KV cache uses ~256MB more VRAM but prevents quantization artifacts on long extractions.
5. **`-c 4096`** — the default. E2B supports 128K. 4096 may truncate long docs. Could raise to 8192 (cost: ~512MB more KV cache).
6. **`--parallel 2`** — two concurrent slots double KV cache VRAM. For serial ingest, `--parallel 1` saves ~256MB.

### Proposed Optimized Flags (test each independently)
```
# Option A: Conservative (keep similar VRAM, lower latency)
--gpu-layers 999 -c 8192 --parallel 1 --no-warmup (remove)
--flash-attn on --reasoning-format none --kv-unified
-t 12 -tb 12 -b 1024 -ub 256

# Option B: VRAM-efficient (free ~500MB for Jina cache)
--gpu-layers 32 -c 4096 --parallel 1
--cache-type-k f16 --cache-type-v f16
--flash-attn on --reasoning-format none
-t 12 -tb 12 -b 512 -ub 128

# Option C: Max throughput (for batch ingestion)
--gpu-layers 999 -c 16384 --parallel 1
--flash-attn on --reasoning-format none
-t 12 -tb 12 -b 2048 -ub 512
```

### Benchmark Matrix
For each option (A, B, C), measure:
1. VRAM after model load (`nvidia-smi`)
2. Single-token generation speed (`tg = X t/s` from logs)
3. Full extraction latency on `/tmp/prove_extraction.md`
4. Output correctness (same 89%+ precision)
5. Context window utilization (does `-c 8192` improve long-doc extraction?)

### Tuning Command Template
```bash
# Stop current E2B
fuser -k 8082/tcp

# Launch with experimental flags
LD_LIBRARY_PATH=... llama-server \
  -m /mnt/data-970-plus/models/gemma-4-E2B_q4_0-it.gguf \
  <tuned flags> \
  --host 0.0.0.0 --port 8082 2>/tmp/e2b-tune.log &

# Run benchmark
cd /mnt/data-970-plus/rag-system/v2
/mnt/data-970-plus/rag-env/bin/python -c "
import ingest, time
text=open('/tmp/prove_extraction.md').read()
t0=time.time()
res=ingest.ingest_text(text,'tune-test',domain='engineering',extract_graph=True)
print(f'time={time.time()-t0:.1f}s entities={res[\"entities\"]}')
"
```

### Expected Outcome
- Find flag combo that maximizes tokens/sec without OOM
- Target: <8s extraction (current 10.5s) with same quality

---

## WORKSTREAM 3: HIGHER QWEN INTELLIGENCE

### Option Roster (all compatible with index_routing if we ever switch back)

| Model | Size | VRAM | Download | Use Case |
|-------|------|------|----------|----------|
| **Qwen2.5-1.5B Q4_K_M** (current, on disk) | 1.1GB | 1.5GB | ✅ | Fast, 20% precision — reference baseline |
| **Qwen2.5-3B Q4_K_M** | 1.9GB | ~2.0GB | DL ~2GB | Medium intelligence, should reach 50-70% precision |
| **Qwen2.5-7B Q4_K_M** | 4.7GB | ~4.5GB | DL ~5GB (split) | High intelligence, 75%+ precision, tight on VRAM |
| **Phi-3-mini-4k Q4** (on disk) | 2.3GB | ~2.5GB | ✅ | Proven extractor, 60-75% precision |

### Test Priority

1. **Phi-3-mini-4k first** (already on disk, zero download, proven in Phase 1 extraction benchmark)
   - Set `EXTRACTION_MODE="llm"` (NOT index_routing — E2B is primary)
   - Point `EXTRACTION_LLM_MODEL="Phi-3-mini-4k-instruct-q4.gguf"`
   - Restart :8082 with Phi-3 GGUF
   - Run extraction + quality check
   - Expected: faster than E2B (~3s), lower precision (~70%), useful as fast-fallback

2. **Qwen2.5-3B Q4_K_M** (download ~2GB, fits VRAM with Jina+GLiNER)
   - Download from `bartowski/Qwen2.5-3B-Instruct-GGUF`
   - Test in both llm mode AND index_routing mode
   - Compare vs Phi-3 and E2B

3. **Qwen2.5-7B Q4_K_M** (only if 3B disappoints, tight VRAM)
   - Download split file (4.7GB total, two parts)
   - VRAM check: 4.5GB + 3.9GB (Jina) + 1.4GB (GLiNER) = 9.8GB. Fits 12GB but no headroom.
   - Must unload GLiNER from GPU daemon first (keep Jina only)
   - Highest intelligence, highest risk

### Benchmark Script (self-contained)
```python
#!/usr/bin/env python3
"""Run in /mnt/data-970-plus/rag-system/v2/ with rag-env venv."""
import ingest, time, json

text = open("/tmp/prove_extraction.md").read()
GOLD = {
    ("PR-482","FIXES","BUG-204"), ("PR-482","AUTHORED","bob"),
    ("alice","REVIEWED","PR-482"), ("checkout-publisher","DEPENDS_ON","checkout database"),
    ("ADR-014","REFERENCES","auth-service"),
}

models = ["gemma-4-E2B_q4_0-it.gguf", "Phi-3-mini-4k-instruct-q4.gguf",
          "qwen2.5-3b-instruct-q4_k_m.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf"]

for model in models:
    # start model, run ingest, count precision
    ...
```

---

## DOCUMENTATION DELIVERABLES

This plan covers updating these files:

1. **`NEXT_PHASE_PLAN.md`** — this file (workstreams 1-3)
2. **`ARCHITECTURE_V3.0.0.md`** — update with:
   - E2B llm mode as primary extraction (replacing index-routing diagram)
   - All benchmark tables (Qwen vs E2B, E2B tuning options, Qwen alternatives)
   - Systemd table (E2B on :8082, not Qwen)
   - VRAM budget (E2B-based, not Qwen-based)
3. **`BENCHMARKS.md`** (new) — comprehensive benchmark data:
   - E2B llm mode: 89% precision, 10.5s, 28 edges across 6 types
   - Qwen index_routing: 20% precision, 1.2s, 5 edges across 3 types
   - E2B llm server log analysis: 129 t/s gen, 396 t/s prompt eval
   - Hybrid retrieval: 127ms (vector+graph)
   - VRAM: 5.8GB (E2B+Jina+GLiNER) / 12GB
4. **`config.py`** — already updated (llm mode + E2B), no changes needed

---

## EXECUTION ORDER (for hy3 agent)

```
Phase A — Documentation first
  1. Read ARCHITECTURE_V3.0.0.md current state
  2. Update with final E2B llm architecture (diagram, systemd, VRAM)
  3. Create BENCHMARKS.md with all data from this plan

Phase B — E2B Tuning
  4. Stop E2B on :8082
  5. Test Option A flags → measure VRAM + latency + quality
  6. Test Option B flags → measure VRAM + latency + quality
  7. Pick best, update systemd unit (keep as -tuned variant)
  8. Restart E2B with winning flags

Phase C — GLiNER + E2B Hybrid
  9. Create hybrid_extraction.py with prompt from Workstream 1
  10. Add "hybrid" branch to ingest.py
  11. Benchmark vs llm mode (same doc)
  12. If hybrid beats llm on both quality AND speed → make it default
  13. If hybrid underperforms → document why, keep llm as primary

Phase D — Higher Qwen Intelligence
  14. Test Phi-3-mini in llm mode (already on disk)
  15. Download Qwen2.5-3B Q4_K_M (~2GB)
  16. Test Qwen2.5-3B in llm mode
  17. Run benchmark comparison E2B vs Phi-3 vs Qwen3B
  18. Report: which model gives best quality/speed tradeoff
```

---

## CONSTRAINTS

- **Never overwrite live config without backup.** Copy to `.bak` first.
- **GPU**: RTX 3060 12GB. 3.9GB Jina v3 + 1.4GB GLiNER permanently resident in rag-gpu-daemon. ~6.7GB free for extraction model.
- **Sudo may fail** — use direct process launch (fuser -k / background llama-server) when systemctl is unavailable.
- **No model downloads during benchmarking** — download first, then benchmark.
- **All benchmarks must use `/tmp/prove_extraction.md`** for consistency (648 chars, PR-482 document).
- **Neo4j credentials:** bolt://localhost:7687, neo4j / ragpassword123
- **Qdrant:** localhost:6333
- **Python venv:** `/mnt/data-970-plus/rag-env/bin/python`
- **llama.cpp bin:** `/home/reinhard/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1/llama-server`
- **LD_LIBRARY_PATH:** `/home/reinhard/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1`

---

## SUCCESS CRITERIA

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| GLiNER+E2B hybrid works | edges extracted ≥ 5, types > 2 | Must beat Qwen index_routing (5 edges, 3 types) |
| Hybrid precision | exact match vs gold | ≥ 80% |
| E2B tuning latency | extraction time | < 8s (from 10.5s baseline) |
| Higher Qwen precision | exact match vs gold | ≥ 60% for 3B, ≥ 75% for 7B |
| All benchmarks documented | BENCHMARKS.md | Complete with all numbers |
| Architecture doc matches reality | ARCHITECTURE_V3.0.0.md | E2B llm, not index-routing |
