# Latency Calibration Guide — synthesize /ask under 3s

This documents how the `polyglot-graphrag` `/ask` endpoint was calibrated to
return a synthesized answer in **< 3s** on the reference hardware (RTX 3060
12GB, i5-11400F, 48GB RAM, Ubuntu 22.04), and — more importantly — **how to
re-calibrate it** if you want lower latency, different hardware, or longer
answers.

> TL;DR: end-to-end `/ask` latency = **retrieval (~0.5s) + rerank (~0.1s) +
> synthesis (~1–2s)**. Synthesis is token-count-bound (≈ tokens / decode_speed).
> Retrieval was the hidden 4s trap (hub-node graph traversal). Both are now
> fixed and tunable.

---

## 1. The latency model (what actually costs time)

A synthesized `/ask` does, in order:

| Stage | What it is | Typical cost (RTX 3060) | Controlled by |
|-------|-----------|------------------------|---------------|
| Embed query | Jina embedder (GPU) | ~70 ms | `EMBED_DEVICE` |
| Retrieve | Qdrant vector + Neo4j graph (parallel) | ~0.4–0.6 s | `GRAPH_TRAVERSAL_LIMIT`, entity index |
| Rerank | BGE reranker (GPU) over candidates | ~100 ms | `RERANK_TOP_K`, `RERANK_DEVICE` |
| **Synthesize** | E2B GGUF generates the answer (GPU) | **~1.2–2.3 s** | `SYNTH_MAX_TOKENS_OUT`, E2B backend, ctx-size |
| Stream/flush | print answer once | <10 ms | (fixed) |

**Key insight:** synthesis wall time ≈ `SYNTH_MAX_TOKENS_OUT / decode_speed +
~0.5s`. The E2B GGUF decodes at **~103–128 tok/s on CUDA** on this card. So:

- 250 tokens → ~2.0 s  (current default, comfortably < 3 s)
- 400 tokens → ~3.9 s  (old default — forced padding + boilerplate)
- 200 tokens → ~1.6 s  (tighter, slightly shorter answers)

Retrieval used to be **~4 s** for most queries (see §3) — that dominated the
budget and is *independent* of synthesis. Fixing it was mandatory before the
3 s goal was reachable.

---

## 2. What was changed to hit < 3 s

### 2.1 Synthesis — CUDA backend + token cap
- **E2B must run on the CUDA llama.cpp build, not Vulkan.**
  `run.sh` auto-detects `llama-server` via `find` and can pick the *Vulkan*
  binary (≈45 tok/s) instead of the *CUDA* one (≈128 tok/s). Always pin
  `LLAMA_BIN` to the CUDA backend, or the synthesis SLO silently breaks:
  ```bash
  LLAMA_BIN=/home/reinhard/.lmstudio/extensions/backends/ \
    llama.cpp-linux-x86_64-nvidia-cuda-avx2-*/llama-server \
    bash run.sh serve
  ```
  Verify: `nvidia-smi` should show the `llama-server` using the **CUDA** binary
  path, and a direct 250-token generation should be ~2 s, not ~5 s.
- **`SYNTH_MAX_TOKENS_OUT = 250`** (was 400). The model stops at EOS well before
  the cap for most grounded queries; 400 just made it pad to the limit with
  generic text. 250 is the sweet spot for grounded, cited answers under 3 s.

### 2.2 Retrieval — kill the 4 s hub-node traversal
- **Server-side entity vector lookup** (`entity_vector_idx`) replaced the old
  fallback that pulled **all ~500 entities and embedded each in Python**
  (~3.7 s for any query with no exact entity match).
- **`GRAPH_TRAVERSAL_LIMIT = 200`** bounds the k-hop expansion so a
  high-degree "hub" entity's 2-hop neighbourhood can't explode to thousands of
  nodes (~3.7 s → ~0.5 s).
- **`GRAPH_ENTRY_MIN_SIM`** gate skips the traversal entirely when no entity is
  semantically relevant (degrades gracefully to Qdrant-only).

**Result (release gate, RTX 3060):**
```
Synthesis benchmark   p95 = 2.3 s   (was 6.8 s)   0 errors
Retrieval p95         = 0.53 s      (was ~4.5 s)
All 10 bench queries  = 0.36–0.57 s (was 4–4.8 s)
Release gate          = 14/14 ALL SYSTEMS GO
```

---

## 3. How to calibrate for *lower* latency

If you need sub-2 s, or you're on weaker hardware, here are the levers in
**order of impact**. Each lists *what to change*, *why it helps*, and *the
tradeoff*.

### Lever A — Lower `SYNTH_MAX_TOKENS_OUT` (biggest, safest)
- **Change:** `SYNTH_MAX_TOKENS_OUT=180` (or `200`) via env or `config.py`.
- **Why:** synthesis ≈ tokens / decode_speed. 180 tokens ≈ 1.4 s at 128 tok/s.
- **Tradeoff:** answers get shorter. Most grounded answers fit in ~170 tokens,
  but complex "explain X in detail" queries may be cut at the cap. **Do not go
  below ~150** or you risk truncating real answers. Always re-run the quality
  benchmark (`scripts/bench_synth_compare.py`) after changing.

### Lever B — Confirm/pin the CUDA E2B backend (correctness, not tuning)
- **Change:** set `LLAMA_BIN` to the CUDA llama.cpp build (see §2.1).
- **Why:** Vulkan on this card is ~3× slower (45 vs 128 tok/s). If you ever see
  synthesis ~5 s for 250 tokens, this is almost certainly why.
- **Tradeoff:** none — CUDA is strictly better here.

### Lever C — Reduce `E2B_CTX` (context window)
- **Change:** `E2B_CTX=8192` (was 32768) when launching the E2B server.
- **Why:** smaller KV cache reduces per-token decode overhead and VRAM; helps
  when the card is memory-constrained.
- **Tradeoff:** the prompt is ~700 tokens + 250 out ≈ 950, so 8192 is safe
  headroom. Don't drop below ~4096 or long retrieved contexts will be truncated
  by the model. Minor latency win (~10–20%) — not the main lever.

### Lever D — Trim retrieval work
- **`GRAPH_HOPS = 1`** (was 2): halves the graph traversal cost. Tradeoff:
  loses 2-hop relationships (weaker graph context for dense KGs).
- **`GRAPH_TRAVERSAL_LIMIT` lower (e.g. 100)**: bounds hub-node expansion
  further. Tradeoff: may drop relevant distant neighbors.
- **`RERANK_TOP_K = 3`** (was 5): reranks fewer candidates. Tradeoff: slightly
  less precise context selection.
- **`MAX_SYNTH_CONTEXTS = 3`** (was 4): feeds fewer chunks to synthesis.
  Tradeoff: less context → shorter, possibly less complete answers.
- Retrieval is already ~0.5 s, so these are only worth it if you're chasing
  sub-2 s end-to-end and retrieval is the bottleneck in your profile.

### Lever E — Smaller / faster synthesis model (biggest lever, biggest tradeoff)
- **Change:** swap `E2B_MODEL` for a smaller or more efficient GGUF (e.g. a
  QAT/Q4_0 1B–2B model, or a model with a faster decode architecture).
- **Why:** decode_speed is the hard ceiling on synthesis latency. A model that
  decodes at 200 tok/s instead of 128 cuts 250-token synthesis from 2 s to 1.25 s.
- **Tradeoff:** answer quality usually drops. This is the right move only if you
  have headroom in the quality benchmark.

### Lever F — Hardware
- A faster GPU (more VRAM bandwidth) raises decode_speed directly. On a 3060
  (12GB) E2B sits at ~128 tok/s; a 4070/4090-class card pushes higher. The
  config knobs above are hardware-agnostic — the gate's `_SYNTH_THRESHOLD_S`
  and `_BENCH_THRESHOLD_MS` (in `scripts/release-gate.py`) are the only
  hardware-specific values and should be re-measured per machine.

---

## 4. Calibration workflow (reproducible)

1. **Start with the CUDA E2B backend** and a small `E2B_CTX`:
   ```bash
   LLAMA_BIN=<cuda-llama-server> E2B_CTX=8192 SYNTH_MAX_TOKENS_OUT=250 \
     bash run.sh serve
   ```
2. **Measure synthesis alone** (decode speed sanity check):
   ```bash
   ./venv/bin/python - <<'PY'
   import urllib.request, json, time
   url="http://localhost:8082/v1/chat/completions"
   p={"model":"gemma-4-E2B-it-QAT-Q4_0.gguf","messages":[{"role":"user","content":"how does the hybrid retrieval pipeline fuse Qdrant and Neo4j"}],"max_tokens":250,"temperature":0.0,"stream":True}
   t0=time.time(); n=0
   for raw in urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(p).encode(),headers={'Content-Type':'application/json'}),timeout=60):
       line=raw.decode().strip()
       if line.startswith('data:') and '[DONE]' not in line:
           try:
               if json.loads(line[5:].strip())['choices'][0]['delta'].get('content'): n+=1
           except: pass
   print(f"{n} tokens in {(time.time()-t0)*1000:.0f}ms => {n/((time.time()-t0)):.0f} tok/s")
   PY
   ```
   Expect **≥100 tok/s** (CUDA). If ~45 tok/s, you're on Vulkan — fix
   `LLAMA_BIN`.
3. **Run the end-to-end benchmark:**
   ```bash
   ./venv/bin/python scripts/bench_synth_compare.py
   ```
   Reads `synth_p95` and `retrieval_p95`. Target: `synth_p95 < 3.0s`, `errors=0`.
4. **Run the release gate:**
   ```bash
   ./venv/bin/python scripts/release-gate.py   # expect 14/14 ALL SYSTEMS GO
   ```
5. **If you changed a threshold**, update `scripts/release-gate.py`
   (`_SYNTH_THRESHOLD_S`, `_BENCH_THRESHOLD_MS`) to match the *measured*
   envelope — never set it below what you can reproduce, and never above what
   users will tolerate. Comment the measured numbers.

---

## 5. Gotchas

- **Vulkan vs CUDA is silent.** `run.sh` picks whichever `llama-server` `find`
  returns first. If synthesis is mysteriously ~3× slow, check the binary path
  in `nvidia-smi` / `ps`.
- **HuggingFace can be unreachable.** The Jina embedder loads from the local HF
  cache; if HF is down, the daemon hangs retrying HEAD requests (~4 min before
  failing). Start with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when offline.
- **Don't lower `SYNTH_MAX_TOKENS_OUT` blindly.** Below ~150 tokens you start
  truncating real answers; verify with the quality benchmark, not just latency.
- **The gate measures full `/ask`**, not synthesis alone. A fast synthesis with
  a slow retrieval still fails the SLO — fix retrieval first (§2.2).
- **Re-measure per machine.** The 3 s / 1100 ms thresholds in
  `scripts/release-gate.py` are calibrated for RTX 3060 12GB. On different
  hardware, re-run §4 steps 3–4 and adjust.
