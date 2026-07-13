# Benchmark: Lighter E2B Model (gemma-4-E2B-it-qat-mobile)

**Date:** 2026-07-13
**Branch:** `bench/lighter-e2b-model`
**Goal:** Evaluate `google/gemma-4-E2B-it-qat-mobile-ct` (QAT mobile variant) as a
drop-in replacement for the current extraction LLM `gemma-4-E2B_q4_0-it.gguf`.

**Verdict:** ❌ NOT a drop-in replacement. See findings below.

---

## Model variants available

| Model | Format | File | Size | Notes |
|-------|--------|------|------|-------|
| Current (prod) | GGUF Q4_0 | `gemma-4-E2B_q4_0-it.gguf` | 3.35 GB | Used by `:8082` llama-server |
| `google/gemma-4-E2B-it-qat-mobile-ct` | **safetensors** | (HF) | — | Multimodal any-to-any; not llama.cpp-compatible |
| `unsloth/gemma-4-E2B-it-qat-mobile-GGUF` | GGUF `UD-Q2_K_XL` | `gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf` | 2.03 GB | Only GGUF build; 2-bit unsloth quant |

The `google/...-ct` model is **safetensors** (transformers), so it cannot run on
the existing llama.cpp `/v1` OpenAI-compatible server. The community GGUF build
(`unsloth/...-mobile-GGUF`) only ships a `UD-Q2_K_XL` (2-bit) variant — no Q4_0.
That 2-bit GGUF was used for this benchmark.

---

## Benchmark results (RTX 3060 12GB)

Test: 5 runs of relation-classification extraction on an engineering sample doc.
Both servers run with identical flags (`--n-gpu-layers 32 --ctx-size 8192`).

| Metric | Current Q4_0 (`:8082`) | Mobile Q2_K_XL (`:8083`) |
|--------|------------------------|---------------------------|
| **File size** | 3.35 GB | 2.03 GB (−39%) |
| **VRAM (GPU alloc)** | 2.25 GB | 3.95 GB (+75% ⚠️) |
| **Latency / extract** | 14.2 s | 21.1 s (+49% ⚠️) |
| **Nodes extracted** | 5 (via daemon) | 4 (schema mismatch) |
| **Edges extracted** | 0 | 3 |
| **JSON schema** | `{name, type}` | `{id, label, type}` ⚠️ |
| **Thinking mode** | emits `<|channel>thought` | clean JSON |

### Key findings

1. **VRAM is WORSE, not better.** The 2-bit mobile model uses **3.95 GB** vs the
   current **2.25 GB** — 75% more VRAM. Counterintuitive, but the 2-bit quant
   needs more KV-cache / layer materialization on GPU than the 4-bit model at the
   same `--n-gpu-layers`.

2. **Latency is SLOWER.** 21.1 s vs 14.2 s per extraction (+49%). The 2-bit model
   does more compute per token to compensate for quantization loss.

3. **JSON schema incompatibility (blocking).** The mobile QAT model emits
   `{"id": "...", "label": "...", "type": "..."}` while `hybrid_extraction.py` /
   `ingest.py` parse `{"name": "...", "type": "..."}`. The production parser reads
   `node.get("name")` → gets `None` → **0 usable entities**. A swap would require
   prompt + parser changes to normalize the schema.

4. **Thinking-mode divergence.** Current E2B emits `<|channel>thought>` blocks
   (the daemon's `extract_graph_llm` strips these correctly). The mobile model
   returns clean JSON. Both are parseable by a robust extractor, but the current
   production path already handles the thinking variant.

---

## Recommendation

**Do NOT replace the current E2B with the mobile QAT model.** It is:
- Slower (1.5× latency)
- Larger in VRAM (1.75×)
- Schema-incompatible (breaks the entity parser without code changes)
- Only available as a 2-bit GGUF (no Q4_0 mobile variant exists)

If a lighter model is still desired, the better path is:
- Wait for a `Q4_0` or `Q4_K_M` mobile GGUF from unsloth, OR
- Use `gemma-4-E2B-it-qat-q4_0-gguf` (google's own QAT Q4_0) if released, OR
- Reduce `--n-gpu-layers` on the current model to free VRAM instead.

Production (`gemma-4-E2B_q4_0-it.gguf` on `:8082`) is **untouched** by this benchmark.

---

## Repro

```bash
# Start mobile model on a SEPARATE port (never touch prod :8082)
LLAMA=~/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1/llama-server
$LLAMA --model models/gemma-4-E2B-it-qat-mobile-UD-Q2_K_XL.gguf \
  --host 127.0.0.1 --port 8083 --n-gpu-layers 32 --ctx-size 8192 \
  --parallel 1 --reasoning-format none --cache-type-k f16 --cache-type-v f16 --no-mmap &

# Benchmark
python benchmark_models.py
```
