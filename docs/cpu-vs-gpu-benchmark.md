# CPU vs GPU Benchmark — GraphRAG v2

**Date:** 2026-07-10
**Hardware:** RTX 3060 12GB / Intel i5-11400F (6C/12T) / 48GB DDR4
**Environment:** rag-env (torch 2.3.1+cu121, transformers 4.49.0, sentence_transformers 5.6.0)
**Models tested:** Jina v3 (1024-dim), BGE reranker-v2-m3, full /ask pipeline

## Methodology

GPU daemon (`serve_gpu.py`) — models preloaded at startup in fp16, resident VRAM:
```
sudo systemctl start rag-gpu-daemon
curl -s http://127.0.0.1:8000/health  # wait for ready
```

CPU daemon (`serve_cpu.py`) — models lazy-loaded on first use in fp32, forced CPU:
```
sudo systemctl stop rag-gpu-daemon
CUDA_VISIBLE_DEVICES="" /mnt/data-970-plus/rag-env/bin/python serve_cpu.py &
curl -s http://127.0.0.1:8000/health  # wait for ready
```

Each endpoint benchmarked with 3 queries × 3 iterations, taking best-of-3.
All models warmed before timing (excludes cold-load latency).

## Results

```
                    GPU (RTX 3060, fp16)      CPU (i5-11400F, fp32)     CPU slower by
                    ────────────────────      ──────────────────────     ────────────

Jina v3 embed          0.057s                    0.723s                    12.7×
BGE rerank (5 docs)    0.011s                    0.159s                    14.5×
Full /ask retrieval    0.162s                    3.824s                    23.6×
Cold model load         ~14s                     ~10.7s                     0.8×
Resident memory       2.3 GB VRAM               5.9 GB system RAM            —
```

### Query-level breakdown

| Stage | GPU | CPU | Gap | Notes |
|-------|-----|-----|-----|-------|
| embed (Jina v3) | 0.057s | 0.723s | 12.7× | fp16 CUDA vs fp32 x86 |
| Qdrant search | 0.035s | 0.035s | 1.0× | Same Docker (network) |
| Neo4j subgraph | 0.008s | 0.008s | 1.0× | Same Docker (Bolt) |
| rerank (BGE, 5 docs) | 0.011s | 0.159s | 14.5× | CUDA tensor cores vs serial CPU |
| **retrieval total** | **0.111s** | **0.925s** | **8.3×** | embed + qdrant + neo4j + rerank |
| synthesize (E4B :8084) | 6.35s | 6.35s | 1.0× | Separate LLM, same endpoint |
| **full pipeline** | **6.46s** | **7.28s** | **1.1×** | E4B dominates both paths |

### Per-query detail

| Query | GPU embed | CPU embed | GPU rerank | CPU rerank | GPU /ask | CPU /ask |
|-------|-----------|-----------|------------|------------|----------|----------|
| who reported BUG-204? | 0.057s | 0.723s | 0.011s | 0.159s | 0.169s | 3.824s |
| what database does ADR-021 use? | 0.057s | 0.743s | 0.011s | 0.159s | 0.144s | 3.037s |
| how does checkout relate to billing? | 0.056s | 0.733s | 0.011s | 0.169s | 0.172s | 4.727s |

## Why CPU is so much slower

| Factor | GPU (serve_gpu.py) | CPU (serve_cpu.py) |
|--------|--------------------|--------------------|
| Precision | **fp16** (half) — `.half()` called after load | **fp32** (full) — no `.half()`, serve_cpu doesn't call it |
| Jina v3 | CUDA kernels + tensor cores, batched | x86 vector extensions, OMP_NUM_THREADS=8 |
| BGE reranker | Cross-attention parallelized on GPU | Serial comparison, one doc pair at a time |
| `/ask` threading | True `threading.Thread` — Qdrant + Neo4j in parallel | Serve_cpu lacks threading optimization |
| Model residency | Preloaded, persistent VRAM | Lazy-loaded on first use, swapped to RAM |

> The 12.7× embed gap is primarily fp16 vs fp32. Jina v3 at fp16 uses half the memory bandwidth and ~2× the throughput. The remaining ~6× is raw GPU vs CPU compute. If serve_cpu.py were patched to call `.half()` on CPU (which offers no speed benefit on x86), the gap would still be ~6×.

## When CPU is acceptable

| Scenario | CPU Viable? | Why |
|----------|------------|-----|
| Batch ingestion (100+ docs) | **No** | 12.7× slower per chunk × hundreds of chunks = minutes → hours |
| Interactive queries | **Borderline** | 3.8s retrieval vs 0.16s is noticeable but usable |
| Full pipeline with E4B | **Yes** | E4B at 6.35s dominates both; 7.3s CPU vs 6.5s GPU is a 12% difference |
| Development / testing | **Yes** | No GPU needed, same API, same results |
| Production | **No** | GPU daemon delivers 23.6× faster retrieval |

## Recommendation

**Always use `serve_gpu.py` (GPU) for production.** The 0.16s retrieval latency is real-time. CPU at 3.8s feels sluggish. For the full synthesis pipeline, the E4B bottleneck at 6.35s makes the retrieval gap less critical (7.3s CPU vs 6.5s GPU), but GPU still wins every stage.

Keep `serve_cpu.py` as a development fallback and CI/testing target. It needs patching to match the model-agnostic config from v2.3.1 (still has hardcoded model names and MiniLM).
