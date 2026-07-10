# GraphRAG v2 — Full Benchmark History

**Date:** 2026-07-10
**Hardware:** RTX 3060 12GB / Intel i5-11400F (6C/12T) / 48GB DDR4 / Ubuntu 22.04
**Git:** `adc1e50` → `b7c51db` (v2.3.1 → v2.4.1)

---

## 1. GPU vs CPU — Raw Model Performance (standalone, no daemon)

| Stage | GPU (fp16) | CPU (fp32) | Gap |
|-------|-----------|-----------|-----|
| Jina v3 embed | 0.056s | 0.776s | 13.8× |
| BGE rerank (5 docs) | 0.012s | 0.155s | 12.9× |
| BGE rerank (27 docs) | 0.061s | 3.205s | 52.5× |

---

## 2. Full /ask Retrieval (synthesize=false, skip_cache=true)

### 2.1 GPU Daemon (serve_gpu.py, BGE v2-m3, full pool)

| Query | Time |
|-------|------|
| who reported BUG-204? | 0.168s |
| what database does ADR-021 use? | 0.141s |
| how does checkout relate to billing? | 0.166s |
| what services impacted by payment? | 0.165s |
| list all developers who authored ADRs | 3.314s |

**Average (4 targeted): 0.160s**

Stage breakdown:
```
embed (Jina v3):  0.056s
Qdrant search:     0.039s
Neo4j subgraph:    0.006s
rerank (BGE, 27d): 0.061s
─────────────────────────
TOTAL:             0.162s
```

### 2.2 CPU Daemon — Evolution

| Version | Reranker | Pool | Time | vs GPU |
|---------|----------|------|------|--------|
| v2.4.0-raw | BGE v2-m3 | 27 | 3.794s | 23.7× |
| v2.4.0-fp16-bug | BGE v2-m3 fp16 | 27 | ~14s | 87× |
| v2.4.0-fp32 | BGE v2-m3 fp32 | 27 | 3.794s | 23.7× |
| v2.4.1-cap10 | BGE v2-m3 | 10 | 1.641s | 10.3× |
| v2.4.1-cap15 | BGE v2-m3 | 15 | 2.197s | 13.7× |

Per-query (v2.4.1-cap15):
| Query | Time |
|-------|------|
| BUG-204 | 2.20s |
| ADR-021 | 1.94s |
| checkout | 2.22s |

---

## 3. Pool Cap vs Quality Tradeoff

Tested across 4 queries, comparing top-5 overlap with full 27-doc reference (BGE v2-m3):

| Cap | Avg Overlap | 5/5 Queries | Est. /ask |
|-----|------------|-------------|-----------|
| 5 | 2.3/5 (46%) | 0 of 4 | 1.17s |
| 8 | 3.0/5 (60%) | 0 of 4 | 1.45s |
| 10 | 3.0/5 (60%) | 0 of 4 | 1.64s |
| 12 | 3.5/5 (70%) | 1 of 4 | 1.94s |
| **15** | **4.5/5 (90%)** | **2 of 4** | **2.20s** |
| 20 | 4.8/5 (96%) | 3 of 4 | 2.69s |
| 27 | 5.0/5 (100%) | 4 of 4 | 3.79s |

**cap=15 chosen: best quality per millisecond.**

---

## 4. Reranker Model Alternatives (CPU only, 27-doc pool)

| Model | Size | Time | Multilingual | Top-3 Overlap vs BGE v2-m3 |
|-------|------|------|-------------|---------------------------|
| BGE v2-m3 (ref) | 568M | 3.205s | ✓ | 3/3 |
| BGE-reranker-base | 278M | 1.190s | ✓ | 2/3 |
| mMARCO-L12 | 118M | 0.341s | ✓ | 3/3 (EN), 2/3 (ID) |
| MiniLM-L-2 | 17M | 0.075s | ✗ | 3/3 (EN only) |

**Verdict:** mMARCO-L12 is the fastest multilingual CPU cross-encoder, but BGE v2-m3 + pool cap produces better quality with the same model family as GPU.

---

## 5. Lighter Model + More Docs vs Heavier Model + Fewer Docs

| Config | Rerank | /ask Total | Overlap | Notes |
|--------|--------|-----------|---------|-------|
| BGE v2-m3, 15d | 1376ms | **2.20s** | **4.5/5** | ✓ chosen |
| BGE base, 27d | 994ms | 1.82s | 4.0/5 | faster, lower quality |
| BGE v2-m3, 10d | 932ms | 1.75s | 3.0/5 | fastest, lowest quality |

**Heavier model + fewer docs beats lighter model + more docs on quality.**

---

## 6. Modularity — Model Swap Impact

| Component | Config Field | Re-ingest? | Daemon Restart? |
|-----------|-------------|-----------|----------------|
| Embedding model | EMBED_MODEL_NAME + VECTOR_DIM + flags | Yes | Yes |
| Reranker (GPU) | RERANK_MODEL_NAME | No | Yes |
| Reranker (CPU) | RERANK_MODEL_NAME_CPU + RERANK_CPU_POOL_CAP | No | Yes |
| GLiNER | GLINER_MODEL_NAME | No | Yes |
| Extraction LLM | EXTRACTION_LLM_* | No | That service |
| Synthesis LLM | SYNTHESIS_LLM_* | No | That service |

---

## 7. Jina v4 & EmbeddingGemma Compatibility

| Model | Dim | Status | Reason |
|-------|-----|--------|--------|
| Jina v3 | 1024 | **Production** | — |
| Jina v4 | 2048 | Blocked | torch 2.3.1 → need ≥2.6, transformers 4.49→ need ≥4.52, need peft |
| EmbeddingGemma 300M | 768 | Blocked | Gated (HF login + Google license required) |
| BGE-M3 | 1024 | Untested | Should work, same config pattern |

---

## 8. Full Pipeline (synthesize=true, E4B :8084)

| Query | GPU | CPU |
|-------|-----|-----|
| BUG-204 | 3.69s | 5.84s |
| ADR-021 | 5.58s | 3.48s* |
| checkout/billing | 6.84s | 11.42s |

*E4B synthesis time variance dominates; retrieval gap invisible.

---

## 9. Resource Utilization

| Component | GPU VRAM | System RAM |
|-----------|---------|-----------|
| serve_gpu (Jina + BGE) | 3.81 GB | ~2 GB |
| gemma-4-e2b (:8082) | 2.27 GB | — |
| gemma-4-e4b (:8084) | 4.17 GB | — |
| Desktop (Xorg + gnome) | 0.13 GB | — |
| **GPU total** | **10.38 GB** | — |
| serve_cpu (Jina + BGE) | 0 (CPU) | 5.2 GB |
| GPU headroom | 1.62 GB | — |

---

## 10. Key Architectural Decisions

1. **Same reranker on both paths** — BGE v2-m3 (568M) for GPU and CPU. Pool cap delivers CPU speed without model downgrade.

2. **fp16 poison fixed** — serve_cpu.py must never call `.half()`. fp16 on x86 destroys performance (14s vs 3.8s). Code now has explicit guard.

3. **No MiniLM router** — removed in v2.3.1. 50% accuracy was worse than parallel Qdrant+Neo4j search.

4. **CANON_MAP removed** — replaced with Neo4j vector entity resolution (v2.3.0). Jina v3 cross-lingual embeddings auto-merge multilingual entities.

5. **Extraction reads content not reasoning_content** — E2B is a non-reasoning model. Config flag `EXTRACTION_READS_REASONING` for future model swaps.
