# How Every Model Is Run

**Last updated:** 2026-07-11

This document is the single source of truth for how every model in the
GraphRAG stack is launched — the exact binary, every flag, every model
file, and why each parameter is set the way it is.

---

## Table of Contents

- [Model Files](#model-files)
- [Gemma 4 E2B (Extraction LLM) — :8082](#gemma-4-e2b-extraction-llm--8082)
- [Gemma 4 E4B (Synthesis LLM) — :8084](#gemma-4-e4b-synthesis-llm--8084)
- [GPU Auxiliary Daemon — :8000](#gpu-auxiliary-daemon--8000)
- [CPU Fallback Daemon](#cpu-fallback-daemon)
- [Neo4j + Qdrant (Docker)](#neo4j--qdrant-docker)
- [Parameter Reference](#parameter-reference)
- [Full Cold-Start Sequence](#full-cold-start-sequence)
- [Troubleshooting](#troubleshooting)

---

## Model Files

All GGUF files live under `/home/reinhard/.lmstudio/models/lmstudio-community/`:

| Model | File | Size | Quant | Architecture | Purpose |
|-------|------|------|-------|-------------|---------|
| Gemma 4 E2B | `gemma-4-E2B-it-QAT-GGUF/gemma-4-E2B-it-QAT-Q4_0.gguf` | 3.2 GB | Q4_0 | Gemma 4 (9B-class) | Entity/edge extraction from documents |
| Gemma 4 E4B | `gemma-4-E4B-it-QAT-GGUF/gemma-4-E4B-it-QAT-Q4_0.gguf` | 4.9 GB | Q4_0 | Gemma 4 (27B-class?) | Answer synthesis from query + context |

The `QAT-Q4_0` suffix indicates a **Quantization-Aware Trained** Q4_0 model —
the model was fine-tuned to expect Q4_0 quantization, so accuracy loss from
quantization is much lower than post-training quantization of the same bitwidth.

---

## Gemma 4 E2B (Extraction LLM) — :8082

### Server command

```bash
/home/reinhard/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1/llama-server \
  -m /home/reinhard/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-QAT-GGUF/gemma-4-E2B-it-QAT-Q4_0.gguf \
  -c 131072 \
  -t 12 \
  -tb 12 \
  --gpu-layers 999 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn auto \
  --parallel 8 \
  --kv-unified \
  --host 0.0.0.0 --port 8082
```

### Parameter breakdown

| Flag | Value | Why |
|------|-------|-----|
| `-m` | `gemma-4-E2B-it-QAT-Q4_0.gguf` | The 3.2 GB Q4_0 quantized model file |
| `-c` | `131072` | 128K context window. Gemma 4 natively supports 128K tokens. Extraction needs it because documents can be long — a full page of engineering text + extraction schema + instruction fits in ~8K tokens, but code files or medical records can be much longer. 128K is the native cap. |
| `-t` | `12` | 12 CPU threads for prompt processing (tokenization, de-tokenization). Matches the i5-11400F's 12 threads. |
| `-tb` | `12` | 12 CPU threads for batch processing. Same reasoning. |
| `--gpu-layers` | `999` | Offload ALL layers to GPU. The model has ~32 layers; `999` means "offload as many as possible" (effectively all). Keeps inference on CUDA, no fallback to CPU. |
| `--cache-type-k` | `q4_0` | Key cache in Q4_0 (4-bit). 128K context with full-precision key cache would be ~8 GB extra. Q4_0 drops this to ~500 MB. The value is acceptable because keys are primarily used for attention scoring, not as output. |
| `--cache-type-v` | `q4_0` | Value cache in Q4_0. Same reasoning — value precision matters more than key precision for output quality, but at 128K context the alternative is running out of VRAM entirely. Q4_0 is the pragmatic choice. |
| `--flash-attn` | `auto` | Flash Attention v2 (or v3 if supported). Reduces attention memory from O(n²) to O(n). For 128K context, this is the difference between fitting and OOM: naive attention = 128K² × 2 bytes ≈ 32 GB; flash attention = ~linear. |
| `--parallel` | `8` | Up to 8 concurrent requests can share the same KV cache (batch). Extraction runs in parallel during ingest — up to 8 documents can be extracted simultaneously if the daemon threads them. |
| `--kv-unified` | (flag) | Single KV cache pool shared across all slots. Without this, each slot allocates its own cache and 8 slots × 128K cache = OOM. Unified means slots borrow from a shared pool. Slot A using 8K of context gives the remaining 120K to slot B. |
| `--port` | `8082` | HTTP server port. Used by `serve_gpu.py` → `/extract_graph` |
| `--host` | `0.0.0.0` | Listen on all interfaces (not just localhost), so other machines on LAN can use it if needed. |

### VRAM breakdown (E2B)

| Component | Memory |
|-----------|--------|
| Model weights (Q4_0, ~9B params × 0.5 bytes) | ~1.7 GB |
| KV cache (128K, Q4_0, 1 slot startup) | ~0.5 GB |
| CUDA context + scratch | ~0.5 GB |
| **Total** | **~2.6 GB** |

### API format (OpenAI-compatible)

```
POST http://localhost:8082/v1/chat/completions
{"model": "gemma-4-E2B-it-QAT-Q4_0.gguf",
 "messages":[{"role":"user","content":"Extract entities from: ..."}],
 "max_tokens": 1024, "temperature": 0.0}
```

### Reasoning content

Gemma 4 E2B is a "thinking" model. It populates `choices[0].delta.reasoning_content`
with chain-of-thought before writing `choices[0].delta.content` with the final
JSON. The extraction pipeline in `extract_graph_llm()` collects both — reasoning
is logged for debugging, content is parsed as JSON.

---

## Gemma 4 E4B (Synthesis LLM) — :8084

### Server command

```bash
/home/reinhard/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1/llama-server \
  -m /home/reinhard/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-QAT-GGUF/gemma-4-E4B-it-QAT-Q4_0.gguf \
  -c 131072 \
  -t 12 \
  -tb 12 \
  --gpu-layers 999 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --parallel 8 \
  --kv-unified \
  --host 0.0.0.0 --port 8084 \
  --mlock --no-warmup
```

### Parameter differences from E2B

| Flag | E2B (:8082) | E4B (:8084) | Why the difference |
|------|-------------|-------------|-------------------|
| `-m` | E2B GGUF | E4B GGUF | Different model (4.9 GB vs 3.2 GB) |
| `--flash-attn` | `auto` | *(omitted)* | E4B's larger model may have flash-attn compatibility issues. Server defaults to safe fallback. |
| `--mlock` | *(omitted)* | ✓ | Lock E4B's memory into RAM so it never gets swapped. E4B is the synthesis model — every user query hits it. Swapping would cause multi-second pauses. |
| `--no-warmup` | *(omitted)* | ✓ | Skip the startup warmup run (which would allocate extra VRAM tokens for a test prompt). E4B already loads in ~90s; warmup adds another ~30s for no benefit since the daemon does its own warmup. |

Everything else is identical: 128K context, Q4_0 KV cache, 12 threads,
all GPU layers, unified KV pool, 8 parallel slots.

### VRAM breakdown (E4B)

| Component | Memory |
|-----------|--------|
| Model weights (Q4_0, ~27B params × 0.5 bytes) | ~3.5 GB |
| KV cache (128K, Q4_0, 1 slot startup) | ~0.8 GB |
| CUDA context + scratch | ~0.7 GB |
| **Total** | **~4.7 GB** |

### API format (OpenAI-compatible, streamed)

```bash
curl -s -N -X POST http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-E4B-it-QAT-Q4_0.gguf",
       "messages":[{"role":"user","content":"...query with context..."}],
       "max_tokens":1024,"temperature":0.1,"stream":true}'
```

Uses **Server-Sent Events (SSE) streaming** — `choices[0].delta.reasoning_content`
is the chain-of-thought, `choices[0].delta.content` is the clean answer. The
daemon collects both: reasoning goes to stdout (debug), content goes to the
response `answer` field.

### Why temperature 0.1 vs 0.0

E4B is a Gemma 4 thinking model. At temperature 0.0, the thinking process can
get stuck in a repetitive loop. 0.1 introduces just enough entropy to break
loops without degrading answer quality. Validated empirically.

---

## GPU Auxiliary Daemon — :8000

### Server command (systemd)

```
ExecStart=/mnt/data-970-plus/rag-env/bin/python serve_gpu.py
```

No CLI flags — all configuration is in `v2/config.py`:

### Models loaded inside the daemon

| Model | Config var | What it does | When it loads |
|-------|-----------|-------------|---------------|
| **Jina v3** | `EMBED_MODEL_NAME=jinaai/jina-embeddings-v3` | Cross-lingual embedding (1024d). Query + passage encoder. Uses two task types: `retrieval.query` and `retrieval.passage`. Half-precision (fp16) on GPU. | On startup (~15s) |
| **BGE v2-m3** | `RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3` | Cross-encoder reranker. Scores query-document pairs. Full-precision (fp32) on GPU. | On startup (~10s) |
| **GLiNER** | `GLINER_MODEL_NAME=urchade/gliner_multi-v2.1` | Zero-shot NER fallback when LLM extraction fails or E2B is down. Multi-lingual (11 languages). | Lazy — only loads when first extraction request has `force_gliner=true` or E2B fails (~5s) |

### VRAM breakdown (daemon)

| Component | Memory |
|-----------|--------|
| Jina v3 (fp16, 560M params) | ~1.1 GB |
| BGE v2-m3 (fp32, 568M params) | ~2.1 GB |
| CUDA context + scratch | ~0.3 GB |
| Python runtime + FastAPI | ~0.2 GB |
| GLiNER (lazy, not included) | (+~0.5 GB when loaded) |
| **Total** | **~3.9 GB** (+0.5 GB if GLiNER loaded) |

### Configuration (from v2/config.py)

```python
EMBED_MODEL_NAME   = "jinaai/jina-embeddings-v3"
EMBED_TASK_QUERY   = "retrieval.query"
EMBED_TASK_PASSAGE = "retrieval.passage"
EMBED_USE_HALF     = True           # fp16 for Jina (GPU only)

RERANK_MODEL_NAME  = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K       = 5              # top contexts after rerank
QDRANT_SEARCH_TOP_K = 10            # candidates from Qdrant

GLINER_MODEL_NAME  = "urchade/gliner_multi-v2.1"
```

---

## CPU Fallback Daemon

Same code, identical API, runs on CPU:

```bash
# Start manually (no systemd service):
cd /mnt/data-970-plus/rag-system/v2
/mnt/data-970-plus/rag-env/bin/python serve_cpu.py
```

Key difference: `EMBED_USE_HALF=False` enforced at line 1 of `serve_cpu.py`
(x86 has no native fp16 — every fp16→fp32 upcast kills performance).

```python
# serve_cpu.py (line ~20)
EMBED_USE_HALF = False  # x86 has no native fp16; half() upcasts and destroys perf
```

---

## Neo4j + Qdrant (Docker)

Neo4j and Qdrant run in Docker. The `docker-compose.yml` at the repo root:

```yaml
services:
  neo4j:
    image: neo4j:5-community
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/<REDACTED>
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_memory_heap_max__size: 2G
    volumes: ["./neo4j_data:/data"]

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333", "6334:6334"]
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
    volumes: ["./qdrant_storage:/qdrant/storage"]
```

### Why each parameter

- **Neo4j heap 2G** — Neo4j's default is too high for a 48 GB workstation that's
  also running three GPU servers. 2 GB is sufficient for our graph size (<10K
  entities).
- **APOC plugin** — needed for `apoc.periodic.commit` and vector index creation.
- **Qdrant gRPC port** — `6334` gRPC is faster than REST for bulk operations
  (ingest). `6333` REST is used for simple queries.

---

## Full Cold-Start Sequence

```
Step 1: docker compose up -d
        Neo4j :7687 + Qdrant :6333
        ~10s

Step 2: (one-time) cd v2 && bash run.sh ingest sample_data
        Reads sample_data/ → extracts entities → writes to Qdrant + Neo4j
        ~2-5 min (depends on number of docs; E2B extraction is GPU-bound)

Step 3: sudo systemctl start gemma-4-e2b.service
        llama-server (E2B) :8082
        ~45s load  →  2.6 GB VRAM
        Verify: curl http://localhost:8082/v1/models

Step 4: sudo systemctl start gemma-4-e4b.service
        llama-server (E4B) :8084
        ~90s load  →  4.7 GB VRAM
        Verify: curl http://localhost:8084/v1/models

Step 5: sudo systemctl start rag-gpu-daemon.service
        python serve_gpu.py :8000
        ~30s load  →  3.9 GB VRAM
        Verify: curl http://127.0.0.1:8000/health

Total VRAM: 2.6 + 4.7 + 3.9 = 11.2 GB (out of 12 GB)
Total time: ~2.5 minutes from cold start
```

### VRAM allocation timeline

```
                    E2B loads            E4B loads            Daemon loads
                 ┌──────────┐        ┌──────────┐        ┌─────────────────┐
11 GB ┤                                             ▄▄▄▄▄█ daemon active
10 GB ┤                                   ▄▄▄▄▄▄▄▄█▄▄▄▄▄▄█ daemon + E4B active
 9 GB ┤                         ▄▄▄▄▄▄▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█ E4B loading + daemon
 8 GB ┤               ▄▄▄▄▄▄▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ E4B loading
 7 GB ┤     ▄▄▄▄▄▄▄▄█ E2B loaded
 6 GB ┤  ▄▄▄▄█ E2B loading
 5 GB ┤
 4 GB ┤
 3 GB ┤
 2 GB ┤  ~1.2 GB already used (Xorg + gnome-shell)
      └──────────────────────────────────────────────────────────────
      t=0   t=45s    t=90s    t=120s   t=150s   t=180s
```

---

## Parameter Reference

### llama-server flags used

| Flag | Full name | Our value | Meaning |
|------|-----------|-----------|---------|
| `-m` | `--model` | path | GGUF model file |
| `-c` | `--ctx-size` | 131072 | Maximum context length in tokens |
| `-t` | `--threads` | 12 | CPU threads for prompt processing |
| `-tb` | `--threads-batch` | 12 | CPU threads for batch processing |
| `--gpu-layers` | | 999 | Number of layers to offload to GPU |
| `--cache-type-k` | | q4_0 | Key cache quantization |
| `--cache-type-v` | | q4_0 | Value cache quantization |
| `--flash-attn` | | auto | Flash Attention mode |
| `--parallel` | | 8 | Max parallel request slots |
| `--kv-unified` | | (flag) | Shared KV cache pool |
| `--mlock` | | (flag) | Lock memory to prevent swapping |
| `--no-warmup` | | (flag) | Skip startup warmup |
| `--host` | | 0.0.0.0 | Listen address |
| `--port` | | 8082/8084 | HTTP port |

### Why 128K context and not smaller

128K is Gemma 4's native context window. Smaller values save memory but
limit what the model can process:

- **E2B (extraction):** needs large context for long documents (patent
  filings, medical records, code files). 128K fits the longest single
  document with room for the prompt schema.
- **E4B (synthesis):** the synthesis prompt includes the query + ALL
  retrieved contexts (5 × ~4K tokens) + instructions + answer format.
  With a hypothetical 32K context, you'd hit the ceiling when contexts
  are long. 128K is future-proof.

The VRAM cost of 128K is ~0.5 GB (E2B) + ~0.8 GB (E4B) in Q4_0 KV cache.
This is acceptable.

### Why `--parallel 8` when VRAM is tight

`--kv-unified` makes this safe. Without unified KV, 8 slots × 128K cache =
4-6.4 GB extra. With unified, all 8 slots share one cache pool — 8 concurrent
requests each using 4K of context = 32K total, which fits in the base
allocation. No extra VRAM consumed unless all 8 slots are simultaneously
using large context, which is unlikely in practice.

### Why `--gpu-layers 999` for both models

Both models fit entirely on the RTX 3060 at Q4_0 quantization:
- E2B (3.2 GB on disk, ~9B params @ Q4_0 = ~4.5 bit params) → all layers
  fit in ~2.6 GB VRAM
- E4B (4.9 GB on disk, ~27B params @ Q4_0 = ~1.5 bit params) → all layers
  fit in ~4.7 GB VRAM

Partial offloading (e.g., 20 layers on GPU, rest on CPU) would slow
inference by 5-10x due to CPU↔GPU transfer for every token.

---

## Troubleshooting

### "CUDA out of memory" on startup

The 12 GB RTX 3060 has only 881 MB headroom with all three services running.
If anything new grabs VRAM (browser, screen recording, another CUDA process),
the daemon will fail on next restart:

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Fix:** stop one service temporarily:
```bash
sudo systemctl stop rag-gpu-daemon.service        # frees ~3.9 GB
sudo systemctl stop gemma-4-e4b.service            # frees ~4.7 GB
sudo systemctl stop gemma-4-e2b.service            # frees ~2.6 GB
```

### "Application startup failed" — serve_gpu.py

The daemon failed during model loading because VRAM was already occupied
by the LLMs. Check the daemon log:

```bash
journalctl -u rag-gpu-daemon.service --no-pager -n 30
```

If you see `torch.cuda.OutOfMemoryError`, the LLMs are already using all
available VRAM. This is normal — start in order (E2B → E4B → daemon) and
verify VRAM stays under 11.5 GB after each step.

### "cannot kill" — process owned by another user

If `kill <PID>` fails with "Operation not permitted", the process belongs
to a different system user. Use sudo:

```bash
sudo kill <PID>
# Or with process name:
sudo pkill -f "llama-server"
```

### LLM server doesn't respond after loading

If `curl http://localhost:8082/v1/models` hangs or returns empty, the
model may have crashed during loading. Check:

```bash
journalctl -u gemma-4-e2b.service --no-pager -n 30
journalctl -u gemma-4-e4b.service --no-pager -n 30
```

Common causes:
- Insufficient VRAM (other process grabbed memory first)
- Corrupted GGUF file (checksum mismatch — re-download)
- `--gpu-layers 999` on a model that doesn't fit (exchange for lower count)
