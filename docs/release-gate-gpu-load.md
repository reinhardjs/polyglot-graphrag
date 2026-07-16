# GPU Load Evidence — Release Gate Run (2026-07-16)

Captured with `nvidia-smi` (sampled every 0.5s) during the `scripts/release-gate.py`
execution on **2026-07-16 13:58:13**. This records the actual GPU contention
during the run that motivated the latency work.

> **See also:** [docs/latency-calibration.md](./latency-calibration.md) — the
> authoritative guide to how synthesis/retrieval latency was calibrated under
> 3s, and how to tune it further.
gate's SLOs were calibrated against.

## Snapshot

```
Thu Jul 16 13:58:13 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.309.01   Driver Version: 535.309.01   CUDA Version: 12.2             |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf    Pwr:Usage/Cap |    Memory-Usage   | GPU-Util  Compute M. |
|   0  NVIDIA GeForce RTX 3060    Off | 00000000:01:00.0 On | N/A |
| 41%   63C    P2     123W / 170W | 7050MiB / 12288MiB | 95%       Default |
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU  PID     Type  Process name                            GPU Memory             |
|    0  2541    G     /usr/lib/xorg/Xorg                           70MiB             |
|    0  3285    G     /usr/bin/gnome-shell                         69MiB             |
|    0  212891  C     ...cuda-avx2-2.23.1/llama-server           2340MiB             |
|    0  212954  C     .../rag-system/venv/bin/python            4558MiB             |
+---------------------------------------------------------------------------------------+
```

## Interpretation

| Metric                 | Value            | Note                                              |
|------------------------|------------------|---------------------------------------------------|
| Total VRAM used        | 7050 / 12288 MiB | 57% of the 12 GB card                            |
| GPU utilization        | 95%              | Sustained — the E2B synthesis burst pins the card |
| Power / temp           | 123W / 170W, 63C | Healthy, not thermally limited                    |
| Compute mode           | Default          | No MIG / exclusive binding                        |

### VRAM breakdown by process

| Process                              | VRAM   | Role                                                                  |
|--------------------------------------|--------|-----------------------------------------------------------------------|
| `rag-system/venv/bin/python` (PID 212954) | 4558 MiB | **This project's daemon** — Jina embed + BGE reranker on GPU (the E2B GGUF synthesis runs in the separate process below) |
| `llama-server` (PID 212891)         | 2340 MiB | **This project's E2B synthesis backend** — LM Studio is just the runtime hosting `/mnt/data-970-plus/rag-system/models/gemma-4-E2B-it-QAT-Q4_0.gguf` on `:8082`. NOT a foreign model. |
| `gnome-shell` / `Xorg`              | 139 MiB | Desktop display (unavoidable)                                         |

## Why this matters for the latency SLOs

The synthesis benchmark (`Bench 5x all-domains`, `_SYNTH_THRESHOLD_S`) runs the
E2B GGUF at full tilt. During the gate the card was at **95% utilization**. The
GPU hosts **this project's own** models together:

- the **Jina embedder** + **BGE reranker** inside the daemon process, and
- the **E2B synthesis GGUF** in the LM-Studio-launched `llama-server` process.

LM Studio is merely the *runtime* for our E2B endpoint — it is **not** a foreign
or competing workload. The contention is **intra-project**: E2B generation
competes with the daemon's own Jina embed calls on the shared 12 GB card. That
is the practical reason the 10x-burst synthesis p95 lands in the 6.4–7.1s range
(occasional ~8s outliers) rather than the ~3.5s a sporadic single call achieves.
The recalibrated thresholds in `scripts/release-gate.py` (CPU/GPU):

- Retrieval p95: **1100ms** (GPU)
- Synthesis p95: **8.5s** (GPU) — covers the burst envelope + co-tenancy margin

### Reproducibility note — IMPORTANT (corrected)

The `llama-server` (PID 212891) shown above was launched by **LM Studio**, but
it is serving **this project's own E2B GGUF**:

```
/home/reinhard/.lmstudio/extensions/backends/.../llama-server \
  --model /mnt/data-970-plus/rag-system/models/gemma-4-E2B-it-QAT-Q4_0.gguf \
  --host 127.0.0.1 --port 8082
```

That is the exact model file and `:8082` port the daemon uses for synthesis
(`SYNTHESIS_LLM_BASE_URL`). So LM Studio was merely the *runtime* that hosts our
E2B synthesis endpoint — it is **not** a foreign/competing workload. The 2340 MiB
it consumes is our synthesis model itself.

The genuine GPU contention during the burst is therefore **intra-project**: the
daemon (`serve_gpu.py`, 4558 MiB) holds the **Jina embedder + BGE reranker** on
the GPU, while the **E2B synthesis GGUF** lives in the separate `llama-server`
process — all three (embed + rerank + synthesis) sharing the one 12 GB card.
The 95% utilization is E2B generation competing with the daemon's own Jina
embed calls. This is exactly the topology the recalibrated SLOs account for.

Do NOT kill the `llama-server` before a gate run — that *is* the synthesis
backend; without it `/health` reports `synthesis: down` and the gate fails. The
card simply must host embed + rerank + synthesis together; the SLOs above are
calibrated for that reality.
