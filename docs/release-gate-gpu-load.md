# GPU Load Evidence — Release Gate Run (2026-07-16)

Captured with `nvidia-smi` (sampled every 0.5s) during the `scripts/release-gate.py`
execution on **2026-07-16 13:58:13**. This records the actual GPU pressure the
synthesis/retrieval benchmarks ran under, and explains the latency envelope the
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

| Process                              | VRAM   | Role                                     |
|--------------------------------------|--------|------------------------------------------|
| `rag-system/venv/bin/python` (PID 212954) | 4558 MiB | **This project's daemon** — Jina embed + E2B GGUF (synthesis) + BGE reranker, all on GPU |
| `llama-server` (PID 212891)         | 2340 MiB | **External LM Studio server** — NOT part of this project; co-resident on the same GPU |
| `gnome-shell` / `Xorg`              | 139 MiB | Desktop display (unavoidable)            |

## Why this matters for the latency SLOs

The synthesis benchmark (`Bench 5x all-domains`, `_SYNTH_THRESHOLD_S`) runs the
E2B GGUF at full tilt. During the gate the card was at **95% utilization** and
the project daemon **shared the GPU with an external LM Studio `llama-server`
(2340 MiB)** that was not part of the test.

That co-tenancy is additional contention beyond the documented Jina-embed + E2B
sharing. It is the practical reason the 10x-burst synthesis p95 lands in the
6.4–7.1s range (occasional ~8s outliers) rather than the ~3.5s a sporadic
single call achieves. The recalibrated thresholds in `scripts/release-gate.py`
(CPU/GPU):

- Retrieval p95: **1100ms** (GPU)
- Synthesis p95: **8.5s** (GPU) — covers the burst envelope + co-tenancy margin

### Reproducibility note

For a *clean* gate run with minimal GPU contention, free the card first:

```bash
# stop this project's daemon
pkill -f "rag-system/venv/bin/python serve_gpu.py"   # or kill by explicit PID
# stop the co-resident LM Studio server if present
pkill -f "llama-server"
# confirm zero compute apps
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Re-running the gate on a card with only the project daemon loaded (no external
llama-server) will typically show a lower synthesis p95. The SLOs above remain
valid headroom for the contested (default desktop) state.
