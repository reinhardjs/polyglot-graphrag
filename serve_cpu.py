#!/usr/bin/env python3
"""
CPU-compatible daemon — thin wrapper around serve_gpu.py (unified).

Design rationale (v2.0 ~v2.6):
  Originally there were two separate files: serve_gpu.py (with CUDA-optimised
  model loading, Prometheus metrics, /admin/reload, /v1/embeddings) and
  serve_cpu.py (a simpler CPU-only implementation that lagged behind in feature
  parity). Maintaining both created a persistent gap: every new endpoint or
  handler fix had to be ported to the other file.

  Now serve_gpu.py IS the unified daemon. It detects device at startup
  (DEVICE = "cuda" | "cpu"), CPU-guards .half() calls, and adjusts its log
  prefix (gpu-daemon vs rag-daemon). This file exists purely as a backward-
  compatible entry point so scripts and cron jobs referencing "serve_cpu.py"
  keep working without changes. It imports and re-exports the FastAPI app from
  serve_gpu.py — zero divergence.

To verify device detection:
    curl -s http://localhost:8000/health | python3 -c "import sys,json; print(json.load(sys.stdin)['device'])"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__) or ".")

from serve_gpu import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
