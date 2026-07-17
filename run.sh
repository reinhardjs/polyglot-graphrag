#!/usr/bin/env bash
# run.sh — one-command entry point for polyglot-graphrag.
#
#   bash run.sh doctor    check the whole setup, tell you what's missing
#   bash run.sh serve      start GPU models + daemon (runs doctor first)
#   bash run.sh ask "..."   ask a question (starts everything if needed)
#   bash run.sh retrieve "..."   retrieval-only (no synthesis)
#   bash run.sh health      show service + VRAM status
#   bash run.sh stop        stop daemon + models
#
# Everything is env-overridable (no machine-specific paths baked in):
#   MODELS_DIR   where GGUF weights live        (default: <root>/models)
#   LLAMA_BIN    path to llama.cpp server        (auto-detected if unset)
#   E2B_MODEL    GGUF for extraction + synthesis (default: $MODELS_DIR/gemma-4-E2B-it-QAT-Q4_0.gguf)
#   RERANK_DEVICE cuda | cpu
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
ENV="$ROOT/venv"
export HF_HOME="${HF_HOME:-$ROOT/.cache/hf}"
PY="$ENV/bin/python"
mkdir -p "$ROOT/.run" "$ROOT/logs" "$ROOT/.cache"

# ── Offline-safe fallback for the Jina embedder ──────────────────────────────
# The daemon loads the Jina embedding model from the local HF cache. If
# HuggingFace is unreachable, `sentence_transformers` still probes HF HEAD
# endpoints and retries for ~4 minutes before failing to start. Auto-enable
# offline mode when HF is not reachable so the daemon starts from cache
# immediately. Override with HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE if needed.
if [ -z "${HF_HUB_OFFLINE:-}" ] && [ -z "${TRANSFORMERS_OFFLINE:-}" ]; then
  if ! timeout 5 curl -sI https://huggingface.co >/dev/null 2>&1; then
    echo "  [warn] HuggingFace unreachable — enabling offline mode (using cached models)"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  fi
fi

MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
E2B_MODEL="${E2B_MODEL:-$MODELS_DIR/gemma-4-E2B-it-QAT-Q4_0.gguf}"
E2B_CTX="${E2B_CTX:-32768}"
export SW_EXTRACT_WORKERS="${SW_EXTRACT_WORKERS:-4}"
DAEMON_PID="$ROOT/.run/rag_gpu_daemon.pid"

# ── Model files (download hint shown if missing) ─────────────────────────────
MODEL_HINT="  Download:  huggingface-cli download lmstudio-community/gemma-4-E2B-it-QAT-GGUF
  then place gemma-4-E2B-it-QAT-Q4_0.gguf in $MODELS_DIR/
  (override with E2B_MODEL=/path/to.gguf or MODELS_DIR=/path)"

# ── llama.cpp server binary: auto-detect if not provided ──────────────────────
# IMPORTANT: prefer the CUDA build. The Vulkan llama.cpp build is ~3× slower
# (~45 tok/s vs ~128 tok/s on RTX 3060) and silently breaks the synthesis SLO.
# Candidates are ordered CUDA-first; the bare `find` last-resort is avoided
# because it can return the Vulkan binary first. Set LLAMA_BIN explicitly to
# override.
detect_llama_bin() {
  # 1) explicit env
  [ -n "${LLAMA_BIN:-}" ] && [ -x "${LLAMA_BIN:-}" ] && { echo "$LLAMA_BIN"; return; }
  # 2) CUDA llama.cpp builds (preferred), then generic, then PATH
  local cands=(
    "$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1/llama-server"
    "$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-cb/llama-server"
    "$ROOT/bin/llama-server"
    "$(command -v llama-server 2>/dev/null || true)"
  )
  local c
  for c in "${cands[@]}"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return; }
  done
  # 3) CUDA-only search (do NOT fall back to a Vulkan binary)
  local found
  found=$(find "$HOME/.lmstudio" -path '*nvidia-cuda*' -name llama-server -type f 2>/dev/null | head -1)
  [ -n "$found" ] && { echo "$found"; return; }
  echo ""
}

LLAMA_BIN="$(detect_llama_bin)"

# ── venv: create + install if missing or broken ──────────────────────────────
ensure_venv() {
  if [ ! -x "$PY" ]; then
    # Prefer Python 3.11 (the verified-working interpreter); fall back to python3.
    local py="${PYTHON3:-$(command -v python3.11 || command -v python3)}"
    echo "Creating venv ($ENV) with $py ..."
    "$py" -m venv "$ENV" || { echo "  FAILED to create venv — install python3.11-venv." >&2; return 1; }
  fi
  # Verify key imports; install only if something is missing.
  if ! "$PY" -c "import fastapi, qdrant_client, neo4j, requests" >/dev/null 2>&1; then
    echo "Installing Python dependencies (pip install -r requirements.txt) ..."
    "$ENV/bin/pip" install --quiet -r "$ROOT/requirements.txt" || {
      echo "  FAILED pip install — check network / requirements.txt." >&2; return 1; }
    "$PY" -m spacy download en_core_web_sm >/dev/null 2>&1 || \
      echo "  (note: spacy model download skipped/failed — run: $PY -m spacy download en_core_web_sm)"
  fi
  return 0
}

# ── doctor: report exactly what is missing, exit non-zero if broken ───────────
cmd_doctor() {
  local fail=0
  echo "=== polyglot-graphrag setup check ==="
  # venv
  if [ -x "$PY" ] && "$PY" -c "import fastapi, qdrant_client, neo4j, requests" >/dev/null 2>&1; then
    echo "  [ok]  venv        ($ENV — deps present)"
  else
    echo "  [MISSING] venv/deps — run:  bash run.sh setup"
    echo "           (or: python3 -m venv venv && pip install -r requirements.txt)"
    fail=1
  fi
  # docker + stores
  if command -v docker >/dev/null 2>&1; then
    if docker compose ps >/dev/null 2>&1; then echo "  [ok]  docker + compose"; else
      echo "  [MISSING] docker compose — install Docker Desktop / docker-compose-plugin"; fail=1; fi
  else echo "  [MISSING] docker — install Docker (Neo4j + Qdrant run via 'docker compose up -d')"; fail=1; fi
  # stores reachable
  if curl -s --max-time 2 http://localhost:6333 >/dev/null 2>&1; then echo "  [ok]  Qdrant :6333"; else
    echo "  [ACTION] Qdrant not up — run:  docker compose up -d"; fi
  if curl -s --max-time 2 http://localhost:7474 >/dev/null 2>&1; then echo "  [ok]  Neo4j :7474"; else
    echo "  [ACTION] Neo4j not up — run:  docker compose up -d"; fi
  # llama binary
  if [ -n "$LLAMA_BIN" ]; then echo "  [ok]  llama-server ($LLAMA_BIN)"; else
    echo "  [MISSING] llama.cpp server — install llama.cpp (provides 'llama-server'),";
    echo "           or set LLAMA_BIN=/path/to/llama-server"; fail=1; fi
  # model weights
  if [ -f "$E2B_MODEL" ]; then echo "  [ok]  E2B model   ($E2B_MODEL)"; else
    echo "  [MISSING] E2B GGUF — $E2B_MODEL"; echo "$MODEL_HINT"; fail=1; fi
  # daemon
  if curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then echo "  [ok]  daemon :8000"; else
    echo "  [ACTION] daemon not running — 'bash run.sh serve' will start it"; fi
  # synthesis backend
  if curl -s --max-time 2 http://127.0.0.1:8082/v1/models >/dev/null 2>&1; then echo "  [ok]  E2B :8082 (synthesis)"; else
    echo "  [ACTION] E2B :8082 not up — 'bash run.sh serve' will start it"; fi
  # doc/code consistency audit (advisory — flags drift, non-fatal)
  if [ -x "$PY" ] && [ -f "$ROOT/scripts/audit_docs.py" ]; then
    if "$PY" "$ROOT/scripts/audit_docs.py" --daemon-url http://127.0.0.1:8000 >/tmp/audit_docs.log 2>&1; then
      echo "  [ok]  doc/code audit (no drift)"
    else
      echo "  [ACTION] doc/code drift detected — run:  ./venv/bin/python scripts/audit_docs.py"
      echo "           (see /tmp/audit_docs.log for the [FAIL] items)"
    fi
  fi
  echo "──────────────────────────────────────"
  if [ "$fail" -ne 0 ]; then echo "Setup INCOMPLETE — fix the [MISSING] items above, then re-run."; return 1; fi
  echo "Setup OK — you can 'bash run.sh ask \"how does the ask pipeline work\"'."
  return 0
}

# ── model / daemon starters ───────────────────────────────────────────────────
start_llm() {
  local port="$1" model="$2" log="$3" ctx="${4:-32768}"
  if curl -s --max-time 2 "http://localhost:$port/v1/models" >/dev/null 2>&1; then
    echo "  :$port already up"; return 0; fi
  if [ -z "$LLAMA_BIN" ]; then
    echo "  :$port SKIP — llama.cpp server not found. Set LLAMA_BIN or install llama.cpp." >&2; return 1; fi
  if [ ! -f "$model" ]; then
    echo "  :$port SKIP — model not found: $model" >&2; echo "$MODEL_HINT" >&2; return 1; fi
  echo "  starting llama-server :$port ($(basename "$model"), ctx=$ctx)"
  # --reasoning off: gemma-4 puts answers in content (not thinking channel)
  nohup "$LLAMA_BIN" --model "$model" --host 127.0.0.1 --port "$port" \
    --gpu-layers 999 --ctx-size "$ctx" --reasoning off > "$log" 2>&1 &
  for i in $(seq 1 90); do
    curl -s --max-time 2 "http://localhost:$port/v1/models" >/dev/null 2>&1 && { echo "  :$port ready (${i}s)"; return 0; }
    sleep 1
  done
  echo "  :$port FAILED to start — see $log" >&2; return 1
}

start_daemon() {
  ensure_venv || { echo "venv setup failed — run 'bash run.sh doctor'." >&2; return 1; }
  if curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "GPU daemon already running"; return 0; fi
  echo "Starting GPU daemon (Jina/BGE on GPU, GLiNER lazy)..."
  nohup "$PY" serve_gpu.py > "$ROOT/logs/daemon_gpu.log" 2>&1 &
  echo $! > "$DAEMON_PID"
  for i in $(seq 1 200); do
    curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "GPU daemon ready (${i}s)"; return 0; }
    sleep 1
  done
  echo "Daemon failed — see logs/daemon_gpu.log" >&2; return 1
}

ensure_up() {
  start_llm 8082 "$E2B_MODEL" "$ROOT/logs/llm_e2b.log" "$E2B_CTX" || true
  start_daemon || { echo "Could not start the stack. Run 'bash run.sh doctor' for guidance." >&2; exit 1; }
}

cmd_health() {
  echo "=== LLMs ==="
  curl -s --max-time 2 http://127.0.0.1:8082/v1/models >/dev/null 2>&1 && echo "E2B :8082  up (extraction + synthesis)" || echo "E2B :8082  DOWN"
  echo "=== Daemon ==="
  curl -s http://127.0.0.1:8000/health 2>/dev/null || echo "Daemon :8000 DOWN"
  echo "=== VRAM ==="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi missing)"
}

cmd_ask() {
  ensure_up
  "$PY" ask.py "${@:-how does the ask pipeline work?}"
}

cmd_retrieve() {
  ensure_up
  local query="${*:-how does the ask pipeline work?}"
  echo "Retrieving: $query"
  curl -s --max-time 30 -X POST http://127.0.0.1:8000/ask \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$query\",\"synthesize\":false}" | "$PY" -m json.tool
}

case "${1:-help}" in
  doctor)   cmd_doctor ;;
  setup)    ensure_venv && echo "Setup complete — next: docker compose up -d && bash run.sh serve" ;;
  serve)    cmd_doctor || true; start_llm 8082 "$E2B_MODEL" "$ROOT/logs/llm_e2b.log" "$E2B_CTX" || true; start_daemon ;;
  ask)      shift; cmd_ask "$@" ;;
  retrieve) shift; cmd_retrieve "$@" ;;
  ingest)   shift; ensure_up; "$PY" scripts/ingest_corpus_docs.py --docs "${DOCS:-corpus}" --domain "${DOMAIN:-enterprise}" --source "${SOURCE:-my-corpus}" "$@" ;;
  health)   cmd_health ;;
  stop)     [ -f "$DAEMON_PID" ] && kill "$(cat "$DAEMON_PID")" 2>/dev/null && rm -f "$DAEMON_PID"
            pkill -f "llama-server.*--port 8082" 2>/dev/null || true
            echo "Stopped daemon + E2B. (E4B :8084 is retired.)" ;;
  *)        echo "Usage:"
            echo "  bash run.sh doctor              check setup, show what's missing"
            echo "  bash run.sh setup               create venv + install dependencies"
            echo "  bash run.sh serve               start models + daemon"
            echo "  bash run.sh ask \"your question\"  ask (starts stack if needed)"
            echo "  bash run.sh retrieve \"query\"    retrieval-only (no synthesis)"
            echo "  bash run.sh ingest              ingest ./corpus/*.md (see corpus/README.md)"
            echo "  bash run.sh health              service + VRAM status"
            echo "  bash run.sh stop                stop everything" ;;
esac
