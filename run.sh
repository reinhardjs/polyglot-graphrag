#!/usr/bin/env bash
# run.sh — orchestrates the GraphRAG v3 pipeline.
#
# GPU LLMs (Gemma 4 family, on the RTX 3060):
#   E2B QAT on :8082 — serves BOTH extraction (ingest) AND synthesis (ask).
#   E4B (:8084) is RETIRED — E2B handles synthesis (p95 ~2.2s) and E4B gave
#   ~22s for no quality gain on this 12 GB card. E2B is managed by systemd
#   (user unit gemma-4-e2b.service); run.sh only starts it as a fallback if
#   the systemd unit isn't already running.
# GPU auxiliary daemon (Jina / MiniLM / BGE-reranker / GLiNER lazy) on :8000.
#
# Services:
#   bash run.sh serve       start LLMs + GPU daemon
#   bash run.sh ingest [dir]  re-populate Qdrant + Neo4j
#   bash run.sh ask "query"  full pipeline: embed → retrieve → rerank → E2B synth
#   bash run.sh retrieve "query"  retrieval-only (contexts, no synthesis)
#   bash run.sh health        check all services
#   bash run.sh stop          stop daemon + LLMs
set -e
# Anchor all paths to the project root (this script's directory) so the project
# is portable and does not depend on a machine-specific absolute prefix.
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
ENV="$ROOT/venv"
export HF_HOME="${HF_HOME:-$ROOT/.cache/hf}"
PY="$ENV/bin/python"
DAEMON_PID="$ROOT/.run/rag_gpu_daemon.pid"
mkdir -p "$ROOT/.run" "$ROOT/logs"

# Model directory: GGUF files go here (default <root>/models). Override with
# MODELS_DIR if your weights live elsewhere. This is the single source of truth
# for model paths — no machine-specific absolute prefixes.
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
LLAMA_BIN="${LLAMA_BIN:-/home/reinhard/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.23.1/llama-server}"

e2b_model="${E2B_MODEL:-$MODELS_DIR/gemma-4-E2B-it-QAT-Q4_0.gguf}"
e4b_model="${E4B_MODEL:-$MODELS_DIR/gemma-4-E4B-it-QAT-Q4_0.gguf}"

start_llm() {
  # $1 = port, $2 = model path, $3 = log file, $4 = ctx-size
  local port="$1" model="$2" log="$3" ctx="${4:-8192}"
  if curl -s --max-time 2 "http://localhost:$port/v1/models" >/dev/null 2>&1; then
    echo "  :$port already up"
    return
  fi
  if [ ! -f "$model" ]; then
    echo "  :$port SKIP — model not found: $model" >&2
    echo "       set E2B_MODEL / E4B_MODEL or place the GGUF in $MODELS_DIR" >&2
    return
  fi
  echo "  starting llama-server on :$port with $(basename "$model") (ctx=$ctx)"
  # --reasoning off: gemma-4 routes answers to reasoning_content (thinking
  # channel) by default, leaving message.content EMPTY. Disabling reasoning
  # puts the answer in content where the extraction code reads it.
  nohup "$LLAMA_BIN" --model "$model" --host 127.0.0.1 --port "$port" \
    --gpu-layers 999 --ctx-size "$ctx" --reasoning off > "$log" 2>&1 &
  for i in $(seq 1 90); do
    curl -s --max-time 2 "http://localhost:$port/v1/models" >/dev/null 2>&1 && { echo "  :$port ready (${i}s)"; return; }
    sleep 1
  done
  echo "  :$port FAILED — see $log" >&2
}

# E2B serves both extraction AND synthesis. Large context: sliding_window
# sends several long windows concurrently, and llama.cpp shares ONE KV cache
# across parallel slots. With the default 8192 the concurrent windows overflow
# → HTTP 500. 32768 gives headroom so SW_EXTRACT_WORKERS>1 actually parallelizes.
E2B_CTX="${E2B_CTX:-32768}"
# Parallel sliding-window extraction workers. Bound by E2B's shared KV cache;
# 4 is the sweet spot on a 12GB card with E2B_CTX=32768 (~2x faster than seq).
export SW_EXTRACT_WORKERS="${SW_EXTRACT_WORKERS:-4}"

start_llms() {
  echo "Starting GPU LLM (E2B :8082 — extraction + synthesis)..."
  start_llm 8082 "$e2b_model" "$ROOT/logs/llm_e2b.log" "$E2B_CTX"
}

start_daemon() {
  if curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "GPU daemon already running ($(curl -s http://127.0.0.1:8000/health 2>/dev/null | head -c 50))"
    return
  fi
  echo "Starting GPU daemon (Jina/MiniLM/BGE on GPU, GLiNER lazy)..."
  nohup "$PY" serve_gpu.py > "$ROOT/logs/daemon_gpu.log" 2>&1 &
  echo $! > "$DAEMON_PID"
  for i in $(seq 1 200); do
    curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "GPU daemon ready (${i}s)"; return; }
    sleep 1
  done
  echo "Daemon failed — see logs/daemon_gpu.log"; exit 1
}

cmd_health() {
  echo "=== LLMs ==="
  curl -s --max-time 2 http://127.0.0.1:8082/v1/models >/dev/null && echo "E2B :8082  up (extraction + synthesis)" || echo "E2B :8082  DOWN"
  echo "=== Daemon ==="
  curl -s http://127.0.0.1:8000/health 2>/dev/null || echo "Daemon :8000 DOWN"
  echo "=== VRAM ==="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi missing)"
}

cmd_retrieve() {
  # Retrieval-only: returns contexts via /ask (synthesize=false). No E4B.
  start_llms; start_daemon
  local query="${*:-who reported BUG-204?}"
  echo "Retrieving: $query"
  curl -s --max-time 30 -X POST http://127.0.0.1:8000/ask \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$query\",\"synthesize\":false}" | "$PY" -m json.tool
}

case "$1" in
  llms)     start_llms ;;
  serve)    start_llms; start_daemon ;;
  ingest)   start_llms; start_daemon
            DIR="${2:-$PWD/sample_data}"
            echo "Ingesting $DIR ..."; "$PY" ingest.py "$DIR" ;;
  ask)     start_llms; start_daemon; shift; "$PY" ask.py "$*" ;;
  retrieve) shift; cmd_retrieve "$@" ;;
  health)  cmd_health ;;
  stop)    [ -f "$DAEMON_PID" ] && kill "$(cat "$DAEMON_PID")" 2>/dev/null && rm -f "$DAEMON_PID"
           # Stop the E2B llama-server (extraction + synthesis). If E2B is
           # running under the systemd user unit, stop it there instead:
           #   systemctl --user stop gemma-4-e2b.service
           pkill -f "llama-server.*--port 8082" 2>/dev/null || true
           echo "Stopped daemon + E2B (E4B :8084 is retired)" ;;
  *)       echo "Usage: bash run.sh {llms|serve|ingest [dir]|ask \"query\"|retrieve \"query\"|health|stop}" ;;
esac
