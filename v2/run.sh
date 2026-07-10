#!/usr/bin/env bash
# run.sh — orchestrates the GraphRAG v2 pipeline.
#
# GPU LLMs (Gemma 4 family, on the RTX 3060):
#   Extraction  -> Gemma 4 E2B QAT on :8082  (systemd gemma-4-e2b.service)
#   Synthesis   -> Gemma 4 E4B QAT on :8084  (systemd gemma-4-e4b.service)
# GPU auxiliary daemon (Jina / MiniLM / BGE-reranker / GLiNER lazy) on :8000.
#
# Services:
#   bash run.sh serve       start LLMs + GPU daemon
#   bash run.sh ingest [dir]  re-populate Qdrant + Neo4j
#   bash run.sh ask "query"  full pipeline: embed → retrieve → rerank → E4B synth
#   bash run.sh retrieve "query"  retrieval-only (contexts, no synthesis)
#   bash run.sh health        check all services
#   bash run.sh stop          stop daemon + LLMs
set -e
cd "$(dirname "$0")"
ENV=/mnt/data-970-plus/rag-env
export HF_HOME=/mnt/data-970-plus/hf_cache
PY="$ENV/bin/python"
DAEMON_PID=/tmp/rag_gpu_daemon.pid

start_llms() {
  echo "Starting GPU LLMs via systemd..."
  # Only start if not already running (idempotent)
  curl -s --max-time 2 "http://localhost:8082/v1/models" >/dev/null 2>&1 || \
    sudo systemctl start gemma-4-e2b.service
  curl -s --max-time 2 "http://localhost:8084/v1/models" >/dev/null 2>&1 || \
    sudo systemctl start gemma-4-e4b.service
  for p in 8082 8084; do
    for i in $(seq 1 60); do
      curl -s --max-time 2 "http://localhost:$p/v1/models" >/dev/null 2>&1 && break
      sleep 1
    done
    echo "  :$p ready"
  done
}

start_daemon() {
  if curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "GPU daemon already running ($(curl -s http://127.0.0.1:8000/health 2>/dev/null | head -c 50))"
    return
  fi
  echo "Starting GPU daemon (Jina/MiniLM/BGE on GPU, GLiNER lazy)..."
  nohup "$PY" serve_gpu.py > /mnt/data-970-plus/rag-system/logs/daemon_gpu_v2.log 2>&1 &
  echo $! > "$DAEMON_PID"
  for i in $(seq 1 200); do
    curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "GPU daemon ready (${i}s)"; return; }
    sleep 1
  done
  echo "Daemon failed — see logs/daemon_gpu_v2.log"; exit 1
}

cmd_health() {
  echo "=== LLMs ==="
  curl -s --max-time 2 http://localhost:8082/v1/models >/dev/null && echo "E2B :8082  up" || echo "E2B :8082  DOWN"
  curl -s --max-time 2 http://localhost:8084/v1/models >/dev/null && echo "E4B :8084  up" || echo "E4B :8084  DOWN"
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
           sudo systemctl stop gemma-4-e2b.service gemma-4-e4b.service
           echo "Stopped daemon + LLMs" ;;
  *)       echo "Usage: bash run.sh {llms|serve|ingest [dir]|ask \"query\"|retrieve \"query\"|health|stop}" ;;
esac
