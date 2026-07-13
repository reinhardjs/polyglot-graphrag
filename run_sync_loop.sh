#!/usr/bin/env bash
# run_sync_loop.sh — keep ingesting until all cleaned docs are in the graph.
# sync_docs.py resumes from .sync_state.json, so re-invoking it after each
# pass continues where it left off. This loop survives the 900s background
# cap by spawning a fresh sync_docs per iteration.
set -u
cd /mnt/data-970-plus/rag-system
export EXTRACTION_MODE=sliding_window
export SYNC_INGEST_TIMEOUT=600
# First arg = docs root to ingest (default: the cleaned test corpus).
ROOT="${1:-/mnt/data-970-plus/test-docs}"
STATE=$ROOT/.sync_state.json
LOG=logs/sync_sliding_loop.log
TOTAL=$(./venv/bin/python -c "import sync_docs as s;from pathlib import Path;r=Path('$ROOT');print(len(s.scan_files(r,s.load_syncignore(r),'$STATE')))")
echo "$(date) TOTAL ingestable=$TOTAL" | tee -a "$LOG"
for i in $(seq 1 100); do
  DONE=$(python3 -c "import json;print(len(json.load(open('$STATE'))))" 2>/dev/null || echo 0)
  if [ "$DONE" -ge "$TOTAL" ]; then
    echo "$(date) DONE all $TOTAL ingested" | tee -a "$LOG"
    break
  fi
  echo "$(date) pass $i: $DONE/$TOTAL ingested, running sync_docs..." | tee -a "$LOG"
  ./venv/bin/python sync_docs.py "$ROOT" >> "$LOG" 2>&1
  sleep 2
done
echo "$(date) LOOP COMPLETE" | tee -a "$LOG"
