#!/usr/bin/env bash
# Run the GraphRAG system. Usage:
#   bash run.sh ingest <file> [--type ADR] [--author alice]
#   bash run.sh ingest-folder <dir>
#   bash run.sh ask "<query>"
#   bash run.sh serve
set -e
export HF_HOME=/mnt/data-970-plus/hf_cache
PY=/mnt/data-970-plus/rag-env/bin/python
cd /mnt/data-970-plus/rag-system
exec $PY main.py "$@"
