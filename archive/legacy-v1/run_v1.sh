#!/usr/bin/env bash
# Run the GraphRAG system. Usage:
#   bash run.sh ingest <file> [--type ADR] [--author alice]
#   bash run.sh ingest-folder <dir>
#   bash run.sh ask "<query>"
#   bash run.sh serve
set -e
export HF_HOME=<hf-cache>
PY=<legacy-venv>/bin/python
cd <project-root>
exec $PY main.py "$@"
