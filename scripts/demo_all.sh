#!/usr/bin/env bash
# scripts/demo_all.sh — one-command v1.0 demo across all 4 domains.
#
# Seeds every domain idempotently (via --demo), health-checks the daemon,
# then runs one sample query per vertical so you can see real results.
#
# Usage:
#   ./scripts/demo_all.sh            # start daemon + run sample queries
#   ./scripts/demo_all.sh --no-start # assume daemon already running
set -euo pipefail
cd "$(dirname "$0")/.."

DAEMON=localhost:8000
NO_START=0
[[ "${1:-}" == "--no-start" ]] && NO_START=1

start_daemon() {
  echo "[demo] starting daemon with --demo seeding ..."
  ./venv/bin/python serve_gpu.py --demo > logs/daemon_gpu.log 2>&1 &
  echo $! > /tmp/rag_demo.pid
}

if [[ "$NO_START" -eq 0 ]]; then
  start_daemon
fi

echo "[demo] waiting for /health ..."
for i in $(seq 1 60); do
  curl -s --max-time 3 "http://$DAEMON/health" >/dev/null 2>&1 && break
  sleep 2
done

echo "[demo] /health:"; curl -s "http://$DAEMON/health" | ./venv/bin/python -c "import sys,json; d=json.load(sys.stdin); print('  status=',d['status'],'| domains=',sorted(d['domains'].keys()),'| backends=',d['backends'])"

run() {
  local dom="$1"; local q="$2"
  echo "──────── $dom ────────"
  curl -s -X POST "http://$DAEMON/ask" -H 'Content-Type: application/json' \
    -d "{\"query\":\"$q\",\"domain\":\"$dom\",\"synthesize\":false}" \
    | ./venv/bin/python -c "import sys,json; d=json.load(sys.stdin); print('  n_contexts=',d['n_contexts'],'| signals=',sorted(set(m['signal'] for m in d['contexts_meta'])))"
}

run healthcare "fever and rash and cough"
run enterprise "what services depend on payment"
run legal     "GDPR obligations for cross-border data transfers"
run fraud     "unusual transaction pattern on account alpha"

echo "──────── cross-domain (all) ────────"
curl -s -X POST "http://$DAEMON/ask" -H 'Content-Type: application/json' \
  -d '{"query":"payment fraud breach contract","domain":"all","synthesize":false}' \
  | ./venv/bin/python -c "import sys,json; d=json.load(sys.stdin); print('  n_contexts=',d['n_contexts'],'| domains=',sorted(set(m.get('domain','') for m in d['contexts_meta'])))"

echo "[demo] done. Daemon PID: $(cat /tmp/rag_demo.pid 2>/dev/null || echo 'external')"
