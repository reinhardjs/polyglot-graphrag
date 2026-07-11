#!/usr/bin/env bash
# run_tests.sh — comprehensive test runner for GraphRAG v2 (all phases).
#
# Modes:
#   bash run_tests.sh            run everything (unit + e2e + eval smoke)
#   bash run_tests.sh unit       fast, no daemon required (pure logic)
#   bash run_tests.sh e2e        live daemon tests (:8000 must be up)
#   bash run_tests.sh eval       Phase 4 eval smoke on golden datasets
#   bash run_tests.sh phase N    tests for one phase (1|2|3|4)
#
# Exit non-zero if any selected suite fails.
set -uo pipefail
cd "$(dirname "$0")"
PY=/mnt/data-970-plus/rag-env/bin/python
BASE=http://127.0.0.1:8000
RC=0

hr() { printf '%s\n' "────────────────────────────────────────────────────────"; }

daemon_up() { curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1; }

run_unit() {
  hr; echo "UNIT TESTS (no daemon required)"; hr
  # Phase-local logic: modulator, pruning, crag FSM, eval metrics, chunking, prompts
  "$PY" -m pytest \
    tests/test_query_modulator.py \
    tests/test_graph_prune.py \
    tests/test_crag_pipeline.py \
    tests/test_evaluate_pipeline.py \
    tests/test_chunking.py \
    tests/test_prompts.py \
    tests/test_condense.py \
    -q || RC=1
}

run_e2e() {
  hr; echo "E2E TESTS (live daemon on :8000)"; hr
  if ! daemon_up; then
    echo "  ! daemon down — starting via run.sh serve ..."
    bash run.sh serve || { echo "  ! could not start daemon"; RC=1; return; }
  fi
  "$PY" -m pytest \
    tests/test_e2e_neuro_symbolic.py \
    tests/test_e2e_chunking.py \
    tests/test_ingest_contract.py \
    -q || RC=1
}

run_eval() {
  hr; echo "PHASE 4 EVAL SMOKE (golden datasets)"; hr
  if ! daemon_up; then
    echo "  ! daemon down — eval --live needs :8000. Running static metric check only."
    "$PY" -m pytest tests/test_evaluate_pipeline.py -q || RC=1
    return
  fi
  for dom in engineering medical; do
    gd="sample_data/golden/${dom}.json"
    [ -f "$gd" ] || continue
    echo ">> eval domain=$dom (standard /ask)"
    "$PY" evaluate_pipeline.py "$gd" --live --domain "$dom" || RC=1
  done
}

run_full() {
  hr; echo "FULL PYTEST SUITE"; hr
  if daemon_up; then
    "$PY" -m pytest tests/ -q || RC=1
  else
    echo "  (daemon down — e2e tests will auto-skip)"
    "$PY" -m pytest tests/ -q || RC=1
  fi
  run_eval
}

case "${1:-all}" in
  unit)  run_unit ;;
  e2e)   run_e2e ;;
  eval)  run_eval ;;
  phase)
    case "${2:-}" in
      1) "$PY" -m pytest tests/test_query_modulator.py -q || RC=1 ;;
      2) "$PY" -m pytest tests/test_graph_prune.py -q || RC=1 ;;
      3) "$PY" -m pytest tests/test_crag_pipeline.py -q || RC=1 ;;
      4) "$PY" -m pytest tests/test_evaluate_pipeline.py -q || RC=1 ;;
      *) echo "Usage: bash run_tests.sh phase {1|2|3|4}"; exit 2 ;;
    esac ;;
  all)   run_full ;;
  *)     echo "Usage: bash run_tests.sh {all|unit|e2e|eval|phase N}"; exit 2 ;;
esac

hr
if [ "$RC" -eq 0 ]; then echo "ALL SELECTED TESTS PASSED ✅"; else echo "SOME TESTS FAILED ❌"; fi
exit "$RC"
