#!/usr/bin/env bash
# Morning one-command check: quick eval + summary display.
# Usage: bash scripts/morning_worker_c_check.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

echo "=== Model timestamps ==="
ls -lt models/segment_*.zip 2>/dev/null | head -12
echo ""

echo "=== Running eval (isolated + chained) ==="
"${PYTHON_BIN}" scripts/eval_torizo_integration.py \
  --headless \
  --mode both \
  --episodes "${WORKER_C_EPISODES:-8}" \
  --trials "${WORKER_C_TRIALS:-5}" \
  --max-steps "${WORKER_C_MAX_STEPS:-18000}" \
  --seed-base "${WORKER_C_SEED_BASE:-1729}" \
  --output-json "logs/overnight_worker_c_eval.json" \
  --output-summary "logs/overnight_worker_c_summary.md" \
  "$@"

echo ""
echo "=== Summary ==="
cat logs/overnight_worker_c_summary.md
