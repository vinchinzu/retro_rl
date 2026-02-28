#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Error: $PY not found. Run ./setup.sh first." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  super_mario_bros/run_8_4.sh diagnose [args...]
  super_mario_bros/run_8_4.sh manifest [args...]
  super_mario_bros/run_8_4.sh train    [args...]
  super_mario_bros/run_8_4.sh eval     [args...]
  super_mario_bros/run_8_4.sh run      [args...]

Examples:
  super_mario_bros/run_8_4.sh diagnose --candidate-runs 2
  super_mario_bros/run_8_4.sh train --segments 3 --mode raw --iterations 1500
  super_mario_bros/run_8_4.sh manifest
  super_mario_bros/run_8_4.sh eval --chain --runs 5
  super_mario_bros/run_8_4.sh run --chain --runs 3
EOF
  exit 1
fi

cmd="$1"
shift

cd "$ROOT"
exec "$PY" super_mario_bros/smb84_pipeline.py "$cmd" "$@"

