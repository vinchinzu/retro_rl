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
  super_mario_bros/run_8_3.sh audit [args...]
  super_mario_bros/run_8_3.sh train [args...]
  super_mario_bros/run_8_3.sh eval  [args...]

Examples:
  super_mario_bros/run_8_3.sh audit
  super_mario_bros/run_8_3.sh train --seed super_mario_bros/optimizer/runs/smb_8_3/recording_000.json --population 12 --generations 6
  super_mario_bros/run_8_3.sh train --resume super_mario_bros/optimizer/runs/smb_8_3/raw_ga_8_3/ga_raw_best.json --generations 6
  super_mario_bros/run_8_3.sh eval --runs 20
EOF
  exit 1
fi

cmd="$1"
shift

cd "$ROOT"
exec "$PY" super_mario_bros/smb83_pipeline.py "$cmd" "$@"
