#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Error: $PY not found. Run ./setup.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}"

PYTHONPATH=. "$PY" -m super_mario_bros.fullgame "$@"
