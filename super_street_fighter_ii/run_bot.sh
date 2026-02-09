#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "No virtual environment found. Run: cd $ROOT_DIR && ./setup.sh"
    exit 1
fi

if [[ "${HEADLESS:-}" == "1" ]]; then
    export SDL_VIDEODRIVER="dummy"
    export SDL_AUDIODRIVER="dummy"
    export SDL_SOFTWARE_RENDERER="1"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
"$VENV_PYTHON" "$SCRIPT_DIR/run_bot.py" "$@"
