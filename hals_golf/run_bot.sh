#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_CANDIDATES=(
    "$ROOT_DIR/.venv/bin/python"
    "$SCRIPT_DIR/.venv/bin/python"
)

VENV_PYTHON=""
for candidate in "${VENV_CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]]; then
        VENV_PYTHON="$candidate"
        break
    fi
done

if [[ -z "$VENV_PYTHON" ]]; then
    echo "No virtual environment found. Run: cd $ROOT_DIR && ./setup.sh"
    exit 1
fi

if [[ "${HEADLESS:-}" == "1" ]]; then
    export SDL_VIDEODRIVER="dummy"
    export SDL_AUDIODRIVER="dummy"
    export SDL_SOFTWARE_RENDERER="1"
fi

export PYTHONPATH="${SCRIPT_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"

if [[ "${1:-}" == "auto" ]]; then
    shift
    exec "$VENV_PYTHON" -m hals_golf.runtime.golf_bot play --autoplay "$@"
fi

exec "$VENV_PYTHON" -m hals_golf.runtime.golf_bot "$@"
