#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Missing venv. Run: uv sync"
    exit 1
fi

if [[ "${HEADLESS:-}" == "1" ]]; then
    export SDL_VIDEODRIVER="dummy"
    export SDL_AUDIODRIVER="dummy"
    export SDL_SOFTWARE_RENDERER="1"
fi

if [[ "${1:-}" == "clear" ]]; then
    shift
    "$VENV_PYTHON" -m harvest.runtime.harvest_bot play --autoplay "$@"
else
    "$VENV_PYTHON" -m harvest.runtime.harvest_bot "$@"
fi
