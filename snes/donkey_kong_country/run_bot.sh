#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find venv python - prefer root, fallback to other projects
VENV_CANDIDATES=(
    "$ROOT_DIR/.venv/bin/python"
    "$ROOT_DIR/super_metroid_rl/retro_env/bin/python"
    "$ROOT_DIR/harvest/.venv/bin/python"
)

VENV_PYTHON=""
for candidate in "${VENV_CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]]; then
        VENV_PYTHON="$candidate"
        break
    fi
done

if [[ -z "$VENV_PYTHON" ]]; then
    echo "No virtual environment found. Run setup.sh from project root:"
    echo "  cd $ROOT_DIR && ./setup.sh"
    exit 1
fi

# Headless mode support
if [[ "${HEADLESS:-}" == "1" ]]; then
    export SDL_VIDEODRIVER="dummy"
    export SDL_AUDIODRIVER="dummy"
    export SDL_SOFTWARE_RENDERER="1"
else
    # Default to X11 to avoid Wayland windowing issues
    if [[ -z "${SDL_VIDEODRIVER:-}" ]]; then
        export SDL_VIDEODRIVER="x11"
    fi
fi

# Add root to PYTHONPATH for retro_harness imports
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

"$VENV_PYTHON" "$SCRIPT_DIR/run_bot.py" "$@"
