#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Missing venv at $VENV_ACTIVATE"
  exit 1
fi

if [[ "${HEADLESS:-}" == "1" ]]; then
  export SDL_VIDEODRIVER="dummy"
  export SDL_AUDIODRIVER="dummy"
  export SDL_SOFTWARE_RENDERER="1"
fi

source "$VENV_ACTIVATE"

if [[ "${1:-}" == "clear" ]]; then
  shift
  python "$SCRIPT_DIR/harvest_bot.py" play --autoplay "$@"
else
  python "$SCRIPT_DIR/harvest_bot.py" "$@"
fi
