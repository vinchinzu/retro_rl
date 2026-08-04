#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONOREPO="$(cd "$ROOT/.." && pwd)"
STATE_DIR="$ROOT/debug_alignment/editor"
PID_FILE="$STATE_DIR/editor.pid"
LOG_FILE="$STATE_DIR/editor.log"

background=0
state_name="latest"
editor_args=()

usage() {
  cat <<'EOF'
Usage:
  ./startup.sh [--background] [--state NAME] [editor args]

Starts the Harvest Moon editor through the parent uv project so PySide6,
stable-retro, and the optional Cursor SDK resolve from one environment.
The editor includes the embedded emulator dock, map overlay, state editor,
and Agent panel (View -> Agent Panel).

Options:
  --background       Start detached, write debug_alignment/editor/editor.pid
                     and debug_alignment/editor/editor.log.
  --foreground       Start attached. This is the default.
  --state NAME       Initial save state name. Defaults to latest.
  --autostart        Start the embedded emulator after loading the editor.
  --log PATH         Background log path.
  --help             Show this help.

Examples:
  ./startup.sh
  ./startup.sh --state Y1_Spring_D1_Farm
  ./startup.sh --state latest --autostart
  ./startup.sh --background --state latest
  ./startup.sh -- --export-dir debug_alignment/editor_exports
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background)
      background=1
      shift
      ;;
    --foreground)
      background=0
      shift
      ;;
    --state)
      if [[ $# -lt 2 ]]; then
        echo "--state requires a value" >&2
        exit 2
      fi
      state_name="$2"
      shift 2
      ;;
    --state=*)
      state_name="${1#--state=}"
      shift
      ;;
    --log)
      if [[ $# -lt 2 ]]; then
        echo "--log requires a value" >&2
        exit 2
      fi
      LOG_FILE="$2"
      shift 2
      ;;
    --log=*)
      LOG_FILE="${1#--log=}"
      shift
      ;;
    --autostart)
      editor_args+=("$1")
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      editor_args+=("$@")
      break
      ;;
    *)
      editor_args+=("$1")
      shift
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for the canonical editor launcher." >&2
  exit 127
fi

if [[ ! -f "$MONOREPO/pyproject.toml" ]]; then
  echo "Parent pyproject.toml not found: $MONOREPO/pyproject.toml" >&2
  exit 2
fi

cmd=(
  uv run --project "$MONOREPO" --extra cursor
  python -m retro_harness.editor_launcher harvest
  -- --state "$state_name"
)
cmd+=("${editor_args[@]}")

export PYTHONPATH="$MONOREPO:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"

mkdir -p "$STATE_DIR" "$(dirname "$LOG_FILE")"

if [[ "$background" -eq 1 ]]; then
  if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Harvest editor already running as PID $old_pid"
      echo "Log: $LOG_FILE"
      exit 0
    fi
  fi
  cd "$ROOT"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "${cmd[@]}" >>"$LOG_FILE" 2>&1 < /dev/null &
  else
    nohup "${cmd[@]}" >>"$LOG_FILE" 2>&1 < /dev/null &
  fi
  pid=$!
  printf '%s\n' "$pid" >"$PID_FILE"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Harvest editor failed to stay running. Log: $LOG_FILE" >&2
    tail -40 "$LOG_FILE" >&2 || true
    exit 1
  fi
  echo "Started Harvest editor as PID $pid"
  echo "Log: $LOG_FILE"
  exit 0
fi

cd "$ROOT"
exec "${cmd[@]}"
