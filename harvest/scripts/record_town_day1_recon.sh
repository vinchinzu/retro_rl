#!/usr/bin/env bash
# Record Spring D1 town recon (six talks → truck → farm → sleep).
#
# Natural entry is power-on → town gate 0x04 @(712,424). For iterative
# recording this script prefers a pinned entry state; use --power-on for a
# clean boot in the same session.
#
# Usage:
#   ./scripts/record_town_day1_recon.sh
#   ./scripts/record_town_day1_recon.sh --power-on
#   ./scripts/record_town_day1_recon.sh --name my_handoff
#   ./scripts/record_town_day1_recon.sh --capture-only
#   ./scripts/record_town_day1_recon.sh --checklist
#
# Controls (interactive window):
#   Walk / talk with A (C on keyboard)
#   Live HUD shows d1_town_event_mask bits (target 0x3F)
#   [ ] = speed | TAB = fast-forward | F5 = save | ESC = cancel
#
# Outputs on F5:
#   tasks/<name>.json
#   tasks/<name>_end.state
#   custom_integrations/HarvestMoon-Snes/<name>_end.state
#
# After recording:
#   HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon replay \
#     --task town_day1_handoff
#
# Docs: docs/town_day1_recon.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

NAME="town_day1_handoff"
STATE="Y1_Spring_D1_Town_Gate"
POWER_ON=0
CAPTURE_ONLY=0
CHECKLIST=0
AUTO_CAPTURE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name|-n)
      NAME="${2:?}"
      shift 2
      ;;
    --state|-s)
      STATE="${2:?}"
      AUTO_CAPTURE=0
      shift 2
      ;;
    --power-on)
      POWER_ON=1
      shift
      ;;
    --capture-only)
      CAPTURE_ONLY=1
      shift
      ;;
    --no-auto-capture)
      AUTO_CAPTURE=0
      shift
      ;;
    --checklist)
      CHECKLIST=1
      shift
      ;;
    -h|--help)
      sed -n '2,35p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (try --help)" >&2
      exit 2
      ;;
  esac
done

run_py() {
  if command -v uv >/dev/null 2>&1; then
    env PYTHONPATH="${PYTHONPATH:-}:.." uv run --project .. python -m harvest.scripts.town_day1_recon "$@"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    "$ROOT/.venv/bin/python" -m harvest.scripts.town_day1_recon "$@"
  else
    echo "Need uv or harvest/.venv — run: uv sync" >&2
    exit 1
  fi
}

if [[ "$CHECKLIST" -eq 1 ]]; then
  run_py checklist
  exit $?
fi

if [[ "$CAPTURE_ONLY" -eq 1 ]]; then
  echo "============================================================"
  echo " Capture Spring D1 town-gate entry state"
  echo "   name: $STATE"
  echo "============================================================"
  HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    run_py capture-entry --name "$STATE"
  exit $?
fi

STATE_PATH="custom_integrations/HarvestMoon-Snes/${STATE}.state"

if [[ "$POWER_ON" -eq 0 ]]; then
  if [[ ! -f "$STATE_PATH" && "$AUTO_CAPTURE" -eq 1 ]]; then
    echo "[setup] missing $STATE_PATH — capturing power-on entry first..."
    HEADLESS=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
      run_py capture-entry --name "$STATE"
  fi
  if [[ ! -f "$STATE_PATH" ]]; then
    echo "Missing state: $STATE_PATH" >&2
    echo "Run: $0 --capture-only   or   $0 --power-on" >&2
    exit 1
  fi
fi

# Interactive pygame window.
unset HEADLESS SDL_VIDEODRIVER SDL_AUDIODRIVER SDL_SOFTWARE_RENDERER || true

echo "============================================================"
echo " Record Spring D1 town recon"
echo "   name  : $NAME"
if [[ "$POWER_ON" -eq 1 ]]; then
  echo "   start : power-on (clean natural entry)"
else
  echo "   state : $STATE"
fi
echo "   goal  : mask 0x3F → truck leave → farm sleep → D2"
echo "============================================================"
echo " Priority still open: flower-shop owner counter (bit 0x08)"
echo " HUD tracks Ann/Eve/Nina/owner/livestock/Maria bits live."
echo " F5 saves → tasks/${NAME}.json + ${NAME}_end.state"
echo "============================================================"
echo ""

if [[ "$POWER_ON" -eq 1 ]]; then
  run_py record --name "$NAME" --power-on --save-entry "$STATE"
  exit $?
fi

run_py record --name "$NAME" --state "$STATE"
exit $?
