#!/usr/bin/env bash
# Record a human hot-spring *bath* from the outdoor mountain pond.
#
# Default start: mountain_fish_power_berry_end — already on mountain 0x10 at
# ~(686,411) / tile(42,25), a few steps from the tent/posts pond lip.
# (Not the west cave 0x29.)
#
# Usage:
#   ./scripts/record_hot_spring_bath.sh
#   ./scripts/record_hot_spring_bath.sh --name my_bath
#   ./scripts/record_hot_spring_bath.sh --state run_to_spa_end
#
# Controls (interactive window):
#   Walk / face pond edge → try B (jump, no D-pad) and/or A
#   [ ] = speed down/up
#   F5  = save recording + end state (also F1 alias)
#   Esc / close window = quit without save (if you didn't hit F5)
#
# Outputs on F5:
#   tasks/<name>.json
#   tasks/<name>_end.state
#   custom_integrations/HarvestMoon-Snes/<name>_end.state (mirrored)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

NAME="hot_spring_bath"
# Closest outdoor pond stand we have on disk (camp pond, not cave).
STATE="mountain_fish_power_berry_end"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name|-n)
      NAME="${2:?}"
      shift 2
      ;;
    --state|-s)
      STATE="${2:?}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (try --help)" >&2
      exit 2
      ;;
  esac
done

STATE_PATH="custom_integrations/HarvestMoon-Snes/${STATE}.state"
if [[ ! -f "$STATE_PATH" ]]; then
  echo "Missing state: $STATE_PATH" >&2
  echo "Nearby options:" >&2
  ls custom_integrations/HarvestMoon-Snes/*mountain*.state \
     custom_integrations/HarvestMoon-Snes/*spa*.state 2>/dev/null | sed 's|.*/||;s|\.state$||' || true
  exit 1
fi

# Unset headless so the pygame window is interactive.
unset HEADLESS SDL_VIDEODRIVER SDL_AUDIODRIVER SDL_SOFTWARE_RENDERER || true

echo "============================================================"
echo " Record outdoor hot-spring bath"
echo "   state  : $STATE"
echo "   name   : $NAME"
echo "   start  : mountain pond area (map 0x10) — NOT cave 0x29"
echo "============================================================"
echo " Goal: intentionally bathe / jump in the outdoor pond and"
echo "       restore stamina (tool-drain first if full)."
echo ""
echo " Tips (verified path: upper 0xF7 pond, NOT camp tent pond):"
echo "   - From fish spot: west along mid path, climb west, then NE to y~201"
echo "   - Stand A0 tile(38,12) ~(619,201); hold A+Right into water 0xF7 (39,12)"
echo "   - Walk through and A+Left back (player_action=3 while in water)"
echo "   - F5 saves → tasks/${NAME}.json + ${NAME}_end.state"
echo "============================================================"
echo ""

# Prefer monorepo env (stable-retro + pygame); fall back to harvest venv.
if command -v uv >/dev/null 2>&1; then
  exec env PYTHONPATH="${PYTHONPATH:-}:.." \
    uv run --project .. python -m harvest.runtime.harvest_bot play \
      --state "$STATE" \
      --record "$NAME" \
      --no-day-plan
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  exec "$ROOT/.venv/bin/python" -m harvest.runtime.harvest_bot play \
    --state "$STATE" \
    --record "$NAME" \
    --no-day-plan
fi

echo "Need uv or harvest/.venv — run: uv sync" >&2
exit 1
