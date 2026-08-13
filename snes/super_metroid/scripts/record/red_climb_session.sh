#!/usr/bin/env bash
# Multi-take Red Tower → Hellway human session (rr-av5s).
# Usage (from repo root):
#   ./snes/super_metroid/scripts/record/red_climb_session.sh
#   ./snes/super_metroid/scripts/record/red_climb_session.sh human
#   ./snes/super_metroid/scripts/record/red_climb_session.sh pure red_climb_v2
#   ./snes/super_metroid/scripts/record/red_climb_session.sh rank red_climb_v1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"

MODE="${1:-pure}"
SERIES="${2:-red_climb_v1}"

case "$MODE" in
  pure|p)
    # Dual pure Bat→Red pin — stable for bot-parity climbs
    uv run python snes/super_metroid/scripts/record/practice_takes.py \
      --segment red-to-hellway \
      --series "$SERIES"
    ;;
  human|h)
    # Live warehouse_to_red room_enter f2012
    uv run python snes/super_metroid/scripts/record/practice_takes.py \
      --segment red-to-hellway-human \
      --series "${SERIES%_v*}_human_v1"
    ;;
  rank|r)
    uv run python snes/super_metroid/scripts/tools/rank_red_climb_takes.py \
      --series "$SERIES" \
      --write-manifest "snes/super_metroid/tasks/${SERIES}/splice_manifest.json"
    ;;
  list|l)
    uv run python snes/super_metroid/scripts/record/practice_takes.py \
      --series "$SERIES" --list
    ;;
  *)
    echo "usage: $0 [pure|human|rank|list] [series_stem]"
    echo "  pure  (default) multi-take from post_ice_bat_to_red_pure"
    echo "  human           multi-take from warehouse_to_red live Red enter"
    echo "  rank  SERIES    rank takes + write splice_manifest.json"
    echo "  list  SERIES    list saved takes"
    exit 1
    ;;
esac
