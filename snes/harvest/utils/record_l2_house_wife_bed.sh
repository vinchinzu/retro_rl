#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_NAME="${1:-${STATE_NAME:-latest}}"
TASK_NAME="${2:-${TASK_NAME:-l2_house_wife_bed}}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/debug_alignment/$TASK_NAME}"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

cat <<EOF
[L2 HOUSE RECORDING]
State: $STATE_NAME
Task:  $TASK_NAME

Use the canonical recorder. Press F5 after the wife talk + sleep flow has
finished, preferably after the next morning or sleep transition is visible.

Before sleeping, trace the L2 house floor by walking every reachable tile you
care about. The post-record step extracts tilemap 0x17 player tiles and A-press
windows for wife/bed landmarks.
EOF

uv run python harvest_bot.py play --state "$STATE_NAME" --record "$TASK_NAME" --no-day-plan

uv run python utils/extract_recording_walkable_tiles.py "$TASK_NAME" \
  --tilemap 0x17 \
  --out "$OUT_DIR/walkable_tiles.json" \
  --text-out "$OUT_DIR/walkable_tiles.txt"

uv run python utils/task_replay_probe.py "$TASK_NAME" \
  --watch tilemap,input_lock,player_state,player_action,wife_pregnancy,day,hour,minute \
  --out "$OUT_DIR/replay_probe.jsonl"

cat <<EOF

[L2 HOUSE RECORDING DONE]
Walkable tile summary: $OUT_DIR/walkable_tiles.txt
JSON summary:           $OUT_DIR/walkable_tiles.json
Replay probe:           $OUT_DIR/replay_probe.jsonl
Task JSON:              $REPO_DIR/tasks/$TASK_NAME.json
End state:              $REPO_DIR/tasks/${TASK_NAME}_end.state
EOF
