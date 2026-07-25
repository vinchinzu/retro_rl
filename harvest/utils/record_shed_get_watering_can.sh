#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

STATE_NAME="${1:-Y1_After_Two_Berries_Before_Shed_Water_Can}"
TASK_NAME="${2:-shed_get_watering_can}"

echo "Recording task: $TASK_NAME"
echo "Start state: $STATE_NAME"
echo "Controls: F5 save recording, ESC cancel"

exec "$REPO_DIR/run_bot.sh" play --state "$STATE_NAME" --record "$TASK_NAME" --no-day-plan
