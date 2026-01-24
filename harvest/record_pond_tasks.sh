#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="${1:-Y1_Spring_Day01_06h}"

echo "Recording pond toss tasks from state: ${STATE}"
echo "1) toss_bush_pond"
"$SCRIPT_DIR/run_recorder.sh" record toss_bush_pond --state "${STATE}"

echo "2) toss_stone_pond"
"$SCRIPT_DIR/run_recorder.sh" record toss_stone_pond --state "${STATE}"
