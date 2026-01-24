#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"

BASE_STATE="Y1_Spring_Day01_06h"

echo "Building save states from $BASE_STATE..."

$PY "$SCRIPT_DIR/state_builder.py" --base "$BASE_STATE" --tasks get_sickle --save Y1_After_Get_Sickle
$PY "$SCRIPT_DIR/state_builder.py" --base "$BASE_STATE" --tasks get_hammer --save Y1_After_Get_Hammer
$PY "$SCRIPT_DIR/state_builder.py" --base "$BASE_STATE" --tasks get_axe --save Y1_After_Get_Axe
$PY "$SCRIPT_DIR/state_builder.py" --base "$BASE_STATE" --tasks ship_berry --save Y1_After_Ship_Berry
$PY "$SCRIPT_DIR/state_builder.py" --base "$BASE_STATE" --tasks toss_bush_pond --save Y1_After_Toss_Bush
$PY "$SCRIPT_DIR/state_builder.py" --base "$BASE_STATE" --tasks get_sickle,toss_bush_pond --save Y1_L6_Mixed --find-liftable-after --radius 12

echo "Searching for a liftable-nearby state during toss_bush_pond..."
$PY "$SCRIPT_DIR/state_builder.py" --base "$BASE_STATE" --tasks toss_bush_pond --save Y1_Liftable_Nearby --find-liftable --radius 2

echo "Done."
