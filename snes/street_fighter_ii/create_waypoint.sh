#!/bin/bash
# Create waypoint at character select (one-time setup)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "No virtual environment found. Run: cd $ROOT_DIR && ./setup.sh"
    exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "$ROOT_DIR"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SF2 TURBO WAYPOINT CREATOR                               ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "This creates a waypoint save at CHARACTER SELECT screen."
echo "You only need to do this ONCE per game."
echo ""
echo "Instructions:"
echo "  1. Hold TAB to speed through intro screens (10x!)"
echo "  2. Navigate to CHARACTER SELECT screen"
echo "  3. Press F1 to save the waypoint"
echo "  4. ESC to quit"
echo ""
echo "After creating the waypoint, use:"
echo "  ./create_character_states_fast.sh"
echo ""
echo "Press ENTER to continue..."
read

"$VENV_PYTHON" "$SCRIPT_DIR/create_waypoint.py"
