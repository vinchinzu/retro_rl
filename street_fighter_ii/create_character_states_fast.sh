#!/bin/bash
# Fast character state creation using waypoint
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

# Check if waypoint exists
WAYPOINT="$SCRIPT_DIR/custom_integrations/StreetFighterIITurbo-Snes/CharSelect_StreetFighterIITurbo.state"
if [[ ! -f "$WAYPOINT" ]]; then
    echo "❌ Waypoint not found!"
    echo ""
    echo "You need to create a waypoint first:"
    echo "  ./create_waypoint.sh"
    echo ""
    exit 1
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SF2 TURBO FAST CHARACTER STATE CREATOR                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Waypoint loaded! This will be FAST."
echo ""
echo "Workflow for each character:"
echo "  1. Select character (use arrows + button)"
echo "  2. Wait for fight to start ('FIGHT!' appears)"
echo "  3. Press F1-F12 to save the character state"
echo "  4. Press R to RESET back to character select"
echo "  5. Repeat for next character!"
echo ""
echo "Character mapping:"
echo "  F1  = Ryu          F7  = Zangief"
echo "  F2  = E.Honda      F8  = Dhalsim"
echo "  F3  = Blanka       F9  = Balrog"
echo "  F4  = Guile        F10 = Vega"
echo "  F5  = Ken          F11 = Sagat"
echo "  F6  = Chun-Li      F12 = M.Bison"
echo ""
echo "🚀 With R reset, you can create all 12 states in ~5-10 minutes!"
echo ""
echo "Press ENTER to start..."
read

"$VENV_PYTHON" "$SCRIPT_DIR/manual_state_creator.py" --from-waypoint
