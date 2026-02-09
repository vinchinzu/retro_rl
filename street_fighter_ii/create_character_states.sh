#!/bin/bash
# Create character starting states for SF2 Turbo (all 12 World Warriors)
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
echo "║  SF2 TURBO CHARACTER STATE CREATOR                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "This tool helps you create starting states for all 12 World Warriors."
echo ""
echo "Instructions:"
echo "  1. Navigate through boot/menus (use TAB to speed up)"
echo "  2. Select the character you want"
echo "  3. When the first fight starts, press F1-F12 to save"
echo ""
echo "Character mapping:"
echo "  F1  = Ryu          F7  = Zangief"
echo "  F2  = E.Honda      F8  = Dhalsim"
echo "  F3  = Blanka       F9  = Balrog (Boxer)"
echo "  F4  = Guile        F10 = Vega (Claw)"
echo "  F5  = Ken          F11 = Sagat"
echo "  F6  = Chun-Li      F12 = M.Bison (Dictator)"
echo ""
echo "Press ENTER to continue..."
read

"$VENV_PYTHON" "$SCRIPT_DIR/manual_state_creator.py" --from-start
