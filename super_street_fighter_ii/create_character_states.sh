#!/bin/bash
# Create character starting states for SSF2 (16 characters in 2 batches)
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

# Parse batch argument
BATCH="${1:-1}"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SSF2 CHARACTER STATE CREATOR (Batch $BATCH)                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

if [[ "$BATCH" == "1" ]]; then
    echo "This tool creates states for the original 12 World Warriors."
    echo ""
    echo "Character mapping:"
    echo "  F1  = Ryu          F7  = Zangief"
    echo "  F2  = E.Honda      F8  = Dhalsim"
    echo "  F3  = Blanka       F9  = Balrog"
    echo "  F4  = Guile        F10 = Vega"
    echo "  F5  = Ken          F11 = Sagat"
    echo "  F6  = Chun-Li      F12 = M.Bison"
elif [[ "$BATCH" == "2" ]]; then
    echo "This tool creates states for the 4 New Challengers."
    echo ""
    echo "Character mapping:"
    echo "  F1 = Cammy"
    echo "  F2 = Fei Long"
    echo "  F3 = Dee Jay"
    echo "  F4 = T.Hawk"
else
    echo "Invalid batch number. Use: $0 [1|2]"
    exit 1
fi

echo ""
echo "Instructions:"
echo "  1. Navigate through boot/menus (use TAB to speed up)"
echo "  2. Select the character you want"
echo "  3. When the first fight starts, press the F-key to save"
echo ""
echo "After batch 1, run: $0 2"
echo ""
echo "Press ENTER to continue..."
read

"$VENV_PYTHON" "$SCRIPT_DIR/manual_state_creator.py" --from-start --batch "$BATCH"
