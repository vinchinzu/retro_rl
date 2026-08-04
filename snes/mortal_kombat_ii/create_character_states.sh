#!/bin/bash
# Create character starting states for MK2 (all 12 characters)
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
echo "║  MK2 CHARACTER STATE CREATOR                              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "This tool helps you create starting states for all 12 MK2 characters."
echo ""
echo "Instructions:"
echo "  1. Navigate through boot/menus (use TAB to speed up)"
echo "  2. Select the character you want"
echo "  3. When the first fight starts, press F1-F12 to save"
echo ""
echo "Character mapping:"
echo "  F1  = Liu Kang       F7  = Kitana"
echo "  F2  = Kung Lao       F8  = Jax"
echo "  F3  = Johnny Cage    F9  = Mileena"
echo "  F4  = Reptile        F10 = Baraka"
echo "  F5  = Sub-Zero       F11 = Scorpion"
echo "  F6  = Shang Tsung    F12 = Raiden"
echo ""
echo "Press ENTER to continue..."
read

"$VENV_PYTHON" "$SCRIPT_DIR/manual_state_creator.py" --from-start
