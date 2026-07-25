#!/bin/bash
# Test the MK2 cheat extractor
# Usage: ./test_extractor.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "No virtual environment found. Run: cd $ROOT_DIR && ./setup.sh"
    exit 1
fi

export SDL_VIDEODRIVER="dummy"
export SDL_AUDIODRIVER="dummy"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "Testing MK2 Cheat Extractor..."
echo ""
echo "Running a quick test for LiuKang (first 2 stages only)..."
echo ""

# Test with --start-from Fight and only run a few stages by limiting characters
"$VENV_PYTHON" "$SCRIPT_DIR/cheat_extractor.py" --char LiuKang --start-from Fight 2>&1 | head -50

echo ""
echo "Test complete! If you see no errors above, the extractor is working."
echo ""
echo "To extract all states for all 12 characters:"
echo "  ./extract_all_states.sh"
