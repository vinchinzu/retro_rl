#!/bin/bash
# Extract MK2 tournament states for all characters
# Usage: ./extract_all_states.sh [start_from]
# Example: ./extract_all_states.sh Match5

cd "$(dirname "$0")"

START_FROM="${1:-}"

echo "========================================"
echo "MK2 Tournament State Extraction"
echo "========================================"
echo ""
echo "This will extract states for all 12 characters through the tournament:"
echo "  - Match 1-8 (regular fights)"
echo "  - Shang Tsung (sub-boss)"
echo "  - Kintaro (boss 1)"
echo "  - Shao Kahn (final boss)"
echo ""
echo "Characters: LiuKang, KungLao, JohnnyCage, Reptile, SubZero, ShangTsung, Kitana, Jax, Mileena, Baraka, Scorpion, Raiden"
echo ""

if [ -n "$START_FROM" ]; then
    echo "Starting from: $START_FROM"
    echo ""
    python3 cheat_extractor.py --char LiuKang --start-from "$START_FROM"
else
    read -p "Extract for ALL characters? This will take ~30-60 minutes. (y/N) " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo ""
        echo "Starting extraction for all 12 characters..."
        python3 cheat_extractor.py --all-chars
    else
        echo "Cancelled."
        echo ""
        echo "To extract for a single character:"
        echo "  python3 cheat_extractor.py --char LiuKang"
        echo ""
        echo "To start from a specific stage:"
        echo "  python3 cheat_extractor.py --char LiuKang --start-from Match5"
        exit 0
    fi
fi
