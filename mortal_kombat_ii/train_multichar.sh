#!/bin/bash
# Train a generic multi-character fighter from scratch for MK2
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

cat << 'EOF'

╔═══════════════════════════════════════════════════════════╗
║  MK2 MULTI-CHARACTER TRAINING - PHASE 1                   ║
╚═══════════════════════════════════════════════════════════╝

This will train a GENERIC FIGHTER from scratch using all 12 MK2 characters.

Training Setup:
  • Characters: Liu Kang, Kung Lao, Johnny Cage, Reptile, Sub-Zero,
                Shang Tsung, Kitana, Jax, Mileena, Baraka, Scorpion, Raiden
  • Steps: 2,000,000 (default, ~8-12 hours on GPU)
  • Random character each episode
  • Learns universal fighting strategies

Models saved to: models/mk2_multichar_ppo_*.zip

Press ENTER to start training, or Ctrl+C to cancel...
EOF

read

echo ""
echo "Starting training..."
echo ""

"$VENV_PYTHON" "$SCRIPT_DIR/train_multi_character.py" \
    --game mk2 \
    --steps 2000000 \
    "$@"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Training Complete!                                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Watch the trained model: ./watch.sh"
echo "  2. Continue to Phase 2: ./train_multi_opponent.sh"
echo ""
