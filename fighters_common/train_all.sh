#!/bin/bash
set -euo pipefail

# Train PPO agents for all fighting games.
# Creates save states if needed, then runs training.
#
# Usage:
#   ./train_all.sh                    # Train all games (500k steps each)
#   ./train_all.sh --steps 100000     # Custom step count
#   ./train_all.sh --game sf2         # Single game only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "No venv found. Run: cd $ROOT_DIR && ./setup.sh"
    exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export SDL_VIDEODRIVER="dummy"
export SDL_AUDIODRIVER="dummy"

STEPS="${1:-500000}"
GAME="${2:-all}"

GAMES=("sf2" "ssf2" "mk1" "mk2")

if [[ "$GAME" != "all" ]]; then
    GAMES=("$GAME")
fi

for g in "${GAMES[@]}"; do
    echo "============================================================"
    echo "Creating save state for: $g"
    echo "============================================================"
    "$VENV_PYTHON" "$SCRIPT_DIR/create_all_states.py" --game "$g" || true

    echo ""
    echo "============================================================"
    echo "Training PPO for: $g ($STEPS steps)"
    echo "============================================================"

    # Find the state file
    STATE="NONE"
    case "$g" in
        sf2)  STATE="Fight_StreetFighterIITurbo" ;;
        ssf2) STATE="Fight_SuperStreetFighterII" ;;
        mk1)  STATE="Fight_MortalKombat" ;;
        mk2)  STATE="Fight_MortalKombatII" ;;
    esac

    "$VENV_PYTHON" "$SCRIPT_DIR/train_ppo.py" \
        --game "$g" \
        --state "$STATE" \
        --steps "$STEPS" \
        --n-envs 4

    echo ""
done

echo "All training complete!"
