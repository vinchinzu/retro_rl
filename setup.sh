#!/bin/bash
# Setup shared retro_rl environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Setting up retro_rl shared environment..."

# Create venv if needed
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

# Install dependencies
echo "Installing dependencies..."
uv sync

echo ""
echo "Setup complete!"
echo ""
echo "Available games:"
for game_dir in "$SCRIPT_DIR"/*/custom_integrations; do
    if [[ -d "$game_dir" ]]; then
        game=$(basename "$(dirname "$game_dir")")
        echo "  - $game"
    fi
done
echo ""
echo "To run a game:"
echo "  cd <game_directory> && ./run_bot.sh"
