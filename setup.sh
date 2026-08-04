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

# Game packages live under snes/ and nes/ but keep import names (alttp, smb, …).
# Install a site .pth so `uv run python -c "import alttp"` works from monorepo root.
if [[ ! -f "$SCRIPT_DIR/.env" && -f "$SCRIPT_DIR/.env.example" ]]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "Created .env from .env.example."
fi
SITE_PACKAGES="$(
    uv run python -c "import site; print(site.getsitepackages()[0])"
)"
cat > "$SITE_PACKAGES/retro_rl_paths.pth" <<EOF
$SCRIPT_DIR
$SCRIPT_DIR/snes
$SCRIPT_DIR/nes
$SCRIPT_DIR/snes/harvest
$SCRIPT_DIR/snes/hals_golf
EOF
echo "Installed $SITE_PACKAGES/retro_rl_paths.pth"

echo ""
echo "Setup complete!"
echo ""
echo "Available games:"
for console in snes nes; do
    for game_dir in "$SCRIPT_DIR/$console"/*/custom_integrations; do
        if [[ -d "$game_dir" ]]; then
            game=$(basename "$(dirname "$game_dir")")
            echo "  - $console/$game"
        fi
    done
done
echo ""
echo "To run a game:"
echo "  uv run python snes/<game>/scripts/...   # or nes/<game>/..."
