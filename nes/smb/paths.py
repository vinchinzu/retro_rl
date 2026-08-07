"""Filesystem and integration constants for Super Mario Bros. (NES)."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent.parent  # monorepo root (game under snes/ or nes/)
# Boot/probe integration (M1 scaffold).
GAME = "SuperMarioBros-Nes"
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME

# Full-run / practice / optimizer integration (shared with snes_editor traces).
GAME_V0 = "SuperMarioBros-Nes-v0"
INTEGRATION_V0_DIR = GAME_DIR / "custom_integrations" / GAME_V0

RECORDINGS_DIR = GAME_DIR / "recordings"
MODELS_DIR = GAME_DIR / "models"
FULLGAME_RECORDINGS_DIR = RECORDINGS_DIR / "fullgame"
FULLGAME_REPLAYS_DIR = RECORDINGS_DIR / "fullgame_replays"
OPTIMIZER_RUNS_DIR = GAME_DIR / "optimizer" / "runs"
ROMS_DIR = GAME_DIR / "roms"
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / "Super Mario Bros..zip"
LEVEL1_STATE = "Level1"

# Historical absolute roots used inside older branch JSON `state_file` fields.
# Resolution rewrites these prefixes onto the live snes_editor tree.
LEGACY_SMB_ROOTS = (
    Path("/home/v/01_projects/11_games/speedrun/retro_rl/super_mario_bros"),
    Path("/home/v/01_projects/11_games/retro_rl/super_mario_bros"),
)
SNES_EDITOR_SMB_ROOT = (
    REPO_ROOT.parent / "snes_editor" / "super_mario_bros"
)
