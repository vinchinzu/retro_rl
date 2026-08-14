"""Filesystem and integration constants for Super Mario Bros. (NES)."""

from __future__ import annotations

from pathlib import Path

from retro_harness.game_layout import game_paths

# Boot/probe integration (M1 scaffold).
_paths = game_paths(__file__, "SuperMarioBros-Nes")
GAME_DIR = _paths.game_dir
REPO_ROOT = _paths.repo_root
INTEGRATION = _paths.integration
GAME = INTEGRATION
INTEGRATION_DIR = _paths.integration_dir
RECORDINGS_DIR = _paths.recordings_dir
ROMS_DIR = _paths.roms_dir
DOCS_DIR = _paths.docs_dir

# Full-run / practice / optimizer integration (shared with snes_editor traces).
GAME_V0 = "SuperMarioBros-Nes-v0"
INTEGRATION_V0_DIR = GAME_DIR / "custom_integrations" / GAME_V0

MODELS_DIR = GAME_DIR / "models"
FULLGAME_RECORDINGS_DIR = RECORDINGS_DIR / "fullgame"
FULLGAME_REPLAYS_DIR = RECORDINGS_DIR / "fullgame_replays"
OPTIMIZER_RUNS_DIR = GAME_DIR / "optimizer" / "runs"
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
