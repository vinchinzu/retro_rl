"""Filesystem and integration constants for Teenage Mutant Ninja Turtles II: The Arcade Game."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent.parent  # monorepo root (game under snes/ or nes/)
GAME = "TeenageMutantNinjaTurtlesII-Nes"
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / "Teenage Mutant Ninja Turtles II - The Arcade Game.zip"
LEVEL1_STATE = "Level1"
