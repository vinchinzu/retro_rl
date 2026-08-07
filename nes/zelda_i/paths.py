"""Filesystem and integration constants for The Legend of Zelda."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent.parent  # monorepo root (game under snes/ or nes/)
GAME = "LegendOfZelda-Nes"
RECORDINGS_DIR = GAME_DIR / "recordings"
ROOM_TIMINGS_DIR = RECORDINGS_DIR / "room_timings"
ROMS_DIR = GAME_DIR / "roms"
MODELS_DIR = GAME_DIR / "models"
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / "Legend of Zelda, The.zip"
LEVEL1_STATE = "Level1"
TAS_DIR = GAME_DIR / "tas"
TAS_REF_DIR = TAS_DIR / "ref"
