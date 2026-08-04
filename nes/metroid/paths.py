"""Filesystem and integration constants for Metroid (NES)."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent.parent  # monorepo root (game under snes/ or nes/)
GAME = "Metroid-Nes"
RECORDINGS_DIR = GAME_DIR / "recordings"
SCREEN_TIMINGS_DIR = RECORDINGS_DIR / "screen_timings"
ROMS_DIR = GAME_DIR / "roms"
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / "Metroid.zip"
LEVEL1_STATE = "Level1"
