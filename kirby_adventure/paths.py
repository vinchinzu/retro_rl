"""Filesystem and integration constants for Kirby's Adventure."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent
GAME = "KirbysAdventure-Nes"
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / "Kirby's Adventure.zip"
LEVEL1_STATE = "Level1"
