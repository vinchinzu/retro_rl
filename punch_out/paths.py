"""Filesystem and integration constants for Mike Tyson's Punch-Out!!."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent
GAME = "PunchOut-Nes"
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / "Mike Tyson's Punch-Out!!.zip"
LEVEL1_STATE = "Level1"
MATCH1_STATE = "Match1"
