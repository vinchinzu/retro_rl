"""Filesystem and integration constants for F-Zero."""

from __future__ import annotations

from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent.parent  # monorepo root (game under snes/ or nes/)
INTEGRATION = "FZero-Snes"
GAME = INTEGRATION
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / INTEGRATION
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
DOCS_DIR = GAME_DIR / "docs"
MUTE_CITY_STATE = "MuteCity"
MUTE_CITY_RUNNING_STATE = "MuteCityRunning"

