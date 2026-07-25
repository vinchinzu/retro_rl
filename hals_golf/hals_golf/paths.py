"""Stable paths for the Hal's Hole in One Golf package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
MONOREPO_DIR = PROJECT_DIR.parent

CUSTOM_INTEGRATIONS_DIR = PROJECT_DIR / "custom_integrations"
GAME = "HalsHoleInOne-Snes"
GAME_DIR = CUSTOM_INTEGRATIONS_DIR / GAME
STATES_DIR = GAME_DIR
TASKS_DIR = PROJECT_DIR / "tasks"
SAVES_DIR = PROJECT_DIR / "saves"
ROMS_DIR = PROJECT_DIR / "roms"
SHARED_ROMS_DIR = MONOREPO_DIR / "roms"
DEBUG_FRAMES_DIR = PROJECT_DIR / "debug_frames"


def ensure_monorepo_on_path() -> None:
    """Make shared ``retro_harness`` imports available."""
    monorepo = str(MONOREPO_DIR)
    if monorepo not in sys.path:
        sys.path.insert(0, monorepo)
