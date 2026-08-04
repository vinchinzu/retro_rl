"""Stable paths for the Hal's Hole in One Golf package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
# Game lives at snes/hals_golf/; monorepo root is two levels up from PROJECT_DIR.
MONOREPO_DIR = PROJECT_DIR.parent.parent

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
    for path in (MONOREPO_DIR, PROJECT_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
