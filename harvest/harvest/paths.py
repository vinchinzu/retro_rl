"""Repository paths used by the Harvest Moon package.

Runtime code needs stable access to game-local assets even after modules move
between subpackages. Keep path discovery here instead of deriving data
locations from individual module files.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
MONOREPO_DIR = PROJECT_DIR.parent

CUSTOM_INTEGRATIONS_DIR = PROJECT_DIR / "custom_integrations"
GAME = "HarvestMoon-Snes"
GAME_DIR = CUSTOM_INTEGRATIONS_DIR / GAME
STATES_DIR = GAME_DIR
TASKS_DIR = PROJECT_DIR / "tasks"
SAVES_DIR = PROJECT_DIR / "saves"
MAPS_DIR = PROJECT_DIR / "maps"
DEBUG_ALIGNMENT_DIR = PROJECT_DIR / "debug_alignment"
DECOMP_DIR = PROJECT_DIR / "HM-Decomp"
ROMS_DIR = PROJECT_DIR / "roms"
SHARED_ROMS_DIR = MONOREPO_DIR / "roms"


def ensure_monorepo_on_path() -> None:
    """Make shared ``retro_harness`` imports available from game-local commands."""
    monorepo = str(MONOREPO_DIR)
    if monorepo not in sys.path:
        sys.path.insert(0, monorepo)
