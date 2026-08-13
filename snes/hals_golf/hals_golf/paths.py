"""Stable paths for the Hal's Hole in One Golf package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
# Nested package: snes/hals_golf/hals_golf/ → monorepo is two parents above PROJECT_DIR.
MONOREPO_DIR = PROJECT_DIR.parent.parent

# ``./run_bot.sh`` only puts the game workspace + ``snes/`` on PYTHONPATH.
# Insert the monorepo root before importing shared layout helpers.
for _bootstrap in (MONOREPO_DIR, PROJECT_DIR):
    _text = str(_bootstrap)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from retro_harness.game_layout import game_paths  # noqa: E402

_paths = game_paths(__file__, "HalsHoleInOne-Snes", workspace_parent=True)
PROJECT_DIR = _paths.game_dir
MONOREPO_DIR = _paths.repo_root
GAME = _paths.integration
GAME_DIR = _paths.integration_dir
CUSTOM_INTEGRATIONS_DIR = PROJECT_DIR / "custom_integrations"
RECORDINGS_DIR = _paths.recordings_dir
ROMS_DIR = _paths.roms_dir
DOCS_DIR = _paths.docs_dir
STATES_DIR = GAME_DIR
TASKS_DIR = PROJECT_DIR / "tasks"
SAVES_DIR = PROJECT_DIR / "saves"
SHARED_ROMS_DIR = MONOREPO_DIR / "roms"
DEBUG_FRAMES_DIR = PROJECT_DIR / "debug_frames"


def ensure_monorepo_on_path() -> None:
    """Make shared ``retro_harness`` + nested hals_golf imports available."""
    for path in (MONOREPO_DIR, PROJECT_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    from retro_harness.repo import ensure_import_paths

    ensure_import_paths(root=MONOREPO_DIR)
