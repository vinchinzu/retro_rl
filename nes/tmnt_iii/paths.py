"""Filesystem and integration constants for Teenage Mutant Ninja Turtles III: The Manhattan Project."""

from __future__ import annotations

from retro_harness.game_layout import game_paths

_paths = game_paths(__file__, "TeenageMutantNinjaTurtlesIII-Nes")
GAME_DIR = _paths.game_dir
REPO_ROOT = _paths.repo_root
INTEGRATION = _paths.integration
GAME = INTEGRATION
INTEGRATION_DIR = _paths.integration_dir
RECORDINGS_DIR = _paths.recordings_dir
ROMS_DIR = _paths.roms_dir
DOCS_DIR = _paths.docs_dir
SHARED_ROM_ZIP = (
    REPO_ROOT
    / "roms"
    / "Nintendo"
    / "NES"
    / "Teenage Mutant Ninja Turtles III - The Manhattan Project.zip"
)
LEVEL1_STATE = "Level1"
