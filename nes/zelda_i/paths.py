"""Filesystem and integration constants for The Legend of Zelda."""

from __future__ import annotations

from retro_harness.game_layout import game_paths

_paths = game_paths(__file__, "LegendOfZelda-Nes")
GAME_DIR = _paths.game_dir
REPO_ROOT = _paths.repo_root
INTEGRATION = _paths.integration
GAME = INTEGRATION
INTEGRATION_DIR = _paths.integration_dir
RECORDINGS_DIR = _paths.recordings_dir
ROMS_DIR = _paths.roms_dir
DOCS_DIR = _paths.docs_dir
ROOM_TIMINGS_DIR = RECORDINGS_DIR / "room_timings"
MODELS_DIR = GAME_DIR / "models"
SHARED_ROM_ZIP = REPO_ROOT / "roms" / "Nintendo" / "NES" / "Legend of Zelda, The.zip"
LEVEL1_STATE = "Level1"
TAS_DIR = GAME_DIR / "tas"
TAS_REF_DIR = TAS_DIR / "ref"
