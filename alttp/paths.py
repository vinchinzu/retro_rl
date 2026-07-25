"""Filesystem constants for A Link to the Past."""

from __future__ import annotations

from pathlib import Path

from retro_harness.snes import GameSpec

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent
INTEGRATION = "Zelda3-Snes"
GAME = INTEGRATION
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / INTEGRATION
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
DOCS_DIR = GAME_DIR / "docs"
BOOT_STATE = "YazeSlot000"
LINKS_HOUSE_OVERWORLD_STATE = "LinksHouseOverworld"
HYRULE_CASTLE_GROUNDS_STATE = "HyruleCastleGrounds"
FIRST_ACTION_STATE = "FirstAction"

GAME_SPEC = GameSpec(GAME, GAME_DIR)
