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
REFS_DIR = GAME_DIR / "refs"
BOOT_STATE = "YazeSlot000"
LINKS_HOUSE_OVERWORLD_STATE = "LinksHouseOverworld"
HYRULE_CASTLE_GROUNDS_STATE = "HyruleCastleGrounds"
FIRST_ACTION_STATE = "FirstAction"
# Dev checkpoint after uncle / fighter-sword (state-load only; not natural-chain proof).
FIGHTER_SWORD_STATE = "FighterSword"

# Local clone of vg-json-data/z3-json-data (gitignored; setup script only).
Z3_JSON_DATA_DIR = REFS_DIR / "z3-json-data"
Z3_JSON_DATA_REPO = "https://github.com/vg-json-data/z3-json-data.git"
# Pinned upstream revision for reproducible local checkouts.
Z3_JSON_DATA_PIN = "1eb7a785bda0d671136316c24f223c7ce12257e6"

GAME_SPEC = GameSpec(GAME, GAME_DIR)
