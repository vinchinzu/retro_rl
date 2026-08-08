"""Filesystem constants for A Link to the Past."""

from __future__ import annotations

from pathlib import Path

from retro_harness.snes import GameSpec

GAME_DIR = Path(__file__).resolve().parent
# Game lives at snes/alttp/; monorepo root is two levels up.
REPO_ROOT = GAME_DIR.parent.parent
INTEGRATION = "Zelda3-Snes"
GAME = INTEGRATION
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / INTEGRATION
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
DOCS_DIR = GAME_DIR / "docs"
REFS_DIR = GAME_DIR / "refs"
# Measured room geometry (JSON authority for room_engine / room_sense).
MAPS_DIR = GAME_DIR / "maps"
BOOT_STATE = "YazeSlot000"
LINKS_HOUSE_OVERWORLD_STATE = "LinksHouseOverworld"
# Controllable on screen 0x1B spawn — NOT bridge-turn or secret-hole approach.
# Semantic name: HyruleCastle_GroundsSpawn_Controllable (see opening_route.anchors).
HYRULE_CASTLE_GROUNDS_STATE = "HyruleCastleGrounds"
FIRST_ACTION_STATE = "FirstAction"
# Dev checkpoint after uncle / fighter-sword (state-load only; not natural-chain proof).
# Semantic name: HyruleCastle_SecretEntrance_FighterSword.
FIGHTER_SWORD_STATE = "FighterSword"

# Semantic aliases (meaning-bearing; filenames stay short for integration).
STATE_SEMANTIC_NAMES: dict[str, str] = {
    HYRULE_CASTLE_GROUNDS_STATE: "HyruleCastle_GroundsSpawn_Controllable",
    FIGHTER_SWORD_STATE: "HyruleCastle_SecretEntrance_FighterSword",
    BOOT_STATE: "TitleBoot_YazeSlot000",
    FIRST_ACTION_STATE: "HyruleCastle_FirstAction_Dev",
}

# Sanctuary-path save-state work queue artifacts (see opening_route.work_queue).
ROOM_WORK_QUEUE_JSON = RECORDINGS_DIR / "room_work_queue.json"
ROOM_WORK_QUEUE_MD = DOCS_DIR / "routes" / "ROOM_WORK_QUEUE.md"

# Local clone of vg-json-data/z3-json-data (gitignored under refs/; setup script).
# Optional committed/workspace copy at GAME_DIR/z3-json-data is also accepted
# (see resolve in z3_json_data.default_data_root) — same US/JP vanilla labels.
Z3_JSON_DATA_DIR = REFS_DIR / "z3-json-data"
Z3_JSON_DATA_WORKSPACE_DIR = GAME_DIR / "z3-json-data"
Z3_JSON_DATA_REPO = "https://github.com/vg-json-data/z3-json-data.git"
# Pinned upstream revision for reproducible local checkouts.
Z3_JSON_DATA_PIN = "1eb7a785bda0d671136316c24f223c7ce12257e6"

GAME_SPEC = GameSpec(GAME, GAME_DIR)
