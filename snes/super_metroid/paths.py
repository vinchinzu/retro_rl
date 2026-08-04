"""Filesystem and integration constants for Super Metroid."""

from __future__ import annotations

from pathlib import Path

GAME = "SuperMetroid-Snes"
GAME_DIR = Path(__file__).resolve().parent
# Game lives at snes/super_metroid/; monorepo root is two levels up.
REPO_DIR = GAME_DIR.parent.parent
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / GAME
# Ephemeral probe save-states (gitignored). Named anchors stay in INTEGRATION_DIR.
SCRATCH_STATE_DIR = INTEGRATION_DIR / "scratch"
POLICY_DIR = GAME_DIR / "policies" / "morph"
EARLY_POLICY_DIR = GAME_DIR / "policies" / "early_game"
RECORDINGS_DIR = GAME_DIR / "recordings"
ROOM_TIMINGS_DIR = RECORDINGS_DIR / "room_timings"
MODELS_DIR = GAME_DIR / "models"
MAPS_DIR = GAME_DIR / "maps"
DEBUG_DIR = GAME_DIR / "debug"
DOCS_DIR = GAME_DIR / "docs"
ROOM_CLEAR_POLICY_DIR = GAME_DIR / "policies" / "room_clears"
ROOM_CLEAR_RECORDINGS_DIR = RECORDINGS_DIR / "room_clears"
FULL_ROOM_GRAPH_PATH = MAPS_DIR / "full_room_graph.json"
ROOM_PROBLEMS_PATH = MAPS_DIR / "room_problems.json"
SHARED_ROM = REPO_DIR / "roms" / "SuperMetroid.sfc"
