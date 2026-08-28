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
# YouTube KPDR reference VODs + button extracts (gitignored).
YT_REFERENCE_DIR = GAME_DIR / "refs" / "yt_reference"
# Default VOD workspace (Kentroid KPDR 59:46.72) when --ref omitted.
YT_DEFAULT_REF_ID = "TFsGVxQReMw"
ROOM_CLEAR_POLICY_DIR = GAME_DIR / "policies" / "room_clears"
ROOM_CLEAR_RECORDINGS_DIR = RECORDINGS_DIR / "room_clears"
FULL_ROOM_GRAPH_PATH = MAPS_DIR / "full_room_graph.json"
ROOM_PROBLEMS_PATH = MAPS_DIR / "room_problems.json"
# Map Rando / sm-json-data canonical names (https://maprando.com/logic)
MAPRANDO_ROOM_CATALOG_PATH = MAPS_DIR / "maprando_room_catalog.json"
MAPRANDO_ROOM_NAMES_PATH = MAPS_DIR / "maprando_room_names.json"
SHARED_ROM = REPO_DIR / "roms" / "SuperMetroid.sfc"
# NTSC unheadered vanilla (tas/oracle + practice patch baseline).
VANILLA_ROM_SHA1 = "da957f0d63d14cb441d215462904c4fa8519c613"
# Community practice hack (tewtal) — build via scripts/setup_practice_rom.py.
# Product continuous/pure evidence still uses SHARED_ROM only.
SHARED_PRACTICE_ROM = REPO_DIR / "roms" / "SuperMetroid_Practice.sfc"
SHARED_PRACTICE_ROM_TINYSTATES = (
    REPO_DIR / "roms" / "SuperMetroid_Practice_tinystates.sfc"
)
# Practice-hack ROM integration (contractor / repertoire capture only).
PRACTICE_GAME = "SuperMetroid-Practice-Snes"
PRACTICE_INTEGRATION_DIR = GAME_DIR / "custom_integrations" / PRACTICE_GAME
PRACTICE_PRESET_ADDR_PATH = MAPS_DIR / "practice_preset_addresses.json"
# Full practice-hack preset menu/save repertoire (JSON catalog).
PRACTICE_REPERTOIRE_PATH = MAPS_DIR / "practice_repertoire.json"
# Canonical harness states/demos keyed by repertoire session id.
# Vanilla integration: product pins. Contractor captures go to the
# practice-ROM integration (PRACTICE_CONTRACTOR_STATE_DIR).
PRACTICE_REPERTOIRE_STATE_DIR = INTEGRATION_DIR / "practice_repertoire"
PRACTICE_CONTRACTOR_STATE_DIR = PRACTICE_INTEGRATION_DIR / "practice_repertoire"
PRACTICE_REPERTOIRE_DEMO_DIR = RECORDINGS_DIR / "practice_repertoire"
# Vendored TAS movies (Sniq any%/100%) + snes12_rle slices — see tas/README.md.
TAS_DIR = GAME_DIR / "tas"
TAS_REF_DIR = TAS_DIR / "ref"
TAS_SLICE_DIR = TAS_DIR / "slices"
