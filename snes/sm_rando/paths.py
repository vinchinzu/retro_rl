"""Filesystem and integration constants for Super Metroid Randomizer."""

from __future__ import annotations

from pathlib import Path

from retro_harness.snes import GameSpec

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent.parent
INTEGRATION = "SMRando-Snes"
GAME = INTEGRATION
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / INTEGRATION
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
SEEDS_DIR = GAME_DIR / "seeds"
DOCS_DIR = GAME_DIR / "docs"
REFS_DIR = GAME_DIR / "refs"
MAPS_DIR = GAME_DIR / "maps"

# Vanilla SM dump (same as super_metroid / smz3).
SHARED_SM_ROM = REPO_ROOT / "roms" / "SuperMetroid.sfc"
LOCAL_SM_ROM = ROMS_DIR / "SuperMetroid.sfc"
SM_XXH32 = 0xCADB4883
# Official Super Metroid (JU) unheadered SHA1 used by stable-retro integrations.
SM_SHA1 = "da957f0d63d14cb441d215462904c4fa8519c613"

# Power-on → first controllable frame save state (Ceres elevator on vanilla).
FIRST_PLAY_STATE = "FirstPlay"

# Vanilla package (skill library substrate — do not fork).
VANILLA_PACKAGE = "super_metroid"
VANILLA_DIR = REPO_ROOT / "snes" / "super_metroid"

TEST_SEED_DIR = SEEDS_DIR / "test_seed"
TEST_SEED_NUMBER = "1337"
DEMO_SEED_DIR = SEEDS_DIR / "demo_seed"
DEMO_SEED_NUMBER = "demo"

GAME_SPEC = GameSpec(GAME, GAME_DIR)
