"""Filesystem and integration constants for ALTTP Randomizer.

Uses **Japanese 1.0** ALttP only (same dump as SMZ3 / samus.link).
Do NOT wire the USA dump at ``roms/zelda3.sfc`` (that is for ``alttp/``).
"""

from __future__ import annotations

from pathlib import Path

from retro_harness.snes import GameSpec

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent.parent
INTEGRATION = "ALTTPRando-Snes"
GAME = INTEGRATION
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / INTEGRATION
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
SEEDS_DIR = GAME_DIR / "seeds"
DOCS_DIR = GAME_DIR / "docs"
REFS_DIR = GAME_DIR / "refs"
MAPS_DIR = GAME_DIR / "maps"

# Japanese 1.0 only (samus.link xxHash32 seed SMZ3 = 0x534D5A33 → 0x8AC8FD15).
SHARED_Z3_JP_ROM = REPO_ROOT / "roms" / "zelda3_jp.sfc"
# Alias used by local setup; always JP.
SHARED_Z3_ROM = SHARED_Z3_JP_ROM
LOCAL_Z3_ROM = ROMS_DIR / "zelda3_jp.sfc"

# USA dump path — never use as primary integration ROM.
SHARED_Z3_US_ROM = REPO_ROOT / "roms" / "zelda3.sfc"

# samus.link Upload.jsx (xxHash32 seed "SMZ3" = 0x534D5A33).
Z3_JP_XXH32 = 0x8AC8FD15  # ALttP JP 1.0 unheadered 1 MiB
Z3_JP_SHA1 = "e7e852f0159ce612e3911164878a9b08b3cb9060"

# First controllable frame after title/file/intro (saved by boot).
FIRST_PLAY_STATE = "FirstPlay"

VANILLA_PACKAGE = "alttp"
VANILLA_DIR = REPO_ROOT / "snes" / "alttp"

TEST_SEED_DIR = SEEDS_DIR / "test_seed"
TEST_SEED_NUMBER = "1337"
DEMO_SEED_DIR = SEEDS_DIR / "demo_seed"

GAME_SPEC = GameSpec(GAME, GAME_DIR)
