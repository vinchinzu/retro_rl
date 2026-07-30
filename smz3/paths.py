"""Filesystem and integration constants for SMZ3."""

from __future__ import annotations

from pathlib import Path

from retro_harness.snes import GameSpec

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent
INTEGRATION = "SMZ3-Snes"
GAME = INTEGRATION
INTEGRATION_DIR = GAME_DIR / "custom_integrations" / INTEGRATION
RECORDINGS_DIR = GAME_DIR / "recordings"
ROMS_DIR = GAME_DIR / "roms"
SEEDS_DIR = GAME_DIR / "seeds"
DOCS_DIR = GAME_DIR / "docs"
REFS_DIR = GAME_DIR / "refs"
MAPS_DIR = GAME_DIR / "maps"

# Shared vanilla ROMs (same bytes as alttp / super_metroid setups).
SHARED_SM_ROM = REPO_ROOT / "roms" / "SuperMetroid.sfc"
SHARED_Z3_ROM = REPO_ROOT / "roms" / "zelda3.sfc"

# Local copies under smz3/roms/ (symlinks via setup_roms.py).
LOCAL_SM_ROM = ROMS_DIR / "SuperMetroid.sfc"
LOCAL_Z3_ROM = ROMS_DIR / "zelda3.sfc"

# Combo base IPS from tewtal SMZ3Randomizer web client resources.
# GameVersion > 11.2 uses zsm.ips; older seeds need zsm.v11.2.ips.
BASE_IPS_GZ = REFS_DIR / "zsm.ips.gz"
BASE_IPS_URL = (
    "https://raw.githubusercontent.com/tewtal/SMZ3Randomizer/master/"
    "WebRandomizer/ClientApp/src/resources/zsm.ips.gz"
)
RANDOMIZER_REPO = "https://github.com/tewtal/SMZ3Randomizer.git"
# Optional local clone for offline CLI builds (gitignored).
RANDOMIZER_DIR = REFS_DIR / "SMZ3Randomizer"

# Known-good fixture seed (deterministic settings; regenerated via generate_seed).
TEST_SEED_DIR = SEEDS_DIR / "test_seed"
TEST_SEED_NUMBER = "1337"

# Combined ROM is ExHiROM-style 6 MiB.
COMBO_ROM_SIZE = 0x600000

# Room timeout: dwell longer than this multiple of the baseline → game over.
ROOM_TIMEOUT_MULTIPLIER = 3.0

GAME_SPEC = GameSpec(GAME, GAME_DIR)
