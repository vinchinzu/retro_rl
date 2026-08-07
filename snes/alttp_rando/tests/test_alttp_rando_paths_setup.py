"""Offline path / setup checks for JP ROM wiring."""

from __future__ import annotations

from alttp_rando.paths import (
    DEMO_SEED_DIR,
    FIRST_PLAY_STATE,
    INTEGRATION_DIR,
    LOCAL_Z3_ROM,
    SHARED_Z3_JP_ROM,
    Z3_JP_SHA1,
    Z3_JP_XXH32,
)


def test_demo_seed_dir_name() -> None:
    assert DEMO_SEED_DIR.name == "demo_seed"


def test_local_rom_is_jp() -> None:
    assert LOCAL_Z3_ROM.name == "zelda3_jp.sfc"
    assert SHARED_Z3_JP_ROM.name == "zelda3_jp.sfc"


def test_documented_hashes() -> None:
    assert Z3_JP_XXH32 == 0x8AC8FD15
    assert len(Z3_JP_SHA1) == 40


def test_integration_layout_constants() -> None:
    assert INTEGRATION_DIR.name == "ALTTPRando-Snes"
    assert FIRST_PLAY_STATE == "FirstPlay"
