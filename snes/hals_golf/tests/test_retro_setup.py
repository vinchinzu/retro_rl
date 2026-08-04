"""Tests for ROM registration helpers."""

from __future__ import annotations

from hals_golf.paths import GAME, GAME_DIR
from hals_golf.runtime import retro_setup


def test_rom_candidates_include_shared_and_local() -> None:
    candidates = retro_setup.golf_rom_candidates()
    assert any("HalsHoleInOneGolf" in str(path) for path in candidates)


def test_ensure_golf_rom_resolves() -> None:
    path = retro_setup.ensure_golf_rom(required=True, quiet=True)
    assert path is not None
    assert path.exists()
    assert GAME_DIR.name == GAME
