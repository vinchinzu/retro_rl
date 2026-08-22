"""Zelda snapshot claims (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.predict import grade_walk, snapshot_fields, walk_claim
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_MODE, ADDR_SCREEN, PLAY_MODE, read_snapshot


def _snap(*, x: int, y: int, screen: int = 0x6B) -> object:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return read_snapshot(ram)


def test_walk_claim_left() -> None:
    assert walk_claim("LEFT") == "move -1,0"


def test_grade_walk_hit_and_wall_miss() -> None:
    before = _snap(x=48, y=149)
    hit = _snap(x=47, y=149)
    miss = _snap(x=48, y=149)
    assert grade_walk("LEFT", before, hit).ok
    stuck = grade_walk("LEFT", before, miss)
    assert not stuck.ok
    assert stuck.missed == ("move -1,0",)


def test_snapshot_fields_include_screen() -> None:
    fields = snapshot_fields(_snap(x=120, y=93, screen=0x6B))
    assert fields["screen"] == 0x6B
    assert fields["x"] == 120
