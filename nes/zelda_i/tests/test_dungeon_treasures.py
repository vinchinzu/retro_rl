"""Wiki dungeon treasures vs Survival spine collection."""

from __future__ import annotations

import numpy as np

from zelda_i.route.treasures import (
    KIND_GATE,
    TREASURES,
    treasure,
)
from zelda_i.level6.door_hop import NORTH2C_SPEC, door_hop_success
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(*, level: int, screen: int, **fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_KEYS] = fields.get("keys", 1)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    ram[ADDR_ROD] = fields.get("rod", 0)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0)
    return ram


def test_bow_is_l1_not_l6() -> None:
    bow = treasure("bow")
    rod = treasure("magical_rod")
    assert bow.dungeon == 1
    assert bow.addr == ADDR_BOW
    assert rod.dungeon == 6
    assert rod.addr == ADDR_ROD
    assert rod.on_default_spine
    assert bow.on_default_spine


def test_gohma_enter_does_not_require_bow() -> None:
    snap = read_snapshot(
        _ram(
            level=6,
            screen=0x1C,
            x=120,
            y=205,
            keys=3,
            bow=0,
            arrows=0,
            rod=1,
            triforce=0x1F,
        )
    )
    assert door_hop_success(NORTH2C_SPEC, snap)
    assert snap.bow == 0


def test_l7_plus_required_gates_have_no_spine_through_yet() -> None:
    later = [
        item
        for item in TREASURES
        if item.kind == KIND_GATE and item.dungeon >= 7
    ]
    assert [item.name for item in later] == ["red_candle", "silver_arrows"]
    assert all(item.through is None for item in later)
    assert all(not item.on_default_spine for item in later)
