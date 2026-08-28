"""Shared L6 leftover / dest / occupancy-walk helpers."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_occupancy import (
    l6_leftover,
    l6_play_dest_success,
    occupancy_new_miss,
    record_l6_walk,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_COLLIDING_TILE,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PASSAGE_MODE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.walk_physics import OccupancyWalker


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x3A)
    ram[ADDR_LINK_X] = fields.get("x", 96)
    ram[ADDR_LINK_Y] = fields.get("y", 157)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ROD] = fields.get("rod", 1)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 118)
    return ram


def test_l6_leftover_keys() -> None:
    leftover = l6_leftover(read_snapshot(_ram()))
    assert leftover == {
        "x": 96,
        "y": 157,
        "mode": PLAY_MODE,
        "screen": 0x3A,
        "tile": 118,
        "rod": 1,
        "bow": 0,
        "arrows": 0,
        "keys": 4,
        "bombs": 8,
        "triforce": 0x1F,
    }


def test_l6_play_dest_success() -> None:
    dest = read_snapshot(_ram(screen=0x3B, x=16, y=141))
    assert l6_play_dest_success(dest, not_room=0x3A)
    still = read_snapshot(_ram())
    assert not l6_play_dest_success(still, not_room=0x3A)
    passage = read_snapshot(_ram(mode=PASSAGE_MODE, screen=0x08))
    assert l6_play_dest_success(passage, not_room=0x3A)
    assert not l6_play_dest_success(passage, not_room=0x3A, passage_ok=False)
    assert not l6_play_dest_success(passage, not_room=0x3A, forbid=(0x08,))
    no_rod = read_snapshot(_ram(screen=0x3B, rod=0))
    assert not l6_play_dest_success(no_rod, not_room=0x3A)


def test_l6_play_dest_success_exact_dest_room() -> None:
    dest = read_snapshot(_ram(screen=0x18, x=208, y=141))
    assert l6_play_dest_success(
        dest, not_room=0x19, dest_room=0x18, passage_ok=False
    )
    other = read_snapshot(_ram(screen=0x29, x=120, y=77))
    assert not l6_play_dest_success(
        other, not_room=0x19, dest_room=0x18, passage_ok=False
    )
    still = read_snapshot(_ram(screen=0x19, x=32, y=141))
    assert not l6_play_dest_success(
        still, not_room=0x19, dest_room=0x18, passage_ok=False
    )
    trans = read_snapshot(_ram(screen=0x18, mode=6, x=208, y=141))
    assert not l6_play_dest_success(
        trans, not_room=0x19, dest_room=0x18, passage_ok=False
    )
    passage = read_snapshot(_ram(mode=PASSAGE_MODE, screen=0x18))
    assert not l6_play_dest_success(
        passage, not_room=0x19, dest_room=0x18, passage_ok=True
    )


def test_occupancy_new_miss_2px_up() -> None:
    walker = OccupancyWalker()
    walker.last_dir = "UP"
    walker.last_xy = (96, 157)
    assert occupancy_new_miss(walker, (96, 155)) == "UP"
    assert walker.misses == 1

    first = OccupancyWalker()
    first.last_dir = "UP"
    first.last_xy = (96, 157)
    assert occupancy_new_miss(first, (96, 155), allow_first=True) is None
    assert first.misses == 1
    first.last_dir = "UP"
    assert occupancy_new_miss(first, (96, 155), allow_first=True) == "UP"


def test_record_l6_walk_samples() -> None:
    snap = read_snapshot(_ram())
    samples: list[dict] = []
    leftover = record_l6_walk(
        samples, snap, reason="door_y", frames=1, period=8, misses=0
    )
    assert leftover["x"] == 96
    assert leftover["y"] == 157
    assert samples[0]["reason"] == "door_y"
    assert samples[0]["misses"] == 0
