"""Shared L6 leftover / dest / occupancy-walk helpers."""

from __future__ import annotations

import numpy as np

from zelda_i.level6.occupancy import (
    l6_play_dest_success,
    occupancy_new_miss,
)
from zelda_i.level6.path import Level6North68Controller
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
from zelda_i.walk.physics import OccupancyWalker


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


def test_north68_peels_south_from_0x78_statue_pocket() -> None:
    """West-clear leftover (104,149): DOWN to y=189, RIGHT to x=144, then UP."""
    ctl = Level6North68Controller()
    snap = read_snapshot(_ram(screen=0x78, x=104, y=149))
    reasons = [ctl.step(snap).reason for _ in range(8)]
    assert reasons[0] == "peel_south_statue"
    mid = Level6North68Controller()
    v2 = read_snapshot(_ram(screen=0x78, x=104, y=158))
    assert mid.step(v2).reason == "peel_south_statue"
    # v3 leftover: CLIP_CLEAR_Y is not far enough; keep DOWN past the statue.
    south = Level6North68Controller()
    cleared = read_snapshot(_ram(screen=0x78, x=104, y=173))
    assert south.step(cleared).reason == "peel_south_statue"
    boxed = Level6North68Controller()
    boxed.walker.last_dir = None
    for x in range(40, 217):
        for y in range(77, 174):
            boxed.walker.grid.blocked.add((x, y))
    boxed_snap = read_snapshot(_ram(screen=0x78, x=104, y=173))
    assert boxed.step(boxed_snap).reason == "peel_south_statue"
    mouth = Level6North68Controller()
    at_mouth = read_snapshot(_ram(screen=0x78, x=104, y=189))
    assert mouth.step(at_mouth).reason == "peel_east_door"
    almost = Level6North68Controller()
    at_119 = read_snapshot(_ram(screen=0x78, x=119, y=189))
    assert almost.step(at_119).reason == "peel_east_door"
    # v4 leftover: do not occupancy-UP the door column. RIGHT to x=144.
    aligned = Level6North68Controller()
    at_door = read_snapshot(_ram(screen=0x78, x=120, y=189))
    assert aligned.step(at_door).reason == "peel_east_aisle"
    mid_col = Level6North68Controller()
    at_mid = read_snapshot(_ram(screen=0x78, x=120, y=149))
    assert mid_col.step(at_mid).reason == "peel_east_aisle"
    almost_aisle = Level6North68Controller()
    at_143 = read_snapshot(_ram(screen=0x78, x=143, y=149))
    assert almost_aisle.step(at_143).reason == "peel_east_aisle"
    east = Level6North68Controller()
    hist = read_snapshot(_ram(screen=0x78, x=144, y=141))
    first = east.step(hist)
    assert first.reason == "north_path"
    assert first.reason not in (
        "peel_south_statue",
        "peel_east_door",
        "peel_east_aisle",
        "north_stand",
    )


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
