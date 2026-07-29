from __future__ import annotations

import numpy as np

from zelda_i.nav_common import (
    on_arrival_edge,
    swing_action,
    track_stuck,
)
from zelda_i.overworld import ScreenHop, path_screens_from_hops
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _snap(*, x: int = 120, y: int = 140, screen: int = 0x37, mode: int = PLAY_MODE):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return read_snapshot(ram)


def test_path_screens_from_hops() -> None:
    hops = (
        ScreenHop(0x38, "RIGHT", align_y=140),
        ScreenHop(0x48, "DOWN", align_x=120),
    )
    assert path_screens_from_hops(0x37, hops) == (0x37, 0x38, 0x48)


def test_swing_pulses_a() -> None:
    slash = swing_action(0, "RIGHT", "walk", period=10, hold=3)
    walk = swing_action(3, "RIGHT", "walk", period=10, hold=3)
    assert slash.reason == "walk_slash"
    assert walk.reason == "walk"


def test_arrival_edge_and_stuck_tracking() -> None:
    edge = _snap(x=230, y=140)
    assert on_arrival_edge("RIGHT", edge)
    interior = _snap(x=120, y=140)
    assert not on_arrival_edge("RIGHT", interior)

    stuck, x, y, sc = track_stuck(
        interior, last_x=120, last_y=140, last_screen=0x37, stuck=2
    )
    assert stuck == 3
    assert (x, y, sc) == (120, 140, 0x37)
    stuck, *_ = track_stuck(
        _snap(x=121, y=140), last_x=120, last_y=140, last_screen=0x37, stuck=3
    )
    assert stuck == 0
