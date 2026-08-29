"""0x6b occupancy seed (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level3_dungeon import ROOM_L3_NORTH_ZOLS as ROOM_6B
from zelda_i.level3_geometry import NORTH_DOOR_X, ROOM_6B_BAND_Y
from zelda_i.walk_physics import WALK_DELTA
from retro_harness.nes import nes_action
from zelda_i.level3_path import Level3NorthExit6bController
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(*, x: int, y: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = ROOM_6B
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return ram


def test_miss_blocks_ahead_and_replans() -> None:
    # Off the door column — x≈120 inland is a leave-column residual (v6).
    ctrl = Level3NorthExit6bController()
    start = read_snapshot(_ram(x=96, y=141))
    first = ctrl.step(start)
    assert first.reason == "north6b_path"
    assert ctrl.walker.last_dir in WALK_DELTA
    blocked_ahead = {
        "UP": (96, 140),
        "DOWN": (96, 142),
        "LEFT": (95, 141),
        "RIGHT": (97, 141),
    }[ctrl.walker.last_dir]
    second = ctrl.step(start)
    assert ctrl.misses == 1
    assert blocked_ahead in ctrl.grid.blocked
    assert second.reason in {"north6b_path", "north6b_thread", "north6b_thread_up"}
    path = ctrl.grid.shortest_path((96, 141), (NORTH_DOOR_X, ROOM_6B_BAND_Y))
    assert path is not None
    assert blocked_ahead not in path


def test_south_mouth_is_not_occupancy_graded() -> None:
    """Combat can leave Link on the south door; do not miss-block inland UP.

    Cardinals stick at (120,181) (v2). LEFT+UP is the door clip that moves
    (v4); once off x≈120, UP inland. Occupancy does not grade the residual.
    """
    ctrl = Level3NorthExit6bController()
    door = ctrl.step(read_snapshot(_ram(x=120, y=181)))
    assert door.reason == "north6b_leave_mouth_clip"
    assert list(door.action) == list(nes_action("LEFT", "UP"))
    assert ctrl.misses == 0
    assert ctrl.walker.last_dir is None
    inland = ctrl.step(read_snapshot(_ram(x=100, y=181)))
    assert inland.reason == "north6b_leave_mouth"
    assert list(inland.action) == list(nes_action("UP"))
    assert ctrl.misses == 0
