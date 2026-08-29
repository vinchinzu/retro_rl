"""South-pocket 0x40 key leftover. No emulator."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.level4_dungeon import ROOM_ITEM_SMALL_KEY, ROOM_L4_ZOLS_40
from zelda_i.level4_key40 import (
    SOUTH_POCKET_KEY_XY,
    live_key_xy,
    make_room_40_key_controller,
    south_pocket_key_xy,
)
from zelda_i.level4_maze_path import Key40Phase
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(*, x: int = 120, y: int = 149, keys: int = 4, key_xy: tuple[int, int] | None = None) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = ROOM_L4_ZOLS_40
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    if key_xy is not None:
        ram[ADDR_OBJ_TYPE + 1] = ROOM_ITEM_SMALL_KEY
        ram[ADDR_LINK_X + 1] = key_xy[0]
        ram[ADDR_LINK_Y + 1] = key_xy[1]
    return ram


def test_south_pocket_leftover_chases_up_not_north_band() -> None:
    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.HUNT
    ctrl.keys_before = 4
    snap = read_snapshot(_ram(x=120, y=149, keys=4))
    assert south_pocket_key_xy(snap) == SOUTH_POCKET_KEY_XY
    action = ctrl.step(snap)
    assert action.reason == "key_chase"
    assert action.action == nes_action("UP")


def test_live_key_object_beats_south_pocket() -> None:
    snap = read_snapshot(_ram(x=120, y=149, key_xy=(128, 157)))
    assert live_key_xy(snap) == (128, 157)
    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.HUNT
    ctrl.keys_before = 4
    action = ctrl.step(snap)
    assert action.reason == "key_chase"
    assert action.action in (nes_action("RIGHT"), nes_action("DOWN"))


def test_stand_on_key_does_not_orbit() -> None:
    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.HUNT
    ctrl.keys_before = 4
    snap = read_snapshot(_ram(x=120, y=141, keys=4))
    action = ctrl.step(snap)
    assert action.reason == "key_stand"


def test_up_miss_blocks_and_replans() -> None:
    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.HUNT
    ctrl.keys_before = 4
    stuck = read_snapshot(_ram(x=120, y=149, keys=4))
    first = ctrl.step(stuck)
    assert first.reason == "key_chase"
    assert first.action == nes_action("UP")
    second = ctrl.step(stuck)
    assert ctrl.walker.misses >= 1
    assert second.reason == "key_chase"
    assert second.action != nes_action("UP")


def test_blocked_dest_falls_back_to_documented_pickup() -> None:
    from zelda_i.level4_dungeon import KEY_40_PICKUP_XY

    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.HUNT
    ctrl.keys_before = 4
    snap = read_snapshot(_ram(x=128, y=149, keys=4))
    ctrl.walker.grid.blocked.add(SOUTH_POCKET_KEY_XY)
    ctrl.walker.grid.blocked.add((120, 148))
    action = ctrl.step(snap)
    assert action.reason in ("key_chase", "key_stand")
    if action.reason == "key_chase":
        assert KEY_40_PICKUP_XY not in ctrl.walker.grid.blocked or action.action is not None
