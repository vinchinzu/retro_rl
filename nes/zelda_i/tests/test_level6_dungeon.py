"""Unit tests for Level 6 leftover walks that would burn again."""

from __future__ import annotations

import numpy as np
import pytest

from zelda_i.dungeon.ids import (
    KEESE_OBJECT_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level6.dungeon import (
    LEVEL6_COMPASS_BIT,
    ROOM_78_SPEC,
    ROOM_7A_SPEC,
    ROOM_L6_COMPASS,
    ROOM_L6_EAST_KEY,
    ROOM_L6_ENTRY,
    ROOM_L6_HARD_38,
    ROOM_L6_KEESE,
    ROOM_L6_MAP,
    ROOM_L6_ROD_WIZZ,
    ROOM_L6_WEST_WIZZROBE,
    ROOM_L6_WIZZROBE_28,
    level6_room_09_clear_success,
    level6_room_19_clear_success,
    level6_room_28_clear_success,
    level6_room_38_clear_success,
    level6_room_58_clear_success,
    level6_room_68_compass_success,
    level6_room_78_clear_success,
    level6_room_7a_key_success,
)
from zelda_i.level6.overworld import (
    Level6WestKeyDoorController,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.ram import (
    ADDR_COMPASS,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    level: int = 6,
    room: int = ROOM_L6_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    wizzrobes: int = 0,
    hp: int = 64,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    for slot in range(1, wizzrobes + 1):
        ram[ADDR_OBJ_TYPE + slot] = WIZZROBE_ORANGE_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
    return ram


def test_live_wizzrobes_type_and_hp() -> None:
    snap = read_snapshot(_ram(room=ROOM_L6_EAST_KEY, wizzrobes=5, hp=64))
    assert len(ROOM_7A_SPEC.live_enemies(snap)) == 5
    snap_dead = read_snapshot(_ram(room=ROOM_L6_EAST_KEY, wizzrobes=5, hp=0))
    assert len(ROOM_7A_SPEC.live_enemies(snap_dead)) == 0
    snap78 = read_snapshot(_ram(room=ROOM_L6_WEST_WIZZROBE, wizzrobes=5, hp=64))
    assert len(ROOM_78_SPEC.live_enemies(snap78)) == 5


@pytest.mark.parametrize(
    "success_fn, room, extra, live_type, live_hp",
    [
        (level6_room_7a_key_success, ROOM_L6_EAST_KEY, {"keys": 1}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_78_clear_success, ROOM_L6_WEST_WIZZROBE, {}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_68_compass_success, ROOM_L6_COMPASS, {"compass": LEVEL6_COMPASS_BIT}, ZOL_OBJECT_TYPE, 64),
        (level6_room_19_clear_success, ROOM_L6_MAP, {}, ZOL_OBJECT_TYPE, 32),
        (level6_room_09_clear_success, ROOM_L6_ROD_WIZZ, {}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_58_clear_success, ROOM_L6_KEESE, {}, KEESE_OBJECT_TYPE, 0),
        (level6_room_38_clear_success, ROOM_L6_HARD_38, {}, LIKE_LIKE_OBJECT_TYPE, 64),
        (level6_room_28_clear_success, ROOM_L6_WIZZROBE_28, {}, WIZZROBE_ORANGE_TYPE, 64),
    ],
    ids=["7a", "78", "68", "19", "09", "58", "38", "28"],
)
def test_clear_success(success_fn, room, extra, live_type, live_hp) -> None:
    kwargs = {k: v for k, v in extra.items() if k != "compass"}
    ram = _ram(room=room, **kwargs)
    if "compass" in extra:
        ram[ADDR_COMPASS] = extra["compass"]
    assert success_fn(ram)
    ram[ADDR_OBJ_TYPE + 1] = live_type
    ram[ADDR_OBJ_HP + 1] = live_hp
    assert not success_fn(ram)


def test_west_key_door_controller_from_east_edge() -> None:
    ctl = Level6WestKeyDoorController()
    # East door channel after free return from 0x7a — must LEFT first
    # (vertical blocked at x≈224).
    snap = read_snapshot(
        _ram(room=ROOM_L6_ENTRY, x=224, y=141, keys=1)
    )
    act = ctl.step(snap)
    assert act.reason == "leave_east_door_channel"

    # At fire-wall column, adjust y before crossing.
    ctl2 = Level6WestKeyDoorController()
    snap2 = read_snapshot(
        _ram(room=ROOM_L6_ENTRY, x=208, y=141, keys=1)
    )
    act2 = ctl2.step(snap2)
    assert act2.reason == "east_to_wall_y"

    arrived = Level6WestKeyDoorController()
    arrived.step(read_snapshot(
        _ram(room=ROOM_L6_WEST_WIZZROBE, x=224, y=141, keys=0)
    ))
    assert arrived.success
