"""Unit tests for Level 6 dungeon helpers (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_dungeon import (
    ROOM_78_SPEC,
    ROOM_79_SPEC,
    ROOM_7A_SPEC,
    ROOM_L6_EAST_KEY,
    ROOM_L6_ENTRY,
    ROOM_L6_WEST_WIZZROBE,
    level6_room_78_clear_success,
    level6_room_7a_key_success,
    make_east_key_controller,
    make_west_wizzrobe_controller,
)
from zelda_i.level6_overworld import (
    ENTRY_LEFT_DOOR_Y,
    ENTRY_LEFT_WALL_Y,
    LEVEL6_OLD_MAN_ROOM,
    LEVEL6_WEST_WIZZROBE_ROOM,
    Level6WestKeyDoorController,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.ram import (
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


def test_room_ids_and_specs() -> None:
    assert ROOM_L6_ENTRY == 0x79
    assert ROOM_L6_EAST_KEY == 0x7A
    assert ROOM_L6_WEST_WIZZROBE == 0x78
    assert LEVEL6_WEST_WIZZROBE_ROOM == 0x78
    assert LEVEL6_OLD_MAN_ROOM == 0x6A
    assert ROOM_79_SPEC.room_id == 0x79
    assert ROOM_7A_SPEC.room_id == 0x7A
    assert ROOM_78_SPEC.room_id == 0x78
    assert ROOM_7A_SPEC.enemy_types == (WIZZROBE_ORANGE_TYPE,)
    assert ROOM_78_SPEC.enemy_types == (WIZZROBE_ORANGE_TYPE,)
    assert ROOM_7A_SPEC.expected_enemy_count == 5
    assert ROOM_78_SPEC.expected_enemy_count == 5
    assert ROOM_7A_SPEC.room_item_id == 0x19
    assert ROOM_78_SPEC.level == 6


def test_live_wizzrobes_type_and_hp() -> None:
    snap = read_snapshot(_ram(room=ROOM_L6_EAST_KEY, wizzrobes=5, hp=64))
    assert len(ROOM_7A_SPEC.live_enemies(snap)) == 5
    snap_dead = read_snapshot(_ram(room=ROOM_L6_EAST_KEY, wizzrobes=5, hp=0))
    assert len(ROOM_7A_SPEC.live_enemies(snap_dead)) == 0
    snap78 = read_snapshot(_ram(room=ROOM_L6_WEST_WIZZROBE, wizzrobes=5, hp=64))
    assert len(ROOM_78_SPEC.live_enemies(snap78)) == 5


def test_7a_key_success_predicate() -> None:
    assert level6_room_7a_key_success(
        _ram(room=ROOM_L6_EAST_KEY, keys=1, wizzrobes=0)
    )
    assert not level6_room_7a_key_success(
        _ram(room=ROOM_L6_EAST_KEY, keys=0, wizzrobes=0)
    )
    assert not level6_room_7a_key_success(
        _ram(room=ROOM_L6_EAST_KEY, keys=1, wizzrobes=2, hp=64)
    )


def test_78_clear_success_predicate() -> None:
    assert level6_room_78_clear_success(
        _ram(room=ROOM_L6_WEST_WIZZROBE, keys=0, wizzrobes=0)
    )
    assert not level6_room_78_clear_success(
        _ram(room=ROOM_L6_WEST_WIZZROBE, wizzrobes=3, hp=64)
    )
    assert not level6_room_78_clear_success(
        _ram(room=ROOM_L6_ENTRY, wizzrobes=0)
    )


def test_factories_bind_specs() -> None:
    east = make_east_key_controller()
    west = make_west_wizzrobe_controller()
    assert east.spec.room_id == 0x7A
    assert west.spec.room_id == 0x78


def test_west_key_door_constants() -> None:
    assert ENTRY_LEFT_WALL_Y == 157
    assert ENTRY_LEFT_DOOR_Y == 141


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


def test_west_key_door_controller_at_door() -> None:
    ctl = Level6WestKeyDoorController()
    snap = read_snapshot(
        _ram(room=ROOM_L6_ENTRY, x=32, y=141, keys=1)
    )
    act = ctl.step(snap)
    assert act.reason == "push_key_left"


def test_west_key_door_arrives_0x78() -> None:
    ctl = Level6WestKeyDoorController()
    snap = read_snapshot(
        _ram(room=ROOM_L6_WEST_WIZZROBE, x=224, y=141, keys=0)
    )
    act = ctl.step(snap)
    assert ctl.success
    assert act.reason == "done"
