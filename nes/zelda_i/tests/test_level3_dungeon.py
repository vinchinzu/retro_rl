"""Unit tests for Level 3 west-key pure helpers (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level3_dungeon import (
    ROOM_7B_SPEC,
    ROOM_L3_ENTRY,
    ROOM_L3_WEST_KEY,
    WEST_DOOR_APPROACH_Y,
    ZOL_OBJECT_TYPE,
    Level3WestDoorController,
    Level3WestKeyController,
    level3_room_7b_key_success,
    west_door_step,
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
    level: int = 3,
    room: int = ROOM_L3_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    zols: int = 0,
    hp: int = 32,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    for slot in range(1, zols + 1):
        ram[ADDR_OBJ_TYPE + slot] = ZOL_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = 64 + slot * 16
        ram[ADDR_LINK_Y + slot] = 141
    return ram


def test_room_ids_and_spec() -> None:
    assert ROOM_L3_ENTRY == 0x7C
    assert ROOM_L3_WEST_KEY == 0x7B
    assert ROOM_7B_SPEC.room_id == 0x7B
    assert ROOM_7B_SPEC.level == 3
    assert ROOM_7B_SPEC.enemy_types == (ZOL_OBJECT_TYPE,)
    assert ROOM_7B_SPEC.expected_enemy_count == 6
    assert ROOM_7B_SPEC.room_item_id == 0x19


def test_live_zols_type_and_hp() -> None:
    snap = read_snapshot(_ram(room=ROOM_L3_WEST_KEY, zols=6, hp=32))
    assert len(ROOM_7B_SPEC.live_enemies(snap)) == 6
    snap_dead = read_snapshot(_ram(room=ROOM_L3_WEST_KEY, zols=6, hp=0))
    assert len(ROOM_7B_SPEC.live_enemies(snap_dead)) == 0


def test_key_success_predicate() -> None:
    assert level3_room_7b_key_success(
        _ram(room=ROOM_L3_WEST_KEY, keys=1, zols=0)
    )
    assert not level3_room_7b_key_success(
        _ram(room=ROOM_L3_WEST_KEY, keys=0, zols=0)
    )
    assert not level3_room_7b_key_success(
        _ram(room=ROOM_L3_WEST_KEY, keys=1, zols=2, hp=32)
    )
    assert not level3_room_7b_key_success(
        _ram(room=ROOM_L3_ENTRY, keys=1, zols=0)
    )


def test_west_door_step_mouth_and_align() -> None:
    mouth = west_door_step(read_snapshot(_ram(y=205, x=120)))
    assert mouth.reason == "west_leave_mouth"

    align = west_door_step(
        read_snapshot(_ram(y=WEST_DOOR_APPROACH_Y + 12, x=100))
    )
    assert align.reason == "west_align_y"

    approach = west_door_step(
        read_snapshot(_ram(y=WEST_DOOR_APPROACH_Y, x=100))
    )
    assert approach.reason == "west_approach"

    diagonal = west_door_step(
        read_snapshot(_ram(y=WEST_DOOR_APPROACH_Y, x=40))
    )
    assert diagonal.reason == "west_diagonal_push"


def test_west_door_controller_arrives() -> None:
    ctrl = Level3WestDoorController()
    action = ctrl.step(
        read_snapshot(_ram(room=ROOM_L3_WEST_KEY, x=200, y=141))
    )
    assert ctrl.success
    assert action.reason == "west_arrived"


def test_west_key_controller_hand_off_notes() -> None:
    ctrl = Level3WestKeyController()
    # Already in 0x7b with no enemies and keys — combat should mark reward.
    snap = read_snapshot(_ram(room=ROOM_L3_WEST_KEY, keys=0, zols=0, x=120, y=141))
    # Door phase first: success on 0x7b
    ctrl.step(snap)
    assert ctrl.phase in {"combat", "done", "failed"} or ctrl.door.success
