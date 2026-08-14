"""Unit tests for Level 5 dungeon helpers (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level5_dungeon import (
    BUBBLE_OBJECT_TYPE,
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_66_EAST_DOOR_BIT,
    ROOM_66_SPEC,
    ROOM_67_SPEC,
    ROOM_67_WEST_DOOR_BIT,
    ROOM_77_SPEC,
    ROOM_ITEM_SMALL_KEY,
    ROOM_L5_EAST_67,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_POLS_77,
    Level5East67Controller,
    Level5PolsVoiceController,
    level5_in_room_66,
    level5_in_room_67,
    level5_in_room_77,
    level5_room_66_cleared,
    level5_room_67_arrived,
    level5_room_77_key_success,
)
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ALL_DEAD,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    level: int = LEVEL_5,
    room: int = ROOM_L5_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    doors: int = 0,
    all_dead: int = 0,
    enemy_type: int = GIBDO_OBJECT_TYPE,
    enemies: int = 0,
    hp: int = 112,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    ram[ADDR_CUR_OPENED_DOORS] = doors
    ram[ADDR_ROOM_ALL_DEAD] = all_dead
    for slot in range(1, enemies + 1):
        ram[ADDR_OBJ_TYPE + slot] = enemy_type
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = 64 + slot * 16
        ram[ADDR_LINK_Y + slot] = 141
    return ram


def test_room_ids_and_specs() -> None:
    assert ROOM_L5_ENTRY == 0x76
    assert ROOM_L5_GIBDO_66 == 0x66
    assert ROOM_L5_EAST_67 == 0x67
    assert ROOM_L5_POLS_77 == 0x77
    assert ROOM_66_SPEC.room_id == 0x66
    assert ROOM_66_SPEC.level == 5
    assert ROOM_66_SPEC.enemy_types == (GIBDO_OBJECT_TYPE,)
    assert ROOM_66_SPEC.expected_enemy_count == 3
    assert ROOM_66_SPEC.required_open_doors == ROOM_66_EAST_DOOR_BIT
    assert ROOM_67_SPEC.enemy_types == (BUBBLE_OBJECT_TYPE,)
    assert ROOM_67_SPEC.expected_enemy_count == 2
    assert ROOM_77_SPEC.enemy_types == (POLS_VOICE_OBJECT_TYPE,)
    assert ROOM_77_SPEC.expected_enemy_count == 5
    assert ROOM_77_SPEC.room_item_id == ROOM_ITEM_SMALL_KEY


def test_room_66_cleared_predicate() -> None:
    assert level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=0, doors=0x08, all_dead=20)
    )
    assert not level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=3, doors=0x08, all_dead=20, hp=112)
    )
    assert not level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=0, doors=0x00, all_dead=20)
    )


def test_in_room_helpers() -> None:
    assert level5_in_room_66(_ram(room=ROOM_L5_GIBDO_66))
    assert not level5_in_room_66(_ram(room=ROOM_L5_ENTRY))
    assert level5_in_room_67(_ram(room=ROOM_L5_EAST_67))
    assert level5_in_room_77(_ram(room=ROOM_L5_POLS_77))


def test_room_67_arrived_needs_west_door() -> None:
    assert level5_room_67_arrived(
        _ram(room=ROOM_L5_EAST_67, doors=ROOM_67_WEST_DOOR_BIT)
    )
    assert not level5_room_67_arrived(_ram(room=ROOM_L5_EAST_67, doors=0x00))
    assert not level5_room_67_arrived(
        _ram(room=ROOM_L5_GIBDO_66, doors=ROOM_67_WEST_DOOR_BIT)
    )


def test_room_77_key_success() -> None:
    assert level5_room_77_key_success(_ram(room=ROOM_L5_POLS_77, keys=1, enemies=0))
    assert not level5_room_77_key_success(_ram(room=ROOM_L5_POLS_77, keys=0, enemies=0))
    assert not level5_room_77_key_success(
        _ram(
            room=ROOM_L5_POLS_77,
            keys=1,
            enemies=2,
            enemy_type=POLS_VOICE_OBJECT_TYPE,
            hp=160,
        )
    )
    assert not level5_room_77_key_success(_ram(room=ROOM_L5_ENTRY, keys=1, enemies=0))


def test_live_pols_type_and_hp() -> None:
    snap = read_snapshot(
        _ram(
            room=ROOM_L5_POLS_77,
            enemies=5,
            enemy_type=POLS_VOICE_OBJECT_TYPE,
            hp=160,
        )
    )
    assert len(ROOM_77_SPEC.live_enemies(snap)) == 5
    snap_dead = read_snapshot(
        _ram(
            room=ROOM_L5_POLS_77,
            enemies=5,
            enemy_type=POLS_VOICE_OBJECT_TYPE,
            hp=0,
        )
    )
    assert len(ROOM_77_SPEC.live_enemies(snap_dead)) == 0


def test_east_67_controller_arrives() -> None:
    ctrl = Level5East67Controller(settle_frames=2)
    snap = read_snapshot(
        _ram(room=ROOM_L5_EAST_67, x=16, y=141, doors=ROOM_67_WEST_DOOR_BIT)
    )
    a1 = ctrl.step(snap)
    assert a1.reason == "settle_67"
    assert not ctrl.success
    a2 = ctrl.step(snap)
    assert a2.reason == "arrived_67"
    assert ctrl.success


def test_east_67_controller_aligns() -> None:
    ctrl = Level5East67Controller()
    action = ctrl.step(read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=120, y=160)))
    assert not ctrl.success
    assert action.reason == "align_east_y"


def test_pols_controller_constructs() -> None:
    ctrl = Level5PolsVoiceController(spec=ROOM_77_SPEC)
    assert ctrl.spec.room_id == ROOM_L5_POLS_77
    assert ctrl.spec.enemy_types == (POLS_VOICE_OBJECT_TYPE,)
