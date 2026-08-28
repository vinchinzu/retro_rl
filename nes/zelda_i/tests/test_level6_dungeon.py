"""Unit tests for Level 6 dungeon helpers (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon_ids import (
    KEESE_OBJECT_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
    WIZZROBE_BLUE_OBJECT_TYPE,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level6_dungeon import (
    LEVEL6_COMPASS_BIT,
    ROOM_09_SPEC,
    ROOM_19_SPEC,
    ROOM_29_SPEC,
    ROOM_28_SPEC,
    ROOM_38_SPEC,
    ROOM_58_SPEC,
    ROOM_68_SPEC,
    ROOM_78_SPEC,
    ROOM_79_SPEC,
    ROOM_7A_SPEC,
    ROOM_L6_COMPASS,
    ROOM_L6_EAST_KEY,
    ROOM_L6_ENTRY,
    ROOM_L6_HARD_38,
    ROOM_L6_KEESE,
    ROOM_L6_WEST_WIZZROBE,
    ROOM_L6_MAP,
    ROOM_L6_ROD_WIZZ,
    ROOM_L6_DARK_29,
    ROOM_L6_WIZZROBE_28,
    level6_room_09_clear_success,
    level6_room_19_clear_success,
    level6_room_28_clear_success,
    level6_room_38_clear_success,
    level6_room_58_clear_success,
    level6_room_68_compass_success,
    level6_room_78_clear_success,
    level6_room_7a_key_success,
    make_clear_09_controller,
    make_clear_29_controller,
    make_clear_19_controller,
    make_clear_28_controller,
    make_compass_68_controller,
    make_east_key_controller,
    make_hard_38_controller,
    make_keese_58_controller,
    make_west_wizzrobe_controller,
)
from zelda_i.level6_overworld import (
    ENTRY_LEFT_DOOR_Y,
    ENTRY_LEFT_WALL_Y,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_OLD_MAN_ROOM,
    LEVEL6_WEST_WIZZROBE_ROOM,
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


def test_room_ids_and_specs() -> None:
    assert ROOM_L6_ENTRY == 0x79
    assert ROOM_L6_EAST_KEY == 0x7A
    assert ROOM_L6_WEST_WIZZROBE == 0x78
    assert LEVEL6_WEST_WIZZROBE_ROOM == 0x78
    assert LEVEL6_COMPASS_ROOM == 0x68
    assert ROOM_L6_COMPASS == 0x68
    assert ROOM_L6_KEESE == 0x58
    assert LEVEL6_COMPASS_BIT == 0x20
    assert LEVEL6_OLD_MAN_ROOM == 0x6A
    assert ROOM_79_SPEC.room_id == 0x79
    assert ROOM_7A_SPEC.room_id == 0x7A
    assert ROOM_78_SPEC.room_id == 0x78
    assert ROOM_68_SPEC.room_id == 0x68
    assert ROOM_68_SPEC.enemy_types[0] == ZOL_OBJECT_TYPE
    assert ROOM_68_SPEC.combat.occupancy_patrol
    assert 0x2B not in ROOM_68_SPEC.enemy_types
    assert 0x68 not in ROOM_68_SPEC.enemy_types
    assert ROOM_58_SPEC.room_id == 0x58
    assert ROOM_58_SPEC.enemy_types == (KEESE_OBJECT_TYPE,)
    assert ROOM_58_SPEC.expected_enemy_count == 8
    assert ROOM_58_SPEC.combat.occupancy_patrol
    assert ROOM_L6_HARD_38 == 0x38
    assert ROOM_38_SPEC.room_id == 0x38
    assert WIZZROBE_ORANGE_TYPE in ROOM_38_SPEC.enemy_types
    assert WIZZROBE_BLUE_OBJECT_TYPE in ROOM_38_SPEC.enemy_types
    assert LIKE_LIKE_OBJECT_TYPE in ROOM_38_SPEC.enemy_types
    assert ROOM_38_SPEC.combat.occupancy_patrol
    assert 0x2B not in ROOM_38_SPEC.enemy_types
    assert 0x68 not in ROOM_38_SPEC.enemy_types
    assert 0x40 not in ROOM_38_SPEC.enemy_types
    assert ROOM_L6_WIZZROBE_28 == 0x28
    assert ROOM_28_SPEC.room_id == 0x28
    assert ROOM_28_SPEC.enemy_types == (WIZZROBE_ORANGE_TYPE,)
    assert ROOM_28_SPEC.expected_enemy_count == 2
    assert ROOM_28_SPEC.combat.occupancy_patrol
    assert 0x2B not in ROOM_28_SPEC.enemy_types
    assert 0x68 not in ROOM_28_SPEC.enemy_types
    assert 0x40 not in ROOM_28_SPEC.enemy_types
    assert LIKE_LIKE_OBJECT_TYPE not in ROOM_28_SPEC.enemy_types
    assert WIZZROBE_BLUE_OBJECT_TYPE not in ROOM_28_SPEC.enemy_types
    assert ROOM_7A_SPEC.enemy_types == (WIZZROBE_ORANGE_TYPE,)
    assert ROOM_78_SPEC.enemy_types == (WIZZROBE_ORANGE_TYPE,)
    assert ROOM_7A_SPEC.expected_enemy_count == 5
    assert ROOM_78_SPEC.expected_enemy_count == 5
    assert ROOM_7A_SPEC.room_item_id == 0x19
    assert ROOM_78_SPEC.level == 6
    assert ROOM_L6_MAP == 0x19
    assert ROOM_L6_ROD_WIZZ == 0x09
    assert ROOM_09_SPEC.room_id == 0x09
    assert ROOM_09_SPEC.combat.occupancy_patrol
    assert WIZZROBE_ORANGE_TYPE in ROOM_09_SPEC.enemy_types
    assert 0x2B not in ROOM_09_SPEC.enemy_types
    assert 0x68 not in ROOM_09_SPEC.enemy_types
    assert ROOM_L6_DARK_29 == 0x29
    assert ROOM_29_SPEC.room_id == 0x29
    assert ROOM_29_SPEC.combat.occupancy_patrol
    assert WIZZROBE_ORANGE_TYPE in ROOM_29_SPEC.enemy_types
    assert WIZZROBE_BLUE_OBJECT_TYPE in ROOM_29_SPEC.enemy_types
    assert 0x2B not in ROOM_29_SPEC.enemy_types
    assert 0x40 not in ROOM_29_SPEC.enemy_types
    assert make_clear_29_controller().spec is ROOM_29_SPEC


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


def test_68_compass_success_predicate() -> None:
    ram = _ram(room=ROOM_L6_COMPASS)
    ram[ADDR_COMPASS] = LEVEL6_COMPASS_BIT
    assert level6_room_68_compass_success(ram)
    ram[ADDR_COMPASS] = 0x1F
    assert not level6_room_68_compass_success(ram)
    ram[ADDR_COMPASS] = LEVEL6_COMPASS_BIT
    ram[ADDR_OBJ_TYPE + 1] = ZOL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_room_68_compass_success(ram)


def test_factories_bind_specs() -> None:
    from zelda_i.level6_wizzrobe import (
        Level6EastKeyController,
        Level6WestWizzrobeController,
    )

    east = make_east_key_controller()
    west = make_west_wizzrobe_controller()
    assert type(east) is Level6EastKeyController
    assert type(west) is Level6WestWizzrobeController
    compass = make_compass_68_controller()
    assert east.spec.room_id == 0x7A
    assert west.spec.room_id == 0x78
    assert compass.spec.room_id == 0x68
    assert compass.spec.combat.occupancy_patrol
    keese = make_keese_58_controller()
    assert keese.spec.room_id == 0x58
    assert keese.spec.combat.occupancy_patrol
    hard = make_hard_38_controller()
    assert hard.spec.room_id == 0x38
    assert hard.spec.combat.occupancy_patrol
    clear28 = make_clear_28_controller()
    assert clear28.spec.room_id == 0x28
    assert clear28.spec.combat.occupancy_patrol


def test_19_clear_success_predicate() -> None:
    ram = _ram(room=ROOM_L6_MAP, x=16, y=141)
    assert level6_room_19_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 1] = ZOL_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 32
    assert not level6_room_19_clear_success(ram)
    ram[ADDR_OBJ_HP + 1] = 0
    ram[ADDR_OBJ_TYPE + 2] = 0x40
    ram[ADDR_OBJ_HP + 2] = 64
    assert level6_room_19_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 3] = 0x2B
    ram[ADDR_OBJ_HP + 3] = 64
    assert level6_room_19_clear_success(ram)
    ctl = make_clear_19_controller()
    assert ctl.spec is ROOM_19_SPEC
    assert ROOM_19_SPEC.room_id == 0x19
    assert ROOM_19_SPEC.combat.occupancy_patrol
    assert ROOM_19_SPEC.combat.inland_dash == 24


def test_09_clear_success_predicate() -> None:
    ram = _ram(room=ROOM_L6_ROD_WIZZ, x=120, y=205)
    assert level6_room_09_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 1] = WIZZROBE_ORANGE_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_room_09_clear_success(ram)
    ram[ADDR_OBJ_HP + 1] = 0
    ram[ADDR_OBJ_TYPE + 2] = WIZZROBE_BLUE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 2] = 64
    assert not level6_room_09_clear_success(ram)
    ram[ADDR_OBJ_HP + 2] = 0
    ram[ADDR_OBJ_TYPE + 3] = 0x40
    ram[ADDR_OBJ_HP + 3] = 64
    assert level6_room_09_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 4] = 0x68
    ram[ADDR_OBJ_HP + 4] = 64
    assert level6_room_09_clear_success(ram)
    ctl = make_clear_09_controller()
    assert ctl.spec is ROOM_09_SPEC
    assert ROOM_09_SPEC.combat.occupancy_patrol
    assert 0x2B not in ROOM_09_SPEC.enemy_types
    assert 0x68 not in ROOM_09_SPEC.enemy_types


def test_58_clear_success_predicate() -> None:
    ram = _ram(room=ROOM_L6_KEESE)
    assert level6_room_58_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 1] = KEESE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 0
    assert not level6_room_58_clear_success(ram)


def test_38_clear_success_predicate() -> None:
    ram = _ram(room=ROOM_L6_HARD_38)
    assert level6_room_38_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 1] = LIKE_LIKE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_room_38_clear_success(ram)
    ram[ADDR_OBJ_HP + 1] = 0
    ram[ADDR_OBJ_TYPE + 2] = WIZZROBE_BLUE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 2] = 64
    assert not level6_room_38_clear_success(ram)
    ram[ADDR_OBJ_HP + 2] = 0
    ram[ADDR_OBJ_TYPE + 3] = 0x40
    ram[ADDR_OBJ_HP + 3] = 64
    assert level6_room_38_clear_success(ram)


def test_28_clear_success_predicate() -> None:
    ram = _ram(room=ROOM_L6_WIZZROBE_28)
    assert level6_room_28_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 1] = WIZZROBE_ORANGE_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_room_28_clear_success(ram)
    ram[ADDR_OBJ_HP + 1] = 0
    ram[ADDR_OBJ_TYPE + 2] = 0x40
    ram[ADDR_OBJ_HP + 2] = 64
    assert level6_room_28_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 3] = 0x2B
    ram[ADDR_OBJ_HP + 3] = 64
    assert level6_room_28_clear_success(ram)
    ram[ADDR_OBJ_TYPE + 4] = 0x68
    ram[ADDR_OBJ_HP + 4] = 64
    assert level6_room_28_clear_success(ram)


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
