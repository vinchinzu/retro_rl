"""Unit tests for Level 4 live interior anchors (rr-5lu / rr-2ysf / rr-9so0)."""

from __future__ import annotations

from zelda_i.dungeon import ROOM_SPECS, ensure_default_specs
from zelda_i.dungeon_ids import VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE, object_name
from zelda_i.level4_dungeon import (
    BOMB_61_NORTH_STAND,
    BOMB_61_OPENS_TO,
    BombWall61North,
    COMPASS_PICKUP_XY,
    KEY_61_EAST_Y,
    KEY_61_OPENS_TO,
    LEVEL4_COMPASS_BIT,
    MAZE_62_RETURN_WEST,
    MAZE_62_TO_COMPASS,
    MAZE_IN_HOLD,
    MAZE_OUT_HOLD,
    ROOM_L4_COMPASS_62,
    ROOM_L4_ENTRY,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    ROOM_50_SPEC,
    ROOM_51_SPEC,
    ROOM_61_SPEC,
    ROOM_62_SPEC,
    ROOM_71_SPEC,
    ROOM_ITEM_COMPASS,
    VIRE_OBJECT_TYPE as L4_VIRE,
    make_bomb_61_north_controller,
    make_compass_62_controller,
    make_entry_up_controller,
    make_key_right_62_controller,
    make_left_50_controller,
    make_room_50_clear_controller,
    make_room_51_key_controller,
    make_room_61_clear_controller,
    make_room_62_clear_controller,
    planning_interior_report,
)
from zelda_i.level4_overworld import LEVEL4


def test_live_room_ids() -> None:
    assert ROOM_L4_ENTRY == 0x71
    assert ROOM_L4_VIRES_61 == 0x61
    assert ROOM_L4_KEESE_KEY_51 == 0x51
    assert ROOM_L4_VIRES_50 == 0x50
    assert ROOM_L4_COMPASS_62 == 0x62
    assert L4_VIRE == VIRE_OBJECT_TYPE == 0x12
    assert VIRE_SPLIT_KEESE_TYPE == 0x1C
    assert ROOM_ITEM_COMPASS == 0x16
    assert LEVEL4_COMPASS_BIT == 0x08
    assert KEY_61_EAST_Y == 141
    assert KEY_61_OPENS_TO == 0x62


def test_object_names() -> None:
    assert object_name(0x12) == "vire"
    assert object_name(0x1C) == "vire_split_keese"


def test_bomb_wall_geometry() -> None:
    wall = BombWall61North()
    assert wall.room == 0x61
    assert wall.stand == BOMB_61_NORTH_STAND == (120, 105)
    assert wall.face == "UP"
    assert wall.opens_to == BOMB_61_OPENS_TO == 0x51


def test_specs_register() -> None:
    ensure_default_specs()
    assert ROOM_71_SPEC.level == LEVEL4
    assert ROOM_61_SPEC.enemy_types[0] == 0x12
    assert VIRE_SPLIT_KEESE_TYPE in ROOM_61_SPEC.type_only_enemy_types
    assert ROOM_61_SPEC.object_slot_max == 12
    assert ROOM_51_SPEC.room_item_id == 0x19
    assert ROOM_50_SPEC.room_id == 0x50
    assert ROOM_50_SPEC.expected_enemy_count == 5
    assert ROOM_62_SPEC.room_id == 0x62
    assert ROOM_62_SPEC.room_item_id == 0x16
    assert ROOM_SPECS[0x71] is ROOM_71_SPEC
    assert ROOM_SPECS[0x61] is ROOM_61_SPEC
    assert ROOM_SPECS[0x51] is ROOM_51_SPEC
    assert ROOM_SPECS[0x50] is ROOM_50_SPEC
    assert ROOM_SPECS[0x62] is ROOM_62_SPEC


def test_factories() -> None:
    up = make_entry_up_controller()
    assert up.max_frames > 0
    clear = make_room_61_clear_controller()
    assert clear.spec is ROOM_61_SPEC
    bomb = make_bomb_61_north_controller(clear_vires=True)
    assert bomb.level == LEVEL4
    assert bomb.to_room == 0x51
    assert bomb.clear_spec is ROOM_61_SPEC
    key = make_room_51_key_controller()
    assert key.spec is ROOM_51_SPEC
    left = make_left_50_controller()
    assert left.max_frames > 0
    c50 = make_room_50_clear_controller()
    assert c50.spec is ROOM_50_SPEC
    kr = make_key_right_62_controller(clear_vires=True)
    assert kr.clear_vires is True
    c62 = make_room_62_clear_controller()
    assert c62.spec is ROOM_62_SPEC
    compass = make_compass_62_controller()
    assert compass.max_frames > 0
    assert compass.phase.name == "MAZE_IN"


def test_maze_62_paths() -> None:
    assert MAZE_IN_HOLD == 6
    assert MAZE_OUT_HOLD == 4
    assert MAZE_62_TO_COMPASS[0] == "DOWN"
    assert "RIGHT" in MAZE_62_TO_COMPASS
    assert MAZE_62_RETURN_WEST[0] == "DOWN"
    assert MAZE_62_RETURN_WEST.count("LEFT") >= 10
    assert COMPASS_PICKUP_XY == (136, 132)


def test_planning_interior_report() -> None:
    r = planning_interior_report()
    assert r["bead"] == "rr-5lu"
    assert r["tip"] == "rr-9so0"
    assert r["entry_room"] == "0x71"
    assert r["live_graph"]["0x71"]["UP"] == "0x61"
    assert r["live_graph"]["0x61"]["BOMB_UP"] == "0x51"
    assert r["live_graph"]["0x61"]["KEY_RIGHT"] == "0x62"
    assert r["live_graph"]["0x51"]["LEFT"] == "0x50"
    assert r["live_graph"]["0x50"]["note"] == "dead_end_pocket"
    assert r["live_graph"]["0x62"]["room_item"] == "0x16"
    assert r["live_graph"]["0x62"]["compass_bit"] == "0x8"
    assert r["segments"]["clear_vires_61"] == "rr-yr77"
    assert r["segments"]["clear_50"] == "rr-2ysf"
    assert r["segments"]["key_right_62"] == "rr-2ysf"
    assert r["segments"]["compass_62"] == "rr-9so0"
    assert r["key_61_east"]["opens_to"] == "0x62"
    assert r["maze_62"]["pickup_xy"] == [136, 132]
