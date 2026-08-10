"""Unit tests for Level 4 live interior anchors (rr-5lu)."""

from __future__ import annotations

from zelda_i.dungeon import ROOM_SPECS, ensure_default_specs
from zelda_i.dungeon_ids import VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE, object_name
from zelda_i.level4_dungeon import (
    BOMB_61_NORTH_STAND,
    BOMB_61_OPENS_TO,
    BombWall61North,
    ROOM_L4_ENTRY,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    ROOM_51_SPEC,
    ROOM_61_SPEC,
    ROOM_71_SPEC,
    VIRE_OBJECT_TYPE as L4_VIRE,
    make_bomb_61_north_controller,
    make_entry_up_controller,
    make_room_51_key_controller,
    make_room_61_clear_controller,
    planning_interior_report,
)
from zelda_i.level4_overworld import LEVEL4


def test_live_room_ids() -> None:
    assert ROOM_L4_ENTRY == 0x71
    assert ROOM_L4_VIRES_61 == 0x61
    assert ROOM_L4_KEESE_KEY_51 == 0x51
    assert ROOM_L4_VIRES_50 == 0x50
    assert L4_VIRE == VIRE_OBJECT_TYPE == 0x12
    assert VIRE_SPLIT_KEESE_TYPE == 0x1C


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
    assert ROOM_SPECS[0x71] is ROOM_71_SPEC
    assert ROOM_SPECS[0x61] is ROOM_61_SPEC
    assert ROOM_SPECS[0x51] is ROOM_51_SPEC


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


def test_planning_interior_report() -> None:
    r = planning_interior_report()
    assert r["bead"] == "rr-5lu"
    assert r["entry_room"] == "0x71"
    assert r["live_graph"]["0x71"]["UP"] == "0x61"
    assert r["live_graph"]["0x61"]["BOMB_UP"] == "0x51"
    assert r["live_graph"]["0x51"]["LEFT"] == "0x50"
    assert r["segments"]["clear_vires_61"] == "rr-yr77"
