
from __future__ import annotations

import numpy as np

from zelda_i.dungeon_ids import KEESE_OBJECT_TYPE
from zelda_i.dungeon_trace import action_button_names
from zelda_i.level9_ganon import LEVEL9, ROOM_BEFORE_GANON
from zelda_i.level9_path import NORTH_DOOR_X
from zelda_i.level9_room62 import (
    LEVEL9_STAIR_SOURCES,
    NORTH_DOOR,
    ROOM52_ROM_NORTH,
    ROOM52_ROM_SOUTH,
    ROOM62_KEESE_COUNT,
    ROOM62_ROM_EAST,
    ROOM62_ROM_NORTH,
    ROOM62_ROM_SOUTH,
    ROOM62_ROM_WEST,
    ROOM_LEVEL9_62,
    door_bits,
    in_room_62,
    room62_is_cardinal_predecessor_of_patra,
    room62_keese,
    room62_to_patra_step,
    uncleared_room62,
)
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_TYPE,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.scripts.run_level9_room62 import _loader_write_rows
from zelda_i.level9_room62 import LOADER_CANDIDATES


def _room62_snapshot(
    *,
    link_x: int = 120,
    link_y: int = 181,
    keese: int = ROOM62_KEESE_COUNT,
    doors: int = 0,
    screen: int = ROOM_LEVEL9_62,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = LEVEL9
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = link_x
    ram[ADDR_LINK_Y] = link_y
    ram[ADDR_HEALTH] = 0xFF
    ram[ADDR_CUR_OPENED_DOORS] = doors
    ram[ADDR_OPEN_DOORWAY_MASK] = 0
    for index in range(keese):
        slot = index + 1
        ram[ADDR_OBJ_TYPE + slot] = KEESE_OBJECT_TYPE
        ram[ADDR_LINK_X + slot] = 48 + index * 16
        ram[ADDR_LINK_Y + slot] = 125
    return read_snapshot(ram)


def test_live_room62_settle_anchors() -> None:
    snap = _room62_snapshot()
    assert ROOM_LEVEL9_62 == 0x62
    assert KEESE_OBJECT_TYPE == 0x1B
    assert ROOM62_KEESE_COUNT == 8
    assert in_room_62(snap)
    assert len(room62_keese(snap)) == 8
    assert uncleared_room62(snap)
    assert door_bits(snap.cur_opened_doors)["north"] is False


def test_room62_nav_recenters_then_pushes_north() -> None:
    align = room62_to_patra_step(_room62_snapshot(link_x=NORTH_DOOR_X - 8))
    assert action_button_names(align.action) == ["RIGHT"]
    assert align.reason == "patra_align_x"

    push = room62_to_patra_step(_room62_snapshot(link_x=NORTH_DOOR_X))
    assert action_button_names(push.action) == ["UP"]
    assert push.reason == "patra_push_north"

    arrived = room62_to_patra_step(
        _room62_snapshot(screen=ROOM_BEFORE_GANON, keese=0)
    )
    assert arrived.reason == "patra_arrived"


def test_rom_and_live_disprove_cardinal_0x62_to_0x52() -> None:
    assert ROOM62_ROM_NORTH == 1
    assert ROOM62_ROM_SOUTH == 1
    assert ROOM62_ROM_WEST == 0
    assert ROOM62_ROM_EAST == 5
    assert ROOM52_ROM_NORTH == 7
    assert ROOM52_ROM_SOUTH == 1
    assert room62_is_cardinal_predecessor_of_patra() is False
    assert 0x52 not in LEVEL9_STAIR_SOURCES
    assert 0x62 not in LEVEL9_STAIR_SOURCES


def test_kill_clear_does_not_count_as_north_door() -> None:
    cleared = _room62_snapshot(keese=0, doors=0)
    assert in_room_62(cleared)
    assert not room62_keese(cleared)
    assert not uncleared_room62(cleared)
    assert not cleared.cur_opened_doors & NORTH_DOOR


def test_loader_fixture_writes_exclude_object_and_door_pokes() -> None:
    names = {row["name"] for row in _loader_write_rows(LOADER_CANDIDATES[0])}
    assert "loader_door_staging" in names
    assert "loader_current_room" in names
    assert "loader_next_room" in names
    assert "clear_final_patra_object_slots" not in names
    assert "mark_room_all_dead" not in names
    assert "open_north_door" not in names
    current = next(
        row for row in _loader_write_rows(LOADER_CANDIDATES[0])
        if row["name"] == "loader_current_room"
    )
    nxt = next(
        row for row in _loader_write_rows(LOADER_CANDIDATES[0])
        if row["name"] == "loader_next_room"
    )
    assert current["value"] == 0x72
    assert nxt["value"] == 0x62
