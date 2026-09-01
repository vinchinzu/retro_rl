"""Unit tests for Level 6 post-Gohma finish (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon.ids import GOHMA_OBJECT_TYPE
from zelda_i.level6.finish import (
    DOOR_NORTH,
    FANFARE_MODE,
    HEART_XY,
    LEAVING_TF,
    Level6HeartController,
    level6_heart_success,
    level6_north0c_success,
    level6_success,
    make_heart_controller,
    make_north0c_controller,
    make_shard_controller,
    north_shutter_open,
)
from zelda_i.level6.spine import L6_STOPS, L6_THROUGH
from zelda_i.screen_glance import (
    HEART_LEAVE,
    LEVEL6_LEAVE,
    NORTH0C_LEAVE,
    grade_final,
)
from zelda_i.ram import (
    ADDR_BOW,
    ADDR_CUR_OPENED_DOORS,
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_TYPE,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_ROD,
    ADDR_ROOM_ITEM_ID,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x1C)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 189)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_ROD] = fields.get("rod", 1)
    ram[ADDR_BOW] = fields.get("bow", 1)
    ram[ADDR_HEALTH] = fields.get("health", 0x66)
    ram[ADDR_ROOM_ITEM_ID] = fields.get("item", 0x1A)
    ram[ADDR_CUR_OPENED_DOORS] = fields.get("doors", 0)
    ram[ADDR_OPEN_DOORWAY_MASK] = fields.get("mask", 0)
    return ram


def test_heart_walks_up_from_south_mouth() -> None:
    ram = _ram()
    ctl = make_heart_controller()
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "occupancy_path"
    assert ctl.incoming_containers == 7


def test_heart_success_needs_container_and_full() -> None:
    ram = _ram()
    assert not level6_heart_success(read_snapshot(ram))
    ram[ADDR_HEALTH] = 0x77
    ram[ADDR_ROOM_ITEM_ID] = 0
    assert level6_heart_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = GOHMA_OBJECT_TYPE
    assert not level6_heart_success(read_snapshot(ram))


def test_heart_arrives_on_plus_one_full() -> None:
    ram = _ram()
    ctl = make_heart_controller()
    ctl.step(read_snapshot(ram))
    ram[ADDR_HEALTH] = 0x77
    ram[ADDR_LINK_X] = HEART_XY[0]
    ram[ADDR_LINK_Y] = HEART_XY[1]
    ctl.step(read_snapshot(ram))
    assert ctl.success
    assert not ctl.failed


def test_heart_fails_if_gohma_live() -> None:
    ram = _ram()
    ram[ADDR_OBJ_TYPE + 1] = GOHMA_OBJECT_TYPE
    ctl = make_heart_controller()
    ctl.step(read_snapshot(ram))
    assert ctl.failed


def test_north_waits_closed_shutter_then_pushes() -> None:
    ram = _ram(x=120, y=141)
    ctl = make_north0c_controller()
    action = ctl.step(read_snapshot(ram))
    assert action.reason == "wait_shutter"
    assert not ctl.success
    ram[ADDR_CUR_OPENED_DOORS] = DOOR_NORTH
    action = ctl.step(read_snapshot(ram))
    assert "north0c" in action.reason
    assert not ctl.failed


def test_north_shutter_open_reads_either_field() -> None:
    ram = _ram()
    assert not north_shutter_open(read_snapshot(ram))
    ram[ADDR_OPEN_DOORWAY_MASK] = DOOR_NORTH
    assert north_shutter_open(read_snapshot(ram))


def test_north_fails_back_to_2c() -> None:
    ram = _ram(screen=0x2C, x=224, y=141)
    ctl = make_north0c_controller()
    ctl.step(read_snapshot(ram))
    assert ctl.failed


def test_north0c_success_is_play_0c_tf_still_1f() -> None:
    ram = _ram(screen=0x0C, x=120, y=205, health=0x77)
    assert level6_north0c_success(read_snapshot(ram))
    ram[ADDR_TRIFORCE] = LEAVING_TF
    assert not level6_north0c_success(read_snapshot(ram))


def test_shard_arrives_on_tf_3f_even_in_fanfare() -> None:
    ram = _ram(
        screen=0x0C, x=120, y=141, triforce=LEAVING_TF, mode=FANFARE_MODE,
        health=0x77,
    )
    ctl = make_shard_controller()
    ctl.step(read_snapshot(ram))
    assert ctl.success
    assert level6_success(read_snapshot(ram))


def test_shard_fails_back_to_1c() -> None:
    ram = _ram(screen=0x1C, x=120, y=93)
    ctl = make_shard_controller()
    ctl.step(read_snapshot(ram))
    assert ctl.failed


def test_finish_through_ids_are_on_the_catalog() -> None:
    assert "level6-heart" in L6_THROUGH
    assert "level6-north0c" in L6_THROUGH
    assert "level6" in L6_THROUGH
    assert L6_STOPS["level6-heart"] == "level6_heart_0x1c"
    assert L6_STOPS["level6-north0c"] == "level6_north_0x0c"
    assert L6_STOPS["level6"] == "level6_triforce_0x20"
    assert L6_THROUGH.index("level6-gohma") < L6_THROUGH.index("level6-heart")
    assert L6_THROUGH.index("level6-heart") < L6_THROUGH.index("level6")


def test_live_leftovers_glance() -> None:
    assert grade_final(
        {
            "room": 0x1C, "xy": [120, 149], "mode": PLAY_MODE, "triforce": 0x1F,
            "keys": 2, "bombs": 8, "health": 0x77,
        },
        HEART_LEAVE,
    ) == []
    assert grade_final(
        {
            "room": 0x0C, "xy": [120, 205], "mode": PLAY_MODE, "triforce": 0x1F,
            "keys": 2, "bombs": 8, "health": 0x77,
        },
        NORTH0C_LEAVE,
    ) == []
    assert grade_final(
        {
            "room": 0x0C, "xy": [120, 149], "mode": FANFARE_MODE,
            "triforce": LEAVING_TF, "keys": 2, "bombs": 8, "health": 0x77,
        },
        LEVEL6_LEAVE,
    ) == []


def test_heart_controller_report_has_leftover() -> None:
    ram = _ram()
    ctl = make_heart_controller()
    ctl.step(read_snapshot(ram))
    report = ctl.report()
    assert report["leftover"]["screen"] == 0x1C
    assert report["leftover"]["health"] == 0x66
    assert isinstance(ctl, Level6HeartController)
