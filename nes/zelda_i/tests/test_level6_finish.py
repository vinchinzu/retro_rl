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
from zelda_i.screen_glance import (
    HEART_LEAVE,
    LEVEL6_LEAVE,
    NORTH0C_LEAVE,
    grade_final,
)
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_HEALTH,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_TYPE,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_ROOM_ITEM_ID,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 6,
    "screen": 0x1C,
    "x": 120,
    "y": 189,
    "triforce": 0x1F,
    "rod": 1,
    "bow": 1,
    "health": 0x66,
    "item": 0x1A,
    "doors": 0,
    "mask": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def test_heart_walks_up_from_south_mouth() -> None:
    ram = _ram()
    ctl = make_heart_controller()
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "occupancy_path"
    assert ctl.incoming_containers == 7


def test_heart_success_is_the_container_not_full_hearts() -> None:
    """Last-heart run 29 took the 0x1C heart (11 -> 12 containers) at 9 of
    12 hearts and stood 4000 frames waiting for a full bar only the
    Survival refill gives."""
    ram = _ram()
    assert not level6_heart_success(read_snapshot(ram))
    ram[ADDR_HEALTH] = 0x75  # 8 containers, 6 whole hearts
    ram[ADDR_ROOM_ITEM_ID] = 0
    assert level6_heart_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = GOHMA_OBJECT_TYPE
    assert not level6_heart_success(read_snapshot(ram))

    ctl = make_heart_controller()
    ram = _ram()
    ctl.step(read_snapshot(ram))
    ram[ADDR_HEALTH] = 0x75
    ram[ADDR_LINK_X], ram[ADDR_LINK_Y] = HEART_XY
    ctl.step(read_snapshot(ram))
    assert ctl.success and not ctl.failed


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
