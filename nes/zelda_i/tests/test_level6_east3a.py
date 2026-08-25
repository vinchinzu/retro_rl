"""Unit tests for Level 6 0x3A east door after cellar08."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_east3a import (
    DATED_SPIT,
    EAST_DOOR,
    level6_east3a_stages,
    level6_east3a_success,
    make_east3a_controller,
)
from zelda_i.level6_spine import L6_STOPS, L6_THROUGH
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x3A)
    ram[ADDR_LINK_X] = fields.get("x", DATED_SPIT[0])
    ram[ADDR_LINK_Y] = fields.get("y", DATED_SPIT[1])
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ROD] = fields.get("rod", 1)
    return ram


def test_east3a_through_and_spit_goes_north_not_east() -> None:
    from retro_harness.nes import nes_action

    assert "level6-east3a" in L6_THROUGH
    assert L6_STOPS["level6-east3a"] == "level6_east_0x3a"
    stages = level6_east3a_stages()
    assert [name for name, _, _ in stages] == [
        "level6_stairs_0x3a_warp",
        "level6_cellar_0x08",
        "level6_east_0x3a",
    ]
    ctl = make_east3a_controller()
    act = ctl.step(read_snapshot(_ram()))
    assert act.reason == "door_y"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("RIGHT"))
    ram2 = _ram(x=96, y=155)
    act2 = ctl.step(read_snapshot(ram2))
    assert act2.reason == "door_y"
    assert not ctl.failed


def test_y143_still_aligns_up_not_right() -> None:
    from retro_harness.nes import nes_action

    ctl = make_east3a_controller()
    ram = _ram(x=96, y=143)
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "door_y"
    assert list(act.action) == list(nes_action("UP"))
    assert not ctl.failed


def test_y141_goes_right_and_dest_play_succeeds() -> None:
    from retro_harness.nes import nes_action

    ram = _ram(x=96, y=EAST_DOOR[1])
    ctl = make_east3a_controller()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "door_x"
    assert list(act.action) == list(nes_action("RIGHT"))
    dest = _ram(mode=PLAY_MODE, screen=0x3B, x=16, y=141)
    assert level6_east3a_success(read_snapshot(dest))
    still = _ram()
    assert not level6_east3a_success(read_snapshot(still))
