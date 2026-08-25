"""Unit tests for Level 6 cellar 0x08 after the 0x3A warp."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_cellar08 import (
    CELLAR_08_ROOM,
    EAST_MOUTH,
    level6_cellar08_stages,
    level6_cellar08_success,
    make_cellar08_controller,
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
from zelda_i.survival_spine import SpineRun


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", 9)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", CELLAR_08_ROOM)
    ram[ADDR_LINK_X] = fields.get("x", EAST_MOUTH[0])
    ram[ADDR_LINK_Y] = fields.get("y", EAST_MOUTH[1])
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ROD] = fields.get("rod", 1)
    return ram


def test_center3a_from_spit_goes_right_not_east() -> None:
    from retro_harness.nes import nes_action
    from zelda_i.level6_center3a import (
        DATED_SPIT,
        level6_center3a_stages,
        level6_center3a_success,
        make_center3a_controller,
    )
    from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y

    stages = level6_center3a_stages()
    assert [name for name, _, _ in stages] == [
        "level6_stairs_0x3a_warp",
        "level6_cellar_0x08",
        "level6_center_0x3a",
    ]
    ram = _ram(mode=PLAY_MODE, screen=0x3A, x=DATED_SPIT[0], y=DATED_SPIT[1])
    ram[ADDR_LINK_X] = DATED_SPIT[0]
    ram[ADDR_LINK_Y] = DATED_SPIT[1]
    ctl = make_center3a_controller()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "hole_x"
    assert list(act.action) == list(nes_action("RIGHT"))
    loop = _ram(mode=9, screen=CELLAR_08_ROOM, x=48, y=93)
    assert not level6_center3a_success(read_snapshot(loop))


def test_cellar08_through_is_wired_after_warp() -> None:
    assert "level6-cellar08" in L6_THROUGH
    assert L6_THROUGH.index("level6-cellar08") == L6_THROUGH.index(
        "level6-stairs3a-warp"
    ) + 1
    assert L6_STOPS["level6-cellar08"] == "level6_cellar_0x08"
    stages = level6_cellar08_stages()
    assert [name for name, _, _ in stages] == [
        "level6_stairs_0x3a_warp",
        "level6_cellar_0x08",
    ]
    run = SpineRun(through="level6-cellar08", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_cellar_0x08"


def test_east_mouth_drops_not_up() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram()
    ctl = make_cellar08_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "drop_clip"
    assert list(act.action) == list(nes_action("LEFT", "DOWN"))
    assert list(act.action) != list(nes_action("UP"))


def test_emerge_play_includes_dated_0x3a_return() -> None:
    emerge = _ram(mode=PLAY_MODE, screen=0x07, x=120, y=205)
    assert level6_cellar08_success(read_snapshot(emerge))
    still = _ram(mode=9, screen=CELLAR_08_ROOM)
    assert not level6_cellar08_success(read_snapshot(still))
    back = _ram(mode=PLAY_MODE, screen=0x3A, x=96, y=157)
    assert level6_cellar08_success(read_snapshot(back))
    ctl = make_cellar08_controller()
    ctl.step(read_snapshot(_ram()))
    act = ctl.step(read_snapshot(back))
    assert ctl.success
    assert act.reason == "emerged"
