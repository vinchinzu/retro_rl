"""Unit tests for Level 6 0x1D south door after cellar 0x08."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_south1d import (
    SOUTH_DOOR_X,
    SOUTH_DOOR_Y,
    level6_south1d_stages,
    level6_south1d_success,
    make_south1d_controller,
)
from zelda_i.level6_spine import L6_STOPS, L6_THROUGH
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
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
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x1D)
    ram[ADDR_LINK_X] = fields.get("x", 96)
    ram[ADDR_LINK_Y] = fields.get("y", 157)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ROD] = fields.get("rod", 1)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    return ram


def test_south1d_through_composes_from_cellar08() -> None:
    assert "level6-south1d" in L6_THROUGH
    assert L6_THROUGH.index("level6-south1d") == L6_THROUGH.index(
        "level6-cellar08"
    ) + 1
    assert L6_STOPS["level6-south1d"] == "level6_south_0x1d"
    stages = level6_south1d_stages()
    assert [name for name, _, _ in stages] == [
        "level6_stairs_0x3a_warp",
        "level6_cellar_0x08",
        "level6_south_0x1d",
    ]
    run = SpineRun(through="level6-south1d", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_south_0x1d"


def test_south1d_occupancy_then_down() -> None:
    from retro_harness.nes import nes_action, nes_idle_action

    leftover = _ram()
    ctl = make_south1d_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "south_path"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    north = _ram(x=96, y=109)
    halt = make_south1d_controller()
    act = halt.step(read_snapshot(north))
    assert act.reason == "south_north_halt"
    assert list(act.action) == list(nes_idle_action())
    band = _ram(x=96, y=181)
    align = make_south1d_controller()
    act = align.step(read_snapshot(band))
    assert act.reason == "south_align"
    assert list(act.action) == list(nes_action("RIGHT"))
    door = _ram(x=SOUTH_DOOR_X, y=SOUTH_DOOR_Y)
    push = make_south1d_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "south_push"
    assert list(act.action) == list(nes_action("DOWN"))


def test_south1d_requires_exact_play_0x2d() -> None:
    dest = _ram(screen=0x2D, x=120, y=77)
    assert level6_south1d_success(read_snapshot(dest))
    ctl = make_south1d_controller()
    ctl.keys = 4
    act = ctl.step(read_snapshot(dest))
    assert ctl.success
    assert not ctl.failed
    assert act.reason == "arrived_2d"
    still = _ram()
    assert not level6_south1d_success(read_snapshot(still))
    gohma = _ram(screen=0x1C, x=120, y=205)
    assert not level6_south1d_success(read_snapshot(gohma))
    west = _ram(screen=0x2C, x=208, y=141)
    assert not level6_south1d_success(read_snapshot(west))
    back = _ram(screen=0x3A, x=96, y=157)
    assert not level6_south1d_success(read_snapshot(back))
    wrong = make_south1d_controller()
    wrong.keys = 4
    fail = wrong.step(read_snapshot(gohma))
    assert wrong.failed
    assert fail.reason.startswith("wrong_room_1c")
    snap = read_snapshot(dest)
    assert snap.keys == 4
    assert snap.bow == 0
    assert snap.arrows == 0
