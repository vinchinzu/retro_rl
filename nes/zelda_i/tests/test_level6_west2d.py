"""Unit tests for Level 6 0x2D west door after south 0x1D."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_spine import L6_STOPS, L6_THROUGH
from zelda_i.level6_west2d import (
    WEST_DOOR_X,
    WEST_DOOR_Y,
    level6_west2d_stages,
    level6_west2d_success,
    make_west2d_controller,
)
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
    ram[ADDR_SCREEN] = fields.get("screen", 0x2D)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 77)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ROD] = fields.get("rod", 1)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    return ram


def test_west2d_through_composes_from_south1d() -> None:
    assert "level6-west2d" in L6_THROUGH
    assert L6_THROUGH.index("level6-west2d") == L6_THROUGH.index(
        "level6-south1d"
    ) + 1
    assert L6_THROUGH[L6_THROUGH.index("level6-west2d") + 1] == "level6-north2c"
    assert L6_STOPS["level6-west2d"] == "level6_west_0x2d"
    stages = level6_west2d_stages()
    assert [name for name, _, _ in stages] == [
        "level6_stairs_0x3a_warp",
        "level6_cellar_0x08",
        "level6_south_0x1d",
        "level6_west_0x2d",
    ]
    run = SpineRun(through="level6-west2d", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_west_0x2d"


def test_west2d_occupancy_y_then_left() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram()
    ctl = make_west2d_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "west_path"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    aisle = _ram(x=120, y=141)
    path = make_west2d_controller()
    act = path.step(read_snapshot(aisle))
    assert act.reason == "west_path"
    assert list(act.action) == list(nes_action("LEFT"))
    door = _ram(x=WEST_DOOR_X, y=WEST_DOOR_Y)
    push = make_west2d_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "west_push"
    assert list(act.action) == list(nes_action("LEFT"))


def test_west2d_requires_exact_play_0x2c() -> None:
    dest = _ram(screen=0x2C, x=208, y=141)
    assert level6_west2d_success(read_snapshot(dest))
    ctl = make_west2d_controller()
    ctl.keys = 4
    act = ctl.step(read_snapshot(dest))
    assert ctl.success
    assert not ctl.failed
    assert act.reason == "arrived_2c"
    still = _ram()
    assert not level6_west2d_success(read_snapshot(still))
    gohma = _ram(screen=0x1C, x=120, y=205)
    assert not level6_west2d_success(read_snapshot(gohma))
    north = _ram(screen=0x1D, x=96, y=157)
    assert not level6_west2d_success(read_snapshot(north))
    back = _ram(screen=0x3A, x=96, y=157)
    assert not level6_west2d_success(read_snapshot(back))
    wrong = make_west2d_controller()
    wrong.keys = 4
    fail = wrong.step(read_snapshot(gohma))
    assert wrong.failed
    assert fail.reason.startswith("wrong_room_1c")
    backtrack = make_west2d_controller()
    backtrack.keys = 4
    fail_n = backtrack.step(read_snapshot(north))
    assert backtrack.failed
    assert fail_n.reason.startswith("backtrack_1d")
    snap = read_snapshot(dest)
    assert snap.keys == 4
    assert snap.bow == 0
    assert snap.arrows == 0
