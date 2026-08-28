"""Unit tests for Level 6 0x2C KEY-UP after west 0x2D."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_north2c import (
    NORTH_DOOR_X,
    NORTH_DOOR_Y,
    level6_north2c_stages,
    level6_north2c_success,
    make_north2c_controller,
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
    ram[ADDR_SCREEN] = fields.get("screen", 0x2C)
    ram[ADDR_LINK_X] = fields.get("x", 224)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 4)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ROD] = fields.get("rod", 1)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    return ram


def test_north2c_through_composes_from_west2d() -> None:
    assert "level6-north2c" in L6_THROUGH
    assert L6_THROUGH.index("level6-north2c") == L6_THROUGH.index(
        "level6-west2d"
    ) + 1
    assert L6_THROUGH[L6_THROUGH.index("level6-north2c") + 1] == "level6-east3a"
    assert L6_STOPS["level6-north2c"] == "level6_north_0x2c"
    stages = level6_north2c_stages()
    assert [name for name, _, _ in stages] == [
        "level6_stairs_0x3a_warp",
        "level6_cellar_0x08",
        "level6_south_0x1d",
        "level6_west_0x2d",
        "level6_north_0x2c",
    ]
    run = SpineRun(through="level6-north2c", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_north_0x2c"


def test_north2c_occupancy_x_then_up() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram()
    ctl = make_north2c_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "north_path"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))
    aisle = _ram(x=120, y=141)
    path = make_north2c_controller()
    act = path.step(read_snapshot(aisle))
    assert act.reason == "north_path"
    assert list(act.action) == list(nes_action("UP"))
    band = _ram(x=120, y=109)
    push = make_north2c_controller()
    act = push.step(read_snapshot(band))
    assert act.reason == "north_push"
    assert list(act.action) == list(nes_action("UP"))
    door = _ram(x=NORTH_DOOR_X, y=NORTH_DOOR_Y)
    door_push = make_north2c_controller()
    act = door_push.step(read_snapshot(door))
    assert act.reason == "north_push"
    assert list(act.action) == list(nes_action("UP"))
    south = _ram(x=120, y=189)
    recover = make_north2c_controller()
    act = recover.step(read_snapshot(south))
    assert list(act.action) != list(nes_action("DOWN"))
    assert act.reason in ("north_path", "north_push")


def test_north2c_requires_exact_play_0x1c() -> None:
    dest = _ram(screen=0x1C, x=120, y=205, keys=3)
    assert level6_north2c_success(read_snapshot(dest))
    ctl = make_north2c_controller()
    ctl.keys = 4
    act = ctl.step(read_snapshot(dest))
    assert ctl.success
    assert not ctl.failed
    assert act.reason == "arrived_1c"
    assert any(n.startswith("key_spent_2c_to_1c_4->3") for n in ctl.notes)
    still = _ram()
    assert not level6_north2c_success(read_snapshot(still))
    back = _ram(screen=0x2D, x=32, y=141)
    assert not level6_north2c_success(read_snapshot(back))
    south = _ram(screen=0x3C, x=120, y=77)
    assert not level6_north2c_success(read_snapshot(south))
    backtrack = make_north2c_controller()
    backtrack.keys = 4
    fail_e = backtrack.step(read_snapshot(back))
    assert backtrack.failed
    assert fail_e.reason.startswith("backtrack_2d")
    wrong = make_north2c_controller()
    wrong.keys = 4
    fail_s = wrong.step(read_snapshot(south))
    assert wrong.failed
    assert fail_s.reason.startswith("wrong_room_3c")
    snap = read_snapshot(dest)
    assert snap.keys == 3
    assert snap.bow == 0
    assert snap.arrows == 0
