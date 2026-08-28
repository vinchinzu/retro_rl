"""Unit tests for the historical Level 6 0x3A east-wall diagnostic."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_east3a import (
    DATED_SPIT,
    EAST_DOOR,
    SOUTH_AROUND_X,
    SOUTH_LANE_Y,
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
    PASSAGE_MODE,
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


def test_east3a_diagnostic_and_spit_stays_on_south_lane() -> None:
    from retro_harness.nes import nes_action

    stages = level6_east3a_stages()
    assert [name for name, _, _ in stages] == ["level6_east_0x3a"]
    assert "level6-east3a" in L6_THROUGH
    assert L6_STOPS["level6-east3a"] == "level6_east_0x3a"
    assert L6_THROUGH[L6_THROUGH.index("level6-cellar08") + 1] == "level6-south1d"
    assert L6_THROUGH[L6_THROUGH.index("level6-south1d") + 1] == "level6-west2d"
    assert L6_THROUGH[L6_THROUGH.index("level6-west2d") + 1] == "level6-north2c"
    assert L6_THROUGH[L6_THROUGH.index("level6-north2c") + 1] == "level6-east3a"
    ctl = make_east3a_controller()
    act = ctl.step(read_snapshot(_ram()))
    assert act.reason == "south_around_path"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    assert ctl.walker.last_dir == "RIGHT"
    ram2 = _ram(x=97, y=SOUTH_LANE_Y)
    act2 = ctl.step(read_snapshot(ram2))
    assert act2.reason == "south_around_path"
    assert not ctl.failed
    assert not ctl.success


def test_south_lane_repairs_y_before_crossing_hole() -> None:
    from retro_harness.nes import nes_action

    ctl = make_east3a_controller()
    ram = _ram(x=96, y=155)
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "south_around_path"
    assert list(act.action) == list(nes_action("DOWN"))
    assert not ctl.failed


def test_east_side_climbs_then_goes_right_and_dest_play_succeeds() -> None:
    from retro_harness.nes import nes_action

    ram = _ram(x=SOUTH_AROUND_X, y=SOUTH_LANE_Y)
    ctl = make_east3a_controller()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "east_side_path"
    assert list(act.action) == list(nes_action("UP"))
    door_band = _ram(x=SOUTH_AROUND_X, y=EAST_DOOR[1])
    ctl = make_east3a_controller()
    act = ctl.step(read_snapshot(door_band))
    assert act.reason == "door_path"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert ctl.walker.last_dir == "RIGHT"
    assert not ctl.failed
    dest = _ram(mode=PLAY_MODE, screen=0x3B, x=16, y=141)
    assert level6_east3a_success(read_snapshot(dest))
    still = _ram()
    assert not level6_east3a_success(read_snapshot(still))
    passage = _ram(mode=PASSAGE_MODE, screen=0x08)
    assert level6_east3a_success(read_snapshot(passage))


def test_south_around_blocks_and_replans_on_a_new_occupancy_miss() -> None:
    ctl = make_east3a_controller()
    start = read_snapshot(_ram())
    ctl.step(start)
    stride = read_snapshot(_ram(x=99))
    replanned = ctl.step(stride)
    assert replanned.reason == "south_around_path"
    assert ctl.walker.misses == 1
    assert (97, SOUTH_LANE_Y) in ctl.walker.grid.blocked
    assert any(note.startswith("miss_f2_RIGHT_99_157") for note in ctl.notes)
    assert not ctl.failed


def test_rom_east_wall_halts_at_the_visual_mouth() -> None:
    ctl = make_east3a_controller()
    wall = read_snapshot(_ram(x=EAST_DOOR[0], y=EAST_DOOR[1]))
    action = ctl.step(wall)
    assert action.reason.startswith("rom_wall_east_208_141")
    assert ctl.failed
    assert not ctl.success
