"""Unit tests for the historical Level 6 0x3A east-wall diagnostic."""

from __future__ import annotations

import numpy as np

from zelda_i.level6.east3a import (
    DATED_SPIT,
    EAST_DOOR,
    SOUTH_AROUND_X,
    SOUTH_LANE_Y,
    level6_east3a_success,
    make_east3a_controller,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 6,
    "screen": 0x3A,
    "x": DATED_SPIT[0],
    "y": DATED_SPIT[1],
    "triforce": 0x1F,
    "keys": 4,
    "bombs": 8,
    "rod": 1,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)
    ram[ADDR_ROD] = fields.get("rod", 1)
    return ram


def test_east3a_diagnostic_and_spit_stays_on_south_lane() -> None:
    from retro_harness.nes import nes_action

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
