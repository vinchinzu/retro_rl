"""Fixture-only L7 pond / entry-area -> L8 bush overworld route."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level8.overworld import (
    L7_POND_TO_LEVEL8_BUSH_HOPS,
    L7_POND_TO_LEVEL8_BUSH_SCREENS,
    POND_42_REFILLED_DEAD_POSE,
    Level7PondToLevel8BushController,
)
from zelda_i.overworld.graph import neighbor_screens
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
    read_snapshot,
)


def _ram(*, level: int = 0, screen: int = 0x42, x: int = 112, y: int = 141):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_SWORD] = 1
    ram[ADDR_HEALTH] = 0x22
    return ram


def test_l7_pond_to_l8_bush_hops_are_contiguous_and_end_0x6d() -> None:
    assert L7_POND_TO_LEVEL8_BUSH_SCREENS == (
        0x42,
        0x52,
        0x53,
        0x54,
        0x64,
        0x65,
        0x55,
        0x56,
        0x57,
        0x58,
        0x59,
        0x5A,
        0x5B,
        0x5C,
        0x5D,
        0x6D,
    )
    for source, target in zip(
        L7_POND_TO_LEVEL8_BUSH_SCREENS,
        L7_POND_TO_LEVEL8_BUSH_SCREENS[1:],
    ):
        assert target in neighbor_screens(source).values(), f"{source:#x}->{target:#x}"
    assert L7_POND_TO_LEVEL8_BUSH_HOPS[-1].direction == "DOWN"
    assert L7_POND_TO_LEVEL8_BUSH_HOPS[-1].align_x == 48


def test_level7_entrance_start_exits_naturally_down() -> None:
    ctl = Level7PondToLevel8BushController()
    snap = read_snapshot(_ram(level=7, screen=0x79, x=120, y=205))
    act = ctl.step(snap)
    assert not ctl.failed
    assert list(act.action) == list(nes_action("DOWN"))
    assert act.reason == "l7_entrance_exit_down"
    assert "fixture_start_l7_entry" in ctl.notes


def test_pond_start_begins_with_predicted_0x42_down_to_0x52() -> None:
    # OW_L7Pond south-shore pose: DOWN scrolls straight to 0x52, bypassing the
    # refilled pool (live probe l7_exit_to_l8_bush A18/A19: 2/2 to 0x6D).
    ctl = Level7PondToLevel8BushController()
    snap = read_snapshot(_ram(screen=0x42, x=112, y=221))
    act = ctl.step(snap)
    assert not ctl.failed
    assert list(act.action) == list(nes_action("DOWN"))
    assert ctl.hops[ctl.hop_index].target == 0x52
    assert "fixture_start_l7_pond" in ctl.notes


def test_refilled_pond_top_strip_fails_closed_anywhere() -> None:
    # Natural Level7Entrance exit strands Link in the y~85-100 strip at any x.
    ctl = Level7PondToLevel8BushController()
    ctl._fixture_start_checked = True
    snap = read_snapshot(_ram(screen=0x42, x=96, y=93))
    act = ctl.step(snap)
    assert ctl.failed and not ctl.success
    assert list(act.action) == list(nes_idle_action())
    assert "42_refilled_pond_straight_down_dead" in ctl.notes


def test_unpredicted_screen_halts_without_input() -> None:
    ctl = Level7PondToLevel8BushController()
    ctl.step(read_snapshot(_ram(screen=0x42)))
    act = ctl.step(read_snapshot(_ram(screen=0x43)))
    assert ctl.failed
    assert not ctl.success
    assert list(act.action) == list(nes_idle_action())
    assert "screen_prediction_miss_42_52_43" in ctl.notes


def test_natural_exit_refilled_pond_dead_pose_halts_without_retry() -> None:
    ctl = Level7PondToLevel8BushController()
    ctl._fixture_start_checked = True
    snap = read_snapshot(
        _ram(
            screen=0x42,
            x=POND_42_REFILLED_DEAD_POSE[0],
            y=POND_42_REFILLED_DEAD_POSE[1],
        )
    )
    act = ctl.step(snap)
    assert ctl.failed
    assert not ctl.success
    assert list(act.action) == list(nes_idle_action())
    assert "42_refilled_pond_straight_down_dead" in ctl.notes


def test_reverse_0x52_west_corridor_then_column_then_east() -> None:
    ctl = Level7PondToLevel8BushController()
    ctl.hop_index = 1
    # On the y~85 corridor, east of the west column: head west.
    top = read_snapshot(_ram(screen=0x52, x=112, y=88))
    ctl._nav_snap = top
    act = ctl._extra_hop_action(top, ctl.hops[1])
    assert act is not None
    assert act.reason == "52r_corridor_west"
    assert list(act.action) == list(nes_action("LEFT"))
    # Down in the bottom corridor: push east toward 0x53.
    bottom = read_snapshot(_ram(screen=0x52, x=48, y=190))
    ctl._nav_snap = bottom
    act = ctl._extra_hop_action(bottom, ctl.hops[1])
    assert act is not None
    assert act.reason == "52r_bottom_east"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_reverse_0x64_crosses_east_on_the_y141_band() -> None:
    ctl = Level7PondToLevel8BushController()
    ctl.hop_index = 4
    snap = read_snapshot(_ram(screen=0x64, x=180, y=141))
    ctl._nav_snap = snap
    act = ctl._extra_hop_action(snap, ctl.hops[4])
    assert act is not None
    assert act.reason == "64r_east_cross"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_fixture_report_never_claims_route_eligibility_or_writes() -> None:
    report = Level7PondToLevel8BushController().report()
    assert report["evidence"] == "fixture-live-prefix"
    assert report["natural_entry"] is False
    assert report["route_eligible"] is False
    assert report["writes"] == 0
