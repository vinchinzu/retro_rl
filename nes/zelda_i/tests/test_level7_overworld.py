"""Unit tests for the Level 7 pond 0x53 inland-left micro (no emulator)."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import (
    SCREEN_BRACELET_ARMOS,
    SCREEN_LEVEL6_ENTRANCE,
)
from zelda_i.level7.overworld import (
    BAIT_32_CORRIDOR_X,
    LEVEL7_POND_HOPS,
    POND_53_INLAND_X,
    POND_53_WEST_GAP_Y,
    POST_L6_TO_BAIT_HOPS,
    POST_L6_TO_BAIT_SCREENS,
    OverworldToBaitShopController,
    OverworldToLevel7PondController,
    at_l6_cave_mouth,
    has_whistle,
)
from zelda_i.overworld.graph import neighbor_screens
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x53)
    ram[ADDR_LINK_X] = fields.get("x", 224)
    ram[ADDR_LINK_Y] = fields.get("y", 173)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_WHISTLE] = fields.get("whistle", 0)
    return ram


def test_pond_hops_are_contiguous_through_53_52_42() -> None:
    screens = [0x77] + [h.target for h in LEVEL7_POND_HOPS]
    assert screens[-3:] == [0x53, 0x52, 0x42]
    for a, b in zip(screens, screens[1:]):
        assert b in neighbor_screens(a).values(), f"{a:#x}->{b:#x}"
    hop52 = LEVEL7_POND_HOPS[10]
    assert hop52.target == 0x52
    assert hop52.direction == "LEFT"
    assert hop52.align_y == POND_53_WEST_GAP_Y


def test_pond_53_east_edge_goes_inland_left_not_down() -> None:
    """v9 leftover (224,173): hop10_ay DOWN is the dead belief."""
    ctl = OverworldToLevel7PondController()
    hop = ctl.hops[10]
    snap = read_snapshot(_ram(x=224, y=173))
    act = ctl._extra_hop_action(snap, hop)
    assert act is not None
    assert "53_inland_left" in act.reason
    assert "DOWN" not in act.reason
    assert "descend" not in act.reason


def test_pond_53_arrival_240_141_also_inland_left() -> None:
    ctl = OverworldToLevel7PondController()
    hop = ctl.hops[10]
    snap = read_snapshot(_ram(x=240, y=141))
    act = ctl._extra_hop_action(snap, hop)
    assert act is not None
    assert "53_inland_left" in act.reason
    assert snap.link_x > POND_53_INLAND_X


def test_pond_53_inland_then_descend_or_left() -> None:
    ctl = OverworldToLevel7PondController()
    hop = ctl.hops[10]
    snap = read_snapshot(_ram(x=150, y=173))
    act = ctl._extra_hop_action(snap, hop)
    assert act is not None
    assert act.reason.startswith("53_")
    assert "inland_left" not in act.reason
    assert any(token in act.reason for token in ("descend", "left_west", "up"))


def test_pond_53_aligned_gap_defers_to_hop_left() -> None:
    ctl = OverworldToLevel7PondController()
    hop = ctl.hops[10]
    snap = read_snapshot(_ram(x=40, y=POND_53_WEST_GAP_Y))
    act = ctl._extra_hop_action(snap, hop)
    assert act is None


def test_pond_53_south_band_leftover_defers_to_hop_left() -> None:
    """l7_dnp_pond_53 leftover (176,205): do not occupancy-UP back to 189."""
    ctl = OverworldToLevel7PondController()
    hop = ctl.hops[10]
    snap = read_snapshot(_ram(x=176, y=205))
    act = ctl._extra_hop_action(snap, hop)
    assert act is None


def test_post_l6_to_bait_hops_are_contiguous() -> None:
    assert POST_L6_TO_BAIT_SCREENS == (
        SCREEN_LEVEL6_ENTRANCE,
        0x32,
        0x33,
        0x23,
        SCREEN_BRACELET_ARMOS,
        0x25,
    )
    assert POST_L6_TO_BAIT_HOPS[0].direction == "DOWN"
    assert POST_L6_TO_BAIT_HOPS[0].align_x == 112
    assert POST_L6_TO_BAIT_HOPS[2].direction == "UP"
    assert POST_L6_TO_BAIT_HOPS[2].align_x == 208
    assert POST_L6_TO_BAIT_HOPS[-1].target == 0x25
    assert POST_L6_TO_BAIT_HOPS[-1].direction == "RIGHT"
    assert POST_L6_TO_BAIT_HOPS[-1].align_y == 141
    for a, b in zip(POST_L6_TO_BAIT_SCREENS, POST_L6_TO_BAIT_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:#x}->{b:#x}"


def test_post_l6_south_leftover_goes_down_not_into_cave() -> None:
    """Live leftover (120,221): first travel is DOWN. Cave mouth is (112,125)."""
    ctl = OverworldToBaitShopController()
    snap = read_snapshot(_ram(screen=0x22, x=120, y=221, sword=1, triforce=0x3F))
    act = ctl.step(snap)
    assert not ctl.failed
    assert list(act.action) == list(nes_action("DOWN"))
    assert not at_l6_cave_mouth(snap)
    assert ctl.report()["route_eligible"] is False
    assert ctl.report()["writes"] == 0


def test_bait_32_north_mouth_aligns_x112_not_down() -> None:
    """l7_bait_from_l6 leftover (120,61): DOWN is the east wall of x=112."""
    ctl = OverworldToBaitShopController()
    hop = ctl.hops[1]
    assert hop.target == 0x33
    snap = read_snapshot(_ram(screen=0x32, x=120, y=61, sword=1))
    act = ctl._extra_hop_action(snap, hop)
    assert act is not None
    assert "32_north_ax" in act.reason
    assert "DOWN" not in act.reason
    aligned = read_snapshot(_ram(screen=0x32, x=BAIT_32_CORRIDOR_X, y=61, sword=1))
    down = ctl._extra_hop_action(aligned, hop)
    assert down is not None
    assert "32_north_down" in down.reason
    inland = read_snapshot(_ram(screen=0x32, x=112, y=141, sword=1))
    assert ctl._extra_hop_action(inland, hop) is None


def test_bait_24_sw_leftover_goes_east_not_down() -> None:
    """l7_bait_33up leftover 0x24 (16,189): DOWN is mountain; UP to y=141 then RIGHT."""
    ctl = OverworldToBaitShopController()
    hop = ctl.hops[-1]
    assert hop.target == 0x25
    snap = read_snapshot(_ram(screen=0x24, x=16, y=189, sword=1))
    act = ctl._extra_hop_action(snap, hop)
    assert act is not None
    assert "24_east_band" in act.reason
    assert "DOWN" not in act.reason
    assert not ctl.failed
    inland = read_snapshot(_ram(screen=0x24, x=0, y=141, sword=1))
    inland_act = ctl._extra_hop_action(inland, hop)
    assert inland_act is not None
    assert "24_west_inland" in inland_act.reason
    band = read_snapshot(_ram(screen=0x24, x=16, y=141, sword=1))
    assert ctl._extra_hop_action(band, hop) is None
    corner = read_snapshot(_ram(screen=0x24, x=25, y=181, sword=1))
    corner_act = ctl._extra_hop_action(corner, hop)
    assert corner_act is not None
    assert "24_east_band" in corner_act.reason
    assert "DOWN" not in corner_act.reason
    belt = read_snapshot(_ram(screen=0x24, x=80, y=173, sword=1))
    belt_act = ctl._extra_hop_action(belt, hop)
    assert belt_act is not None
    assert "24_east_band" in belt_act.reason
    assert "DOWN" not in belt_act.reason
    ladder_x = read_snapshot(_ram(screen=0x24, x=160, y=189, sword=1))
    ladder_act = ctl._extra_hop_action(ladder_x, hop)
    assert ladder_act is not None
    assert "DOWN" not in ladder_act.reason
    se = read_snapshot(_ram(screen=0x24, x=208, y=189, sword=1))
    se_act = ctl._extra_hop_action(se, hop)
    assert se_act is not None
    assert "24_east_band" in se_act.reason
    assert "DOWN" not in se_act.reason
    assert not ctl.failed
    se_band = read_snapshot(_ram(screen=0x24, x=208, y=141, sword=1))
    assert ctl._extra_hop_action(se_band, hop) is None


def test_l6_cave_mouth_112_125_refuses() -> None:
    """Dead belief: (112,125) on 0x22 is the leave. It starts L6 enter."""
    ctl = OverworldToBaitShopController()
    snap = read_snapshot(_ram(screen=0x22, x=112, y=125, sword=1))
    act = ctl.step(snap)
    assert at_l6_cave_mouth(snap)
    assert ctl.failed
    assert act.reason == "l6_cave_mouth"
    assert list(act.action) == list(nes_idle_action())
    act2 = ctl.step(snap)
    assert list(act2.action) == list(nes_idle_action())


def test_pond_geometry_does_not_require_whistle() -> None:
    ram = _ram(whistle=0)
    assert not has_whistle(ram)
    assert read_u8(ram, ADDR_WHISTLE) == 0
    ctl = OverworldToLevel7PondController()
    act = ctl._extra_hop_action(read_snapshot(ram), ctl.hops[10])
    assert act is not None
    assert "53_inland_left" in act.reason
