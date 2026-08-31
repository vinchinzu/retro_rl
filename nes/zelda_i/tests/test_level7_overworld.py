"""Unit tests for the Level 7 pond 0x53 inland-left micro (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level7.overworld import (
    LEVEL7_POND_HOPS,
    POND_53_INLAND_X,
    POND_53_WEST_GAP_Y,
    OverworldToLevel7PondController,
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


def test_pond_geometry_does_not_require_whistle() -> None:
    ram = _ram(whistle=0)
    assert not has_whistle(ram)
    assert read_u8(ram, ADDR_WHISTLE) == 0
    ctl = OverworldToLevel7PondController()
    act = ctl._extra_hop_action(read_snapshot(ram), ctl.hops[10])
    assert act is not None
    assert "53_inland_left" in act.reason
