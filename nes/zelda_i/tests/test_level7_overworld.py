"""Unit tests for the Level 7 pond 0x53 inland-left micro (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import (
    SCREEN_BRACELET_ARMOS,
    SCREEN_LEVEL6_ENTRANCE,
    SCREEN_LEVEL7_BAIT_SHOP_HYP,
    SCREEN_LEVEL7_POND_HYP,
    SCREEN_MAGICAL_SWORD_GRAVE,
)
from zelda_i.level7.entry import (
    MEASURED_POST_L6_EXIT,
    make_post_l6_overworld_controller,
)
from zelda_i.level7.overworld import (
    BAIT_32_CORRIDOR_X,
    LEVEL7_POND_HOPS,
    POND_53_INLAND_X,
    POND_53_WEST_GAP_Y,
    POST_L6_22_WEST_HOP,
    POST_L6_TO_BAIT_HOPS,
    POST_L6_TO_BAIT_SCREENS,
    POST_L6_TO_POND_HOPS,
    POST_L6_TO_POND_SCREENS,
    SHOP_44_NORTH_X,
    SHOP_54_NORTH_X,
    WARP_JOIN_TO_POND_HOPS,
    WARP_JOIN_TO_SHOP_HOPS,
    WARP_JOIN_TO_SHOP_SCREENS,
    OverworldToBaitShopController,
    OverworldToLevel7PondController,
    at_l6_cave_mouth,
    has_whistle,
    make_pond_22_walker,
    pond_22_to_21_action,
)
from zelda_i.overworld.graph import neighbor_screens
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 0,
    "screen": 0x53,
    "x": 224,
    "y": 173,
    "sword": 1,
    "whistle": 0,
    "triforce": 0,
    "keys": 0,
    "bombs": 0,
    "arrows": 0,
    "health": 0x77,
    "food": 0,
    "rod": 0,
    "bow": 0,
    "candle": 0,
    "rupees": 0,
    "selected": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


def _measured_leave_ram() -> np.ndarray:
    return _ram(
        screen=SCREEN_LEVEL6_ENTRANCE,
        x=112,
        y=125,
        triforce=0x3F,
        keys=2,
        bombs=7,
        arrows=1,
        health=0xAA,
        whistle=1,
        food=0,
        rod=1,
        bow=1,
        candle=1,
        rupees=42,
        selected=2,
        sword=1,
    )


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


def test_post_l6_to_pond_hops_are_contiguous_neighbors() -> None:
    """Greened L6-reverse prefix to 0x24. 0x22 west→0x21 is dead mountain."""
    screens = POST_L6_TO_POND_SCREENS
    assert screens == (
        SCREEN_LEVEL6_ENTRANCE,
        0x32,
        0x33,
        0x23,
        SCREEN_BRACELET_ARMOS,
        0x14,
        0x13,
        0x12,
    )
    for a, b in zip(screens, screens[1:]):
        assert b in neighbor_screens(a).values(), f"{a:#x}->{b:#x}"
    assert POST_L6_TO_POND_HOPS[0].direction == "DOWN"
    assert POST_L6_TO_POND_HOPS[0].align_x == 112
    assert POST_L6_TO_POND_HOPS[-1].target == 0x12
    assert POST_L6_TO_POND_HOPS[-1].direction == "LEFT"
    assert POST_L6_TO_POND_HOPS[-1].y_band == (165, 189)
    assert 0x25 not in screens
    assert POST_L6_22_WEST_HOP.target == SCREEN_MAGICAL_SWORD_GRAVE
    assert POST_L6_22_WEST_HOP.direction == "LEFT"


def _post_l6_left_hop(ram, *, target: int):
    ctl = make_post_l6_overworld_controller(handoff=MEASURED_POST_L6_EXIT)
    ctl.bind_env(_env(ram))
    ctl._handoff_checked = True
    ctl.hop_index = [h.target for h in POST_L6_TO_POND_HOPS].index(target)
    return ctl.step(read_snapshot(ram))


def test_post_l6_0x13_east_mouth_left_not_down() -> None:
    """l7_p14w leftover (240,189): LEFT the south sand. Not DOWN the SE."""
    ram = _ram(screen=0x13, x=240, y=189, sword=1, whistle=1)
    act = _post_l6_left_hop(ram, target=0x12)
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert act.reason == "off_east"


def test_post_l6_0x12_west_wall_is_not_a_hop_target() -> None:
    """l7_p12w: 0x12 LEFT y=165-189 is west wall tile 218. Do not add 0x11."""
    assert 0x11 not in POST_L6_TO_POND_SCREENS


def test_default_post_l6_controller_does_not_end_at_0x25() -> None:
    ctl = make_post_l6_overworld_controller()
    assert ctl.hops == POST_L6_TO_POND_HOPS
    assert ctl.hops != POST_L6_TO_BAIT_HOPS
    if ctl.hops:
        assert ctl.hops[-1].target != 0x25
    ram = _measured_leave_ram()
    live = make_post_l6_overworld_controller(handoff=MEASURED_POST_L6_EXIT)
    live.bind_env(_env(ram))
    act = live.step(read_snapshot(ram))
    if not POST_L6_TO_POND_HOPS:
        assert live.failed
        assert act.reason == "post_l6_path_unmeasured"
    else:
        assert not live.failed


def test_post_l6_cave_mouth_reentry_still_refused() -> None:
    """2026-09-04: ``hop.target == 0x21`` no longer gets the dedicated
    ``pond_22_to_21_action`` mouth-leaving controller integration (that
    branch was confirmed dead -- 0x21 is not in ``POST_L6_TO_POND_HOPS``
    and was removed from ``_extra_hop_action``; the standalone function is
    still directly tested in ``test_pond_22_cave_mouth_goes_down_not_up``).
    The generic ``align_and_push`` engine still drives DOWN off the mouth
    toward ``POST_L6_22_WEST_HOP``'s ``align_y``, and mouth-reentry refusal
    is unrelated to which function produced the DOWN action.
    """
    ram = _measured_leave_ram()
    ctl = make_post_l6_overworld_controller(
        handoff=MEASURED_POST_L6_EXIT, hops=(POST_L6_22_WEST_HOP,)
    )
    ctl.bind_env(_env(ram))
    snap = read_snapshot(ram)
    act = ctl.step(snap)
    assert not ctl.failed
    assert act.reason != "l6_cave_mouth"
    assert list(act.action) == list(nes_action("DOWN"))
    assert "post_l6_handoff_accepted" in ctl.notes
    ctl._left_mouth = True
    act = ctl.step(snap)
    assert ctl.failed
    assert act.reason == "l6_cave_mouth_reentry"
    assert list(act.action) == list(nes_idle_action())


def test_pond_22_cave_mouth_goes_down_not_up() -> None:
    """Measured leave (112,125): first action is DOWN, never UP into L6."""
    snap = read_snapshot(_ram(screen=0x22, x=112, y=125, sword=1))
    assert at_l6_cave_mouth(snap)
    walker = make_pond_22_walker()

    def swing(direction: str, reason: str) -> FrameAction:
        return FrameAction(nes_action(direction), reason)

    act = pond_22_to_21_action(snap, walker=walker, swing=swing)
    assert act is not None
    assert act.reason == "22_leave_mouth"
    assert list(act.action) == list(nes_action("DOWN"))
    assert "UP" not in act.reason


def test_after_hops_succeeds_on_pond_0x42() -> None:
    ctl = make_post_l6_overworld_controller(handoff=MEASURED_POST_L6_EXIT)
    snap = read_snapshot(
        _ram(screen=SCREEN_LEVEL7_POND_HYP, x=128, y=160, sword=1, mode=PLAY_MODE)
    )
    act = ctl._after_hops(snap)
    assert ctl.success
    assert not ctl.failed
    assert act.reason == "done"
    inland = read_snapshot(_ram(screen=0x21, x=120, y=141, sword=1))
    ctl2 = make_post_l6_overworld_controller(handoff=MEASURED_POST_L6_EXIT)
    act2 = ctl2._after_hops(inland)
    assert ctl2.failed
    assert act2.reason == "post_l6_path_exhausted_unmeasured"


def test_shop_join_peels_north_at_0x54_not_west_to_pond() -> None:
    """rr-8t4.4: warp-join through 0x54, then UP to 0x44/0x34. Not LEFT to 0x53."""
    assert WARP_JOIN_TO_SHOP_HOPS[:4] == WARP_JOIN_TO_POND_HOPS[:4]
    assert WARP_JOIN_TO_SHOP_SCREENS == (0x45, 0x55, 0x65, 0x64, 0x54, 0x44, 0x34)
    assert WARP_JOIN_TO_SHOP_HOPS[-1].target == SCREEN_LEVEL7_BAIT_SHOP_HYP
    assert 0x53 not in WARP_JOIN_TO_SHOP_SCREENS
    assert 0x42 not in WARP_JOIN_TO_SHOP_SCREENS
    for a, b in zip(WARP_JOIN_TO_SHOP_SCREENS, WARP_JOIN_TO_SHOP_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:#x}->{b:#x}"
    assert WARP_JOIN_TO_SHOP_HOPS[-2].align_x == SHOP_54_NORTH_X
    assert WARP_JOIN_TO_SHOP_HOPS[-1].align_x == SHOP_44_NORTH_X


def test_shop_54_south_aligns_x116_not_left_to_pond() -> None:
    """0x54 leftover from 0x64 UP is x≈60 south. Shop gap is x≈116; pond is LEFT."""
    ctl = OverworldToLevel7PondController(hops=WARP_JOIN_TO_SHOP_HOPS)
    hop = ctl.hops[4]
    assert hop.target == 0x44
    snap = read_snapshot(_ram(screen=0x54, x=60, y=205, sword=1))
    act = ctl._extra_hop_action(snap, hop)
    assert act is not None
    assert "54_shop_ax" in act.reason
    assert "LEFT" not in act.reason
    aligned = read_snapshot(_ram(screen=0x54, x=SHOP_54_NORTH_X, y=205, sword=1))
    up = ctl._extra_hop_action(aligned, hop)
    assert up is not None
    assert "54_shop_north" in up.reason
    assert "LEFT" not in up.reason


def test_shop_44_south_aligns_x132_then_up() -> None:
    ctl = OverworldToLevel7PondController(hops=WARP_JOIN_TO_SHOP_HOPS)
    hop = ctl.hops[5]
    assert hop.target == SCREEN_LEVEL7_BAIT_SHOP_HYP
    snap = read_snapshot(_ram(screen=0x44, x=SHOP_54_NORTH_X, y=205, sword=1))
    act = ctl._extra_hop_action(snap, hop)
    assert act is not None
    assert "44_shop_ax" in act.reason
    aligned = read_snapshot(_ram(screen=0x44, x=SHOP_44_NORTH_X, y=205, sword=1))
    up = ctl._extra_hop_action(aligned, hop)
    assert up is not None
    assert "44_shop_north" in up.reason
