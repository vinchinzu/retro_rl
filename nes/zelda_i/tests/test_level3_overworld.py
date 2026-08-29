"""Unit tests for Level 3 overworld hop tables and stop predicates."""

from __future__ import annotations

from zelda_i.level3_overworld import (
    LEVEL3_HOPS_FROM_POST_L2,
    LEVEL3_POST_L2_SCREENS,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_POST_L2_RETURN,
    OverworldPostL2ToLevel3Controller,
)
from zelda_i.overworld import neighbor_screens
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot


def test_post_l2_path_screens_chain() -> None:
    assert LEVEL3_POST_L2_SCREENS[0] == SCREEN_POST_L2_RETURN == 0x3C
    assert LEVEL3_POST_L2_SCREENS[-1] == SCREEN_LEVEL3_ENTRANCE == 0x74
    assert len(LEVEL3_HOPS_FROM_POST_L2) == len(LEVEL3_POST_L2_SCREENS) - 1
    for a, b in zip(LEVEL3_POST_L2_SCREENS, LEVEL3_POST_L2_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


def test_post_l2_leave_64_inland_when_wrong_y() -> None:
    """West rock face at wrong y must step inland before band align."""
    nav = OverworldPostL2ToLevel3Controller()
    hop = next(h for h in nav.hops if h.target == 0x63)
    snap = ZeldaSnapshot(
        mode=PLAY_MODE,
        level=0,
        screen=0x64,
        next_screen=0x64,
        link_x=24,
        link_y=109,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=127,
        triforce=0x03,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )
    act = nav._extra_hop_action(snap, hop)
    assert act is not None
    assert "64" in act.reason or "inland" in act.reason
