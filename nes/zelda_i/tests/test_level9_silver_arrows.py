"""Unit tests for NaturalSilverArrowsController and level9_silver_arrows_chapter wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level9.dungeon import (
    FULL_TRIFORCE,
    LEVEL9,
    MEASURED_POST_L8_HANDOFF,
    MISSING_SILVER_ARROW_ROOM,
    ROOM_LEVEL9_ENTRY,
    ROOM_SILVER_ARROWS_HYP,
    SILVER_ARROWS,
    TRIFORCE_NOT_FULL,
    UNMEASURED_POST_L8_HANDOFF,
    level9_silver_arrows_stop,
)
from zelda_i.level9.hops import (
    Level9NaturalRouteSelection,
    level9_silver_arrows_chapter,
)
from zelda_i.level9.natural_path import (
    NaturalRouteUnavailableController,
    NaturalSilverArrowsController,
    make_natural_silver_arrows_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot


def _make_snap(
    *,
    level: int = LEVEL9,
    screen: int = ROOM_LEVEL9_ENTRY,
    mode: int = PLAY_MODE,
    triforce: int = FULL_TRIFORCE,
    bombs: int = 14,
    bow: int = 1,
    arrows: int = 1,
    link_x: int = 120,
    link_y: int = 205,
) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=mode,
        level=level,
        screen=screen,
        next_screen=screen,
        link_x=link_x,
        link_y=link_y,
        facing=0x08,
        sword=3,
        bombs=bombs,
        rupees=56,
        keys=1,
        health=0x4F,
        triforce=triforce,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
        submode=0,
        is_updating_mode=1,
        bow=bow,
        arrows=arrows,
    )


def test_natural_silver_arrows_fail_closed_contracts():
    # 1. Triforce not full
    ctrl = make_natural_silver_arrows_controller(handoff=MEASURED_POST_L8_HANDOFF)
    act = ctrl.step(_make_snap(triforce=0x7F))
    assert ctrl.failed
    assert act.reason == TRIFORCE_NOT_FULL

    # 2. Unmeasured handoff
    ctrl2 = make_natural_silver_arrows_controller(handoff=UNMEASURED_POST_L8_HANDOFF)
    act2 = ctrl2.step(_make_snap(triforce=FULL_TRIFORCE))
    assert ctrl2.failed
    assert act2.reason == MISSING_SILVER_ARROW_ROOM
    assert ctrl2.blocked_reason == MISSING_SILVER_ARROW_ROOM

    # 3. Wrong screen (not 0x76)
    ctrl3 = make_natural_silver_arrows_controller(handoff=MEASURED_POST_L8_HANDOFF)
    act3 = ctrl3.step(_make_snap(screen=0x66))
    assert ctrl3.failed
    assert "contract_miss" in act3.reason

    # 4. Link death
    ctrl4 = make_natural_silver_arrows_controller(handoff=MEASURED_POST_L8_HANDOFF)
    act4 = ctrl4.step(_make_snap(mode=17))
    assert ctrl4.failed
    assert act4.reason == "link_death"


def test_natural_silver_arrows_hop_stepping_sequence():
    mock_hops = []
    for i in range(16):
        m = MagicMock()
        m.spec_id = f"hop_{i}"
        m.success = False
        m.failed = False
        m.notes = []
        m.step.return_value = FrameAction(nes_action("UP"), f"step_{i}")
        mock_hops.append(m)

    ctrl = NaturalSilverArrowsController(
        handoff=MEASURED_POST_L8_HANDOFF,
        _hops=tuple(mock_hops),
    )
    snap = _make_snap()

    # Step frame 1 on hop 0
    act1 = ctrl.step(snap)
    assert not ctrl.failed and not ctrl.success
    assert ctrl.hop_i == 0
    assert act1.reason == "prefix_hop_0_step_0"

    # Mark hop 0 success; next step triggers hop 1
    mock_hops[0].success = True
    act2 = ctrl.step(snap)
    assert not ctrl.failed
    assert ctrl.hop_i == 1
    assert act2.reason == "prefix_hop_1_step_1"

    # Advance all remaining hops to success
    for i in range(1, 16):
        mock_hops[i].success = True

    act_final = ctrl.step(snap)
    assert ctrl.success and not ctrl.failed
    assert act_final.reason == "silver_arrows_arrived"
    report = ctrl.report()
    assert report["current_hop"] == "done"
    assert report["total_hops"] == 16
    assert report["hop_i"] == 16


def test_natural_silver_arrows_hop_failure_propagates():
    mock_hop = MagicMock()
    mock_hop.spec_id = "failing_hop"
    mock_hop.success = False
    mock_hop.failed = True
    mock_hop.notes = ["test_bomb_failed"]
    mock_hop.step.return_value = FrameAction(nes_idle_action(), "bomb_failed")

    ctrl = NaturalSilverArrowsController(
        handoff=MEASURED_POST_L8_HANDOFF,
        _hops=(mock_hop,),
    )
    snap = _make_snap()
    act = ctrl.step(snap)
    assert ctrl.failed
    assert "test_bomb_failed" in ctrl.notes
    assert act.reason == "test_bomb_failed"


def test_silver_arrows_chapter_wiring():
    # With measured handoff and selected route
    stages = level9_silver_arrows_chapter(handoff=MEASURED_POST_L8_HANDOFF)
    assert len(stages) == 1
    name, ctrl, max_f = stages[0]
    assert name == "level9_natural_silver_arrows"
    assert isinstance(ctrl.inner, NaturalSilverArrowsController)
    assert max_f == 44000

    # With unmeasured handoff
    unmeasured_stages = level9_silver_arrows_chapter(handoff=UNMEASURED_POST_L8_HANDOFF)
    assert len(unmeasured_stages) == 1
    _, u_ctrl, u_max = unmeasured_stages[0]
    assert isinstance(u_ctrl.inner, NaturalSilverArrowsController)
    assert u_max == 1

    # When silver arrow room is None, returns unavailable controller
    empty_route = Level9NaturalRouteSelection(silver_arrow_room=None)
    unavail_stages = level9_silver_arrows_chapter(empty_route)
    assert len(unavail_stages) == 1
    u_name, u_unavail, _ = unavail_stages[0]
    assert u_name == "level9_natural_silver_arrows"
    assert isinstance(u_unavail, NaturalRouteUnavailableController)
    assert u_unavail.reason == "silver_arrow_room_not_selected"


def test_level9_silver_arrows_stop_predicate():
    good = _make_snap(screen=ROOM_SILVER_ARROWS_HYP, arrows=SILVER_ARROWS, bow=1)
    assert level9_silver_arrows_stop(good, room=ROOM_SILVER_ARROWS_HYP)

    # Missing bow
    assert not level9_silver_arrows_stop(_make_snap(screen=ROOM_SILVER_ARROWS_HYP, arrows=SILVER_ARROWS, bow=0), room=ROOM_SILVER_ARROWS_HYP)

    # Wooden arrows only
    assert not level9_silver_arrows_stop(_make_snap(screen=ROOM_SILVER_ARROWS_HYP, arrows=1, bow=1), room=ROOM_SILVER_ARROWS_HYP)

    # Wrong screen
    assert not level9_silver_arrows_stop(_make_snap(screen=0x20, arrows=SILVER_ARROWS, bow=1), room=ROOM_SILVER_ARROWS_HYP)

    # In transition
    trans = ZeldaSnapshot(
        mode=6,
        level=LEVEL9,
        screen=ROOM_SILVER_ARROWS_HYP,
        next_screen=ROOM_SILVER_ARROWS_HYP,
        link_x=120,
        link_y=189,
        facing=0x08,
        sword=3,
        bombs=10,
        rupees=0,
        keys=0,
        health=0x4F,
        triforce=FULL_TRIFORCE,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
        submode=0,
        is_updating_mode=1,
        bow=1,
        arrows=SILVER_ARROWS,
    )
    assert not level9_silver_arrows_stop(trans, room=ROOM_SILVER_ARROWS_HYP)
