"""Contracts and telemetry for the reusable reactive SMB route layer."""

from __future__ import annotations

import json

import pytest

from smb.ram import SmbSnapshot
from smb.reactive_late import (
    M8_83_START,
    M8_84_START,
    PATCHES,
    LateRouteController,
    stage_frames,
)
from smb.reactive_route import (
    GateWaiter,
    RouteProgressTracker,
    StateGate,
    level_control_gate,
    missing_policies,
)
from smb.routes import ROUTE_ALL_EXITS, ROUTE_WARP_ANY_PERCENT
from smb.reactive_route import (
    KNOWN_41_CONTROL_RESUME,
    KNOWN_42_CONTROL_RESUME,
    DEFAULT_CONTINUATION_START,
    continuation_frames as _continuation_frames,
    in_4_1_exit_auto as _in_4_1_exit_auto,
)


def _snap(
    *,
    world: int = 0,
    level: int = 0,
    oper_mode: int = 1,
    player_state: int = 8,
    lives: int = 2,
) -> SmbSnapshot:
    return SmbSnapshot(
        frame=0,
        player_state=player_state,
        player_x=40,
        player_y=176,
        x_page=0,
        x_offset=40,
        lives=lives,
        world=world,
        level=level,
        level_id=world * 4 + level,
        oper_mode=oper_mode,
        player_power=0,
        timer_hundreds=4,
        timer=400,
        area_pointer=0,
        x_speed=0,
        y_speed=0,
        facing=1,
        screen_x=0,
        player_screen_x=40,
        in_air=False,
    )


def test_warp_and_normal_routes_declare_different_1_2_destinations() -> None:
    warp_12 = ROUTE_WARP_ANY_PERCENT.exits[1]
    normal_12 = ROUTE_ALL_EXITS.exits[1]
    assert warp_12.policy_id == "smb_1_2_warp"
    assert normal_12.policy_id == "smb_1_2_flag"
    assert warp_12.accepts_successor(_snap(world=3, level=0))
    assert not warp_12.accepts_successor(_snap(world=0, level=2))
    assert normal_12.accepts_successor(_snap(world=0, level=2))
    assert not normal_12.accepts_successor(_snap(world=3, level=0))


def test_route_tracker_requires_declared_successors_in_order() -> None:
    tracker = RouteProgressTracker(ROUTE_WARP_ANY_PERCENT, start_lives=2)
    successors = (
        _snap(world=0, level=1),
        _snap(world=3, level=0),
        _snap(world=3, level=1),
        _snap(world=7, level=0),
        _snap(world=7, level=1),
        _snap(world=7, level=2),
        _snap(world=7, level=3),
        _snap(world=7, level=3, oper_mode=2),
    )
    for frame, snap in enumerate(successors, start=1):
        assert tracker.observe(snap, frame=frame)
    assert tracker.complete
    assert [row["exit_id"] for row in tracker.completed] == [
        "1-1",
        "1-2",
        "4-1",
        "4-2",
        "8-1",
        "8-2",
        "8-3",
        "8-4",
    ]
    assert tracker.completed[1]["successor"] == "warp_world_4"
    assert tracker.completed[-1]["successor"] == "ending"


def test_gate_waiter_captures_control_state_and_times_out() -> None:
    gate = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[0])
    waiter = GateWaiter(gate, max_frames=2)
    assert waiter.observe(_snap(world=1, level=0)) is False
    assert waiter.observe(_snap(world=0, level=0, player_state=1)) is False
    assert waiter.timed_out

    immediate = GateWaiter(StateGate("always", lambda snap: snap.lives == 2), 3)
    assert immediate.observe(_snap())
    assert immediate.report()["match_snapshot"]["lives"] == 2


def test_coverage_reports_missing_policies_without_skipping_stages() -> None:
    missing = missing_policies(ROUTE_ALL_EXITS, {"smb_1_1", "smb_1_2_flag"})
    assert len(missing) == 30
    assert missing[0] == {"exit_id": "1-3", "policy_id": "smb_1_3"}


def test_4_1_control_resume_is_after_transition_before_run() -> None:
    """Resume index is the post-control idle before intentional RIGHT+B."""
    assert KNOWN_41_CONTROL_RESUME == 218
    # Global seed index of first post-control input when start=3981.
    assert DEFAULT_CONTINUATION_START + KNOWN_41_CONTROL_RESUME == 4_199


def test_4_2_control_resume_and_exit_auto_gate() -> None:
    """4-2 body starts at cont index 2487; exit-auto ignores mid-level state 3/4."""
    assert KNOWN_42_CONTROL_RESUME == 2_487
    assert DEFAULT_CONTINUATION_START + KNOWN_42_CONTROL_RESUME == 6_468
    # Mid-level pipe/alive state must not freeze the 4-1 body.
    mid = SmbSnapshot(
        frame=0,
        player_state=4,
        player_x=3593,
        player_y=176,
        x_page=14,
        x_offset=5,
        lives=2,
        world=3,
        level=0,
        level_id=12,
        oper_mode=1,
        player_power=0,
        timer_hundreds=3,
        timer=340,
        area_pointer=0,
        x_speed=0,
        y_speed=0,
        facing=1,
        screen_x=0,
        player_screen_x=40,
        in_air=False,
    )
    assert not _in_4_1_exit_auto(mid)
    score = SmbSnapshot(
        frame=0,
        player_state=5,
        player_x=3698,
        player_y=176,
        x_page=14,
        x_offset=110,
        lives=2,
        world=3,
        level=1,
        level_id=13,
        oper_mode=1,
        player_power=0,
        timer_hundreds=0,
        timer=0,
        area_pointer=41,
        x_speed=2,
        y_speed=0,
        facing=1,
        screen_x=0,
        player_screen_x=40,
        in_air=False,
    )
    assert _in_4_1_exit_auto(score)


def test_continuation_frame_drop_is_in_memory_and_coordinates_are_global(
    tmp_path,
) -> None:
    seed = tmp_path / "seed.json"
    frame = [0] * 9
    frames = [frame[:], [1, 0, 0, 0, 0, 0, 0, 0, 0], frame[:], frame[:]]
    seed.write_text(
        json.dumps(
            {
                "format": "nes9_rle",
                "segments": [
                    {"b": frames[0], "n": 1},
                    {"b": frames[1], "n": 1},
                    {"b": frames[2], "n": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _continuation_frames(seed, start=1, drop_at=2, drop_count=1) == [
        frames[1],
        frames[3],
    ]
    with pytest.raises(ValueError, match="inside the continuation"):
        _continuation_frames(seed, start=2, drop_at=1, drop_count=1)


def test_late_stage_frames_are_control_relative_and_patch_only_declared_ranges() -> None:
    from smb.reactive_late import LEAD_IDLES

    frames_83 = stage_frames("8-3")
    frames_84 = stage_frames("8-4")
    lead_83 = int(LEAD_IDLES.get("8-3", 0))
    lead_84 = int(LEAD_IDLES.get("8-4", 0))
    assert len(frames_83) == M8_84_START - M8_83_START + lead_83 == 2_206 + lead_83
    assert len(frames_84) > 3_405 + lead_84
    for stage_id, frames in (("8-3", frames_83), ("8-4", frames_84)):
        lead = int(LEAD_IDLES.get(stage_id, 0))
        for start, end, buttons in PATCHES[stage_id]:
            # Patches are applied pre-lead; played indices are shifted by lead.
            assert frames[start + lead : end + lead] == [buttons] * (end - start)


def test_late_controller_requires_8_3_control_and_hands_off_at_8_4_control() -> None:
    controller = LateRouteController()
    with pytest.raises(ValueError, match="natural 8-3 control"):
        controller.begin(_snap(world=7, level=3))

    controller.begin(_snap(world=7, level=2))
    # Handoff is on first natural 8-4 control, not forced exhaust.
    controller.next_frame()
    assert controller.observe(_snap(world=7, level=2)) is None
    controller.next_frame()
    assert controller.observe(_snap(world=7, level=3)) == "8-4"
    assert controller.stage_id == "8-4"
    assert controller.index == 0
    assert controller.report()["completed"][0]["stage_id"] == "8-3"
    assert controller.report()["completed"][0]["frames"] == 2
