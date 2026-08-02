"""Unit tests for shared combat primitives (no emulator)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from super_metroid.combat.primitives import (
    lane_hold_action,
    lane_hold_window,
    settle_standing,
)
from super_metroid.ram import parse_state


class _FakeSession:
    """Minimal session: optional state sequence on each step; records holds."""

    def __init__(self, state: Any, transitions: list[Any] | None = None) -> None:
        self.state = state
        self.frame = int(getattr(state, "frame", 0))
        self._transitions = list(transitions or [])
        self._step_i = 0
        self.reasons: list[str] = []
        self.actions: list[np.ndarray] = []

    def step(self, action: np.ndarray, reason: str = "") -> Any:
        self.actions.append(np.asarray(action).copy())
        self.reasons.append(reason)
        self.frame += 1
        if self._step_i < len(self._transitions):
            self.state = self._transitions[self._step_i]
            self._step_i += 1
        else:
            self.state = replace(self.state, frame=self.frame)
        return self.state


def _state(**kwargs: Any) -> Any:
    base = parse_state(np.zeros(0x10000, dtype=np.uint8))
    return replace(base, phase=base.phase, **kwargs)


def test_lane_hold_action_one_frame_helpers() -> None:
    """Sanity: one-frame builder still matches band semantics."""
    assert "RIGHT" in lane_hold_action(30, min_x=50, max_x=260)
    assert "LEFT" in lane_hold_action(300, min_x=50, max_x=260)
    assert lane_hold_action(120, min_x=50, max_x=260, face="RIGHT") == ("RIGHT",)
    assert lane_hold_action(70_000, min_x=50, max_x=260) == ()


def test_lane_hold_window_holds_then_skips_recovery() -> None:
    """hold_frames advance session; recovery_frames=0 skips settle."""
    start = _state(samus_x=10, samus_y=400, pose=1, frame=0)
    session = _FakeSession(start)
    out = lane_hold_window(
        session,
        min_x=50,
        max_x=260,
        hold_frames=5,
        recovery_frames=0,
        reason="test_lane",
    )
    assert session.frame == 5
    assert out.frame == 5
    assert len(session.reasons) == 5
    assert all(r == "test_lane" for r in session.reasons)
    # Left of band → RIGHT (+ dash B) each frame.
    assert all(a.sum() >= 1 for a in session.actions)


def test_lane_hold_window_in_band_faces() -> None:
    """Inside band, each hold frame faces the requested direction."""
    start = _state(samus_x=120, samus_y=400, pose=1, frame=0)
    session = _FakeSession(start)
    lane_hold_window(
        session,
        min_x=50,
        max_x=260,
        hold_frames=3,
        face="LEFT",
        dash=False,
        recovery_frames=0,
        reason="in_band",
    )
    assert session.frame == 3
    assert session.reasons == ["in_band", "in_band", "in_band"]


def test_lane_hold_window_recovery_settle_immediate_when_standing() -> None:
    """Standing pose + recovery_frames>0: settle returns without extra idle."""
    start = _state(samus_x=100, samus_y=400, pose=1, frame=0)
    session = _FakeSession(start)
    out = lane_hold_window(
        session,
        min_x=50,
        max_x=260,
        hold_frames=4,
        recovery_frames=30,
        settle_min_y=390,
        reason="lane_ok",
    )
    # Hold only — settle_standing sees good pose/y and returns immediately.
    assert session.frame == 4
    assert out.pose == 1
    assert all(r == "lane_ok" for r in session.reasons)


def test_lane_hold_window_recovery_waits_for_land() -> None:
    """Mid-air entry pose is settled during recovery (not during lane hold)."""
    # hold_frames=2 leave pose 81; then recovery transitions to standing.
    start = _state(samus_x=100, samus_y=350, pose=81, frame=0)
    mid_air = _state(samus_x=100, samus_y=380, pose=81, frame=1)
    still_air = _state(samus_x=100, samus_y=395, pose=81, frame=2)
    landed = _state(samus_x=100, samus_y=400, pose=1, frame=3)
    # Steps: hold0→mid_air, hold1→still_air, recover0→landed (settle checks after step)
    # settle_standing: check before hold; start is pose 81 so holds; after first
    # recovery step state becomes landed... actually transitions apply on step.
    # Sequence of states after each step:
    # step0 (hold): mid_air, step1 (hold): still_air,
    # settle check still_air bad → step2: landed → settle check ok.
    session = _FakeSession(start, transitions=[mid_air, still_air, landed])
    out = lane_hold_window(
        session,
        min_x=50,
        max_x=260,
        hold_frames=2,
        recovery_frames=10,
        settle_min_y=390,
        settle_bad_poses=frozenset({81, 164}),
        reason="door_lane",
    )
    assert session.frame == 3  # 2 hold + 1 recover idle
    assert out.pose == 1
    assert out.samus_y >= 390
    assert session.reasons[:2] == ["door_lane", "door_lane"]
    assert session.reasons[2] == "door_lane_recover"


def test_settle_standing_used_by_window_contract() -> None:
    """Direct settle_standing still idles until pose clears (window dependency)."""
    start = _state(samus_x=100, samus_y=400, pose=81, frame=0)
    landed = _state(samus_x=100, samus_y=400, pose=1, frame=1)
    session = _FakeSession(start, transitions=[landed])
    out = settle_standing(session, max_frames=5, reason="unit_settle")
    assert out.pose == 1
    assert session.reasons == ["unit_settle"]
