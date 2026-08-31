"""Unit coverage for the release-edged ``canPreciseWallJump`` builder."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from super_metroid.ram import parse_state
from super_metroid.routes.skills import walljump as wj


def _state(**values: Any):
    return replace(parse_state(np.zeros(0x2000, dtype=np.uint8)), **values)


class _Session:
    def __init__(self, states: list[Any]) -> None:
        self.state = states[0]
        self.frame = 0
        self._states = states[1:]

    def step(self, action, reason: str = ""):
        del action, reason
        self.frame += 1
        if self._states:
            self.state = self._states.pop(0)
        return self.state


def test_precise_walljump_releases_jump_and_proves_landing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        _state(frame=0, samus_x=216, samus_y=628, pose=26),
        _state(frame=1, samus_x=214, samus_y=620, pose=26),
        _state(frame=2, samus_x=212, samus_y=610, pose=26),
        _state(frame=3, samus_x=211, samus_y=600, pose=26),
        _state(frame=4, samus_x=211, samus_y=590, pose=132),
        _state(frame=5, samus_x=210, samus_y=580, pose=132),
        _state(frame=6, samus_x=205, samus_y=550, pose=132),
        _state(frame=7, samus_x=190, samus_y=500, pose=25),
        _state(frame=8, samus_x=163, samus_y=475, pose=167),
    ]
    session = _Session(states)
    calls: list[tuple[tuple[str, ...], str]] = []

    def _hold(sess, frames: int, *names: str, reason: str = ""):
        assert frames == 1
        calls.append((names, reason))
        return sess.step(None, reason)

    monkeypatch.setattr(wj, "hold", _hold)
    timing = wj.PreciseWallJumpTiming(
        into="RIGHT",
        away="LEFT",
        coast_frames=1,
        into_frames=2,
        release_frames=2,
        jump_frames=2,
    )
    out = wj.precise_walljump_once(
        session,
        timing,
        start_when=lambda st: st.samus_y == 628,
        contact_when=lambda st: st.pose == 132,
        success_when=lambda st: st.samus_y == 475 and st.pose == 167,
        landing_timeout=2,
        reason="ceres_test",
    )

    assert out.samus_y == 475
    assert [names for names, _ in calls] == [
        ("A",),
        ("RIGHT", "A"),
        ("RIGHT", "A"),
        ("LEFT",),
        ("LEFT",),
        ("LEFT", "A"),
        ("LEFT", "A"),
        (),
    ]
    assert calls[3][1] == "ceres_test_release"


def test_precise_walljump_rejects_unproved_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session([_state(pose=26), _state(pose=26), _state(pose=26)])

    def _hold(sess, frames: int, *names: str, reason: str = ""):
        del names, reason
        assert frames == 1
        return sess.step(None)

    monkeypatch.setattr(wj, "hold", _hold)
    timing = wj.PreciseWallJumpTiming(
        into="RIGHT",
        away="LEFT",
        into_frames=1,
        release_frames=1,
    )
    with pytest.raises(RuntimeError, match="missed wall contact"):
        wj.precise_walljump_once(
            session,
            timing,
            contact_when=lambda st: st.pose == 132,
        )
