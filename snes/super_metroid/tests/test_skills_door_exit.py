"""Unit tests for blue/gray door-exit and morph-bomb skills (no emulator)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from super_metroid.ram import parse_state
from super_metroid.routes.skills import door_exit as de
from super_metroid.routes.skills import morph_bomb as mb


class _FakeSession:
    def __init__(self, state: Any, transitions: list[Any] | None = None) -> None:
        self.state = state
        self.frame = int(getattr(state, "frame", 0))
        self._transitions = list(transitions or [])
        self._step_i = 0

    def step(self, action: np.ndarray, reason: str = "") -> Any:
        del action, reason
        self.frame += 1
        if self._step_i < len(self._transitions):
            self.state = self._transitions[self._step_i]
            self._step_i += 1
        else:
            self.state = replace(self.state, frame=self.frame)
        return self.state


def _state(**kwargs: Any) -> Any:
    base = parse_state(np.zeros(0x10000, dtype=np.uint8))
    return replace(base, **kwargs)


def test_skills_package_exports_door_exit() -> None:
    from super_metroid.routes import skills as prim

    assert prim.jump_enter_exit is de.jump_enter_exit
    assert prim.beam_open_door is de.beam_open_door
    assert prim.lip_stage is de.lip_stage
    assert prim.align_x is mb.align_x
    assert prim.JUMP_ENTER_PERIOD == 30


def test_jump_enter_default_windows() -> None:
    assert de.JUMP_ENTER_PERIOD == 30
    assert de.JUMP_ENTER_JUMP_END == 4
    assert de.JUMP_ENTER_SPIN_END == 10
    assert de.JUMP_ENTER_RESHOT_END == 14


def test_lip_stage_and_beam_open(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(_state(samus_x=120, frame=0))
    calls: list[tuple[int, tuple[str, ...], str]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(de, "hold", _hold)
    de.lip_stage(
        session,
        label="eye_to_baby",
        settle_frames=8,
        backoff_frames=8,
        face_frames=8,
        release_frames=6,
    )
    de.beam_open_door(session, label="eye_to_baby", shots=2, shot_frames=4, fuse_frames=14)
    reasons = [c[2] for c in calls]
    assert "eye_to_baby_approach_settle" in reasons
    assert "eye_to_baby_lip_backoff" in reasons
    assert "eye_to_baby_face" in reasons
    assert "eye_to_baby_door_shot" in reasons
    assert "eye_to_baby_door_fuse" in reasons
    assert calls.count((4, ("X",), "eye_to_baby_door_shot")) == 2


def test_period_exit_push_reaches_target(monkeypatch: pytest.MonkeyPatch) -> None:
    start = _state(room_id=0xA56B, frame=0)
    mid = _state(room_id=0xA56B, frame=1, door_transition=1)
    done = _state(room_id=0xA521, frame=2, door_transition=0)
    session = _FakeSession(start, transitions=[mid, done])
    calls: list[str] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        del frames, buttons
        calls.append(reason)
        return sess.step(np.zeros(12, dtype=np.int8), reason)

    monkeypatch.setattr(de, "hold", _hold)
    out = de.period_exit_push(
        session,
        0xA521,
        label="eye_to_baby",
        max_frames=10,
        period=30,
        windows=(
            (4, ("LEFT", "A"), "jump"),
            (30, ("LEFT", "B"), "exit"),
        ),
        transition_drain=5,
    )
    assert out.room_id == 0xA521
    assert any(r.endswith("_jump") or r.endswith("_exit") for r in calls)


def test_align_x_walks_toward_band(monkeypatch: pytest.MonkeyPatch) -> None:
    # Start left of band; after a few RIGHT steps land in band.
    xs = [370, 372, 375, 376]
    states = [_state(samus_x=x, frame=i) for i, x in enumerate(xs)]
    session = _FakeSession(states[0], transitions=states[1:])
    dirs: list[str] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        del frames, reason
        if buttons:
            dirs.append(buttons[0])
        return sess.step(np.zeros(12, dtype=np.int8), "tick")

    monkeypatch.setattr(mb, "hold", _hold)
    out = mb.align_x(session, x_lo=374, x_hi=380, label="hole", max_frames=10)
    assert out.samus_x >= 374
    assert dirs and all(d == "RIGHT" for d in dirs)


def test_period_exit_push_empty_windows_raises() -> None:
    session = _FakeSession(_state())
    with pytest.raises(ValueError, match="requires at least one window"):
        de.period_exit_push(
            session,
            1,
            label="t",
            max_frames=1,
            period=10,
            windows=(),
        )
