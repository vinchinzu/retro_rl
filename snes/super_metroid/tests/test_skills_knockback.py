"""Unit tests for knockback / simple Super-door skills (no emulator)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from super_metroid.ram import parse_state
from super_metroid.routes.skills import knockback as kb
from super_metroid.routes.skills.door import super_door_pressure_frame
from super_metroid.routes.skills.geometry import POSE_KNOCKBACK


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


def test_is_knockback_default_poses() -> None:
    assert POSE_KNOCKBACK == frozenset({137, 138})
    assert kb.is_knockback(_state(pose=137))
    assert kb.is_knockback(_state(pose=138))
    assert not kb.is_knockback(_state(pose=2))
    assert not kb.is_knockback(_state(pose=25))


def test_is_knockback_custom_poses() -> None:
    assert kb.is_knockback(_state(pose=99), poses=frozenset({99}))
    assert not kb.is_knockback(_state(pose=137), poses=frozenset({99}))


def test_skills_package_exports_knockback_helpers() -> None:
    from super_metroid.routes import skills as prim

    assert prim.is_knockback is kb.is_knockback
    assert callable(prim.hold_through_knockback)
    assert callable(prim.escape_knockback_spin)
    assert callable(prim.super_door_pressure_frame)
    assert prim.bubble_is_knockback is kb.is_knockback


def test_hold_through_knockback_idle_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_state(pose=137, frame=0))
    calls: list[tuple[int, tuple[str, ...], str]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(kb, "hold", _hold)
    out = kb.hold_through_knockback(session, 12, label="cath_climb")
    assert calls == [(12, (), "cath_climb_kb")]
    assert session.frame == 12
    assert out.frame == 12


def test_escape_knockback_spin_rising_tide_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_state(pose=138, samus_x=100, room_id=0xAFA3, frame=0))
    calls: list[tuple[int, tuple[str, ...], str]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(kb, "hold", _hold)
    kb.escape_knockback_spin(
        session,
        prefer_dir="RIGHT",
        run_frames=6,
        spin_frames=20,
        label="rising_tide_to_bubble",
        stop_room_id=0xACB3,
    )
    run = [c for c in calls if c[2] == "rising_tide_to_bubble_kb_run"]
    spin = [c for c in calls if c[2] == "rising_tide_to_bubble_kb_spin"]
    assert len(run) == 6
    assert all(c[1] == ("RIGHT", "B") for c in run)
    assert len(spin) == 20
    assert all(c[1] == ("RIGHT", "B", "A") for c in spin)
    assert session.frame == 26


def test_escape_knockback_spin_cathedral_clear_and_motion_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # After 6 clear + 3 spin frames, leave KB and move past start_x.
    start = _state(
        pose=137, samus_x=400, room_id=0xA788, selected_item=2, frame=0
    )
    transitions = [
        replace(start, pose=137, samus_x=400, frame=i + 1) for i in range(6)
    ] + [
        replace(start, pose=2, samus_x=410, selected_item=0, frame=7),
        replace(start, pose=2, samus_x=412, selected_item=0, frame=8),
        replace(start, pose=2, samus_x=415, selected_item=0, frame=9),
    ]
    session = _FakeSession(start, transitions=transitions)
    calls: list[tuple[int, tuple[str, ...], str]] = []
    selects: list[int] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    def _select(sess: Any, target: int, *, max_cycles: int = 8) -> None:
        del max_cycles
        selects.append(target)
        sess.state = replace(sess.state, selected_item=target)

    monkeypatch.setattr(kb, "hold", _hold)
    monkeypatch.setattr(kb, "select_weapon", _select)
    kb.escape_knockback_spin(
        session,
        prefer_dir="RIGHT",
        run_frames=6,
        spin_frames=24,
        label="cathedral_to_rising_tide",
        run_with=("B", "X"),
        spin_with=("B", "A"),
        run_reason="kb_clear",
        spin_reason="kb_spin",
        stop_room_id=0xAFA3,
        break_on_motion_clear=True,
        ensure_beam=True,
    )
    assert selects == [0]
    clear = [c for c in calls if c[2].endswith("_kb_clear")]
    spin = [c for c in calls if c[2].endswith("_kb_spin")]
    assert len(clear) == 6
    assert all(c[1] == ("RIGHT", "B", "X") for c in clear)
    # break after first spin frame where pose not KB and |dx| > 2
    assert len(spin) == 1
    assert spin[0][1] == ("RIGHT", "B", "A")


def test_escape_knockback_spin_stops_on_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _state(pose=138, samus_x=1000, room_id=0xAFA3, frame=0)
    mid = replace(start, pose=25, samus_x=1005, room_id=0xAFA3, frame=1)
    done = replace(start, pose=2, samus_x=1010, room_id=0xACB3, frame=2)
    session = _FakeSession(start, transitions=[mid, done])

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        del buttons, reason
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(kb, "hold", _hold)
    out = kb.escape_knockback_spin(
        session,
        prefer_dir="RIGHT",
        run_frames=0,
        spin_frames=20,
        label="rt",
        stop_room_id=0xACB3,
    )
    assert out.room_id == 0xACB3
    assert session.frame == 2


def test_super_door_pressure_frame_cathedral_entrance_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """period=28: X 0-4, face 5-11, run 12-19, jump 20-27."""
    from super_metroid.routes.skills import door as door_mod

    session = _FakeSession(_state(selected_item=2, frame=0))
    calls: list[tuple[str, ...]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        del frames, reason
        calls.append(buttons)
        sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(door_mod, "hold", _hold)
    expected = {
        0: ("RIGHT", "X"),
        4: ("RIGHT", "X"),
        5: ("RIGHT",),
        11: ("RIGHT",),
        12: ("RIGHT", "B"),
        19: ("RIGHT", "B"),
        20: ("RIGHT", "B", "A"),
        27: ("RIGHT", "B", "A"),
    }
    for frame, want in expected.items():
        calls.clear()
        super_door_pressure_frame(
            session,
            frame,
            label="cathedral_entrance_to_cathedral",
            period=28,
            shoot_end=5,
            face_end=12,
            run_end=20,
            ensure_weapon=False,
        )
        assert calls == [want], f"frame {frame}: {calls} != {[want]}"


def test_super_door_pressure_frame_cathedral_green_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """period=40: X 0-3, idle 4-19, face 20-27, run 28-33, jump 34-39."""
    from super_metroid.routes.skills import door as door_mod

    session = _FakeSession(_state(selected_item=2, frame=0))
    calls: list[tuple[str, ...]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        del frames, reason
        calls.append(buttons)
        sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(door_mod, "hold", _hold)
    expected = {
        0: ("RIGHT", "X"),
        3: ("RIGHT", "X"),
        4: (),
        19: (),
        20: ("RIGHT",),
        27: ("RIGHT",),
        28: ("RIGHT", "B"),
        33: ("RIGHT", "B"),
        34: ("RIGHT", "B", "A"),
        39: ("RIGHT", "B", "A"),
    }
    for frame, want in expected.items():
        calls.clear()
        super_door_pressure_frame(
            session,
            frame,
            label="cathedral_to_rising_tide",
            period=40,
            shoot_end=4,
            idle_end=20,
            face_end=28,
            run_end=34,
            ensure_weapon=False,
        )
        assert calls == [want], f"frame {frame}: {calls} != {[want]}"


def test_super_door_pressure_frame_selects_weapon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.skills import door as door_mod

    session = _FakeSession(_state(selected_item=0, frame=0))
    selects: list[int] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        del buttons, reason
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    def _select(sess: Any, target: int, *, max_cycles: int = 8) -> None:
        del max_cycles
        selects.append(target)
        sess.state = replace(sess.state, selected_item=target)

    monkeypatch.setattr(door_mod, "hold", _hold)
    monkeypatch.setattr(door_mod, "select_weapon", _select)
    super_door_pressure_frame(
        session,
        0,
        label="door",
        period=28,
        shoot_end=5,
        face_end=12,
        run_end=20,
        weapon=2,
    )
    assert selects == [2]


def test_k4_modules_import_knockback_skills() -> None:
    from super_metroid.routes.kpdr import k4_cathedral, k4_rising_tide
    from super_metroid.routes.kpdr.wave import bubble_to_single

    src_c = open(k4_cathedral.__file__, encoding="utf-8").read()
    src_r = open(k4_rising_tide.__file__, encoding="utf-8").read()
    src_w = open(bubble_to_single.__file__, encoding="utf-8").read()
    assert "escape_knockback_spin" in src_c
    assert "super_door_pressure_frame" in src_c
    assert "escape_knockback_spin" in src_r
    assert "is_knockback" in src_r
    # Wave corridor hops use shared escape_kb (rr-7sn.5), not private triples.
    assert "escape_kb(" in src_w
    assert "escape_kb_bsc" not in src_w


def test_escape_kb_defaults_to_wave_shaped_spin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """escape_kb is the shared corridor helper (rr-7sn.5); defaults 6/18."""
    session = _FakeSession(_state(pose=137, room_id=0xAD5E, frame=0))
    calls: list[dict[str, object]] = []

    def _spin(sess: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return sess.state  # type: ignore[attr-defined]

    monkeypatch.setattr(kb, "escape_knockback_spin", _spin)
    out = kb.escape_kb(session, "bubble_to_single", "RIGHT", stop_room_id=0xAD5E)
    assert len(calls) == 1
    assert calls[0]["prefer_dir"] == "RIGHT"
    assert calls[0]["run_frames"] == 6
    assert calls[0]["spin_frames"] == 18
    assert calls[0]["label"] == "bubble_to_single"
    assert calls[0]["stop_room_id"] == 0xAD5E
    assert out is session.state


def test_skills_package_exports_escape_kb() -> None:
    from super_metroid.routes import skills as prim

    assert prim.escape_kb is kb.escape_kb

