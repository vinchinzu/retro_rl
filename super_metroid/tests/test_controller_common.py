"""Unit tests for composable controller primitives (no emulator)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from super_metroid.policy import StateRequirement
from super_metroid.ram import GameplayPhase, MORPH_BALL_MASK, VARIA_MASK, parse_state
from super_metroid.routes import controller_common as cc


class _FakeSession:
    """Minimal session: optional state sequence on each step."""

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


def test_varia_to_kraid_controller_importable() -> None:
    from super_metroid.routes.kpdr import get_segment, play_varia_to_kraid
    from super_metroid.routes.kpdr.varia_return import play_varia_to_kraid as direct

    assert play_varia_to_kraid is direct
    assert get_segment("varia_to_kraid") is play_varia_to_kraid


def test_kraid_to_eye_return_controller_importable() -> None:
    from super_metroid.routes.kpdr import get_segment, play_kraid_to_eye_return
    from super_metroid.routes.kpdr.varia_return import (
        play_kraid_to_eye_return as direct,
    )

    assert play_kraid_to_eye_return is direct
    assert get_segment("kraid_to_eye_return") is play_kraid_to_eye_return


@pytest.mark.parametrize(
    ("segment_id", "name"),
    [
        ("eye_to_baby_return", "play_eye_to_baby_return"),
        ("baby_to_kihunter_return", "play_baby_to_kihunter_return"),
        ("kihunter_to_zeela_return", "play_kihunter_to_zeela_return"),
        ("zeela_to_warehouse_return", "play_zeela_to_warehouse_return"),
    ],
)
def test_kraid_return_controller_importable(segment_id: str, name: str) -> None:
    from super_metroid.routes.kpdr import get_segment
    from super_metroid.routes.kpdr import kraid_return

    controller = getattr(kraid_return, name)
    assert callable(controller)
    assert get_segment(segment_id) is controller


def test_require_state_reports_failures() -> None:
    session = _FakeSession(
        _state(room_id=0xA253, samus_x=10, samus_y=20, collected_items=0)
    )
    req = StateRequirement(
        room_id=0xA253,
        x_range=(100, 120),
        collected_items_mask=MORPH_BALL_MASK,
    )
    with pytest.raises(RuntimeError, match="missing"):
        cc.require_state(session, req, "test_hop")


def test_wait_requirement_succeeds_when_state_already_matches() -> None:
    session = _FakeSession(
        _state(
            room_id=0xA6E2,
            game_state=8,
            phase=GameplayPhase.ORDINARY_GAMEPLAY,
            collected_items=VARIA_MASK,
            samus_x=108,
            samus_y=126,
        )
    )
    req = StateRequirement(
        room_id=0xA6E2,
        game_states=frozenset({8}),
        collected_items_mask=VARIA_MASK,
        x_range=(100, 120),
        y_range=(120, 140),
    )
    out = cc.wait_requirement(session, req, timeout=5, reason="already_ok")
    assert out.room_id == 0xA6E2
    assert session.frame == 0  # no idle steps when already matching


def test_wait_requirement_timeout_includes_failures() -> None:
    session = _FakeSession(_state(room_id=0xA253, samus_x=1, game_state=8))
    req = StateRequirement(room_id=0xA6E2, x_range=(50, 60))
    with pytest.raises(TimeoutError, match="room"):
        cc.wait_requirement(session, req, timeout=3, reason="never")


def test_settle_hold_advances_frames_and_preserves_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_state(frame=4))
    reasons: list[str] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        del buttons
        reasons.append(reason)
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(cc, "hold", _hold)
    out = cc.settle_hold(session, 12, reason="platform_settle")

    assert session.frame == 16
    assert out.frame == 16
    assert reasons == ["platform_settle"]


def test_short_hop_advances_frames_and_forwards_buttons_and_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_state(frame=7))
    calls: list[tuple[int, tuple[str, ...], str]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(cc, "hold", _hold)
    out = cc.short_hop(
        session,
        "LEFT",
        24,
        buttons_extra=("A",),
        reason="kraid_return_short_hop",
    )

    assert session.frame == 31
    assert out.frame == 31
    assert calls == [(24, ("LEFT", "A"), "kraid_return_short_hop")]


def test_vertical_hop_advances_frames_and_forwards_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_state(frame=7))
    calls: list[tuple[int, tuple[str, ...], str]] = []

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(cc, "hold", _hold)
    out = cc.vertical_hop(session, 24, reason="ghz_pillar_vertical_jump")

    assert session.frame == 31
    assert out.frame == 31
    assert calls == [(24, ("A",), "ghz_pillar_vertical_jump")]


def test_collect_item_mask_waits_for_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    start = _state(collected_items=MORPH_BALL_MASK, frame=0)
    mid = _state(collected_items=MORPH_BALL_MASK, frame=1)
    done = _state(collected_items=MORPH_BALL_MASK | VARIA_MASK, frame=2)
    session = _FakeSession(start, transitions=[mid, done])

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        del buttons, reason
        for _ in range(frames):
            sess.step(np.zeros(12, dtype=np.int8), "tick")
        return sess.state

    monkeypatch.setattr(cc, "hold", _hold)
    out = cc.collect_item_mask(session, VARIA_MASK, timeout=5, reason="varia")
    assert out.collected_items & VARIA_MASK


def test_exports_hybrid_surface() -> None:
    assert callable(cc.wait_requirement)
    assert callable(cc.hold_until)
    assert callable(cc.traverse_door)
    assert callable(cc.collect_item_mask)
    assert callable(cc.require_state)
    assert callable(cc.short_hop)
    assert callable(cc.vertical_hop)
