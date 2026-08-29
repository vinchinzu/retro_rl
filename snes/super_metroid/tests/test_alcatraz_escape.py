"""Alcatraz left-chimney WJ + instant-morph roll-out. No emulator except the pin test."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from super_metroid.paths import SCRATCH_STATE_DIR, SHARED_ROM
from super_metroid.ram import FACING_LEFT, GameplayPhase, parse_state
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.alcatraz_escape import (
    ROLLOUT_MAX_X,
    ROLLOUT_MAX_Y,
    SHAFT_LIP_Y,
    at_alcatraz_rollout,
    at_left_wall_base,
    at_mid_ledge,
    at_shaft_lip,
    play_alcatraz_escape,
)
from super_metroid.routes.kpdr.room_ids import ROOM_PARLOR

_PIN = SCRATCH_STATE_DIR / "post_torizo_parlor_continuous.state"


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_PARLOR,
        "samus_x": 968,
        "samus_y": 651,
        "pose": 2,
        "facing": FACING_LEFT,
        "health": 99,
        "max_health": 99,
        "game_state": 8,
        "door_transition": 0,
        "collected_items": 0x1004,
        "velocity_x": 0,
        "velocity_y": 0,
    }
    values.update(overrides)
    return replace(base, **values)


class _Session:
    def __init__(self, state):
        self.state = state
        self.frame = int(state.frame)
        self.actions: list[tuple[object, str]] = []

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        self.state = replace(self.state, frame=self.frame)
        return self.state


def test_left_wall_base_and_mid_ledge_bands() -> None:
    assert at_left_wall_base(_state(samus_x=805, samus_y=545, pose=2))
    assert not at_left_wall_base(_state(samus_x=805, samus_y=545, pose=26))
    assert at_mid_ledge(_state(samus_x=828, samus_y=459, pose=2))
    assert not at_mid_ledge(_state(samus_x=968, samus_y=651, pose=2))


def test_shaft_lip_is_morph_hole_height() -> None:
    lip = _state(samus_x=834, samus_y=183, pose=50)
    assert at_shaft_lip(lip)
    assert lip.samus_y <= SHAFT_LIP_Y
    assert not at_shaft_lip(_state(samus_x=834, samus_y=243, pose=132))
    assert not at_shaft_lip(_state(samus_x=759, samus_y=185, pose=31))


def test_rollout_is_morph_left_of_chimney() -> None:
    out = _state(samus_x=759, samus_y=185, pose=31)
    assert at_alcatraz_rollout(out)
    assert out.samus_x <= ROLLOUT_MAX_X
    assert out.samus_y <= ROLLOUT_MAX_Y
    still_in_shaft = _state(samus_x=837, samus_y=197, pose=50)
    assert not at_alcatraz_rollout(still_in_shaft)
    standing = _state(samus_x=759, samus_y=185, pose=2)
    assert not at_alcatraz_rollout(standing)


def test_base_approach_is_one_dash_jump_not_a_retry_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.kpdr import alcatraz_escape as alcatraz

    calls: list[tuple[int, tuple[str, ...], str]] = []
    session = _Session(_state())

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        sess.frame += frames
        if reason == "alcatraz_base_land":
            sess.state = replace(sess.state, samus_x=805, samus_y=545, pose=2)
        else:
            sess.state = replace(sess.state, frame=sess.frame)
        return sess.state

    monkeypatch.setattr(alcatraz, "hold", _hold)
    monkeypatch.setattr(alcatraz, "_unmorph_probe_pose", lambda _s: None)
    alcatraz._land_left_wall_base(session)
    hops = [c for c in calls if c[2] == "alcatraz_base_hop"]
    assert hops == [(18, ("LEFT", "A"), "alcatraz_base_hop")]
    assert calls[0][2] == "alcatraz_base_face"
    assert calls[1] == (30, ("LEFT", "B"), "alcatraz_base_run")


def test_play_rejects_wrong_entry_seat() -> None:
    session = _Session(_state(samus_x=900, samus_y=651, pose=2))
    with pytest.raises(RuntimeError, match="natural entry"):
        play_alcatraz_escape(session)


def test_instant_morph_is_single_down_while_holding_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.kpdr import alcatraz_escape as alcatraz

    calls: list[tuple[int, tuple[str, ...], str]] = []
    session = _Session(
        _state(samus_x=838, samus_y=224, pose=POSE_WALL_LATCH)
    )

    def _hold(sess: Any, frames: int, *buttons: str, reason: str = "") -> Any:
        calls.append((frames, buttons, reason))
        sess.frame += frames
        if reason == "alcatraz_instant_morph":
            sess.state = replace(sess.state, pose=50, samus_y=197)
        elif reason == "alcatraz_escape":
            sess.state = replace(sess.state, samus_x=759, samus_y=185, pose=31)
        else:
            sess.state = replace(sess.state, frame=sess.frame)
        return sess.state

    monkeypatch.setattr(alcatraz, "hold", _hold)
    morph_frame = alcatraz._instant_morph_rollout(session)
    assert morph_frame == 1
    assert calls[0] == (1, ("DOWN", "A"), "alcatraz_instant_morph")
    assert all(c[1] == ("LEFT",) for c in calls[1:])
    downs = [c for c in calls if "DOWN" in c[1]]
    assert downs == [(1, ("DOWN", "A"), "alcatraz_instant_morph")]
    assert at_alcatraz_rollout(session.state)


@pytest.mark.skipif(not SHARED_ROM.is_file(), reason="vanilla ROM not present")
@pytest.mark.skipif(not _PIN.is_file(), reason="post_torizo_parlor_continuous.state missing")
def test_natural_pin_lands_at_lip_and_rolls_out() -> None:
    from super_metroid.assist import UnlimitedResourcesAssist
    from super_metroid.combat.probe import ProbeSession, open_state_env

    env, _resolved = open_state_env(_PIN, settle=5)
    try:
        session = ProbeSession(env, UnlimitedResourcesAssist())
        min_y = int(session.state.samus_y)

        orig_step = session.step

        def _step(action, reason: str = ""):
            state = orig_step(action, reason)
            nonlocal min_y
            min_y = min(min_y, int(state.samus_y))
            return state

        session.step = _step  # type: ignore[method-assign]
        evidence = play_alcatraz_escape(session)
        assert at_alcatraz_rollout(session.state)
        assert min_y <= SHAFT_LIP_Y
        assert evidence.base_frame <= 80
        assert evidence.ledge_frame <= 330
        assert evidence.exit_frame <= 540
        assert evidence.exit_x <= ROLLOUT_MAX_X
        assert evidence.exit_y <= ROLLOUT_MAX_Y
        assert evidence.exit_x == session.state.samus_x
        assert session.state.pose in {29, 30, 31, 32, 49, 50, 65, 66}
    finally:
        env.close()
