"""Shared blue ceiling-door buttons + session loop. No emulator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from super_metroid.ram import GameplayPhase, parse_state
from super_metroid.routes.kpdr.wrecked_ship.ws_ceiling_door import (
    ceiling_door_action,
    play_ceiling_door,
    settle_ceiling_dest,
    tap_up_action,
)
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_MAIN


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_WS_MAIN,
        "samus_x": 1135,
        "samus_y": 80,
        "pose": 1,
        "facing": 8,
        "health": 299,
        "max_health": 299,
        "game_state": 8,
        "door_transition": 0,
        "velocity_y": 0,
    }
    values.update(overrides)
    return replace(base, **values)


class _Session:
    def __init__(self, state):
        self.state = state
        self.frame = state.frame
        self.actions = []
        self.env = None

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        self.state = replace(self.state, frame=self.frame)
        return self.state


def test_tap_up_charges_then_jumps() -> None:
    assert tap_up_action(0, hold_charge=True) == ("UP", "X")
    assert tap_up_action(60, hold_charge=True) == ("UP",)
    assert "A" not in tap_up_action(64, hold_charge=True)
    assert tap_up_action(239, hold_charge=True) == ("UP",)
    assert tap_up_action(308, hold_charge=True) == ("UP", "A")
    assert tap_up_action(68, hold_charge=False) == ("UP", "A")


def test_ceiling_door_action_none_below_lip() -> None:
    assert (
        ceiling_door_action(
            1135, 200, 1, 0, seat_x=1135, lip_y=160, shaft_y=50, slack=12, hold_charge=True
        )
        is None
    )
    assert ceiling_door_action(
        1135, 80, 1, 0, seat_x=1135, lip_y=160, shaft_y=50, slack=12, hold_charge=True
    ) == ("UP", "X")
    assert ceiling_door_action(
        1135, 36, 21, 0, seat_x=1135, lip_y=160, shaft_y=50, slack=12, hold_charge=True
    ) == ("A",)
    assert ceiling_door_action(
        1135, 80, 138, 0, seat_x=1135, lip_y=160, shaft_y=50, slack=12, hold_charge=True
    ) == ()


def test_play_ceiling_door_noop_in_dest() -> None:
    session = _Session(_state(room_id=ROOM_WS_ATTIC))
    play_ceiling_door(
        session,
        label="test",
        dest_room=ROOM_WS_ATTIC,
        lip_y=160,
        remount=lambda st: ("A",),
        door_action=lambda st, i: ("UP", "X"),
        guard=lambda sess, label: None,
        on_knockback=lambda sess, label: None,
    )
    assert session.actions == []


def test_play_ceiling_door_remounts_below_lip_then_taps() -> None:
    session = _Session(_state(samus_y=200, pose=1))
    seen: list[str] = []

    def _remount(st):
        seen.append("remount")
        session.state = replace(session.state, samus_y=80)
        return ("RIGHT", "A")

    def _door(st, i):
        seen.append(f"door{i}")
        session.state = replace(session.state, room_id=ROOM_WS_ATTIC)
        return ("UP", "X")

    play_ceiling_door(
        session,
        label="test",
        dest_room=ROOM_WS_ATTIC,
        lip_y=160,
        remount=_remount,
        door_action=_door,
        guard=lambda sess, label: None,
        on_knockback=lambda sess, label: None,
    )
    reasons = [r for _, r in session.actions]
    assert "test_remount" in reasons
    assert "test_door" in reasons
    assert seen == ["remount", "door0"]
    assert session.state.room_id == ROOM_WS_ATTIC


def test_play_ceiling_door_times_out() -> None:
    session = _Session(_state(samus_y=80))
    with pytest.raises(TimeoutError, match="ceiling door missed"):
        play_ceiling_door(
            session,
            label="test",
            dest_room=ROOM_WS_ATTIC,
            lip_y=160,
            remount=lambda st: ("A",),
            door_action=lambda st, i: ("UP",),
            guard=lambda sess, label: None,
            on_knockback=lambda sess, label: None,
            budget=3,
        )


def test_settle_ceiling_dest_waits_for_stand() -> None:
    session = _Session(
        _state(room_id=ROOM_WS_ATTIC, pose=21, velocity_y=4, game_state=8)
    )
    land_at = {"n": 0}

    def _step(action, reason):
        session.actions.append((action, reason))
        session.frame += 1
        land_at["n"] += 1
        pose = 1 if land_at["n"] >= 4 else 21
        vy = 0 if land_at["n"] >= 4 else 4
        session.state = replace(
            session.state,
            frame=session.frame,
            pose=pose,
            velocity_y=vy,
            room_id=ROOM_WS_ATTIC,
            game_state=8,
            door_transition=0,
        )
        return session.state

    session.step = _step  # type: ignore[method-assign]
    out = settle_ceiling_dest(
        session, ROOM_WS_ATTIC, label="test", settle_frames=2, land_frames=10
    )
    assert out.pose == 1
    assert abs(out.velocity_y) <= 1
    assert any(r.endswith("_land") for _, r in session.actions)
