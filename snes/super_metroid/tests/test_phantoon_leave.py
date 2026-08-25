"""Unit tests for Phantoon loot + left-door exit (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from super_metroid.ram import GameplayPhase, parse_state
from super_metroid.routes.kpdr.k6.phantoon_leave import (
    DOOR_X_MAX,
    SWEEP_X,
    door_jump_action,
    loot_walk_action,
    play_phantoon_loot_exit,
    require_phantoon_left,
)
from super_metroid.routes.kpdr.room_ids import ROOM_PHANTOON, ROOM_WS_BASEMENT


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_PHANTOON,
        "samus_x": 37,
        "samus_y": 187,
        "pose": 1,
        "facing": 4,
        "enemy0_hp": 0,
        "health": 299,
        "max_health": 299,
        "game_state": 8,
        "door_transition": 0,
        "boss_bits": (0, 0, 0, 0x01, 0, 0, 0, 0),
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


def test_loot_walks_toward_pickup_then_sweeps() -> None:
    assert loot_walk_action(40, 120) == ("RIGHT", "B")
    assert loot_walk_action(160, 40) == ("LEFT", "B")
    assert loot_walk_action(100, 100) == ()
    assert loot_walk_action(40, None, swept=False) == ("RIGHT", "B")
    assert loot_walk_action(SWEEP_X, None, swept=False) == ()
    assert loot_walk_action(40, None, swept=True) == ()


def test_door_jump_is_left_a_not_floor_hug() -> None:
    assert "B" in door_jump_action(DOOR_X_MAX + 10, 1, 0)
    assert door_jump_action(37, 1, 0) == ("LEFT", "A")
    assert door_jump_action(37, 1, 20) == ("LEFT", "X")
    assert door_jump_action(37, 138, 0) == ()


def test_loot_exit_already_in_basement_is_noop() -> None:
    session = _Session(_state(room_id=ROOM_WS_BASEMENT, samus_x=1240, samus_y=139))
    out = play_phantoon_loot_exit(session)
    assert out.room_id == ROOM_WS_BASEMENT
    assert session.actions == []


def test_loot_exit_wrong_room() -> None:
    session = _Session(_state(room_id=0xCA08))
    with pytest.raises(RuntimeError, match="phantoon_loot_exit"):
        play_phantoon_loot_exit(session)


def test_loot_exit_requires_boss_bit() -> None:
    session = _Session(_state(boss_bits=(0, 0, 0, 0, 0, 0, 0, 0)))
    with pytest.raises(RuntimeError, match="not defeated"):
        play_phantoon_loot_exit(session)


def test_require_phantoon_left_after_hook() -> None:
    ok = _Session(_state(room_id=ROOM_WS_BASEMENT, samus_x=1240, samus_y=139))
    require_phantoon_left(ok, [], None)
    with pytest.raises(RuntimeError, match="Basement"):
        require_phantoon_left(_Session(_state()), [], None)
    dead = _Session(
        _state(
            room_id=ROOM_WS_BASEMENT,
            samus_x=1240,
            samus_y=139,
            boss_bits=(0, 0, 0, 0, 0, 0, 0, 0),
        )
    )
    with pytest.raises(RuntimeError, match=r"\$D82B"):
        require_phantoon_left(dead, [], None)
