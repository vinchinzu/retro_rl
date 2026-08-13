"""Unit tests for the optional Golden Torizo combat scaffold."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.golden_torizo import (
    ROOM_GOLDEN_TORIZO,
    GoldenTorizoEvidence,
    fight_golden_torizo_action,
)
from super_metroid.ram import GameplayPhase, parse_state

def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_GOLDEN_TORIZO,
        "enemy0_x": 240,
        "enemy0_y": 188,
        "enemy0_hp": 13_500,
        "enemy0_spritemap": 0xAA12,
        "num_enemies": 1,
    }
    values.update(overrides)
    return replace(base, **values)

def test_active_action_approaches_and_fires() -> None:
    state = _state(samus_x=50, enemy0_x=240)

    action = fight_golden_torizo_action(state, frame_index=0)

    assert action == ("RIGHT", "X")

def test_active_action_retreats_when_too_close() -> None:
    state = _state(samus_x=180, enemy0_x=240)

    action = fight_golden_torizo_action(state, frame_index=1)

    assert action == ("LEFT", "A")

def test_zero_enemy_hp_returns_idle_action() -> None:
    state = _state(enemy0_hp=0)

    assert fight_golden_torizo_action(state, frame_index=0) == ()

def test_evidence_to_dict_keys_are_stable() -> None:
    evidence = GoldenTorizoEvidence(
        start_frame=100,
        body_zero_frame=250,
        end_frame=250,
        peak_body_hp=13_500,
        min_body_hp=0,
        action_frames=150,
        final_body_hp=0,
        outcome="golden_torizo_defeated",
    )

    assert set(evidence.to_dict()) == {
        "start_frame",
        "body_zero_frame",
        "end_frame",
        "peak_body_hp",
        "min_body_hp",
        "action_frames",
        "final_body_hp",
        "outcome",
    }

