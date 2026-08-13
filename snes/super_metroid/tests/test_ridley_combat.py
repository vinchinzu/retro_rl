"""Unit tests for the development-only Ridley strategy (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.ridley import (
    ROOM_RIDLEY,
    RidleyEvidence,
    RidleyStrategy,
    fight_ridley_action,
)
from super_metroid.ram import GameplayPhase, parse_state

def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    return replace(
        base,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=ROOM_RIDLEY,
        enemy0_spritemap=0xABCD,
        num_enemies=1,
        **overrides,
    )

def test_active_enemy_action_faces_and_fires_sometimes() -> None:
    state = _state(samus_x=100, enemy0_x=300, enemy0_y=200, enemy0_hp=18000)
    strategy = RidleyStrategy(fire_period=3)
    actions = [fight_ridley_action(state, frame, strategy) for frame in range(3)]
    assert any("RIGHT" in action for action in actions)
    assert any("X" in action for action in actions)

def test_defeated_enemy_returns_empty_actions() -> None:
    state = _state(enemy0_hp=0)
    assert fight_ridley_action(state, frame_index=0) == ()

def test_evidence_dict_keys_are_stable() -> None:
    evidence = RidleyEvidence(
        start_frame=10,
        body_zero_frame=80,
        boss_bit_frame=90,
        end_frame=90,
        peak_body_hp=18000,
        min_body_hp=0,
        action_frames=80,
        final_body_hp=0,
        boss_bit_set=True,
        outcome="ridley_defeated",
    )
    assert set(evidence.to_dict()) == {
        "start_frame",
        "body_zero_frame",
        "boss_bit_frame",
        "end_frame",
        "peak_body_hp",
        "min_body_hp",
        "action_frames",
        "final_body_hp",
        "boss_bit_set",
        "outcome",
    }

def test_active_enemy_action_becomes_empty_at_hp_zero() -> None:
    active = _state(samus_x=100, enemy0_x=300, enemy0_hp=1)
    assert "X" in fight_ridley_action(active, frame_index=0)
    assert fight_ridley_action(replace(active, enemy0_hp=0), frame_index=0) == ()
