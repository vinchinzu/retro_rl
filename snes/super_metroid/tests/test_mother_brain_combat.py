"""Unit tests for the development-only Mother Brain strategy scaffold."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.features import mother_brain_catalog
from super_metroid.combat.mother_brain import (
    ROOM_MOTHER_BRAIN,
    WEAPON_MISSILES,
    MotherBrainEvidence,
    MotherBrainStrategy,
    fight_mother_brain_action,
    mother_brain_phase,
)
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    base = parse_state(np.zeros(0x2000, dtype=np.uint8), frame=0)
    return replace(
        base,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=ROOM_MOTHER_BRAIN,
        enemy0_spritemap=0xABCD,
        num_enemies=1,
        **overrides,
    )


def test_mother_brain_catalog_is_three_phase_and_deferred() -> None:
    catalog = mother_brain_catalog()
    assert catalog.room_id == ROOM_MOTHER_BRAIN
    assert catalog.continuous_status == "deferred"
    assert [phase.phase_id for phase in catalog.phases] == ["mb1", "mb2", "mb3"]


def test_active_action_faces_and_fires() -> None:
    state = _state(samus_x=100, enemy0_hp=3000)
    action = fight_mother_brain_action(state, frame_index=0)
    assert "RIGHT" in action
    assert "X" in action


def test_defeated_action_is_empty() -> None:
    assert fight_mother_brain_action(_state(enemy0_hp=0), frame_index=0) == ()


def test_event_set_action_is_empty() -> None:
    state = _state(enemy0_hp=3000, event_flags=(0x00, 0x40, 0, 0, 0, 0, 0, 0))
    assert fight_mother_brain_action(state, frame_index=0) == ()


def test_strategy_fire_period_changes_spray_timing() -> None:
    state = _state(enemy0_hp=3000)
    strategy = MotherBrainStrategy(fire_period=5)
    assert "X" in fight_mother_brain_action(state, frame_index=0, strategy=strategy)
    assert "X" not in fight_mother_brain_action(state, frame_index=1, strategy=strategy)


def test_phase_labels_follow_catalog_thresholds() -> None:
    strategy = MotherBrainStrategy()
    assert mother_brain_phase(_state(enemy0_hp=3000), strategy) == "mb1"
    assert mother_brain_phase(_state(enemy0_hp=12000), strategy) == "mb2"
    assert mother_brain_phase(_state(enemy0_hp=30000), strategy) == "mb3"


def test_strategy_weapon_and_period_defaults() -> None:
    strategy = MotherBrainStrategy()
    assert strategy.weapon == WEAPON_MISSILES
    assert strategy.fire_period == 3
    assert strategy.max_fight_frames == 30_000


def test_evidence_dict_contains_phase_and_defeat_fields() -> None:
    evidence = MotherBrainEvidence(
        start_frame=10,
        body_zero_frame=80,
        boss_bit_frame=90,
        event_frame=None,
        end_frame=90,
        peak_body_hp=3000,
        min_body_hp=0,
        action_frames=80,
        final_body_hp=0,
        boss_bit_set=True,
        event_set=False,
        phase_timeline=({"phase": "mb1", "frame": 10, "hp": 3000},),
        outcome="mother_brain_defeated",
    )
    payload = evidence.to_dict()
    assert payload["phase_timeline"] == [{"phase": "mb1", "frame": 10, "hp": 3000}]
    assert payload["event_set"] is False
    assert payload["outcome"] == "mother_brain_defeated"
