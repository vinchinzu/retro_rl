"""Unit tests for the development-only Crocomire strategy scaffold."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.crocomire import (
    ROOM_CROCOMIRE,
    fight_crocomire_action,
)
from super_metroid.combat.features import crocomire_catalog
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    return replace(
        base,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=ROOM_CROCOMIRE,
        enemy0_spritemap=0xABCD,
        num_enemies=1,
        **overrides,
    )


def test_crocomire_catalog_uses_hp_zero_and_acid_push() -> None:
    catalog = crocomire_catalog()
    assert catalog.room_id == ROOM_CROCOMIRE
    assert catalog.max_hp == 0
    assert catalog.primary_weapon == "acid_push"
    assert catalog.continuous_status == "side"


def test_active_crocomire_action_pushes_and_fires() -> None:
    state = _state(enemy0_hp=0)
    action = fight_crocomire_action(state, frame_index=0)
    assert action == ("RIGHT", "X")


def test_defeated_boss_bit_returns_empty_action() -> None:
    state = _state(boss_bits=(0, 0, 0x02, 0, 0, 0, 0))
    assert fight_crocomire_action(state, frame_index=0) == ()


def test_evidence_schema_tracks_push_outcomes() -> None:
    from super_metroid.combat.crocomire import CrocomireEvidence

    evidence = CrocomireEvidence(
        start_frame=10,
        boss_bit_frame=20,
        end_frame=20,
        action_frames=10,
        final_enemy_hp=0,
        boss_bit_set=True,
        outcome="pushed",
    )
    payload = evidence.to_dict()
    assert {"boss_bit_frame", "boss_bit_set", "outcome"} <= payload.keys()
    assert payload["outcome"] == "pushed"
