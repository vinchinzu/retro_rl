"""Unit tests for the development-only Botwoon strategy (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.botwoon import (
    ROOM_BOTWOON,
    BotwoonEvidence,
    BotwoonStrategy,
    fight_botwoon_action,
    play_botwoon_fight,
)
from super_metroid.combat.protocol import wrap_botwoon_as_boss_strategy
from super_metroid.ram import GameplayPhase, parse_state

def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    return replace(
        base,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=ROOM_BOTWOON,
        enemy0_spritemap=0xABCD,
        num_enemies=1,
        **overrides,
    )

class _Session:
    """Deterministic session double for bounded Botwoon evidence tests."""

    def __init__(self, state):
        self.state = state
        self.frame = state.frame
        self.actions = []

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        self.state = replace(self.state, frame=self.frame)
        return self.state

def test_active_enemy_action_faces_and_fires() -> None:
    state = _state(samus_x=100, enemy0_x=300, enemy0_y=200, enemy0_hp=1500)
    action = fight_botwoon_action(state, frame_index=0)
    assert "RIGHT" in action
    assert "X" in action

def test_kite_prefers_distance_band() -> None:
    strategy = BotwoonStrategy(min_range=100, max_range=200, jump_range=0)
    far = _state(samus_x=100, enemy0_x=350, enemy0_hp=1500)
    near = _state(samus_x=100, enemy0_x=120, enemy0_hp=1500)
    in_band = _state(samus_x=100, enemy0_x=220, enemy0_hp=1500)

    assert "RIGHT" in fight_botwoon_action(far, 1, strategy)
    assert "LEFT" in fight_botwoon_action(near, 1, strategy)
    assert fight_botwoon_action(in_band, 1, strategy) == ("RIGHT",)

def test_defeated_enemy_returns_empty_actions() -> None:
    state = _state(enemy0_hp=0)
    assert fight_botwoon_action(state, frame_index=0) == ()

def test_timeout_evidence_has_explicit_label_and_hp_extrema() -> None:
    session = _Session(
        _state(enemy0_hp=1500, selected_item=2, max_super_missiles=5)
    )
    evidence = play_botwoon_fight(
        session,
        strategy=BotwoonStrategy(max_fight_frames=3),
    )

    assert evidence.outcome == "botwoon_timeout"
    assert evidence.peak_enemy_hp == 1500
    assert evidence.min_enemy_hp == 1500
    assert evidence.final_enemy_hp == 1500
    assert evidence.action_frames == 3

def test_botwoon_evidence_dict_captures_hp_extrema() -> None:
    evidence = BotwoonEvidence(
        start_frame=10,
        defeat_frame=40,
        boss_bit_frame=41,
        end_frame=41,
        peak_enemy_hp=1500,
        min_enemy_hp=0,
        action_frames=31,
        final_enemy_hp=0,
        boss_bit_set=True,
        outcome="botwoon_defeated",
    )

    payload = evidence.to_dict()
    assert payload["peak_enemy_hp"] == 1500
    assert payload["min_enemy_hp"] == 0
    assert payload["outcome"] == "botwoon_defeated"

def test_wrapper_boss_id_and_entry_room() -> None:
    strategy = wrap_botwoon_as_boss_strategy()
    assert strategy.boss_id == "botwoon"
    assert strategy.entry.room_id == ROOM_BOTWOON
