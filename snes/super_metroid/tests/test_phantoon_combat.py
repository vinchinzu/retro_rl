"""Unit tests for the development-only Phantoon strategy (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.features import phantoon_catalog
from super_metroid.combat.phantoon import (
    PHANTOON_INVISIBLE,
    PHANTOON_VULNERABLE,
    ROOM_PHANTOON,
    WEAPON_MISSILES,
    PhantoonEvidence,
    PhantoonStrategy,
    fight_phantoon_action,
    phantoon_phase,
    play_phantoon_fight,
)
from super_metroid.combat.protocol import wrap_phantoon_as_boss_strategy
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_PHANTOON,
        "enemy0_spritemap": 0xABCD,
        "num_enemies": 1,
        **overrides,
    }
    return replace(
        base,
        **values,
    )


class _Session:
    """Small deterministic session double for the bounded fight loop."""

    def __init__(self, state, *, hp_after_step=None, set_boss_bit=False):
        self.state = state
        self.frame = state.frame
        self.actions = []
        self.hp_after_step = hp_after_step
        self.set_boss_bit = set_boss_bit

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        updates = {"frame": self.frame}
        if self.state.selected_item != WEAPON_MISSILES:
            updates["selected_item"] = WEAPON_MISSILES
        if self.hp_after_step is not None and len(self.actions) >= 27:
            updates["enemy0_hp"] = self.hp_after_step
        if self.set_boss_bit:
            bits = list(self.state.boss_bits)
            bits[3] |= 0x01
            updates["boss_bits"] = tuple(bits)
        self.state = replace(self.state, **updates)
        return self.state


def test_phantoon_catalog_facts_via_strategy() -> None:
    strategy = wrap_phantoon_as_boss_strategy()
    assert strategy.boss_id == "phantoon"
    assert strategy.catalog == phantoon_catalog()
    assert strategy.catalog.room_id == ROOM_PHANTOON
    assert strategy.catalog.max_hp == 2500


def test_active_enemy_action_faces_and_fires() -> None:
    state = _state(
        samus_x=100,
        enemy0_x=300,
        enemy0_y=200,
        enemy0_hp=2500,
    )
    action = fight_phantoon_action(state, frame_index=0)
    assert "RIGHT" in action
    assert "X" in action


def test_phantoon_phase_labels_live_spritemap_vulnerable() -> None:
    state = _state(enemy0_hp=2500, enemy0_spritemap=0xABCD)
    assert phantoon_phase(state) == PHANTOON_VULNERABLE


def test_phantoon_phase_labels_missing_spritemap_invisible() -> None:
    state = _state(enemy0_hp=2500, enemy0_spritemap=0)
    assert phantoon_phase(state) == PHANTOON_INVISIBLE


def test_invisible_phase_gates_fire_but_keeps_movement() -> None:
    state = _state(
        samus_x=100,
        enemy0_x=300,
        enemy0_hp=2500,
        enemy0_spritemap=0,
    )
    action = fight_phantoon_action(state, frame_index=0)
    assert "RIGHT" in action
    assert "X" not in action


def test_vulnerable_phase_allows_periodic_fire() -> None:
    state = _state(samus_x=100, enemy0_x=300, enemy0_hp=2500)
    assert "X" in fight_phantoon_action(state, frame_index=0)


def test_defeated_enemy_returns_empty_actions() -> None:
    state = _state(enemy0_hp=0)
    assert fight_phantoon_action(state, frame_index=0) == ()


def test_strategy_tuning_changes_fire_period() -> None:
    state = _state(samus_x=100, enemy0_x=300, enemy0_y=200, enemy0_hp=2500)
    strategy = PhantoonStrategy(fire_period=5)
    assert "X" in fight_phantoon_action(state, frame_index=0, strategy=strategy)
    assert "X" not in fight_phantoon_action(state, frame_index=1, strategy=strategy)


def test_catalog_exposes_single_open_eye_phase() -> None:
    phase = phantoon_catalog().phases[0]
    assert phase.phase_id == "round"
    assert phase.max_hp == 2500
    assert "vulnerable" in phase.notes


def test_action_transitions_from_active_spray_to_idle_at_zero_hp() -> None:
    active = _state(samus_x=100, enemy0_x=300, enemy0_hp=1)
    assert "X" in fight_phantoon_action(active, frame_index=0)
    defeated = replace(active, enemy0_hp=0)
    assert fight_phantoon_action(defeated, frame_index=0) == ()


def test_fight_selects_missiles_before_bounded_timeout() -> None:
    session = _Session(
        _state(enemy0_hp=2500, selected_item=0, max_missiles=5)
    )
    evidence = play_phantoon_fight(
        session,
        strategy=PhantoonStrategy(max_fight_frames=0, weapon=WEAPON_MISSILES),
    )
    assert evidence.outcome == "timeout"
    assert evidence.action_frames == 26
    assert session.state.selected_item == WEAPON_MISSILES


def test_fight_labels_body_zero_without_boss_bit() -> None:
    session = _Session(
        _state(enemy0_hp=2500, max_missiles=5),
        hp_after_step=0,
    )
    evidence = play_phantoon_fight(
        session,
        strategy=PhantoonStrategy(max_fight_frames=1),
    )
    assert evidence.outcome == "phantoon_body_zero_no_boss_bit"
    assert evidence.body_zero_frame == 27
    assert evidence.boss_bit_frame is None
    assert evidence.boss_bit_set is False


def test_phantoon_evidence_dict_preserves_phase_metrics() -> None:
    evidence = PhantoonEvidence(
        start_frame=10,
        body_zero_frame=80,
        boss_bit_frame=90,
        end_frame=90,
        peak_body_hp=2500,
        min_body_hp=0,
        action_frames=80,
        final_body_hp=0,
        boss_bit_set=True,
        outcome="phantoon_defeated",
        phase_transitions=(("vulnerable", 10), ("defeated", 80)),
    )
    payload = evidence.to_dict()
    assert payload["body_zero_frame"] == 80
    assert payload["boss_bit_frame"] == 90
    assert payload["outcome"] == "phantoon_defeated"
    assert payload["phase_transitions"] == [
        {"phase": "vulnerable", "frame": 10},
        {"phase": "defeated", "frame": 80},
    ]
    assert set(payload) == {
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
        "phase_transitions",
    }


def test_fight_evidence_records_phase_transition_timestamp() -> None:
    session = _Session(
        _state(enemy0_hp=2500, max_missiles=5, enemy0_spritemap=0),
        hp_after_step=0,
    )
    evidence = play_phantoon_fight(
        session,
        strategy=PhantoonStrategy(max_fight_frames=1),
    )
    assert evidence.phase_transitions[0] == (PHANTOON_INVISIBLE, 0)
    assert evidence.phase_transitions[-1] == ("defeated", 27)


def test_wrapper_entry_room_and_catalog() -> None:
    strategy = wrap_phantoon_as_boss_strategy()
    assert strategy.entry.room_id == ROOM_PHANTOON
    assert strategy.catalog.name == "Phantoon"
