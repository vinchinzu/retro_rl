"""Unit tests for the full-knowledge Bomb Torizo action strategy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.bomb_torizo import (
    ROOM_BOMB_TORIZO,
    SPAWN_SPRITEMAP,
    STATUE_SPRITEMAP,
    BombTorizoEvidence,
    BombTorizoStrategy,
    fight_bomb_torizo_action,
)
from super_metroid.combat.features import bomb_torizo_catalog
from super_metroid.combat.protocol import wrap_bomb_torizo_as_boss_strategy
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_BOMB_TORIZO,
        "enemy0_x": 200,
        "enemy0_y": 188,
        "enemy0_hp": 800,
        "enemy0_spritemap": 0xAA12,
        "num_enemies": 1,
    }
    values.update(overrides)
    return replace(
        base,
        **values,
    )


def test_bomb_torizo_catalog_facts() -> None:
    catalog = bomb_torizo_catalog()

    assert catalog.max_hp == 800
    assert catalog.room_id == ROOM_BOMB_TORIZO
    assert catalog.primary_weapon == "missiles"


def test_statue_spritemap_seeks_activation() -> None:
    state = _state(enemy0_spritemap=STATUE_SPRITEMAP, samus_x=81)

    assert fight_bomb_torizo_action(state, frame_index=0) == ("RIGHT",)


def test_spawn_spritemap_seeks_activation() -> None:
    state = _state(enemy0_spritemap=SPAWN_SPRITEMAP, samus_x=81)

    assert fight_bomb_torizo_action(state, frame_index=0) == ("RIGHT",)


def test_active_fight_retreats_when_too_close() -> None:
    state = _state(samus_x=150, enemy0_x=200)

    action = fight_bomb_torizo_action(state, frame_index=1)

    assert "LEFT" in action
    assert "RIGHT" not in action


def test_active_fight_approaches_when_too_far() -> None:
    state = _state(samus_x=50, enemy0_x=200)

    action = fight_bomb_torizo_action(state, frame_index=1)

    assert "RIGHT" in action
    assert "LEFT" not in action


def test_mid_range_fires_on_fire_period_frames() -> None:
    strategy = BombTorizoStrategy(fire_period=3, jump_period=50, jump_hold_frames=0)
    state = _state(samus_x=100, enemy0_x=200)

    firing = fight_bomb_torizo_action(state, frame_index=0, strategy=strategy)
    cooldown = fight_bomb_torizo_action(state, frame_index=1, strategy=strategy)

    assert "RIGHT" in firing
    assert "X" in firing
    assert "RIGHT" in cooldown
    assert "X" not in cooldown


def test_zero_enemy_hp_returns_idle_actions() -> None:
    state = _state(enemy0_hp=0)

    assert fight_bomb_torizo_action(state, frame_index=0) == ()
    assert fight_bomb_torizo_action(state, frame_index=1) == ()


def test_evidence_to_dict_keys_are_stable() -> None:
    evidence = BombTorizoEvidence(
        start_frame=100,
        activation_seen=True,
        defeat_frame=250,
        end_frame=300,
        peak_hp=800,
        min_enemy_hp=0,
        action_frames=200,
        final_enemy_hp=0,
        outcome="bomb_torizo_defeated",
        boss_bit_frame=280,
    )

    assert set(evidence.to_dict()) == {
        "start_frame",
        "activation_seen",
        "defeat_frame",
        "boss_bit_frame",
        "end_frame",
        "peak_hp",
        "min_enemy_hp",
        "action_frames",
        "final_enemy_hp",
        "outcome",
    }


def test_strategy_defaults_have_positive_fight_budget() -> None:
    strategy = BombTorizoStrategy()

    assert strategy.min_range > 0
    assert strategy.max_range > strategy.min_range
    assert strategy.fire_period > 0
    assert strategy.max_fight_frames > 0


def test_strategy_defaults_are_clean_economy_kite() -> None:
    """Clean low-ammo defaults: wider band + longer jump holds."""
    strategy = BombTorizoStrategy()

    assert strategy.min_range >= 100
    assert strategy.max_range >= 160
    assert strategy.jump_hold_frames >= 28
    assert strategy.jump_period <= 40
    assert strategy.max_fight_frames >= 12_000


def test_open_bus_boss_bits_do_not_idle_active_fight() -> None:
    """Low-WRAM open-bus boss_bits must not suppress fire (enemy_defeated trap)."""
    # features_from_state would mark defeated if boss_bits[0] & 0x04 — open bus
    # often has 0xFF. Action path keys on enemy0_hp only.
    state = _state(
        samus_x=100,
        enemy0_x=200,
        enemy0_hp=800,
        boss_bits=(255, 0, 255, 0, 255, 0, 255, 0),
    )
    action = fight_bomb_torizo_action(state, frame_index=0)
    assert "X" in action
    assert action != ()


def test_bomb_torizo_boss_strategy_adapter_imports() -> None:
    strategy = wrap_bomb_torizo_as_boss_strategy()

    assert strategy.boss_id == "bomb_torizo"
    assert strategy.catalog.room_id == ROOM_BOMB_TORIZO
    assert strategy.entry.room_id == ROOM_BOMB_TORIZO
