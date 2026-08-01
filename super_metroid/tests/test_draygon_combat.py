"""Unit tests for the development-only Draygon strategy (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.draygon import (
    ROOM_DRAYGON,
    DraygonStrategy,
    draygon_gunk_clear_action,
    fight_draygon_action,
)
from super_metroid.combat import wrap_draygon_as_boss_strategy
from super_metroid.combat.features import draygon_catalog
from super_metroid.combat.protocol import BossStrategy
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    return replace(
        base,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=ROOM_DRAYGON,
        enemy0_spritemap=0xABCD,
        num_enemies=1,
        **overrides,
    )


def test_draygon_catalog_facts() -> None:
    catalog = draygon_catalog()
    assert catalog.room_id == ROOM_DRAYGON
    assert catalog.max_hp == 6000


def test_wrapper_is_pure_boss_strategy_for_draygon() -> None:
    strategy = wrap_draygon_as_boss_strategy()
    assert isinstance(strategy, BossStrategy)
    assert strategy.boss_id == "draygon"
    assert strategy.entry.room_id == ROOM_DRAYGON
    assert strategy.catalog == draygon_catalog()
    assert strategy.catalog.continuous_status == "deferred"


def test_active_enemy_action_faces_and_fires() -> None:
    state = _state(
        samus_x=100,
        enemy0_x=300,
        enemy0_y=200,
        enemy0_hp=6000,
    )
    action = fight_draygon_action(state, frame_index=0)
    assert "RIGHT" in action
    assert "X" in action


def test_defeated_enemy_returns_empty_actions() -> None:
    state = _state(enemy0_hp=0)
    assert fight_draygon_action(state, frame_index=0) == ()


def test_strategy_tuning_changes_fire_period() -> None:
    state = _state(samus_x=100, enemy0_x=300, enemy0_y=200, enemy0_hp=6000)
    strategy = DraygonStrategy(fire_period=5)
    assert "X" in fight_draygon_action(state, frame_index=0, strategy=strategy)
    assert "X" not in fight_draygon_action(state, frame_index=1, strategy=strategy)


def test_gunk_clear_is_noop_outside_contact_overlap() -> None:
    state = _state(
        samus_x=100,
        samus_y=100,
        enemy0_x=400,
        enemy0_y=100,
        enemy0_hp=6000,
    )
    assert draygon_gunk_clear_action(state, frame_index=0) == ()


def test_gunk_clear_jumps_away_and_fires_on_cadence() -> None:
    state = _state(
        samus_x=100,
        samus_y=100,
        enemy0_x=120,
        enemy0_y=100,
        enemy0_hp=6000,
    )
    strategy = DraygonStrategy(fire_period=3)
    assert draygon_gunk_clear_action(state, frame_index=0, strategy=strategy) == (
        "LEFT",
        "A",
        "X",
    )
    assert draygon_gunk_clear_action(state, frame_index=1, strategy=strategy) == (
        "LEFT",
        "A",
    )


def test_gunk_clear_escapes_left_when_enemy_is_left() -> None:
    state = _state(
        samus_x=200,
        samus_y=100,
        enemy0_x=180,
        enemy0_y=100,
        enemy0_hp=6000,
    )
    assert draygon_gunk_clear_action(state, frame_index=0)[:2] == ("RIGHT", "A")


def test_gunk_clear_is_empty_after_body_defeat() -> None:
    state = _state(
        samus_x=100,
        samus_y=100,
        enemy0_x=120,
        enemy0_y=100,
        enemy0_hp=0,
    )
    assert draygon_gunk_clear_action(state, frame_index=0) == ()
