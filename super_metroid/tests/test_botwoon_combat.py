"""Unit tests for the development-only Botwoon strategy (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.botwoon import (
    ROOM_BOTWOON,
    fight_botwoon_action,
)
from super_metroid.combat.features import botwoon_catalog
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


def test_botwoon_catalog_facts_via_strategy() -> None:
    strategy = wrap_botwoon_as_boss_strategy()
    assert strategy.boss_id == "botwoon"
    assert strategy.catalog == botwoon_catalog()
    assert strategy.catalog.room_id == ROOM_BOTWOON
    assert strategy.catalog.max_hp == 1500
    assert strategy.catalog.primary_weapon == "supers"


def test_active_enemy_action_faces_and_fires() -> None:
    state = _state(samus_x=100, enemy0_x=300, enemy0_y=200, enemy0_hp=1500)
    action = fight_botwoon_action(state, frame_index=0)
    assert "RIGHT" in action
    assert "X" in action


def test_defeated_enemy_returns_empty_actions() -> None:
    state = _state(enemy0_hp=0)
    assert fight_botwoon_action(state, frame_index=0) == ()


def test_wrapper_boss_id_and_entry_room() -> None:
    strategy = wrap_botwoon_as_boss_strategy()
    assert strategy.boss_id == "botwoon"
    assert strategy.entry.room_id == ROOM_BOTWOON
