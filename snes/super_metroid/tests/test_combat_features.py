"""Unit tests for full-knowledge combat features and Torizo action policy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.actions import (
    N_COMBAT_ACTIONS,
    action_names,
    action_vector,
    nearest_action_id,
)
from super_metroid.combat.bomb_torizo import fight_bomb_torizo_action
from super_metroid.combat.features import (
    FEATURE_DIM,
    AxisAlignedBox,
    bomb_torizo_catalog,
    feature_vector,
    features_from_state,
)
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    return replace(base, phase=GameplayPhase.ORDINARY_GAMEPLAY, **overrides)


def test_aabb_overlap_and_separation() -> None:
    a = AxisAlignedBox(0, 0, 10, 10)
    b = AxisAlignedBox(15, 0, 10, 10)
    assert a.overlaps(b)
    c = AxisAlignedBox(30, 0, 5, 5)
    assert not a.overlaps(c)
    dx, dy = a.separation(c)
    assert dx == 30
    assert dy == 0


def test_bomb_torizo_features_active_and_vector_shape() -> None:
    catalog = bomb_torizo_catalog()
    state = _state(
        room_id=0x9804,
        samus_x=100,
        samus_y=188,
        enemy0_x=200,
        enemy0_y=179,
        enemy0_hp=800,
        enemy0_spritemap=0xAA12,
        num_enemies=1,
        health=99,
        max_health=99,
        missiles=10,
        max_missiles=10,
        selected_item=1,
    )
    feat = features_from_state(state, catalog)
    assert feat.enemy_active
    assert not feat.enemy_defeated
    assert feat.dx == 100
    assert feat.boss_name == "Bomb Torizo"
    vec = feature_vector(feat)
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert 0.0 <= vec[6] <= 1.0  # enemy hp fraction


def test_statue_spritemap_not_active() -> None:
    catalog = bomb_torizo_catalog()
    state = _state(
        room_id=0x9804,
        enemy0_hp=800,
        enemy0_spritemap=0x87D0,
        enemy0_x=219,
        samus_x=81,
        num_enemies=1,
    )
    feat = features_from_state(state, catalog)
    assert not feat.enemy_active
    # Strategy walks into the statue when inactive.
    action = fight_bomb_torizo_action(state, 0)
    assert action == ("RIGHT",)


def test_room_load_garbage_not_active() -> None:
    """Flyway leftovers (many slots, low HP) must not look like Torizo active."""
    catalog = bomb_torizo_catalog()
    state = _state(
        room_id=0x9804,
        enemy0_hp=9,
        enemy0_spritemap=0xB1FD,
        num_enemies=12,
        samus_x=748,
    )
    feat = features_from_state(state, catalog)
    assert not feat.enemy_active


def test_spawn_spritemap_not_active() -> None:
    """0x804F is chozo spawn — not combat AI yet."""
    catalog = bomb_torizo_catalog()
    state = _state(
        room_id=0x9804,
        enemy0_hp=800,
        enemy0_spritemap=0x804F,
        num_enemies=1,
        samus_x=5,
    )
    feat = features_from_state(state, catalog)
    assert not feat.enemy_active
    assert fight_bomb_torizo_action(state, 0) == ("RIGHT",)


def test_active_fight_action_faces_and_fires() -> None:
    state = _state(
        room_id=0x9804,
        samus_x=100,
        samus_y=188,
        enemy0_x=200,
        enemy0_y=179,
        enemy0_hp=500,
        enemy0_spritemap=0xAA12,
        num_enemies=1,
        selected_item=1,
    )
    action = fight_bomb_torizo_action(state, frame_index=0)
    assert "RIGHT" in action  # face boss
    assert "X" in action  # fire on even frames


def test_discrete_action_table_roundtrip() -> None:
    assert N_COMBAT_ACTIONS == 13
    assert action_names(0) == ()
    assert action_names(6) == ("RIGHT", "X")
    vec = action_vector(6)
    assert vec.shape == (12,)
    assert int(vec.sum()) == 2
    assert nearest_action_id(("RIGHT", "X")) == 6
    assert nearest_action_id(()) == 0
    # Strategy projection lands on a fire+face entry.
    assert nearest_action_id(("RIGHT", "X", "A")) == 8


def test_strategy_projects_into_discrete_table() -> None:
    state = _state(
        room_id=0x9804,
        samus_x=100,
        samus_y=188,
        enemy0_x=200,
        enemy0_y=179,
        enemy0_hp=500,
        enemy0_spritemap=0xAA12,
        num_enemies=1,
        selected_item=1,
    )
    names = fight_bomb_torizo_action(state, frame_index=0)
    action_id = nearest_action_id(names)
    assert 0 <= action_id < N_COMBAT_ACTIONS
    projected = set(action_names(action_id))
    # Face + fire should survive projection.
    assert "RIGHT" in projected
    assert "X" in projected
