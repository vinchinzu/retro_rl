"""Pure parser and policy tests for Super Double Dragon."""

from __future__ import annotations

import numpy as np

from retro_harness.controls import (
    SNES_A,
    SNES_B,
    SNES_DOWN,
    SNES_LEFT,
    SNES_RIGHT,
    SNES_UP,
    SNES_Y,
)
from retro_harness.ram_state import EnemyState, GameMode, GameState
from super_double_dragon.policy import Stage1Policy
from super_double_dragon.ram import ADDR_PLAYER_PAGE, ACTOR_BASES, parse_game_state


def _ram() -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[0x001C] = 0x10
    ram[0x00DC] = 2
    ram[ADDR_PLAYER_PAGE] = 0x12
    ram[0x1200] = 3
    ram[0x1202] = 9
    ram[0x1227] = 61
    ram[0x1274] = 128
    ram[0x1210] = 60
    return ram


def _state(
    *,
    enemies: tuple[EnemyState, ...] = (),
    frame: int = 0,
    stage: int = 0x10,
    player_x: int = 100,
    player_y: int = 195,
) -> GameState:
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING,
        stage=stage,
        player_x=player_x,
        player_y=player_y,
        health=60,
        lives=2,
        enemies=enemies,
    )


def test_parse_player_and_drawn_enemy() -> None:
    ram = _ram()
    ram[0x0A00] = 3
    ram[0x0A02] = 6
    ram[0x0A27] = 35
    ram[0x0A74] = 166
    ram[0x0A10] = 64
    state = parse_game_state(ram, frame=7)
    assert state.mode is GameMode.PLAYING
    assert (state.frame, state.player_x, state.player_y) == (7, 128, 195)
    assert state.health == 61
    assert state.extras["mission"] == 1
    assert len(state.living_enemies) == 1
    assert (state.living_enemies[0].x, state.living_enemies[0].y) == (
        166,
        191,
    )


def test_parse_hp0_drawn_fighter_remains_targetable() -> None:
    ram = _ram()
    ram[0x0800] = 3
    ram[0x0802] = 4
    ram[0x0874] = 90
    state = parse_game_state(ram)
    assert len(state.living_enemies) == 1
    assert state.living_enemies[0].health == 1
    assert state.screen_locked is True


def test_parse_down_fighter_is_not_a_living_target() -> None:
    ram = _ram()
    ram[0x1400] = 2
    ram[0x1402] = 7
    ram[0x1427] = 20
    assert parse_game_state(ram).living_enemies == ()


def test_area_11_remains_mission_1_gameplay() -> None:
    ram = _ram()
    ram[0x001C] = 0x11
    state = parse_game_state(ram)
    assert state.mode is GameMode.PLAYING
    assert state.extras["mission"] == 1


def test_parser_follows_player_pointer_to_a_new_actor_page() -> None:
    ram = _ram()
    ram[0x1200] = 3
    ram[0x1202] = 7
    ram[0x1227] = 30
    ram[0x0C00] = 3
    ram[0x0C02] = 9
    ram[0x0C27] = 16
    ram[0x0C74] = 84
    ram[0x0C10] = 64
    ram[ADDR_PLAYER_PAGE] = 0x0C
    state = parse_game_state(ram)
    assert state.extras["player_base"] == 0x0C00
    assert (state.player_x, state.health) == (84, 16)
    assert any(enemy.kind == 7 for enemy in state.living_enemies)


def test_parser_reads_page_17_drawn_fighter() -> None:
    ram = _ram()
    ram[0x1700] = 3
    ram[0x1702] = 7
    ram[0x1727] = 82
    ram[0x170C] = 0xAA
    ram[0x170D] = 0x01
    ram[0x1710] = 61
    ram[0x1774] = 171
    ram[0x0018] = 1
    ram[0x00DE] = 0x20
    state = parse_game_state(ram)
    assert 0x1700 in ACTOR_BASES
    assert len(state.living_enemies) == 1
    enemy = state.living_enemies[0]
    assert (enemy.kind, enemy.x, enemy.health) == (7, 171, 82)
    assert state.extras["player_world_x"] == 0
    assert state.extras["scene_lock"] == 1
    assert state.extras["floor"] == 0x20
    assert state.extras["drawn_actors"][0]["base"] == 0x1700
    assert state.extras["drawn_actors"][0]["world_x"] == 0x01AA


def test_parser_does_not_assume_player_kind_is_stable() -> None:
    ram = _ram()
    ram[0x1100] = 3
    ram[0x1102] = 5
    ram[0x1127] = 29
    ram[0x1174] = 88
    ram[ADDR_PLAYER_PAGE] = 0x11
    state = parse_game_state(ram)
    assert state.extras["player_base"] == 0x1100
    assert (state.extras["player_kind"], state.health) == (5, 29)


def test_walk_right_when_clear() -> None:
    result = Stage1Policy().tick(_state())
    assert result.action is not None
    assert result.action.reason == "walk_right"
    assert result.action.action[SNES_RIGHT] == 1


def test_walk_left_in_mission_1_top_floor() -> None:
    result = Stage1Policy().tick(_state(stage=0x13))
    assert result.action is not None
    assert result.action.reason == "walk_left"
    assert result.action.action[SNES_LEFT] == 1


def test_airport_stairs_snake_toward_alternating_landings() -> None:
    policy = Stage1Policy()
    actions = [policy.tick(_state(stage=0x15)).action for _ in range(1202)]
    top = actions[0]
    top_exit = actions[800]
    next_landing = actions[1200]
    assert top is not None and top.action[SNES_LEFT] == 1
    assert top_exit is not None and top_exit.action[SNES_DOWN] == 1
    assert next_landing is not None
    assert next_landing.action[SNES_RIGHT] == 1


def test_runway_advances_down_and_right_when_clear() -> None:
    action = Stage1Policy().tick(_state(stage=0x16)).action
    assert action is not None
    assert action.action[SNES_DOWN] == 1
    assert action.action[SNES_RIGHT] == 1


def test_align_approach_then_attack_without_block() -> None:
    policy = Stage1Policy()
    high = EnemyState(0, 100, 180, 20, True)
    far = EnemyState(0, 140, 195, 20, True)
    near = EnemyState(0, 118, 195, 20, True)
    aligned = policy.tick(_state(enemies=(high,)))
    approached = policy.tick(_state(enemies=(far,)))
    attacks = [policy.tick(_state(enemies=(near,))).action for _ in range(12)]
    assert aligned.action is not None
    assert aligned.action.action[SNES_UP] == 1
    assert approached.action is not None
    assert approached.action.action[SNES_RIGHT] == 1
    assert any(action and action.action[SNES_Y] for action in attacks)
    assert any(action and action.action[SNES_A] for action in attacks)
    assert all(action is None or action.action[SNES_B] == 0 for action in attacks)


def test_attacks_offscreen_enemy_when_pinned_to_left_edge() -> None:
    policy = Stage1Policy()
    offscreen = EnemyState(0, 0, 195, 1, True)
    actions = [
        policy.tick(
            _state(enemies=(offscreen,), player_x=32, player_y=195)
        ).action
        for _ in range(4)
    ]
    assert any(action and action.action[SNES_Y] for action in actions)


def test_gym_stairs_walk_left_then_up_when_clear() -> None:
    policy = Stage1Policy()
    actions = [policy.tick(_state(stage=0x19)).action for _ in range(1101)]
    approach = actions[0]
    climb = actions[500]
    landing = actions[1100]
    assert approach is not None and approach.action[SNES_LEFT] == 1
    assert climb is not None
    assert climb.action[SNES_UP] == 1 and climb.action[SNES_LEFT] == 1
    assert landing is not None and landing.action[SNES_UP] == 1


def test_chin_finisher_uses_block_counter() -> None:
    policy = Stage1Policy()
    chin = EnemyState(0, 110, 195, 3, True)
    action = policy.tick(_state(stage=0x19, enemies=(chin,))).action
    assert action is not None
    assert action.reason == "counter_block"
    assert action.action[SNES_B] == 1
