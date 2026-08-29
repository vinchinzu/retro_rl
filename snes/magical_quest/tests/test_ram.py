from __future__ import annotations

import numpy as np

from magical_quest.ram import (
    ADDR_GAMEPLAY_ACTIVE,
    ADDR_HEALTH,
    ADDR_HEALTH_MAX,
    ADDR_LIVES,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    ADDR_PROGRESS_X,
    FIRST_DOOR_X,
    FIRST_DOOR_Y,
    first_door_reached,
    parse_game_state,
)
from retro_harness.ram_state import GameMode, GameState


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = value >> 8


def _stage1_ram(
    *,
    player_x: int = 360,
    player_y: int = 34,
    health: int = 3,
    progress: int = 2696,
) -> np.ndarray:
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_GAMEPLAY_ACTIVE] = 1
    _put_u16(ram, ADDR_PLAYER_X, player_x)
    ram[ADDR_PLAYER_Y] = player_y
    _put_u16(ram, ADDR_PROGRESS_X, progress)
    ram[ADDR_HEALTH_MAX] = 3
    ram[ADDR_HEALTH] = health
    ram[ADDR_LIVES] = 2
    return ram


def test_parse_stage1_state() -> None:
    state = parse_game_state(_stage1_ram(), frame=4)

    assert state.mode is GameMode.PLAYING
    assert state.player_x == 360
    assert state.player_y == 34
    assert state.health == 3
    assert state.lives == 2
    assert state.camera_x == 2696
    assert state.frame == 4
    assert state.extras["health_max"] == 3
    assert state.extras["at_first_door"] is False


def test_parse_menu_before_gameplay_is_active() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)
    _put_u16(ram, ADDR_PLAYER_X, 360)

    assert parse_game_state(ram).mode is GameMode.MENU


def test_parse_zero_health_is_dead() -> None:
    state = parse_game_state(_stage1_ram(health=0))
    assert state.player_dead is True
    assert first_door_reached(state) is False


def test_first_door_stop_requires_wall_x_house_y_and_hp() -> None:
    spawn = parse_game_state(_stage1_ram())
    door = parse_game_state(
        _stage1_ram(player_x=FIRST_DOOR_X, player_y=FIRST_DOOR_Y, health=1)
    )
    too_left = parse_game_state(
        _stage1_ram(player_x=FIRST_DOOR_X - 1, player_y=FIRST_DOOR_Y, health=3)
    )
    too_high = parse_game_state(
        _stage1_ram(player_x=FIRST_DOOR_X, player_y=FIRST_DOOR_Y - 1, health=3)
    )

    assert first_door_reached(spawn) is False
    assert first_door_reached(too_left) is False
    assert first_door_reached(too_high) is False
    assert first_door_reached(door) is True
    assert door.extras["at_first_door"] is True
    assert door.room == 1


def test_first_door_reached_rejects_non_playing_state() -> None:
    state = GameState(
        frame=0,
        mode=GameMode.MENU,
        player_x=FIRST_DOOR_X,
        player_y=FIRST_DOOR_Y,
        health=3,
    )
    assert first_door_reached(state) is False
