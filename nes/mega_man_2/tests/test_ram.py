from __future__ import annotations

import numpy as np

from mega_man_2.ram import (
    ADDR_CAMERA_X,
    ADDR_CAMERA_X_SCREEN,
    ADDR_HEALTH,
    ADDR_LIVES,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    camera_progress_x,
    is_fallen,
    is_level1_ready,
    parse_game_state,
)
from retro_harness.ram_state import GameMode


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is True
    assert state.mode == GameMode.MENU
    assert state.extras["level1_ready"] is False


def test_level1_ready_and_progress() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = 28
    ram[ADDR_LIVES] = 3
    ram[ADDR_CAMERA_X] = 40
    ram[ADDR_CAMERA_X_SCREEN] = 0
    ram[ADDR_PLAYER_X] = 168
    ram[ADDR_PLAYER_Y] = 116
    assert is_level1_ready(ram)
    assert camera_progress_x(ram) == 40
    state = parse_game_state(ram, frame=10, obs_mean=120.0)
    assert state.mode == GameMode.PLAYING
    assert state.health == 28
    assert state.player_x == 168
    assert state.extras["progress_x"] == 40


def test_fallen_marks_dead() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = 26
    ram[ADDR_LIVES] = 3
    ram[ADDR_PLAYER_Y] = 220
    assert is_fallen(ram)
    state = parse_game_state(ram, frame=1)
    assert state.player_dead is True
    assert state.mode == GameMode.GAME_OVER


def test_screen_progress() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_CAMERA_X_SCREEN] = 1
    ram[ADDR_CAMERA_X] = 12
    assert camera_progress_x(ram) == 256 + 12
