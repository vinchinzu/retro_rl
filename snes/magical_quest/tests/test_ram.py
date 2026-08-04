from __future__ import annotations

import numpy as np

from magical_quest.ram import (
    ADDR_GAMEPLAY_ACTIVE,
    ADDR_PLAYER_X,
    ADDR_PROGRESS_X,
    parse_game_state,
)
from retro_harness.ram_state import GameMode


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = value >> 8


def test_parse_stage1_state() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_GAMEPLAY_ACTIVE] = 1
    _put_u16(ram, ADDR_PLAYER_X, 360)
    _put_u16(ram, ADDR_PROGRESS_X, 2696)

    state = parse_game_state(ram, frame=4)

    assert state.mode is GameMode.PLAYING
    assert state.player_x == 360
    assert state.camera_x == 2696
    assert state.frame == 4


def test_parse_menu_before_gameplay_is_active() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)
    _put_u16(ram, ADDR_PLAYER_X, 360)

    assert parse_game_state(ram).mode is GameMode.MENU

