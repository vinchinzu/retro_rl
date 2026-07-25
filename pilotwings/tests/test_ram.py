from __future__ import annotations

import numpy as np

from pilotwings.ram import (
    ADDR_ALTITUDE,
    ADDR_HEADING_RAW,
    ADDR_PITCH_CONTROL,
    parse_game_state,
)
from snes_oneshot.game_state import GameMode


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = value >> 8


def test_parse_active_lesson1_flight() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)
    _put_u16(ram, ADDR_ALTITUDE, 299)
    _put_u16(ram, ADDR_HEADING_RAW, 652)
    ram[ADDR_PITCH_CONTROL] = 10

    state = parse_game_state(ram, frame=12)

    assert state.mode is GameMode.PLAYING
    assert state.player_x == 652
    assert state.player_y == 299
    assert state.extras["pitch_control"] == 10
    assert state.extras["heading_raw"] == 652


def test_zero_altitude_is_not_active_flight() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)

    assert parse_game_state(ram).mode is GameMode.MENU
