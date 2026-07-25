from __future__ import annotations

import numpy as np

from f_zero.ram import (
    ADDR_LATERAL,
    ADDR_LATERAL_FINE,
    ADDR_RACE_STATE,
    ADDR_SPEED_RAW,
    ADDR_TRACK_STATE,
    parse_game_state,
)
from snes_oneshot.game_state import GameMode


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = value >> 8


def test_parse_mute_city_race_state() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_RACE_STATE] = 1
    ram[ADDR_TRACK_STATE] = 1
    _put_u16(ram, ADDR_SPEED_RAW, 3196)
    _put_u16(ram, ADDR_LATERAL, 344)
    _put_u16(ram, ADDR_LATERAL_FINE, 3928)

    state = parse_game_state(ram, frame=9)

    assert state.mode is GameMode.PLAYING
    assert state.player_x == 344
    assert state.extras["speed_raw"] == 3196
    assert state.extras["lateral_fine"] == 3928


def test_parse_menu_before_track_is_live() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_RACE_STATE] = 4

    assert parse_game_state(ram).mode is GameMode.MENU

