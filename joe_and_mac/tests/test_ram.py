from __future__ import annotations

import numpy as np

from joe_and_mac.ram import (
    ADDR_ACTOR_STATE,
    ADDR_GAMEPLAY_ACTIVE,
    ADDR_HORIZONTAL_PROGRESS,
    parse_game_state,
)
from snes_oneshot.game_state import GameMode


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = value >> 8


def test_parse_active_stage1_state() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_GAMEPLAY_ACTIVE] = 1
    ram[ADDR_ACTOR_STATE] = 4
    _put_u16(ram, ADDR_HORIZONTAL_PROGRESS, 204)

    state = parse_game_state(ram, frame=17)

    assert state.mode is GameMode.PLAYING
    assert state.player_x == 204
    assert state.extras["actor_state"] == 4


def test_map_is_not_active_gameplay() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)

    assert parse_game_state(ram).mode is GameMode.MENU
