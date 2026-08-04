from __future__ import annotations

import numpy as np

from rival_turf.ram import (
    ADDR_RUN_STATE,
    OFF_ACTIVE,
    OFF_X,
    OFF_Y,
    PLAYER_BASE,
    parse_game_state,
)
from retro_harness.ram_state import GameMode


def test_parse_stage1_player() -> None:
    ram = np.zeros(0x3000, dtype=np.uint8)
    ram[ADDR_RUN_STATE] = 1
    ram[PLAYER_BASE + OFF_ACTIVE] = 1
    ram[PLAYER_BASE + OFF_X] = 96
    ram[PLAYER_BASE + OFF_Y] = 146

    state = parse_game_state(ram, frame=12)

    assert state.mode is GameMode.PLAYING
    assert (state.player_x, state.player_y) == (96, 146)
    assert state.frame == 12
    assert state.extras["ram_map_partial"] is True


def test_parse_menu_when_run_state_is_inactive() -> None:
    ram = np.zeros(0x3000, dtype=np.uint8)
    ram[PLAYER_BASE + OFF_ACTIVE] = 1
    ram[PLAYER_BASE + OFF_X] = 96
    ram[PLAYER_BASE + OFF_Y] = 146

    assert parse_game_state(ram).mode is GameMode.MENU

