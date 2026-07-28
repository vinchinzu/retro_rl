from __future__ import annotations

import numpy as np

from tmnt_ii.ram import (
    ADDR_HEALTH,
    ADDR_LIVES,
    ADDR_SCORE,
    is_level1_ready,
    parse_game_state,
    player_screen_x,
)


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is True
    assert state.mode.name == "MENU"


def test_parse_game_state_playing_when_ready() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = 60
    ram[ADDR_LIVES] = 2
    ram[ADDR_SCORE] = 3
    # OAM sprite in play band
    ram[0x200] = 140  # y
    ram[0x203] = 160  # x
    assert is_level1_ready(ram) is True
    state = parse_game_state(ram, frame=10)
    assert state.mode.name == "PLAYING"
    assert state.health == 60
    assert state.lives == 2
    assert state.extras["score"] == 3
    assert state.player_x == 160
    assert player_screen_x(ram) == 160
