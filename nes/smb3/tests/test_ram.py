from __future__ import annotations

import numpy as np

from smb3.ram import (
    ADDR_MAP_OPERATION,
    MAP_OPERATION_NORMAL,
    is_goal_auto,
    is_in_level,
    is_map_controllable,
    parse_game_state,
    player_progress_x,
)


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is True


def test_player_progress_and_goal_helpers() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[0x75] = 2
    ram[0x90] = 40
    assert player_progress_x(ram) == 2 * 256 + 40
    assert is_in_level(ram) is True
    ram[0x559] = 1
    assert is_goal_auto(ram) is True
    ram[0x75] = 0x20
    assert player_progress_x(ram) == 0.0


def test_map_controllable_is_operation_normal() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    assert is_map_controllable(ram) is False
    ram[ADDR_MAP_OPERATION] = MAP_OPERATION_NORMAL
    assert is_map_controllable(ram) is True
    extras = parse_game_state(ram).extras
    assert extras["map_operation"] == MAP_OPERATION_NORMAL
    assert extras["map_controllable"] is True
