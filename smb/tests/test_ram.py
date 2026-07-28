from __future__ import annotations

import numpy as np

from smb.ram import (
    ADDR_LEVEL,
    ADDR_LIVES,
    ADDR_OPER_MODE,
    ADDR_WORLD,
    is_dying,
    parse_game_state,
    player_x,
    reached_ending,
    read_snapshot,
)


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is False
    assert state.extras["level_id"] == 0
    assert state.player_dead is False


def test_player_x_combines_page_and_offset() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[0x006D] = 3
    ram[0x0086] = 40
    assert player_x(ram) == 3 * 256 + 40
    snap = read_snapshot(ram)
    assert snap.player_x == 808


def test_dying_state() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    assert is_dying(ram) is False
    ram[0x000E] = 0x0B
    assert is_dying(ram) is True


def test_reached_ending_requires_8_4_end_mode_and_lives() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_WORLD] = 7
    ram[ADDR_LEVEL] = 3
    ram[ADDR_OPER_MODE] = 2
    ram[ADDR_LIVES] = 2
    assert reached_ending(ram, start_lives=2)
    ram[ADDR_OPER_MODE] = 1
    assert not reached_ending(ram, start_lives=2)
    ram[ADDR_OPER_MODE] = 2
    ram[ADDR_LIVES] = 1
    assert not reached_ending(ram, start_lives=2)
