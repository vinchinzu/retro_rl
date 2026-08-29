from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode
from zelda_ii.ram import (
    ADDR_ENGINE_MODE,
    ADDR_HEALTH,
    ADDR_LIFE,
    ADDR_OW_X,
    ADDR_OW_Y,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    MODE_OVERWORLD,
    MODE_SIDESCROLL,
    is_dead,
    is_level1_ready,
    is_overworld,
    is_side_scroll,
    palace_exit_success,
    parse_game_state,
    read_snapshot,
)


def _ram(bytes_at: dict[int, int]) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    for addr, value in bytes_at.items():
        ram[addr] = value
    return ram


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is True
    assert state.mode == GameMode.MENU
    assert state.extras["level1_ready"] is False
    assert palace_exit_success(ram) is False
    assert is_dead(ram) is False


def test_side_scroll_snapshot_and_ready() -> None:
    ram = _ram(
        {
            ADDR_LIFE: 127,
            ADDR_HEALTH: 127,
            ADDR_ENGINE_MODE: MODE_SIDESCROLL,
            ADDR_PLAYER_X: 159,
            ADDR_PLAYER_Y: 176,
        }
    )
    assert is_level1_ready(ram, obs_mean=90.0)
    assert is_side_scroll(ram)
    assert is_overworld(ram) is False
    snap = read_snapshot(ram)
    assert snap.player_x == 159
    state = parse_game_state(ram, frame=10, obs_mean=90.0)
    assert state.mode == GameMode.PLAYING
    assert state.player_x == 159
    assert state.health == 127
    assert state.extras["palace_exit"] is False


def test_overworld_is_palace_exit_stop() -> None:
    ram = _ram(
        {
            ADDR_LIFE: 127,
            ADDR_HEALTH: 127,
            ADDR_ENGINE_MODE: MODE_OVERWORLD,
            ADDR_OW_X: 23,
            ADDR_OW_Y: 52,
        }
    )
    assert palace_exit_success(ram)
    assert is_overworld(ram)
    state = parse_game_state(ram, frame=331, obs_mean=110.0)
    assert state.mode == GameMode.PLAYING
    assert state.player_x == 23
    assert state.player_y == 52
    assert state.extras["palace_exit"] is True


def test_death_requires_play_mode() -> None:
    ram = _ram({ADDR_HEALTH: 0, ADDR_LIFE: 127, ADDR_ENGINE_MODE: MODE_SIDESCROLL})
    assert is_dead(ram)
    state = parse_game_state(ram, frame=1)
    assert state.player_dead is True
    assert state.mode == GameMode.GAME_OVER
    menu = _ram({ADDR_HEALTH: 0, ADDR_LIFE: 0, ADDR_ENGINE_MODE: 0})
    assert is_dead(menu) is False
