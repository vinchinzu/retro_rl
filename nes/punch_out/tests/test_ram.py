from __future__ import annotations

import numpy as np

from punch_out.ram import (
    ADDR_CLOCK_ON,
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_OPP_HEALTH,
    ADDR_OPP_PATTERN_SET,
    FIGHT_IN_RING,
    hearts,
    is_level1_ready,
    is_match_live,
    is_taunt_window,
    parse_game_state,
    stars,
)


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is True
    assert state.extras["level1_ready"] is False


def test_level1_ready_requires_both_health_bars() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = 96
    ram[ADDR_OPP_HEALTH] = 96
    assert is_level1_ready(ram) is True
    assert is_level1_ready(ram, obs_mean=10.0) is False


def test_match_live_needs_clock_and_fight_flag() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = 96
    ram[ADDR_OPP_HEALTH] = 96
    ram[ADDR_CLOCK_ON] = 1
    ram[ADDR_FIGHT_FLAG] = FIGHT_IN_RING
    assert is_match_live(ram, obs_mean=100.0) is True


def test_hearts_and_stars() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[0x0323] = 2
    ram[0x0324] = 0
    ram[0x034A] = 0x42  # 2 stars
    assert hearts(ram) == 20
    assert stars(ram) == 2


def test_taunt_window() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_OPP_PATTERN_SET] = 150
    assert is_taunt_window(ram) is True


def test_parse_playing_extras() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = 80
    ram[ADDR_OPP_HEALTH] = 48
    ram[ADDR_CLOCK_ON] = 1
    ram[ADDR_FIGHT_FLAG] = FIGHT_IN_RING
    ram[0x0006] = 2
    state = parse_game_state(ram, frame=10, obs_mean=100.0)
    assert state.health == 80
    assert state.extras["opp_health"] == 48
    assert state.extras["round"] == 2
    assert state.extras["match_live"] is True
