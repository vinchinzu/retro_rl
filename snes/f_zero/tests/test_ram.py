from __future__ import annotations

import numpy as np

from f_zero.ram import (
    ADDR_CHECKPOINT_FACING,
    ADDR_FINISH_STATE,
    ADDR_GAMESTATE,
    ADDR_GAMESTATE_SUB,
    ADDR_HEADING,
    ADDR_LATERAL,
    ADDR_LATERAL_FINE,
    ADDR_POWER,
    ADDR_RACE_STATE,
    ADDR_SCREEN_TEXT,
    ADDR_SPEED_RAW,
    ADDR_TRACK_STATE,
    FINISH_EXPLODED,
    LapWatch,
    SCREEN_LAPS_LEFT,
    crashed_out,
    heading_error,
    parse_game_state,
)
from retro_harness.ram_state import GameMode


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def _ram() -> np.ndarray:
    return np.zeros(0x1000, dtype=np.uint8)


def test_parse_mute_city_race_state() -> None:
    ram = _ram()
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
    ram = _ram()
    ram[ADDR_RACE_STATE] = 4

    assert parse_game_state(ram).mode is GameMode.MENU


def test_parse_live_race_finish_and_heading() -> None:
    ram = _ram()
    ram[ADDR_GAMESTATE] = 2
    ram[ADDR_GAMESTATE_SUB] = 3
    ram[ADDR_HEADING] = 48
    ram[ADDR_CHECKPOINT_FACING] = 52
    ram[ADDR_SCREEN_TEXT] = SCREEN_LAPS_LEFT
    _put_u16(ram, ADDR_POWER, 177)
    state = parse_game_state(ram, frame=2724)
    assert state.mode is GameMode.PLAYING
    assert state.level_complete is True
    assert state.extras["racing"] is True
    assert state.extras["heading_error"] == -4
    assert state.extras["power"] == 177
    assert state.player_dead is False


def test_parse_explosion_is_dead() -> None:
    ram = _ram()
    ram[ADDR_GAMESTATE] = 2
    ram[ADDR_GAMESTATE_SUB] = 3
    ram[ADDR_FINISH_STATE] = FINISH_EXPLODED
    _put_u16(ram, ADDR_POWER, 0xFFFC)
    state = parse_game_state(ram)
    assert state.player_dead is True
    assert state.mode is GameMode.MENU
    assert state.extras["power"] == -4


def test_heading_error_wraps_angle8() -> None:
    assert heading_error(48, 52) == -4
    assert heading_error(191, 0) == -1
    assert heading_error(0, 191) == 1


def test_lap_watch_counts_rising_laps_left_only() -> None:
    watch = LapWatch()
    assert watch.update(0x01) is False
    assert watch.update(SCREEN_LAPS_LEFT | 0x20) is True
    assert watch.update(SCREEN_LAPS_LEFT | 0x20) is False
    assert watch.update(0x20) is False
    assert watch.update(SCREEN_LAPS_LEFT) is True
    assert watch.laps == 2


def test_crashed_out_ignores_zero_power_without_finish_bit() -> None:
    assert crashed_out(0, 0) is False
    assert crashed_out(FINISH_EXPLODED, 100) is True
    assert crashed_out(0, -1) is True
