"""Confirmed early WRAM fields for Pilotwings.

Addresses are stable-retro WRAM offsets. Altitude is the value rendered by the
Lesson 1 HUD; pitch and heading were isolated with held directional input from
the same light-plane checkpoint.
"""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

ADDR_ALTITUDE = 0x0058
ADDR_PITCH_CONTROL = 0x005D
ADDR_HEADING_RAW = 0x0060


def read_u16(ram: np.ndarray, address: int) -> int:
    """Read one little-endian unsigned word from WRAM."""
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def read_i8(ram: np.ndarray, address: int) -> int:
    """Read one signed byte from WRAM."""
    value = int(ram[address])
    return value - 0x100 if value & 0x80 else value


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project the confirmed Lesson 1 flight fields into ``GameState``."""
    altitude = read_u16(ram, ADDR_ALTITUDE)
    pitch_control = read_i8(ram, ADDR_PITCH_CONTROL)
    heading_raw = read_u16(ram, ADDR_HEADING_RAW)
    playing = 0 < altitude < 2000
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if playing else GameMode.MENU,
        stage=1,
        room=1,
        player_x=heading_raw,
        player_y=altitude,
        health=0,
        lives=0,
        enemies=(),
        extras={
            "altitude": altitude,
            "pitch_control": pitch_control,
            "heading_raw": heading_raw,
            "ram_map_partial": True,
        },
    )
