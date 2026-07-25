"""Confirmed early race-state fields for F-Zero."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

# The raw speed word tracks the on-screen km/h value at roughly 10 raw units
# per displayed km/h. Exact scaling remains to be calibrated.
ADDR_SPEED_RAW = 0x0002
ADDR_RACE_STATE = 0x0046
ADDR_TRACK_STATE = 0x0047
# Both fields respond monotonically to controlled LEFT/RIGHT probes. The
# lower-resolution word is convenient for an initial centerline controller.
ADDR_LATERAL = 0x007F
ADDR_LATERAL_FINE = 0x00A6


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned byte from WRAM."""
    return int(ram[address])


def read_u16le(ram: np.ndarray, address: int) -> int:
    """Read one little-endian unsigned word from WRAM."""
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project the confirmed Mute City fields into ``GameState``."""
    race_state = read_u8(ram, ADDR_RACE_STATE)
    track_state = read_u8(ram, ADDR_TRACK_STATE)
    playing = race_state == 1 and track_state == 1
    lateral = read_u16le(ram, ADDR_LATERAL)
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if playing else GameMode.MENU,
        stage=0,
        room=0,
        player_x=lateral,
        player_y=0,
        health=0,
        lives=0,
        enemies=(),
        extras={
            "race_state": race_state,
            "track_state": track_state,
            "speed_raw": read_u16le(ram, ADDR_SPEED_RAW),
            "lateral": lateral,
            "lateral_fine": read_u16le(ram, ADDR_LATERAL_FINE),
            "ram_map_partial": True,
        },
    )

