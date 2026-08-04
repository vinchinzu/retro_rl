"""Confirmed early Stage 1 WRAM fields for The Magical Quest."""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

ADDR_PLAYER_X = 0x0024
ADDR_PROGRESS_X = 0x002A
ADDR_GAMEPLAY_ACTIVE = 0x02C0


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned byte from WRAM."""
    return int(ram[address])


def read_u16le(ram: np.ndarray, address: int) -> int:
    """Read one little-endian unsigned word from WRAM."""
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project the currently confirmed level fields into ``GameState``."""
    active = read_u8(ram, ADDR_GAMEPLAY_ACTIVE)
    player_x = read_u16le(ram, ADDR_PLAYER_X)
    progress_x = read_u16le(ram, ADDR_PROGRESS_X)
    playing = active == 1 and 0 < player_x < 0x4000
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if playing else GameMode.MENU,
        stage=0,
        room=0,
        player_x=player_x,
        player_y=0,
        health=0,
        lives=0,
        camera_x=progress_x,
        enemies=(),
        extras={
            "gameplay_active": active,
            "progress_x": progress_x,
            "ram_map_partial": True,
        },
    )

