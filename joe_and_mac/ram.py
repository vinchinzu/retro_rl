"""Confirmed early WRAM fields for Joe & Mac Stage 1."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

ADDR_HORIZONTAL_PROGRESS = 0x006C
ADDR_GAMEPLAY_ACTIVE = 0x0081
ADDR_ACTOR_STATE = 0x0082


def read_u16(ram: np.ndarray, address: int) -> int:
    """Read one little-endian unsigned word from WRAM."""
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project the confirmed Stage 1 fields into ``GameState``."""
    active = int(ram[ADDR_GAMEPLAY_ACTIVE])
    progress = read_u16(ram, ADDR_HORIZONTAL_PROGRESS)
    actor_state = int(ram[ADDR_ACTOR_STATE])
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if active == 1 else GameMode.MENU,
        stage=1,
        room=0,
        player_x=progress,
        player_y=0,
        health=0,
        lives=0,
        enemies=(),
        extras={
            "gameplay_active": active,
            "horizontal_progress": progress,
            "actor_state": actor_state,
            "ram_map_partial": True,
        },
    )
