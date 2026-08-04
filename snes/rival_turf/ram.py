"""Confirmed early WRAM fields for Rival Turf!.

Addresses are stable-retro WRAM offsets. Player coordinates were isolated by
replaying LEFT/RIGHT and UP/DOWN from the same Stage 1 state.
"""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

ADDR_RUN_STATE = 0x00AB
PLAYER_BASE = 0x0200
OFF_ACTIVE = 0x00
OFF_X = 0x02
OFF_Y = 0x05


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned byte from WRAM."""
    return int(ram[address])


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project the currently confirmed Stage 1 fields into ``GameState``."""
    run_state = read_u8(ram, ADDR_RUN_STATE)
    active = read_u8(ram, PLAYER_BASE + OFF_ACTIVE)
    player_x = read_u8(ram, PLAYER_BASE + OFF_X)
    player_y = read_u8(ram, PLAYER_BASE + OFF_Y)
    playing = (
        run_state == 1
        and active == 1
        and 0 < player_x < 0xF0
        and 80 <= player_y < 0xE0
    )
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if playing else GameMode.MENU,
        stage=0,
        room=0,
        player_x=player_x,
        player_y=player_y,
        health=0,
        lives=0,
        enemies=(),
        extras={
            "run_state": run_state,
            "player_active": active,
            "ram_map_partial": True,
        },
    )

