"""Confirmed Stage 1 WRAM fields for The Magical Quest."""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

ADDR_PLAYER_X = 0x0024
ADDR_PLAYER_Y = 0x0027
ADDR_PROGRESS_X = 0x002A
ADDR_HEALTH_MAX = 0x02B0
ADDR_HEALTH = 0x02B1
ADDR_GAMEPLAY_ACTIVE = 0x02C0
ADDR_LIVES = 0x0372

# Screen X pins here once Mickey is blocked by the 1-1 house wall.
# Y 36 is the house ground (spawn is 34); together they are the first door.
FIRST_DOOR_X = 374
FIRST_DOOR_Y = 36


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned byte from WRAM."""
    return int(ram[address])


def read_u16le(ram: np.ndarray, address: int) -> int:
    """Read one little-endian unsigned word from WRAM."""
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def first_door_reached(state: GameState) -> bool:
    """True when Mickey is pinned on the 1-1 house door with HP remaining."""
    if state.mode is not GameMode.PLAYING or state.player_dead:
        return False
    if state.health <= 0:
        return False
    return state.player_x >= FIRST_DOOR_X and state.player_y >= FIRST_DOOR_Y


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project the currently confirmed level fields into ``GameState``."""
    active = read_u8(ram, ADDR_GAMEPLAY_ACTIVE)
    player_x = read_u16le(ram, ADDR_PLAYER_X)
    player_y = read_u8(ram, ADDR_PLAYER_Y)
    progress_x = read_u16le(ram, ADDR_PROGRESS_X)
    health = read_u8(ram, ADDR_HEALTH)
    health_max = read_u8(ram, ADDR_HEALTH_MAX)
    lives = read_u8(ram, ADDR_LIVES)
    playing = active == 1 and 0 < player_x < 0x4000
    at_door = (
        playing
        and health > 0
        and player_x >= FIRST_DOOR_X
        and player_y >= FIRST_DOOR_Y
    )
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if playing else GameMode.MENU,
        stage=1,
        room=1 if at_door else 0,
        player_x=player_x,
        player_y=player_y,
        health=health,
        lives=lives,
        camera_x=progress_x,
        player_dead=playing and health <= 0,
        extras={
            "gameplay_active": active,
            "progress_x": progress_x,
            "health_max": health_max,
            "at_first_door": at_door,
            "ram_map_partial": True,
        },
    )
