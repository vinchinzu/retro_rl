"""Early RAM fields for Mega Man 2 (NES)."""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

ADDR_LIVES = 0x00A8
ADDR_HEALTH = 0x06C0  # full bar often 28


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once in-stage energy is live (not title/stage select)."""
    health = int(ram[ADDR_HEALTH])
    lives = int(ram[ADDR_LIVES])
    if not (0 < health <= 28 and 0 < lives < 10):
        return False
    if obs_mean is not None and obs_mean <= 50.0:
        return False
    return True


def parse_game_state(ram: np.ndarray, frame: int = 0, obs_mean: float | None = None) -> GameState:
    """Project confirmed early fields into ``GameState``."""
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    extras = {
        "level1_ready": ready,
        "ram_map_partial": True,
        "health": int(ram[ADDR_HEALTH]),
        "lives": int(ram[ADDR_LIVES]),
    }
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if ready else GameMode.MENU,
        stage=1,
        room=0,
        player_x=0,
        player_y=0,
        health=extras["health"],
        lives=extras["lives"],
        enemies=(),
        extras=extras,
    )
