"""Early RAM fields for DuckTales (NES)."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

ADDR_LIVES = 0x000E  # probe: stable 4 during stage play


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once stage lives counter is live."""
    lives = int(ram[ADDR_LIVES])
    if not (0 < lives < 20):
        return False
    if obs_mean is not None and not (40.0 < obs_mean < 120.0):
        return False
    return True


def parse_game_state(
    ram: np.ndarray, frame: int = 0, obs_mean: float | None = None
) -> GameState:
    """Project confirmed early fields into ``GameState``."""
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    extras = {
        "level1_ready": ready,
        "ram_map_partial": True,
        "lives": int(ram[ADDR_LIVES]),
    }
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if ready else GameMode.MENU,
        stage=1,
        room=0,
        player_x=0,
        player_y=0,
        health=0,
        lives=extras["lives"],
        enemies=(),
        extras=extras,
    )
