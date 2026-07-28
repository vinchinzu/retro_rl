"""Early RAM fields for Castlevania (NES)."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

ADDR_LIVES = 0x002A
ADDR_HEALTH = 0x0044


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once Stage 1 health and lives are live."""
    health = int(ram[ADDR_HEALTH])
    lives = int(ram[ADDR_LIVES])
    if not (0 < health < 100 and 0 < lives < 20):
        return False
    if obs_mean is not None and obs_mean <= 15.0:
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
