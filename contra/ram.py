"""Early RAM fields for Contra (NES)."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

ADDR_LIVES = 0x0032
ADDR_FLAG = 0x0008


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once Stage 1 lives/flag are live."""
    lives = int(ram[ADDR_LIVES])
    flag = int(ram[ADDR_FLAG])
    if not (0 < lives < 10 and flag > 0):
        return False
    if obs_mean is not None and obs_mean <= 20.0:
        return False
    return True


def parse_game_state(ram: np.ndarray, frame: int = 0, obs_mean: float | None = None) -> GameState:
    """Project confirmed early fields into ``GameState``."""
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    extras = {
        "level1_ready": ready,
        "ram_map_partial": True,
        "lives": int(ram[ADDR_LIVES]),
        "flag": int(ram[ADDR_FLAG]),
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
