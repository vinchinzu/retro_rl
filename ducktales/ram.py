"""Early RAM fields for DuckTales (NES)."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

ADDR_LIVES = 0x0000


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once first controllable play is RAM-verified."""
    # Filled by bootstrap probe; until then require bright frame only.
    if obs_mean is not None and obs_mean <= 15.0:
        return False
    return int(ram[ADDR_LIVES]) > 0 if ADDR_LIVES else False


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project confirmed early fields into ``GameState``."""
    ready = is_level1_ready(ram)
    extras = {"level1_ready": ready, "ram_map_partial": True}
    for name in (
        "ADDR_HEALTH_1",
        "ADDR_HEALTH",
        "ADDR_LIFE",
        "ADDR_LIVES",
        "ADDR_MODE",
        "ADDR_SCORE",
        "ADDR_GAMEOVER",
        "ADDR_SCREEN",
        "ADDR_FLAG",
        "ADDR_OPP_HEALTH",
        "ADDR_LEVEL_LO",
        "ADDR_WORLD",
        "ADDR_HPOS",
    ):
        addr = globals().get(name)
        if addr is not None and addr < len(ram):
            extras[name.removeprefix("ADDR_").lower()] = int(ram[addr])
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING if ready else GameMode.MENU,
        stage=1,
        room=0,
        player_x=0,
        player_y=0,
        health=int(extras.get("health_1", extras.get("health", extras.get("life", 0)))),
        lives=int(extras.get("lives", 0)),
        enemies=(),
        extras=extras,
    )
