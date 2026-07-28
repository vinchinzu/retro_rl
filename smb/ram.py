"""Early RAM fields for Super Mario Bros. (NES)."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

ADDR_LIVES = 0x075a
ADDR_LEVEL_LO = 0x075c
ADDR_WORLD = 0x075f


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once first controllable play is RAM-verified."""
    lives = int(ram[ADDR_LIVES])
    # SMB stores remaining lives as a counter that is live once gameplay starts.
    if lives < 0 or lives > 98:
        return False
    if obs_mean is not None and obs_mean <= 20.0:
        return False
    # Reject pure black / title by requiring non-trivial lives or bright frame.
    if lives == 0 and (obs_mean is None or obs_mean < 40.0):
        return False
    return True


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
