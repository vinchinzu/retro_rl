"""Early RAM fields for TMNT III (NES)."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

ADDR_LIVES = 0x006A
ADDR_SCORE = 0x006C


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once lives are allocated and the scene is Stage-like (not a cutscene).

    Intro cutscenes can also show lives>0 with a dark frame mean; require a
    brighter outdoor/stage mean when observation is available.
    """
    lives = int(ram[ADDR_LIVES])
    if lives <= 0:
        return False
    if obs_mean is not None and obs_mean <= 55.0:
        return False
    return True


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project confirmed early fields into ``GameState``."""
    ready = is_level1_ready(ram)
    extras = {"level1_ready": ready, "ram_map_partial": True}
    # Attach known scalar fields when present on this module.
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
