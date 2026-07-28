"""Early RAM fields for TMNT II (NES)."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

ADDR_LIVES = 0x004D
ADDR_HEALTH = 0x0568
ADDR_SCORE = 0x03F0


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True once Stage 1 combat health/lives are live."""
    health = int(ram[ADDR_HEALTH])
    lives = int(ram[ADDR_LIVES])
    if not (0 < health < 200 and lives >= 0):
        return False
    if obs_mean is not None and obs_mean <= 30.0:
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
