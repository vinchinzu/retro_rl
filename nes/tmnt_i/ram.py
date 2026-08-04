"""Early RAM fields for TMNT (NES)."""

from __future__ import annotations

import numpy as np

from retro_harness.ram_state import GameMode, GameState

ADDR_HEALTH_1 = 0x0077  # selected turtle health (probe: 128 on Area 1 map)
ADDR_GAMEOVER = 0x009E
ADDR_SCORE = 0x00C2


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True on the outdoor Area 1 map (bright frame + health initialized)."""
    health = int(ram[ADDR_HEALTH_1])
    if health <= 0:
        return False
    if obs_mean is not None and obs_mean <= 100.0:
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
