"""Early RAM fields for Super Mario Bros. 3 (NES)."""

from __future__ import annotations

import numpy as np

from snes_oneshot.game_state import GameMode, GameState

# Player / progress (in-level)
ADDR_HPOS = 0x0090  # on-screen horizontal position
ADDR_VPOS = 0x00A2  # vertical position in levels
ADDR_X_PAGE = 0x0075  # coarse horizontal (8-block units in levels; dual-use on map)
ADDR_IN_AIR = 0x00D8
ADDR_HVEL = 0x00BD

# Map
ADDR_MAP_Y = 0x0078
ADDR_MAP_X = 0x0079

# Meta
ADDR_LIVES = 0x0736
ADDR_WORLD = 0x0727  # world number - 1
ADDR_FORM = 0x0746
ADDR_AUTO_CONTROL = 0x0559  # non-zero during goal card / cutscene
ADDR_RETURN_MAP = 0x0014


def player_progress_x(ram) -> float:
    """Approximate absolute X while in a level; 0 if page looks like map/death."""
    page = int(ram[ADDR_X_PAGE])
    if page >= 0x18:
        return 0.0
    return float(page * 256 + int(ram[ADDR_HPOS]))


def is_in_level(ram) -> bool:
    """Heuristic: controllable in-level pose (screen X live, low page)."""
    page = int(ram[ADDR_X_PAGE])
    hpos = int(ram[ADDR_HPOS])
    return page < 0x18 and hpos > 0


def is_goal_auto(ram) -> bool:
    """True while the game auto-controls Mario (goal grab / card)."""
    return int(ram[ADDR_AUTO_CONTROL]) != 0


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True on World 1 map (lives live; reject curtain/solid frames)."""
    lives = int(ram[ADDR_LIVES])
    if not (0 < lives < 20):
        return False
    if obs_mean is None:
        return True
    # World map is structured mid-range; reject curtain red and flat greys.
    if not (90.0 < obs_mean < 150.0):
        return False
    return True


def parse_game_state(ram: np.ndarray, frame: int = 0, obs_mean: float | None = None) -> GameState:
    """Project confirmed early fields into ``GameState``."""
    ready = is_level1_ready(ram, obs_mean=obs_mean)
    in_level = is_in_level(ram)
    progress = player_progress_x(ram)
    extras = {
        "level1_ready": ready,
        "in_level": in_level,
        "ram_map_partial": True,
        "lives": int(ram[ADDR_LIVES]),
        "hpos": int(ram[ADDR_HPOS]),
        "vpos": int(ram[ADDR_VPOS]),
        "x_page": int(ram[ADDR_X_PAGE]),
        "progress_x": progress,
        "world": int(ram[ADDR_WORLD]),
        "auto_control": int(ram[ADDR_AUTO_CONTROL]),
        "map_x": int(ram[ADDR_MAP_X]),
        "map_y": int(ram[ADDR_MAP_Y]),
    }
    if in_level:
        mode = GameMode.PLAYING
    elif ready:
        mode = GameMode.PLAYING  # map control counts as playable
    else:
        mode = GameMode.MENU
    return GameState(
        frame=frame,
        mode=mode,
        stage=1,
        room=0,
        player_x=int(progress) if in_level else extras["hpos"],
        player_y=extras["vpos"],
        health=0,
        lives=extras["lives"],
        enemies=(),
        extras=extras,
    )
