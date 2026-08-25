"""Travel walkability helpers shared by Pathfinder, MultiNav, and NavTask.

Live pin (Y1_Front_House + mountain_grape_stand_end, 2026-08-15):
``player_action`` stays **0** for idle, walk, run, and cliff/house push.
Jump / water hop is 3; dialogue is 9. There is no distinct push code at
``0x00D4`` — callers must also require no pixel movement while a direction
is held. Facing comes from ``player_direction`` at ``0x00DA``:
0=down, 1=up, 2=right, 3=left.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import (
    LARGE_ROCK_DAMAGE_TILES,
    LARGE_ROCK_TL,
    MAP_HEIGHT,
    MAP_WIDTH,
    STUMP_TL,
    TRAVEL_SOLID_TILES,
    debris_footprint,
    get_tile_at,
)

ADDR_PLAYER_ACTION = field_spec("player_action").address
ADDR_PLAYER_DIRECTION = field_spec("player_direction").address

# Measured live; do not invent a push id that is not in RAM.
PLAYER_ACTION_IDLE = 0  # idle + walk + run + push (same byte)
PLAYER_ACTION_JUMP = 3
PLAYER_ACTION_DIALOGUE = 9

PLAYER_DIR_DOWN = 0
PLAYER_DIR_UP = 1
PLAYER_DIR_RIGHT = 2
PLAYER_DIR_LEFT = 3

DIR_FROM_CODE = {
    PLAYER_DIR_DOWN: "down",
    PLAYER_DIR_UP: "up",
    PLAYER_DIR_RIGHT: "right",
    PLAYER_DIR_LEFT: "left",
}
DIR_DELTA = {
    "down": (0, 1),
    "up": (0, -1),
    "right": (1, 0),
    "left": (-1, 0),
}

# Frames of zero pixel motion while charging a cell before it is no-go.
# Must beat MultiNav's 48f pixel-stuck, but 2f false-triggers on waypoint
# L/R centering and map-settle (D2 path 0x0C (232,128) sealed mountain).
PUSH_HOLD_FRAMES = 20


def is_travel_solid(tile_id: int) -> bool:
    """True when travel BFS must refuse this metatile (weed/stump/rock/damage)."""
    return int(tile_id) in TRAVEL_SOLID_TILES


_TWO_BY_TWO_ANCHORS = frozenset(
    {STUMP_TL, LARGE_ROCK_TL, min(LARGE_ROCK_DAMAGE_TILES)}
)


def is_travel_occupied(ram: np.ndarray, tx: int, ty: int) -> bool:
    """Refuse the cell and every sibling of a 2x2 stump/rock TL.

    Live RAM often keeps 0x0D/0x09 only on the top-left; the other three
    metatiles stay dirt/0x00 while the sprite still occupies the quad.
    """
    if is_travel_solid(get_tile_at(ram, tx, ty)):
        return True
    for dx, dy in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
        ax, ay = tx + dx, ty + dy
        if ax < 0 or ay < 0 or ax >= MAP_WIDTH or ay >= MAP_HEIGHT:
            continue
        tid = get_tile_at(ram, ax, ay)
        if tid not in _TWO_BY_TWO_ANCHORS:
            continue
        if (tx, ty) in debris_footprint((ax, ay), tid):
            return True
    return False


def read_player_action(ram: np.ndarray) -> int:
    if ADDR_PLAYER_ACTION >= len(ram):
        return PLAYER_ACTION_IDLE
    return int(ram[ADDR_PLAYER_ACTION])


def read_player_direction(ram: np.ndarray) -> int:
    if ADDR_PLAYER_DIRECTION >= len(ram):
        return PLAYER_DIR_DOWN
    return int(ram[ADDR_PLAYER_DIRECTION]) & 0x03


def facing_tile(
    tile: Tuple[int, int],
    direction: str | int,
) -> Tuple[int, int]:
    if isinstance(direction, int):
        name = DIR_FROM_CODE.get(direction, "down")
    else:
        name = direction
    dx, dy = DIR_DELTA[name]
    return tile[0] + dx, tile[1] + dy


def is_push_action(action: int) -> bool:
    """True for the measured push/locomotion byte (0).

    Jump (3) and dialogue (9) are not push. Walk/run also read as 0, so
    callers must require a held direction with no pixel movement.
    """
    return int(action) == PLAYER_ACTION_IDLE


def block_push_facing(
    temp_blocked: set,
    ram: np.ndarray,
    facing: Tuple[int, int],
    *,
    pixel_moved: bool,
) -> bool:
    """Add ``facing`` to ``temp_blocked`` when a push is measured.

    Returns True when the tile is (now) non-walkable for travel BFS.
    """
    if pixel_moved:
        return False
    if not is_push_action(read_player_action(ram)):
        return False
    temp_blocked.add(facing)
    return True
