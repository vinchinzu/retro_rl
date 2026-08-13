"""Coop interior layout constants for CoopChoresTask.

Extracted from ``coop_task`` so chore FSM composition stays thin.
Constants and pure stand helpers here must stay free of task state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from harvest.maps.map_config import find_landmark
from harvest.tasks.nav import TILE_SIZE

# ── Coop interior layout (tilemap 0x28) ─────────────────────────
# Positions discovered via coop_chores / coop_sell_egg recording traces.

FEED_BIN_STAND: Tuple[int, int] = (2, 6)
FEED_BIN_FACE: str = "left"
FEED_CLEAR_STAND: Tuple[int, int] = (2, 3)
VISIBLE_EGG_SPRITE = 0x00F3


@dataclass(frozen=True)
class ChickenFeedSpot:
    stand: Tuple[int, int]
    face: str
    interact_px: Tuple[int, int]
    flag: int


CHICKEN_FEED_FLAGS: Tuple[int, ...] = (
    0x0001,
    0x0002,
    0x0004,
    0x0008,
    0x0010,
    0x0020,
    0x0040,
    0x0080,
    0x0100,
    0x0200,
    0x0400,
    0x0800,
)

# Coop feed trough tiles are the top row tile properties 0xE2..0xED.
# The reliable interaction point, from recordings, is near the lower-left
# of the stand tile while holding Up+A into the trough.
CHICKEN_FEED_SPOTS: Tuple[ChickenFeedSpot, ...] = tuple(
    ChickenFeedSpot(
        stand=(x, 3),
        face="up",
        interact_px=(x * TILE_SIZE + (6 if x == 2 else 10), 3 * TILE_SIZE + 14),
        flag=flag,
    )
    for x, flag in zip(range(2, 14), CHICKEN_FEED_FLAGS)
)

EGG_PICKUP_STAND: Tuple[int, int] = (2, 4)
EGG_PICKUP_FACE: str = "left"
CHICKEN_EGG_SPAWN_PIXELS: Tuple[Tuple[int, int], ...] = (
    (0x18, 0x48),
    (0x38, 0x58),
    (0x48, 0x98),
    (0x58, 0x78),
    (0x68, 0xA8),
    (0x78, 0x88),
    (0x88, 0x58),
    (0x98, 0x98),
    (0xA8, 0x78),
    (0xB8, 0xA8),
    (0xC8, 0x68),
    (0xD8, 0x88),
    (0x28, 0xA8),
)
CHICKEN_EGG_FLAGS: Tuple[int, ...] = tuple(1 << slot for slot in range(len(CHICKEN_EGG_SPAWN_PIXELS)))


def egg_recording_stand(px: int, py: int) -> Tuple[Tuple[int, int], str]:
    """Stand tile/face for a spawn pixel, avoiding the false-open x=5 column."""
    tx, ty = px // TILE_SIZE, py // TILE_SIZE
    stand = (tx + 1, ty)
    if stand[0] == 5:
        return (tx, ty + 1), "up"
    return stand, "left"


EGG_PICKUP_SPOTS: Tuple[Tuple[int, Tuple[int, int], str], ...] = (
    (0x02, (4, 5), "left"),
    (0x01, (2, 4), "left"),
    *(
        (flag, *egg_recording_stand(px, py))
        for flag, (px, py) in zip(CHICKEN_EGG_FLAGS[2:], CHICKEN_EGG_SPAWN_PIXELS[2:])
    ),
)

COOP_ENTRY_STAND: Tuple[int, int] = (8, 12)
COOP_MAIN_AISLE_TOP: Tuple[int, int] = (8, 6)
COOP_LEFT_TOP_APPROACH: Tuple[int, int] = (4, 5)

INCUBATOR_STAND: Tuple[int, int] = (13, 11)
INCUBATOR_FACE: str = "right"
INCUBATOR_APPROACH: Tuple[Tuple[int, int], ...] = ((8, 10), (10, 11), INCUBATOR_STAND)

_EGG_SHIPPING_BIN_LANDMARK = find_landmark("egg_shipping_bin", tilemap_id=0x28)

# Bottom-left coop shipping bin is interacted with from the aisle tile just
# above the bin frontage, facing down into the bin.
SHIP_BIN_STAND: Tuple[int, int] = (2, 10)
SHIP_BIN_INTERACT_STAND: Tuple[int, int] = (
    _EGG_SHIPPING_BIN_LANDMARK[1].tile
    if _EGG_SHIPPING_BIN_LANDMARK is not None
    else (1, 10)
)
SHIP_BIN_FACE: str = (
    _EGG_SHIPPING_BIN_LANDMARK[1].face
    if _EGG_SHIPPING_BIN_LANDMARK is not None and _EGG_SHIPPING_BIN_LANDMARK[1].face
    else "down"
)
SHIP_LANE_X = 38
SHIP_APPROACH_Y = 165
SHIP_INTERACT_PX = (22, 169)
SHIP_RIGHT_LANE_CORNER: Tuple[int, int] = (3, 10)

# Stage at the same door tile EXIT_COOP expects. Handing off at the left
# service lane (3,11) left EXIT_COOP to cross the false-open x=5 edge and time
# out (seen at pos=(57,184) tile=(3,11)).
EXIT_PREP_STAND: Tuple[int, int] = COOP_ENTRY_STAND
# The coop tilemap reports a walkable vertical strip around x=5 that is not
# actually passable. Long-run stalls consistently pin at (5, 11) / (86, 183).
COOP_FALSE_OPEN_COLUMN_X = 5
COOP_FALSE_OPEN_MIN_Y = 8
EXIT_PREP_ESCAPE_ROUTE: Tuple[Tuple[int, int], ...] = (
    (5, 12),
    (2, 12),
    EXIT_PREP_STAND,
)
EXIT_PREP_LEFT_ROUTE: Tuple[Tuple[int, int], ...] = (
    (2, 12),
    EXIT_PREP_STAND,
)
MAX_EXIT_PREP_FRAMES = 360
MAX_EGG_NAV_FRAMES = 480
MAX_EGG_ATTEMPTS = 4
MAX_EGG_DEFERRALS = 1
MAX_FEED_PLACE_FRAMES = 480
MAX_FEED_SLOT_DEFERRALS = 1
MAX_FLOCK_SIZE = 12
