"""Barn cow stands, routes, and pure geometry helpers.

Extracted from ``cow_task`` so the chore FSM owns composition only.
Constants and pure functions here must stay free of ``CowChoresTask`` state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Optional, Sequence, Tuple

from harvest.core.animal_probe import BARN_TILEMAP
from harvest.maps.map_config import find_landmark
from harvest.tasks.nav import MAP_WIDTH


Tile = Tuple[int, int]
Pixel = Tuple[int, int]
StandFace = Tuple[Tile, str]


# ── landmarks / static stands ─────────────────────────────────────────

_COW_TALK = find_landmark("cow_talk_stand", tilemap_id=BARN_TILEMAP)
_FODDER = find_landmark("fodder_dispenser", tilemap_id=BARN_TILEMAP)
_TROUGH = find_landmark("cow_feed_trough", tilemap_id=BARN_TILEMAP)
_BARN_BIN = find_landmark("barn_shipping_bin", tilemap_id=BARN_TILEMAP)

COW_TALK_STAND: Tile = _COW_TALK[1].tile if _COW_TALK else (10, 17)
COW_TALK_FACE: str = _COW_TALK[1].face if _COW_TALK and _COW_TALK[1].face else "left"
COW_TALK_ROUTE: Tuple[Tile, ...] = ((11, 21), COW_TALK_STAND)
COW_TALK_ANCHOR: Tile = COW_TALK_ROUTE[0]
COW_BAD_INTERACT_STANDS: set[Tile] = {(9, 17), (10, 16), (10, 18), (13, 18)}

FACE_VECTORS: dict[str, Tile] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

FODDER_STAND: Tile = _FODDER[1].tile if _FODDER else (13, 11)
FODDER_FACE: str = _FODDER[1].face if _FODDER and _FODDER[1].face else "right"
FODDER_ROUTE: Tuple[Tile, ...] = ((11, 11), FODDER_STAND)
FODDER_TROUGH_ROUTE: Tuple[Tile, ...] = ((9, 11), (11, 11), FODDER_STAND)
LEFT_TROUGH_RETURN_X = 130
LEFT_TROUGH_LANE_Y = FODDER_STAND[1] * 16 + 8

COW_EXIT_PREP_STAND: Tile = COW_TALK_ANCHOR
COW_EXIT_PREP_PX: Pixel = (
    COW_EXIT_PREP_STAND[0] * 16 + 8,
    COW_EXIT_PREP_STAND[1] * 16 + 8,
)
COW_INTERACT_X_OFFSET = 13
COW_INTERACT_Y_OFFSET = 3
COW_LEFT_INTERACT_X = 163
LEFT_COW_VERTICAL_LANE_X = 38
LEFT_COW_LOWER_LANE_X = 55
LEFT_COW_LANE_SWITCH_Y = 315
COW_UPPER_RIGHT_ROUTE_MAX_Y = 100
COW_RIGHT_AISLE_X = 204
UPPER_BARN_SHIP_ESCAPE_Y = 184
UPPER_BARN_RIGHT_AISLE_X = 216
UPPER_BARN_SHIP_CROSS_Y = 217
UPPER_BARN_SHIP_AISLE_X = 205
UPPER_BARN_SHIP_LOWER_LANE_Y = 315
LEFT_BARN_SHIP_LANE_X = 55
LEFT_BARN_SHIP_LOWER_Y = 346

BARN_SHIP_BIN_STAND: Tile = _BARN_BIN[1].tile if _BARN_BIN else (2, 22)
BARN_SHIP_BIN_INTERACT_STAND: Tile = BARN_SHIP_BIN_STAND
BARN_SHIP_BIN_FACE = _BARN_BIN[1].face if _BARN_BIN and _BARN_BIN[1].face else "left"
MILK_SHIP_ROUTE: Tuple[Tile, ...] = ((11, 21), (5, 22), BARN_SHIP_BIN_STAND)
MILK_SHIP_PIXEL_ROUTE: Tuple[Pixel, ...] = (
    (183, 328),
    (139, 328),
    (139, 346),
    (55, 346),
    (55, 358),
    (38, 361),
)


@dataclass(frozen=True)
class CowFeedSpot:
    stand: Tile
    face: str
    interact_px: Pixel
    flag: int


# Feed trough coordinates decoded from the barn replacement table
# (DATA16_81B0ED) and Cow_Feed_Flags in the decomp.
COW_FEED_SPOTS: Tuple[CowFeedSpot, ...] = (
    CowFeedSpot((7, 9), "right", (113, 149), 0x0008),
    CowFeedSpot((7, 13), "right", (113, 213), 0x0004),
    CowFeedSpot((7, 15), "right", (113, 245), 0x0002),
    CowFeedSpot((7, 17), "right", (113, 277), 0x0001),
    CowFeedSpot((7, 7), "right", (113, 117), 0x0010),
    CowFeedSpot((7, 5), "right", (113, 85), 0x0020),
    CowFeedSpot((6, 17), "left", (111, 277), 0x0040),
    CowFeedSpot((6, 15), "left", (111, 245), 0x0080),
    CowFeedSpot((6, 13), "left", (111, 213), 0x0100),
    CowFeedSpot((6, 9), "left", (111, 149), 0x0200),
    CowFeedSpot((6, 7), "left", (111, 117), 0x0400),
    CowFeedSpot((6, 5), "left", (111, 85), 0x0800),
)

CARE_TROUGH_EXIT_X = COW_FEED_SPOTS[0].interact_px[0]
CARE_TROUGH_EXIT_MIN_Y = COW_FEED_SPOTS[0].interact_px[1] - 8
CARE_TROUGH_EXIT_ANCHOR_X = COW_TALK_ANCHOR[0] * 16 + 8
CARE_TROUGH_EXIT_BOTTOM_Y = COW_TALK_ANCHOR[1] * 16 + 8
FEED_TROUGH_STAND: Tile = COW_FEED_SPOTS[0].stand if not _TROUGH else _TROUGH[1].tile
FEED_TROUGH_FACE: str = (
    COW_FEED_SPOTS[0].face if not _TROUGH else (_TROUGH[1].face or COW_FEED_SPOTS[0].face)
)
FEED_TROUGH_ROUTE: Tuple[Tile, ...] = ((9, 11), FEED_TROUGH_STAND)
FEED_TROUGH_INTERACT_PX: Pixel = COW_FEED_SPOTS[0].interact_px


# ── pure geometry ─────────────────────────────────────────────────────

def facing_tile(stand: Tile, face: str) -> Tile:
    """Return the tile the player faces from ``stand`` looking ``face``."""
    dx, dy = FACE_VECTORS.get(face, (-1, 0))
    return stand[0] + dx, stand[1] + dy


def cow_body_tile(cow_tile: Tile) -> Tile:
    """Cow hitbox body tile one row below the head/slot tile."""
    return cow_tile[0], cow_tile[1] + 1


def is_adjacent_to_cow_tile(stand: Tile, face: str, cow_tile: Tile) -> bool:
    """True when facing the cow head tile or a valid body-side pin."""
    facing = facing_tile(stand, face)
    if facing == cow_tile:
        return True
    body = cow_body_tile(cow_tile)
    # Horizontal sides and below-body (face up) are valid talk/brush pins.
    return facing == body and face in ("left", "right", "up")


def feed_route_for_spot(spot: CowFeedSpot) -> Tuple[Tile, ...]:
    """Tile route to a feed trough spot from the barn aisle."""
    if spot.stand[0] <= 7:
        return ((9, 11), spot.stand)
    return ((11, 11), spot.stand)


def fodder_route_from(current_tile: Tile) -> Tuple[Tile, ...]:
    """Fodder dispenser approach route from the player's current barn tile."""
    tx, ty = current_tile
    if tx == 10 and 13 <= ty <= 18:
        return ((11, ty),) + FODDER_ROUTE
    if (ty >= 18 and tx <= 9) or (ty >= 13 and tx <= 8):
        return FODDER_TROUGH_ROUTE
    return FODDER_ROUTE


def talk_route_to(stand: Tile) -> Tuple[Tile, ...]:
    """Standard talk/brush approach: lower-aisle anchor then interact stand."""
    return (COW_TALK_ANCHOR, stand)


def left_cow_lane_x(current_y: int) -> int:
    """Vertical lane X for left-wall cow care (switches near the lower corridor)."""
    if current_y > LEFT_COW_LANE_SWITCH_Y:
        return LEFT_COW_LOWER_LANE_X
    return LEFT_COW_VERTICAL_LANE_X


def preferred_cow_stands(cx: int, cy: int) -> list[StandFace]:
    """Geometric interact stand candidates ordered by cow column preference.

    Does not include the player's current tile or path/walkability filters —
    callers prepend adjacency pins and score/filter for reachability.
    """
    preferred: list[StandFace] = []
    if cx <= 4:
        # Wall-side cows: stay on the right/body column. Same-x stands
        # (face up/down) trap the player on the cow's tile column.
        preferred.extend(
            [
                ((cx + 1, cy + 1), "left"),
                ((cx + 1, cy), "left"),
                ((cx - 1, cy), "right"),
                ((cx - 1, cy + 1), "right"),
            ]
        )
    elif cx <= 10:
        preferred.append(((cx + 1, cy), "left"))
        preferred.append(((cx + 1, cy + 1), "left"))
    elif cx >= 12:
        preferred.append(((cx - 1, cy), "right"))
        preferred.append(((cx - 1, cy + 1), "right"))
        # Prefer staying on the right aisle for right-side cows instead of
        # flipping to a left-face stand that pixel-nav used to mis-aim.
        preferred.append(((cx + 1, cy), "left"))
        preferred.append(((cx + 1, cy + 1), "left"))
    else:
        preferred.extend(
            [
                ((cx + 1, cy), "left"),
                ((cx - 1, cy), "right"),
                ((cx + 1, cy + 1), "left"),
                ((cx - 1, cy + 1), "right"),
            ]
        )

    preferred.extend(
        [
            ((cx, cy + 1), "up"),
            ((cx, cy - 1), "down"),
            ((cx + 1, cy + 1), "left"),
            ((cx - 1, cy + 1), "right"),
            ((cx + 1, cy), "left"),
            ((cx - 1, cy), "right"),
        ]
    )
    return preferred


def body_side_stand_candidates(cx: int, cy: int) -> list[StandFace]:
    """Body-side interact stands preferred for talk/brush pinning."""
    if cx <= 4:
        return [
            ((cx + 1, cy + 1), "left"),
            ((cx + 1, cy), "left"),
        ]
    if cx <= 10:
        return [((cx + 1, cy + 1), "left"), ((cx + 1, cy), "left")]
    if cx >= 12:
        return [((cx - 1, cy + 1), "right"), ((cx - 1, cy), "right")]
    return [
        ((cx + 1, cy + 1), "left"),
        ((cx - 1, cy + 1), "right"),
        ((cx + 1, cy), "left"),
        ((cx - 1, cy), "right"),
    ]


def geometric_fallback_stands(
    cx: int,
    cy: int,
    cow_tiles: Collection[Tile],
    *,
    current: Tile,
    current_face: str,
) -> list[StandFace]:
    """Last-resort stands when path/walkability filters reject all candidates."""
    if cx >= 12:
        return [((cx - 1, cy), "right"), ((cx - 1, cy + 1), "right")]
    if cx <= 10:
        return [((cx + 1, cy), "left"), ((cx + 1, cy + 1), "left")]
    if COW_TALK_STAND not in cow_tiles and COW_TALK_STAND not in COW_BAD_INTERACT_STANDS:
        return [(COW_TALK_STAND, COW_TALK_FACE)]
    return [(current, current_face)]


def cow_push_escape_tile(cow_tile: Tile, stand: Tile, face: str) -> Optional[Tile]:
    """Tile the cow would be pushed into when interacting from ``stand``/``face``.

    Returns ``None`` when the stand does not face the cow head tile.
    """
    dx, dy = FACE_VECTORS.get(face, (0, 0))
    if (stand[0] + dx, stand[1] + dy) != cow_tile:
        return None
    return cow_tile[0] + dx, cow_tile[1] + dy


def face_for_cow_at_stand(
    stand: Tile,
    cow_tile: Optional[Tile],
    *,
    default_face: str = COW_TALK_FACE,
    talk_stand: Optional[Tile] = None,
    talk_face: Optional[str] = None,
) -> str:
    """Choose face direction for a stand next to a cow tile.

    When the stand is not orthogonally adjacent to head or body, returns
    ``talk_face`` if ``stand == talk_stand``, else ``default_face``.
    """
    if cow_tile is None:
        return default_face
    dx = cow_tile[0] - stand[0]
    dy = cow_tile[1] - stand[1]
    body = cow_body_tile(cow_tile)
    body_dx = body[0] - stand[0]
    body_dy = body[1] - stand[1]
    if abs(dx) + abs(dy) != 1 and not (body_dy == 0 and abs(body_dx) == 1):
        if talk_stand is not None and stand == talk_stand and talk_face is not None:
            return talk_face
        return default_face
    if dx > 0:
        return "right"
    if dx < 0:
        return "left"
    if body_dy == 0:
        if body_dx > 0:
            return "right"
        if body_dx < 0:
            return "left"
    if dy > 0:
        return "down"
    return "up"


def cow_interact_pixel(
    cow_px: Pixel,
    face: str,
    *,
    tool: bool,
    cow_tile: Optional[Tile] = None,
) -> Optional[Pixel]:
    """Pixel target for talk/brush/milk approach from a known cow pixel."""
    px, py = cow_px
    if face == "left":
        # Recorded left-aisle clamp (x=163) for left/center stall cows.
        # Right-side cows (tile x >= 12) must keep px+offset or pixel nav
        # aims through the cow toward the wrong aisle.
        target_x = px + COW_INTERACT_X_OFFSET
        if cow_tile is None or cow_tile[0] <= 10:
            target_x = min(target_x, COW_LEFT_INTERACT_X)
        if cow_tile is not None:
            if cow_tile[1] == 17 and tool:
                return target_x, 278
            if cow_tile[1] == 15:
                return target_x, 249
        return target_x, py
    if face == "right":
        return px - COW_INTERACT_X_OFFSET, py + COW_INTERACT_Y_OFFSET
    if face == "up":
        return px, py + COW_INTERACT_X_OFFSET
    if face == "down":
        return px, py - COW_INTERACT_X_OFFSET
    return None


def stand_in_bounds(stand: Tile) -> bool:
    sx, sy = stand
    return 0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH


def stand_blocked(stand: Tile, cow_tiles: Collection[Tile]) -> bool:
    return stand in cow_tiles or stand in COW_BAD_INTERACT_STANDS


def next_unfed_spot(
    flags: int,
    goal: int,
    spots: Sequence[CowFeedSpot] = COW_FEED_SPOTS,
) -> CowFeedSpot:
    """First feed spot within ``goal`` whose trough flag is clear."""
    goal = max(0, goal)
    for spot in spots[:goal]:
        if not (flags & spot.flag):
            return spot
    return spots[max(0, goal - 1)]


def count_fed_trough_flags(
    flags: int,
    goal: int,
    spots: Sequence[CowFeedSpot] = COW_FEED_SPOTS,
) -> int:
    """How many of the first ``goal`` trough flags are set."""
    return min(goal, sum(1 for spot in spots if flags & spot.flag))
