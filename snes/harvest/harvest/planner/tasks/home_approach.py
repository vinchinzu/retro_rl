"""Farm → house-door approach geometry for ReturnHomeTask.

Named free-lane constants and zone predicates so return-home routing is not
inline spaghetti in the FSM. Waypoint densify (south-of-fence / east-of-pond /
west free side / fence gaps) lives here; ReturnHomeTask owns phase/dispatch.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

from harvest.core.tile_catalog import FENCE
from harvest.maps.map_config import (
    FARM_POND_ACCESS_FENCE_ROW,
    FARM_POND_ACCESS_FENCE_X_RANGE,
    Waypoint,
)
from harvest.planner.day_plan_status import (
    FARM_TILEMAP,
    HOUSE_TILEMAP,
    HOUSE_TILEMAPS,
)
from harvest.planner.tasks.transitions import (
    DirectionalTransitionTask,
    HOUSE_ENTER_DOOR_X,
    HOUSE_ENTER_OVERSHOOT_Y,
    HOUSE_ENTER_STAND_TILE,
)
from harvest.tasks.nav import Point, get_tile_at

# y=31 fence wall (x=11–29) blocks northbound return from south field after
# water/CLEAR. East end (tile x≥30 → px≥480) clears the wall; west end is
# x≤10 → px≤160. Never route mid-corridor x≈248 (tile 15) — that is solid fence.
# East free lane must be *east of the pond* (tile x≥36 → px≥576). Using the
# pond column (x=512 / tile 32) makes multi_nav lateral-align through water
# (rr-5in D12 return_home stuck ~(854,527)→(774,521)).
FENCE_ROW_Y = FARM_POND_ACCESS_FENCE_ROW
FENCE_PX_Y = FENCE_ROW_Y * 16  # 496
EAST_AROUND_FENCE_X = 576  # tile x=36, east of pond + past fence wall
EAST_LANE_X_MAX = 640  # cap northbound lane; farther east is shipping scrub
WEST_AROUND_FENCE_X = 96  # tile x=6, west of wall
WEST_FREE_X_MAX = 176  # west of fence wall body (tile x<11)


class ApproachZone(str, Enum):
    """Named geometry bands for farm→door routing."""

    NORTH_OF_DOOR = "north_of_door"
    AT_DOOR = "at_door"
    NEAR_DOOR = "near_door"
    MID_YARD = "mid_yard"
    DEEP_SOUTH = "deep_south"
    SOUTH_OF_FENCE_WEST = "south_of_fence_west"
    SOUTH_OF_FENCE_EAST = "south_of_fence_east"
    SOUTH_OF_FENCE_MID = "south_of_fence_mid"
    FAR_EAST_POND = "far_east_pond"
    OPEN_FIELD = "open_field"


def deep_south_of_house(pos: Point, front: Point) -> bool:
    """True when CLEAR left us far south of the door (viewport-BFS risk)."""
    return pos.y > front.y + 120


def south_of_fence_wall(pos: Point) -> bool:
    """South of the y=31 plant-pocket fence wall (px y≥~504)."""
    return pos.y >= FENCE_PX_Y + 8


def far_east_of_pond_lane(pos: Point) -> bool:
    """East of the free-lane cap — shipping scrub / pond latitude thrash zone."""
    return pos.x > EAST_LANE_X_MAX + 40


def open_fence_gap_tiles(ram: np.ndarray) -> List[int]:
    """x tiles on fence row that are open gaps (require confirmed wall).

    Only trust gaps when at least one solid fence is visible on the row —
    empty/stale unit-test RAM would otherwise treat every tile as a gap and
    route through mid-wall x≈11–15.
    """
    x0, x1 = FARM_POND_ACCESS_FENCE_X_RANGE
    fences: List[int] = []
    gaps: List[int] = []
    for x in range(x0, x1 + 1):
        try:
            tid = int(get_tile_at(ram, x, FENCE_ROW_Y))
        except Exception:
            continue
        if tid == FENCE:
            fences.append(x)
        elif tid not in {0x72, 0x75, 0x76, 0xFF, 0x00}:
            # Non-stale, non-empty open tile — candidate gap.
            gaps.append(x)
        elif tid == 0x00 and fences:
            # Untilled soil after lift counts once wall is confirmed.
            gaps.append(x)
    if not fences:
        return []
    return gaps


def classify_approach_zone(
    pos: Point,
    front: Point,
    *,
    gaps: Optional[Sequence[int]] = None,
) -> ApproachZone:
    """Classify player px relative to door stand and fence free lanes."""
    dx = abs(pos.x - front.x)
    dy = abs(pos.y - front.y)
    if dx <= 16 and dy <= 16:
        return ApproachZone.AT_DOOR
    if abs(pos.x - front.x) <= 24 and pos.y < front.y - 10:
        return ApproachZone.NORTH_OF_DOOR
    if dx <= 28 and dy <= 28:
        return ApproachZone.NEAR_DOOR
    if south_of_fence_wall(pos):
        if far_east_of_pond_lane(pos):
            return ApproachZone.FAR_EAST_POND
        if pos.x < WEST_FREE_X_MAX:
            return ApproachZone.SOUTH_OF_FENCE_WEST
        if pos.x >= EAST_AROUND_FENCE_X - 16:
            return ApproachZone.SOUTH_OF_FENCE_EAST
        if gaps:
            return ApproachZone.SOUTH_OF_FENCE_MID
        return ApproachZone.SOUTH_OF_FENCE_MID
    if deep_south_of_house(pos, front):
        return ApproachZone.DEEP_SOUTH
    if pos.y > front.y + 24:
        return ApproachZone.MID_YARD
    return ApproachZone.OPEN_FIELD


def build_house_approach_waypoints(
    base: List[Waypoint],
    front: Point,
    pos: Point,
    ram: np.ndarray,
) -> List[Waypoint]:
    """Route farm→door, densifying when south of house / fence wall.

    Direct multi_nav to (136,424) from south-of-fence (y≥496) has no path
    through the solid y=31 wall (x=11–29). Prefer:
    1) open gap on the fence row if water/CLEAR already cut one, else
    2) east around the wall (x≥480 / tile 30+), then north, then to door.
    Never densify mid-wall x≈248 (tile 15) — that is the fence body.
    """
    if not base:
        base = [Waypoint(tilemap=0x00, target_px=(front.x, front.y), radius=12)]

    if not deep_south_of_house(pos, front) and not south_of_fence_wall(pos):
        return list(base)

    stages: List[Waypoint] = []
    south_of_wall = south_of_fence_wall(pos)
    gaps = open_fence_gap_tiles(ram) if south_of_wall else []
    corridor_x = front.x

    if south_of_wall and pos.x < WEST_FREE_X_MAX:
        # West of the fence wall (tile x<11): run north on the free side.
        # Prefer current x when already past the west corridor so we do not
        # pull left into the SW rock pocket (D12 residual ~(122,518)).
        corridor_x = min(max(pos.x, WEST_AROUND_FENCE_X), 160)
        for y in (600, 520, 440):
            if pos.y > y + 24:
                stages.append(
                    Waypoint(
                        tilemap=0x00,
                        target_px=(corridor_x, y),
                        radius=22,
                        run_direction="up",
                    )
                )
    elif south_of_wall and gaps and pos.x < EAST_AROUND_FENCE_X + 80:
        # Nearest open gap — approach from south then push north through.
        # Skip gap routing when already far east of the pond (prefer free
        # east lane; gap at x≈14 would force a long west crawl through water).
        gap_x = min(gaps, key=lambda x: abs(x * 16 + 8 - pos.x))
        gap_px = gap_x * 16 + 8
        if abs(pos.x - gap_px) > 20:
            stages.append(
                Waypoint(
                    tilemap=0x00,
                    target_px=(gap_px, min(pos.y, FENCE_PX_Y + 48)),
                    radius=20,
                    run_direction="right" if pos.x < gap_px else "left",
                )
            )
        stages.append(
            Waypoint(
                tilemap=0x00,
                target_px=(gap_px, FENCE_PX_Y + 24),
                radius=16,
                run_direction="up",
            )
        )
        stages.append(
            Waypoint(
                tilemap=0x00,
                target_px=(gap_px, FENCE_PX_Y - 24),
                radius=16,
                run_direction="up",
            )
        )
        corridor_x = gap_px
    elif south_of_wall:
        # Past fence wall on the east free lane (east of pond), then north.
        # If already east of the lane, north first at a capped east x — do
        # not lateral-align onto the pond column while still at pond y.
        if pos.x >= EAST_AROUND_FENCE_X - 16:
            corridor_x = min(max(pos.x, EAST_AROUND_FENCE_X), EAST_LANE_X_MAX)
        else:
            corridor_x = EAST_AROUND_FENCE_X
            stages.append(
                Waypoint(
                    tilemap=0x00,
                    target_px=(corridor_x, min(pos.y, 720)),
                    radius=24,
                    run_direction="right",
                )
            )
        for y in (600, 520, 440):
            if pos.y > y + 24:
                stages.append(
                    Waypoint(
                        tilemap=0x00,
                        target_px=(corridor_x, y),
                        radius=22,
                        run_direction="up",
                    )
                )
        # After clearing fence latitude at east x, slide west above the
        # wall (y≈440) before the final door approach — avoids pond y.
        if corridor_x > front.x + 40:
            stages.append(
                Waypoint(
                    tilemap=0x00,
                    target_px=(min(corridor_x, front.x + 80), 440),
                    radius=20,
                    run_direction="left",
                )
            )
    else:
        # North of fence but still deep south of door — mid-field open.
        # Far-east (post-water/CLEAR): slide west first above the wall.
        if pos.x > EAST_AROUND_FENCE_X + 40:
            corridor_x = min(pos.x, EAST_LANE_X_MAX)
            stages.append(
                Waypoint(
                    tilemap=0x00,
                    target_px=(EAST_AROUND_FENCE_X, min(pos.y, 460)),
                    radius=22,
                    run_direction="left",
                )
            )
            corridor_x = EAST_AROUND_FENCE_X
        else:
            corridor_x = max(pos.x, front.x)
            corridor_x = min(360, max(160, corridor_x))
        for y in (520, 460):
            if pos.y > y + 32:
                stages.append(
                    Waypoint(
                        tilemap=0x00,
                        target_px=(corridor_x, y),
                        radius=18,
                        run_direction="up",
                    )
                )

    # From north-of-fence latitude, slide to door x then stand.
    approach_y = max(front.y + 40, FENCE_PX_Y - 40)
    if abs(corridor_x - front.x) > 20 or stages:
        stages.append(
            Waypoint(
                tilemap=0x00,
                target_px=(front.x, approach_y),
                radius=16,
            )
        )
    stages.append(
        Waypoint(tilemap=0x00, target_px=(front.x, front.y), radius=12)
    )
    return stages


def drop_spot_px(front: Point, *, deep: bool = False) -> Point:
    """Open ground south of the house door — not mid-field debris."""
    if deep:
        return Point(front.x, min(560, front.y + 112))
    return Point(front.x, min(520, front.y + 56))


def house_enter_task(front: Point) -> DirectionalTransitionTask:
    """Build the outdoor→house doorway push from a door-front stand."""
    if front.y <= 360:
        stand_tile = (front.x // 16, front.y // 16)
        overshoot_y = min(HOUSE_ENTER_OVERSHOOT_Y, front.y - 12)
    else:
        stand_tile = HOUSE_ENTER_STAND_TILE
        overshoot_y = HOUSE_ENTER_OVERSHOOT_Y
    return DirectionalTransitionTask(
        name="enter_house",
        direction="up",
        origin_tilemap=FARM_TILEMAP,
        target_tilemap=HOUSE_TILEMAP,
        target_tilemaps=tuple(sorted(HOUSE_TILEMAPS)),
        timeout=2500,
        min_frames_before_success=15,
        settle_frames=20,
        stand_tile=stand_tile,
        stand_tolerance=0,
        door_align_px=front.x if front.x else HOUSE_ENTER_DOOR_X,
        overshoot_limit_px=overshoot_y,
        require_empty_hands=True,
        clear_hands_limit=6,
    )


__all__ = [
    "ApproachZone",
    "EAST_AROUND_FENCE_X",
    "EAST_LANE_X_MAX",
    "FENCE_PX_Y",
    "FENCE_ROW_Y",
    "WEST_AROUND_FENCE_X",
    "WEST_FREE_X_MAX",
    "build_house_approach_waypoints",
    "classify_approach_zone",
    "deep_south_of_house",
    "drop_spot_px",
    "far_east_of_pond_lane",
    "house_enter_task",
    "open_fence_gap_tiles",
    "south_of_fence_wall",
]
