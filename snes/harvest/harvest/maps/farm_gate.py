"""Start-aware farm → west-gate corridors.

``map_routes`` keeps the named hop lists. This module picks which farm
prefix to prepend so viewport BFS never first-hops 40 tiles (Partial spa
was holding RIGHT toward (8,37) from (54,42) and immediately overshot).
"""

from __future__ import annotations

from typing import List, Optional

from harvest.maps.farm_pond import EAST_SPUR_FA_SOUTH_OPEN_X, FARM_TILEMAP_IDS
from harvest.maps.map_types import Waypoint

# Dirt row between the D2 leftover stump belts (y=36–38 and y=40–43).
SOUTH_FIELD_CLEAR_ROW_Y_PX = 39 * 16 + 8
# South-field join column (13, 37) / (216, 600).
SOUTH_FIELD_DIRT_COL_X_PX = 13 * 16 + 8
# After_Rocks / wood-checkpoint join. Authored as (624, 272), not tile-center.
NORTH_EAST_JOIN_PX = (624, 272)


def farm_to_west_gate_waypoints(
    px: int,
    py: int,
    tilemap: Optional[int] = None,
) -> List[Waypoint]:
    """Farm → path crossroads. South crop field uses the dirt-row corridor."""
    from harvest.maps import map_routes as routes

    if tilemap in FARM_TILEMAP_IDS and py >= routes.SOUTH_FIELD_MIN_Y_PX:
        if px >= SOUTH_FIELD_DIRT_COL_X_PX:
            return routes.densify_waypoints(
                _south_east_to_dirt_column(px, py)
                + list(routes._FARM_SOUTH_FIELD_TO_WEST_GATE[2:])
            )
        return list(routes._FARM_SOUTH_FIELD_TO_WEST_GATE)
    if (
        tilemap in FARM_TILEMAP_IDS
        and py < routes.NORTH_FARM_MAX_Y_PX
        and px >= routes.EAST_FARM_MIN_X_PX
    ):
        # Skip house-south (137,375): NE prefix already joins at (136,392).
        return routes.densify_waypoints(
            _north_east_to_house_join(px, py)
            + list(routes._NORTH_EAST_FARM_TO_HOUSE[1:])
            + list(routes._FARM_TO_PATH[1:])
        )
    if tilemap in FARM_TILEMAP_IDS and _on_ditch_north_lip(px, py):
        return routes.densify_waypoints(
            _ditch_lip_to_pinch(px, py) + list(routes._FARM_SOUTH_FIELD_TO_WEST_GATE[5:])
        )
    return list(routes._FARM_TO_PATH)


def _on_y13_south_wall(px: int, py: int) -> bool:
    """True on the FA-east bank where DOWN at x=46–50 slides back to y=13."""
    return py // 16 == 13 and 46 <= px // 16 <= 50


def _north_east_to_house_join(px: int, py: int) -> List[Waypoint]:
    """Onto y=17, then west to (39,17). Wood checkpoint ~(48,13) cannot DOWN
    in place — open south is x=51 (farm_pond EAST_SPUR_FA_SOUTH_OPEN_X).
    """
    join_x, join_y = NORTH_EAST_JOIN_PX
    hops = [Waypoint(tilemap=0x00, target_px=(px, py), radius=16)]
    if _on_y13_south_wall(px, py):
        open_x = EAST_SPUR_FA_SOUTH_OPEN_X * 16 + 8
        hops.append(Waypoint(tilemap=0x00, target_px=(open_x, py), radius=12))
        hops.append(Waypoint(tilemap=0x00, target_px=(open_x, join_y), radius=12))
    else:
        hops.append(Waypoint(tilemap=0x00, target_px=(px, join_y), radius=12))
    hops.append(Waypoint(tilemap=0x00, target_px=(join_x, join_y), radius=12))
    return hops


def _on_ditch_north_lip(px: int, py: int) -> bool:
    """West-pocket stand north of the shipping ditch (11,26–28) no-go."""
    tx, ty = px // 16, py // 16
    return 10 <= tx <= 14 and 24 <= ty <= 26


def _ditch_lip_to_pinch(px: int, py: int) -> List[Waypoint]:
    """Join south-field (12,25)/(8,24). House-south (137,375) cuts the well."""
    return [
        Waypoint(tilemap=0x00, target_px=(px, py), radius=16),
        Waypoint(tilemap=0x00, target_px=(200, 408), radius=8),
    ]


def _south_east_to_dirt_column(px: int, py: int) -> List[Waypoint]:
    """Onto y=39 at current x, then west to x=13. Densify fills 7-tile hops.

    South-field hop 0 is (136,600) run-right. From x>=216 that is already
    an overshoot, so MultiNav skips it and BFS-hugs the first untimed hop.
    """
    return [
        Waypoint(tilemap=0x00, target_px=(px, py), radius=16),
        Waypoint(
            tilemap=0x00,
            target_px=(px, SOUTH_FIELD_CLEAR_ROW_Y_PX),
            radius=12,
        ),
        Waypoint(
            tilemap=0x00,
            target_px=(SOUTH_FIELD_DIRT_COL_X_PX, SOUTH_FIELD_CLEAR_ROW_Y_PX),
            radius=12,
        ),
    ]
