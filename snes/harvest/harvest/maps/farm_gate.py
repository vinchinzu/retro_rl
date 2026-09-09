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
# House-column dirt row. y=25 x=9-10 is A6; LEFT from (11,25) hugs the pond.
HOUSE_COLUMN_DIRT_Y_PX = 24 * 16 + 8
# After_Rocks / wood-checkpoint join. Authored as (624, 272), not tile-center.
NORTH_EAST_JOIN_PX = (624, 272)
# Above y=9, x=39 descends through the standard stump at (38,9). Move one
# 2x2 footprint west before joining y=17.
UPPER_NORTH_EAST_MAX_Y_PX = 9 * 16
UPPER_NORTH_EAST_LANE_X_PX = 37 * 16
# Shipping bin tile y=28. y>=27 is south of the house pinch but still north
# of the y=31 fence — not the berry y=60 south-field route.
HOUSE_SOUTH_MIN_Y_PX = 27 * 16
HOUSE_PINCH_PX = (136, 424)
# NW boulder stands can land inside the house-paddock collision rectangle.
# Walking cannot cross its y=17 edge, while a B-run west onto x=4 and then
# south follows the open outer lane to the west-gate row.
NORTH_WEST_MAX_X_PX = 11 * 16
NORTH_WEST_MAX_Y_PX = 20 * 16
NORTH_WEST_OUTER_LANE_X_PX = 4 * 16 + 8


def farm_wp(px: int, py: int, radius: int = 12, **kwargs) -> Waypoint:
    """Farm-tilemap hop. Prefixes share this so radius/run kwargs stay local."""
    return Waypoint(tilemap=0x00, target_px=(px, py), radius=radius, **kwargs)


def farm_to_west_gate_waypoints(
    px: int,
    py: int,
    tilemap: Optional[int] = None,
) -> List[Waypoint]:
    """Farm → path crossroads. South crop field uses the dirt-row corridor."""
    from harvest.maps import map_routes as routes

    if tilemap not in FARM_TILEMAP_IDS:
        return list(routes._FARM_TO_PATH)
    if py >= routes.SOUTH_FIELD_MIN_Y_PX:
        if px >= SOUTH_FIELD_DIRT_COL_X_PX:
            return routes.densify_waypoints(
                _south_east_to_dirt_column(px, py)
                + list(routes._FARM_SOUTH_FIELD_TO_WEST_GATE[2:])
            )
        return list(routes._FARM_SOUTH_FIELD_TO_WEST_GATE)
    if py < routes.NORTH_FARM_MAX_Y_PX and px >= routes.EAST_FARM_MIN_X_PX:
        # Skip house-south (137,375): NE prefix already joins at (136,392).
        return routes.densify_waypoints(
            _north_east_to_house_join(px, py)
            + list(routes._NORTH_EAST_FARM_TO_HOUSE[1:])
            + list(routes._FARM_TO_PATH[1:])
        )
    if px < NORTH_WEST_MAX_X_PX and py < NORTH_WEST_MAX_Y_PX:
        return routes.densify_waypoints(
            _north_west_to_gate_row(px, py) + [routes._PATH_PLAZA_FROM_FARM]
        )
    if _on_ditch_north_lip(px, py):
        return routes.densify_waypoints(
            _ditch_lip_to_pinch(px, py) + list(routes._FARM_SOUTH_FIELD_TO_WEST_GATE[5:])
        )
    if HOUSE_SOUTH_MIN_Y_PX <= py < routes.SOUTH_FIELD_MIN_Y_PX:
        # Bin / house-south: join the y=26 pinch. Do not rewind to house
        # (8,23) and do not first-hop berry (55,60) north of the y=31 fence.
        return routes.densify_waypoints(
            _house_south_to_pinch(px, py)
            + list(routes._FARM_GATE_PINCH_TO_EXIT[1:])
            + [routes._PATH_PLAZA_FROM_FARM]
        )
    return list(routes._FARM_TO_PATH)


def farm_exit_waypoints(
    px: int,
    py: int,
    tilemap: Optional[int] = None,
) -> List[Waypoint]:
    """Farm → west-gate stand only. NAV_FARM_EXIT must not walk path 0x0C."""
    from harvest.maps import map_routes as routes

    if tilemap in FARM_TILEMAP_IDS and py >= routes.SOUTH_FIELD_MIN_Y_PX:
        return list(routes.ROUTES["farm_south_to_west_gate"])
    route = farm_to_west_gate_waypoints(px, py, tilemap)
    out: List[Waypoint] = []
    for wp in route:
        if wp.tilemap != 0x00:
            break
        out.append(wp)
        if wp.is_exit:
            break
    return out or list(routes._FARM_GATE_PINCH_TO_EXIT)


def _on_y13_south_wall(px: int, py: int) -> bool:
    """True on the FA-east bank where DOWN at x=46–50 slides back to y=13."""
    return py // 16 == 13 and 46 <= px // 16 <= 50


def _north_east_to_house_join(px: int, py: int) -> List[Waypoint]:
    """Onto y=17, then west to (39,17). Wood checkpoint ~(48,13) cannot DOWN
    in place — open south is x=51 (farm_pond EAST_SPUR_FA_SOUTH_OPEN_X).
    """
    join_x, join_y = NORTH_EAST_JOIN_PX
    hops = [farm_wp(px, py, 16)]
    if py < UPPER_NORTH_EAST_MAX_Y_PX:
        # Live rock refill (38,8) sits on the stump at x=38-39/y=9-10.
        join_x = UPPER_NORTH_EAST_LANE_X_PX
        if abs(px - join_x) > 8:
            hops.append(farm_wp(join_x, py))
    elif _on_y13_south_wall(px, py):
        open_x = EAST_SPUR_FA_SOUTH_OPEN_X * 16 + 8
        hops.append(farm_wp(open_x, py))
        hops.append(farm_wp(open_x, join_y))
    elif abs(px - join_x) > 8:
        # Leftover 2x2s sit on due-south from NE stands. Live spa pin
        # (41,11) has a boulder at (40-41,12-13); sidestep onto the join
        # column first, then south on x=39 dirt.
        hops.append(farm_wp(join_x, py))
    hops.append(farm_wp(join_x, join_y))
    return hops


def _north_west_to_gate_row(px: int, py: int) -> List[Waypoint]:
    """B-run outside the house paddock, then south onto the west-gate row."""
    from harvest.maps import map_routes as routes

    lane_x = min(px, NORTH_WEST_OUTER_LANE_X_PX)
    hops = [farm_wp(px, py, 16)]
    if px - lane_x > 8:
        hops.append(farm_wp(lane_x, py, 8, run_direction="left", force_run=True))
    hops.append(
        farm_wp(
            lane_x,
            routes._FARM_WEST_EXIT.target_px[1],
            8,
            run_direction="down",
            force_run=True,
        )
    )
    hops.append(routes._FARM_WEST_EXIT)
    return hops


def _house_south_to_pinch(px: int, py: int) -> List[Waypoint]:
    """Shipping-bin / house-south onto the y=26 west-gate pinch.

    West of the F2 ditch (x<=9): up onto y=26 at the current x, then the
    pinch column. Sidestep on y=27 then ``run_left`` to (72,424) treats
    dy==radius as aligned and charges into (1,27)=FF (power-on (7,27)).
    East of the ditch: leftover 2x2s still sit on the bin row (live stump
    (18,28) west of (29,28)). LEFT on y=28 walks through them. (12,26)
    LEFT crosses A6. North onto y=24 dirt, then west to house-column A0.
    """
    pinch_x, pinch_y = HOUSE_PINCH_PX
    hops = [farm_wp(px, py, 16)]
    if px // 16 <= 9:
        if abs(py - pinch_y) >= 8:
            hops.append(farm_wp(px, pinch_y, 6))
        if hops[-1].target_px != (pinch_x, pinch_y):
            hops.append(farm_wp(pinch_x, pinch_y))
        return hops
    dirt_y = HOUSE_COLUMN_DIRT_Y_PX
    if abs(py - dirt_y) > 8:
        hops.append(farm_wp(px, dirt_y))
    if abs(px - pinch_x) > 8:
        hops.append(farm_wp(pinch_x, dirt_y))
    hops.append(farm_wp(pinch_x, pinch_y))
    return hops


def _on_ditch_north_lip(px: int, py: int) -> bool:
    """West-pocket stand north of the shipping ditch (11,26–28) no-go.

    Live stump spa (13,23) is one row above y=24 dirt. Default _FARM_TO_PATH
    first-hops house (8,23) and 10k-hugs. Join the dirt row instead.
    """
    tx, ty = px // 16, py // 16
    return 10 <= tx <= 14 and 23 <= ty <= 26


def _ditch_lip_to_pinch(px: int, py: int) -> List[Waypoint]:
    """Join house-column A0, then south-field pinch. Not (12,25) LEFT.

    y=23 x=9-21 is A8. South-then-west on y=24 still push-faces that wall
    (live spa (13,23) → (11,24)). North onto y=22 path, then west to x=8.
    Stands already south of the wall keep the y=24 dirt join.
    """
    hops = [farm_wp(px, py, 16)]
    if py // 16 <= 23:
        north_y = 22 * 16 + 8
        hops.append(farm_wp(px, north_y, 8))
        hops.append(farm_wp(HOUSE_PINCH_PX[0], north_y, 8))
        return hops
    hops.append(farm_wp(px, HOUSE_COLUMN_DIRT_Y_PX, 8))
    return hops


def _south_east_to_dirt_column(px: int, py: int) -> List[Waypoint]:
    """Onto y=39 at current x, then west to x=13. Densify fills 7-tile hops.

    South-field hop 0 is (136,600) run-right. From x>=216 that is already
    an overshoot, so MultiNav skips it and BFS-hugs the first untimed hop.
    """
    return [
        farm_wp(px, py, 16),
        farm_wp(px, SOUTH_FIELD_CLEAR_ROW_Y_PX),
        farm_wp(SOUTH_FIELD_DIRT_COL_X_PX, SOUTH_FIELD_CLEAR_ROW_Y_PX),
    ]
