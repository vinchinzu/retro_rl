"""Named multi-map waypoint routes and route helpers.

Coordinate lists are hand-authored from recordings; do not "simplify" hops.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from harvest.core.tile_catalog import PATH_TILEMAP_IDS, TILE_SIZE
from harvest.maps.farm_gate import farm_to_west_gate_waypoints
from harvest.maps.farm_pond import FARM_TILEMAP_IDS
from harvest.maps.map_types import Waypoint

# Path 0x0C is a ~16x10-tile fork. Farm (y~422) and mountain (y~740)
# pixels leak across the transition; Manhattan then prefers a later
# exit over the plaza center.
PATH_TILEMAP_ID = 0x0C
MOUNTAIN_TILEMAP_ID = 0x10
PATH_ONMAP_MAX_X = 280
PATH_ONMAP_MAX_Y = 200


def path_coords_leaked(px: int, py: int) -> bool:
    """True when (px, py) is not a real path-0x0C stand."""
    return px < 0 or py < 0 or px > PATH_ONMAP_MAX_X or py > PATH_ONMAP_MAX_Y


# Path farm-gate stands sit near (232–244, 128). Those pixels are also a
# valid north-farm stand (shed row). After path→farm the tilemap flips
# first and BFS treats (244,118) as house-north — wall-hug to the bin.
_PATH_FARM_GATE_X = (180, 280)
_PATH_FARM_GATE_Y = (80, 160)


def farm_coords_look_like_path(px: int, py: int) -> bool:
    """True when farm RAM still holds the path farm-gate pixel."""
    return _PATH_FARM_GATE_X[0] <= px <= _PATH_FARM_GATE_X[1] and (
        _PATH_FARM_GATE_Y[0] <= py <= _PATH_FARM_GATE_Y[1]
    )


def slice_route_from_position(
    waypoints: List[Waypoint],
    px: int,
    py: int,
    *,
    tilemap: Optional[int] = None,
) -> List[Waypoint]:
    """Return a suffix of ``waypoints`` starting at the nearest relevant hop.

    MultiMapNav always begins at index 0. Mid-route stands must not rewind to
    the first hop. Leaked 0x0C coords start at the first path hop; mountain
    north-edge y<80 is handled by callers (Gotz).
    """
    if not waypoints:
        return []
    if tilemap in PATH_TILEMAP_IDS and path_coords_leaked(px, py):
        # Farm-gate (232,128) is on-map but east toward the farm. A leaked
        # west-edge landing (10,422) cannot BFS there; start at plaza.
        for i, wp in enumerate(waypoints):
            if wp.tilemap not in PATH_TILEMAP_IDS:
                continue
            tx, ty = wp.target_px
            if path_coords_leaked(tx, ty) or farm_coords_look_like_path(tx, ty):
                continue
            return list(waypoints[i:])
        return list(waypoints)
    if tilemap in FARM_TILEMAP_IDS and farm_coords_look_like_path(px, py):
        for i, wp in enumerate(waypoints):
            if wp.tilemap in FARM_TILEMAP_IDS and not farm_coords_look_like_path(
                wp.target_px[0], wp.target_px[1]
            ):
                return list(waypoints[i:])
        return list(waypoints)
    best_i = 0
    best_d = None
    for i, wp in enumerate(waypoints):
        if tilemap is not None and wp.tilemap != tilemap:
            continue
        d = abs(wp.target_px[0] - px) + abs(wp.target_px[1] - py)
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    for i in range(len(waypoints) - 1, best_i - 1, -1):
        wp = waypoints[i]
        if tilemap is not None and wp.tilemap != tilemap:
            continue
        if (
            abs(wp.target_px[0] - px) <= wp.radius
            and abs(wp.target_px[1] - py) <= wp.radius
        ):
            return list(waypoints[i:])
    start = max(0, best_i - 1)
    return list(waypoints[start:])


def mountain_downhill_escape(
    px: int,
    py: int,
    *,
    tilemap: Optional[int] = None,
    hops: Optional[Sequence[Waypoint]] = None,
) -> List[Waypoint]:
    """Mountain hops strictly south of the live pose, toward the south exit.

    The grape return reverses through the carpenter terrace at (520, 632).
    A pin at ~(474, 630) (run6 D15, 10 entities) cannot BFS that hop.
    South means higher y: skip the terrace and drop to (520, 712).
    From the grape stand (y=409) this still includes the x=20 cliff drop.
    """
    hops_list = list(hops) if hops is not None else list(_FIRST_BERRY_TO_MOUNTAIN_EXIT)
    south = [wp for wp in hops_list if wp.target_px[1] > py + 16]
    if south:
        return list(south)
    return slice_route_from_position(
        hops_list, px, py, tilemap=MOUNTAIN_TILEMAP_ID if tilemap is None else tilemap
    )


# Row 7 on path 0x0C. (7,7) and (8,7) are 0xFF; a descent aimed at the
# plaza (132, 128) walks into them (run13 return_to_bin pin, 6x).
_PATH_SEAL_Y_PX = 7 * TILE_SIZE


def path_stand_is_mountain_gate(
    px: int,
    py: int,
    tilemap: Optional[int] = None,
) -> bool:
    """True when a path stand is north of the row-7 seal (or a mountain leak).

    Farm-west leak ~(10, 422) stays on the plaza road. Mountain pixels leak
    past PATH_ONMAP_MAX_* and must not start at the plaza.
    """
    if tilemap is not None and tilemap not in PATH_TILEMAP_IDS and tilemap != MOUNTAIN_TILEMAP_ID:
        return False
    if path_coords_leaked(px, py):
        return px > PATH_ONMAP_MAX_X or px > 80
    return py < _PATH_SEAL_Y_PX


def path_return_to_farm(
    px: int,
    py: int,
    *,
    tilemap: Optional[int] = None,
) -> List[Waypoint]:
    """0x0C hops to the farm gate from the live stand.

    Mountain-gate stands use the inbound climb lane. Plaza/town stands stay
    on the east road. Callers must not concatenate ``path_to_farm`` after a
    mountain 0x10 exit — that list aims at the sealed column.
    """
    hops = (
        _PATH_MOUNTAIN_GATE_TO_FARM
        if path_stand_is_mountain_gate(px, py, tilemap)
        else _PATH_PLAZA_TO_FARM
    )
    return slice_route_from_position(
        list(hops),
        px,
        py,
        tilemap=PATH_TILEMAP_ID if tilemap is None else tilemap,
    )


def mountain_exit_then_farm(mountain_hops: Sequence[Waypoint]) -> List[Waypoint]:
    """Append the mountain-gate descent, never the plaza path."""
    return list(mountain_hops) + list(_PATH_MOUNTAIN_GATE_TO_FARM)


def farm_to_spa_waypoints(
    px: int,
    py: int,
    tilemap: Optional[int] = None,
) -> List[Waypoint]:
    """Farm/path/mountain hops to the outdoor spa lip, start-aware on farm."""
    return (
        farm_to_west_gate_waypoints(px, py, tilemap)
        + list(_PATH_TO_MOUNTAIN)
        + list(_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA)
    )


def densify_waypoints(
    waypoints: Sequence[Waypoint],
    *,
    max_hop_tiles: int = 7,
    tile_size: int = TILE_SIZE,
) -> List[Waypoint]:
    """Insert intermediate hops so same-map targets stay within viewport BFS.

    SNES only loads ~16x14 tiles around the player. BFS beyond ~7–10 tiles
    sees stale tile IDs. Map transitions and exit/action waypoints are kept.
    """
    if max_hop_tiles < 1:
        raise ValueError("max_hop_tiles must be >= 1")
    if not waypoints:
        return []

    max_hop_px = max_hop_tiles * tile_size
    result: List[Waypoint] = [waypoints[0]]
    for nxt in waypoints[1:]:
        prev = result[-1]
        if (
            prev.tilemap != nxt.tilemap
            or prev.is_exit
            or nxt.is_exit
            or prev.action_on_arrive
            or nxt.action_on_arrive
            or prev.run_direction
            or nxt.run_direction
        ):
            result.append(nxt)
            continue

        dx = nxt.target_px[0] - prev.target_px[0]
        dy = nxt.target_px[1] - prev.target_px[1]
        dist = max(abs(dx), abs(dy))
        if dist <= max_hop_px:
            result.append(nxt)
            continue

        steps = int((dist + max_hop_px - 1) // max_hop_px)
        for i in range(1, steps):
            t = i / steps
            ix = int(round(prev.target_px[0] + dx * t))
            iy = int(round(prev.target_px[1] + dy * t))
            result.append(
                Waypoint(
                    tilemap=prev.tilemap,
                    target_px=(ix, iy),
                    radius=min(prev.radius, nxt.radius),
                )
            )
        result.append(nxt)
    return result


# West-mid y=360 → spa lip. Join after grape inbound's (312, 360); do not
# walk south onto the grape stand. Avoid D0 hut at tile(36,13): y=201 is
# tile 12 (y=192–207); radius 6 stays off hut row 13 (y=208).
_WEST_MID_TO_OUTDOOR_SPA: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(348, 361), radius=12, run_direction="right"),
    Waypoint(tilemap=0x10, target_px=(430, 345), radius=12),
    Waypoint(tilemap=0x10, target_px=(433, 255), radius=12),
    Waypoint(tilemap=0x10, target_px=(529, 246), radius=12),
    Waypoint(tilemap=0x10, target_px=(560, 214), radius=8),
    Waypoint(tilemap=0x10, target_px=(560, 201), radius=6),
    Waypoint(tilemap=0x10, target_px=(619, 201), radius=16, run_direction="right"),
]

# Fish/camp stand only. Farm→spa uses grape dirt — never the east Gotz/fish
# pocket or camp boulder (38,28). Tight climb radius: a loose (70,377)
# "arrives" at y~390 still below the ridge. No down-run from (640,428)
# (pins boulder/log ~(654,441)); west on dirt y~478, not grass ledge y=454.
_FISH_TO_OUTDOOR_SPA: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(640, 428), radius=14),
    Waypoint(tilemap=0x10, target_px=(650, 454), radius=10),
    Waypoint(tilemap=0x10, target_px=(650, 478), radius=10),
    Waypoint(tilemap=0x10, target_px=(560, 478), radius=10),
    Waypoint(tilemap=0x10, target_px=(480, 478), radius=10),
    Waypoint(tilemap=0x10, target_px=(380, 470), radius=10),
    Waypoint(tilemap=0x10, target_px=(280, 470), radius=10),
    Waypoint(tilemap=0x10, target_px=(185, 456), radius=12),
    Waypoint(tilemap=0x10, target_px=(103, 442), radius=12),
    Waypoint(tilemap=0x10, target_px=(70, 413), radius=10),
    Waypoint(tilemap=0x10, target_px=(70, 361), radius=10),
    Waypoint(tilemap=0x10, target_px=(148, 361), radius=12, run_direction="right"),
    Waypoint(tilemap=0x10, target_px=(248, 361), radius=12, run_direction="right"),
] + list(_WEST_MID_TO_OUTDOOR_SPA)

# Path 0x0C is the farm / town / mountain fork. Town and mountain share
# farm→crossroads instead of forking the farm exit.
# Live Wood_Progress dump + spa probe: MapExit lists x=0 y=24–28, but the
# trigger is 0xC0 at x=0 y=26–28 (not in FARM_WALKABLE). Stand ~px(11,424)
# tile (0,26). (0,24)=0xA1 is the gate wall — LEFT+B pins at x=22, no map
# change. (1,27)=0xFF, so LEFT on y=27 never reaches C0. Open face is
# (1,26)=0xA8. Do not BFS onto C0; is_exit walk left into it.
_FARM_WEST_EXIT = Waypoint(
    tilemap=0x00, target_px=(40, 424), radius=6, is_exit=True, exit_direction="left"
)
_PATH_FARM_GATE = Waypoint(tilemap=0x0C, target_px=(232, 128), radius=16)
_PATH_CROSSROADS = Waypoint(tilemap=0x0C, target_px=(132, 128), radius=16)
# Exit then plaza. PATH_FARM_GATE (232,128) pins leaked (10,422) landings.
# force_run left: BFS from the east landing walks back onto the farm.
_PATH_PLAZA_FROM_FARM = Waypoint(
    tilemap=0x0C,
    target_px=(132, 128),
    radius=16,
    run_direction="left",
    force_run=True,
)
_PATH_TOWN_EXIT = Waypoint(
    tilemap=0x0C, target_px=(10, 128), radius=8, is_exit=True, exit_direction="left"
)
_PATH_MOUNTAIN_EXIT = Waypoint(
    tilemap=0x0C, target_px=(132, 30), radius=10, is_exit=True, exit_direction="up"
)
# Live return crosses the right boundary around y≈102 before y=128.
_PATH_FARM_EXIT = Waypoint(
    tilemap=0x0C,
    target_px=(244, 128),
    radius=32,
    is_exit=True,
    exit_direction="right",
    exit_push_frames=18,
)
# Pond A6 is y=25 x=0-6. House column x=8 is A0 y=24–25 then A8 y=26 (door).
# Down that column onto y=26, then west on A8. Do not LEFT-hold on y=24
# (A1 gate), y=25 (pond), or y=27 (FF at x=1).
_FARM_GATE_PINCH_TO_EXIT: List[Waypoint] = [
    Waypoint(
        tilemap=0x00,
        target_px=(136, 424),
        radius=6,
        run_direction="down",
        force_run=True,
    ),
    Waypoint(tilemap=0x00, target_px=(72, 424), radius=8, run_direction="left"),
    _FARM_WEST_EXIT,
]
_FARM_WEST_TO_TOWN: List[Waypoint] = [
    *_FARM_GATE_PINCH_TO_EXIT,
    _PATH_PLAZA_FROM_FARM,
    _PATH_TOWN_EXIT,
]

# L1 house front ~(136,344) BFS-cuts the NW ledge. Drop south first onto
# A0 above the pond, then the house-column pinch down to y=26.
_FARM_TO_PATH: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(137, 375), radius=12),
    Waypoint(tilemap=0x00, target_px=(136, 392), radius=8),  # (8,24) A0
    *_FARM_GATE_PINCH_TO_EXIT,
    _PATH_PLAZA_FROM_FARM,
]

# Sunday south crop ~(78,598). Viewport BFS to (136,424) cuts 0x5E beds and
# F2 ditch (9/11,26–28). Stay on untilled dirt, x=13 through y=31 fence,
# then UP to y=24 and west on dirt onto house-column A0. y=25 x=9-10 is A6.
SOUTH_FIELD_MIN_Y_PX = 520
# After_Rocks ~(633,223). House-first hop (137,375) is a 31-tile BFS; barn
# A1 push-faces (30,19-21). Drop south east of the barn onto y=24 dirt.
NORTH_FARM_MAX_Y_PX = 320
EAST_FARM_MIN_X_PX = 480
_NORTH_EAST_FARM_TO_HOUSE: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(624, 272), radius=12),  # (39,17)
    Waypoint(tilemap=0x00, target_px=(624, 384), radius=12),  # (39,24)
    Waypoint(tilemap=0x00, target_px=(512, 384), radius=12),  # (32,24)
    Waypoint(tilemap=0x00, target_px=(384, 384), radius=12),  # (24,24)
    Waypoint(tilemap=0x00, target_px=(256, 384), radius=12),  # (16,24)
]
_FARM_SOUTH_FIELD_TO_WEST_GATE: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(136, 600), radius=8, run_direction="right"),
    # Radius 6 keeps arrival on tile 13. Radius 12 arrived on tile 12
    # and the up-run charged the 0x5E crop at (12,36).
    Waypoint(tilemap=0x00, target_px=(216, 600), radius=6, run_direction="right"),
    Waypoint(tilemap=0x00, target_px=(216, 536), radius=6),  # (13,33)
    Waypoint(tilemap=0x00, target_px=(216, 440), radius=6),  # (13,27)
    Waypoint(tilemap=0x00, target_px=(216, 392), radius=8),  # (13,24) dirt; not y=25 LEFT
    Waypoint(tilemap=0x00, target_px=(136, 392), radius=8),  # (8,24) A0 house column
    *_FARM_GATE_PINCH_TO_EXIT,
    # Crossroads next — not PATH_FARM_GATE (232,128), which is east
    # toward the farm and pins leaked (10,422) landings.
    _PATH_PLAZA_FROM_FARM,
]
# South-west field (the west-strip rings) to the west gate by the steps at
# (3-4,30-31), tile 0xE0. The x=13 climb above runs through the (13,28) ring,
# and a grown ring stopped 18 of 44 spa trips at (223,425) (v2_d3). The hops
# stay on the west strip's open lanes: x=6-7, y=37-38 and y=33.
_FARM_SOUTH_WEST_TO_WEST_GATE: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(104, 600), radius=8),  # (6,37) lane crossing
    Waypoint(tilemap=0x00, target_px=(104, 536), radius=8),  # (6,33)
    Waypoint(tilemap=0x00, target_px=(72, 488), radius=12),  # (4,30) steps
    # Up x=4 onto the centre of the exit row y=26 before the left walk: arriving
    # at y=430, the bottom of the row, the walk caught the 0xFF corner at (1,27)
    # for its whole 500 frames (v5b64 Summer D2, LN_spa_whatif_exit_walk_FAIL).
    Waypoint(tilemap=0x00, target_px=(72, 424), radius=3),  # (4,26)
    _FARM_WEST_EXIT,
    _PATH_PLAZA_FROM_FARM,
]
_PATH_TO_TOWN: List[Waypoint] = [_PATH_TOWN_EXIT]
# Already on 0x0C: plaza center first, then the north exit. Do not
# BFS-diagonal from the east landing / leaked farm y toward (132,30).
_PATH_TO_MOUNTAIN: List[Waypoint] = [_PATH_PLAZA_FROM_FARM, _PATH_MOUNTAIN_EXIT]
# Seed-shop town gate: stand on the open face, not (0,8)/(1,8) doorframe hugs.
_PATH_TO_TOWN_SHOP: List[Waypoint] = [
    _PATH_PLAZA_FROM_FARM,
    Waypoint(tilemap=0x0C, target_px=(40, 128), radius=8, is_exit=True, exit_direction="left"),
]
# shop_door landmark tile (37,13). buy_potato_seeds_d2: east road sealed
# west of ~x=684. Open face just inside the gate, north at x=684, west on
# y=288, then shop column. Tape stand south of shop (37,17) — (601,246) pins.
_TOWN_TO_SHOP_DOOR: List[Waypoint] = [
    Waypoint(tilemap=0x04, target_px=(728, 424), radius=18),
    Waypoint(tilemap=0x04, target_px=(692, 424), radius=14, run_direction="left", force_run=True),
    Waypoint(tilemap=0x04, target_px=(684, 400), radius=14, run_direction="up", force_run=True),
    Waypoint(tilemap=0x04, target_px=(684, 340), radius=14, run_direction="up", force_run=True),
    Waypoint(tilemap=0x04, target_px=(684, 288), radius=10),
    Waypoint(tilemap=0x04, target_px=(640, 288), radius=12),
    Waypoint(tilemap=0x04, target_px=(608, 288), radius=10),
    Waypoint(tilemap=0x04, target_px=(602, 274), radius=10, is_exit=True, exit_direction="up"),
]
# buy_potato_seeds_d2 clerk A-stand (182,342) tile (11,21) face up.
_SEED_CLERK_PX = (182, 342)
_SHOP_TO_COUNTER: List[Waypoint] = [
    Waypoint(tilemap=0x1C, target_px=_SEED_CLERK_PX, radius=4),
]
# Shop exit open face is (8,28); (8,29) is the doorframe stasis tile.
_SHOP_TO_TOWN: List[Waypoint] = [
    Waypoint(tilemap=0x1C, target_px=_SEED_CLERK_PX, radius=10),
    Waypoint(tilemap=0x1C, target_px=(154, 400), radius=14),
    Waypoint(
        tilemap=0x1C,
        target_px=(8 * 16 + 8, 28 * 16 + 8),
        radius=8,
        is_exit=True,
        exit_direction="down",
    ),
]
# After shop, walk the square to the east gate open face — not (46,26)/(47,26).
# Shop exit lands ~(601,232). Step south first — (37,14) pathing east seals.
_TOWN_SHOP_TO_PATH: List[Waypoint] = [
    Waypoint(tilemap=0x04, target_px=(602, 274), radius=12),
    Waypoint(tilemap=0x04, target_px=(608, 288), radius=12),
    Waypoint(tilemap=0x04, target_px=(640, 288), radius=14, run_direction="right"),
    Waypoint(tilemap=0x04, target_px=(684, 288), radius=12),
    Waypoint(tilemap=0x04, target_px=(684, 340), radius=14, run_direction="down"),
    Waypoint(tilemap=0x04, target_px=(684, 400), radius=14, run_direction="down"),
    Waypoint(tilemap=0x04, target_px=(692, 424), radius=14),
    Waypoint(tilemap=0x04, target_px=(728, 424), radius=12, is_exit=True, exit_direction="right"),
]
_FARM_TO_MOUNTAIN_GATE: List[Waypoint] = list(_FARM_TO_PATH) + list(_PATH_TO_MOUNTAIN)

# mountain_grape_stand tape: land ~(328, 728) tile (20, 45). Cliff blocks
# due-north; recorded gap is east to (32, 44) then north to (32, 39)
# (carpenter corridor — walk through, never A / never step onto Gotz).
# Then west to x=20 and north to the stand (326, 409) tile (20, 25).
# Short x=19 wrap is NPC-blocked; tape uses the long west loop.
_MOUNTAIN_ENTRY_TO_FIRST_BERRY: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(328, 728), radius=20),
    Waypoint(tilemap=0x10, target_px=(424, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(520, 712), radius=16, run_direction="right", force_run=True),
    Waypoint(tilemap=0x10, target_px=(520, 632), radius=16, run_direction="up", force_run=True),
    Waypoint(tilemap=0x10, target_px=(472, 600), radius=16),
    Waypoint(tilemap=0x10, target_px=(392, 568), radius=16),
    Waypoint(tilemap=0x10, target_px=(328, 568), radius=16, run_direction="left", force_run=True),
    Waypoint(tilemap=0x10, target_px=(328, 488), radius=8),
    Waypoint(tilemap=0x10, target_px=(240, 488), radius=10, run_direction="left", force_run=True),
    Waypoint(tilemap=0x10, target_px=(192, 464), radius=10),
    Waypoint(tilemap=0x10, target_px=(144, 448), radius=10),
    Waypoint(tilemap=0x10, target_px=(80, 432), radius=10),
    Waypoint(tilemap=0x10, target_px=(72, 368), radius=10),
    Waypoint(tilemap=0x10, target_px=(168, 360), radius=12),
    Waypoint(tilemap=0x10, target_px=(312, 360), radius=12, run_direction="right", force_run=True),
    Waypoint(tilemap=0x10, target_px=(326, 409), radius=10),
]

# First grape → south mountain exit. Inbound cannot climb the x=20 cliff
# under the grape, but the return can jump it onto ~(328, 568). A second
# cliff still blocks due-south; finish on the recorded east/south dirt.
# Do not reuse the outdoor-spa return here (wrong ridge).
_FIRST_BERRY_TO_MOUNTAIN_EXIT: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(326, 409), radius=10),
    Waypoint(tilemap=0x10, target_px=(328, 568), radius=24, run_direction="down", force_run=True),
    Waypoint(tilemap=0x10, target_px=(392, 568), radius=16),
    Waypoint(tilemap=0x10, target_px=(472, 600), radius=16),
    Waypoint(tilemap=0x10, target_px=(520, 632), radius=16),
    # NOTE: 960585b3 briefly forced run_direction="down"/force_run here; the
    # grape *return* slices this same list and the force-run pinned it at
    # ~(505,633) into the carpenter terrace wall (run10). Plain waypoint;
    # MultiMapNavTask's run_direction stall guard covers a genuine pin.
    Waypoint(tilemap=0x10, target_px=(520, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(424, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(328, 728), radius=20),
    Waypoint(tilemap=0x10, target_px=(312, 744), radius=16, is_exit=True, exit_direction="down"),
]

# These pixels are the east ladder from the mushroom stand.
_MUSHROOM_EAST_LADDER_PX: Tuple[Tuple[int, int], ...] = (
    (616, 712),
    (616, 696),
    (632, 696),
    (632, 664),
    (648, 664),
    (648, 632),
    (662, 633),
)
# North pull after landing, as on the spa route: a pixel-stuck replan at the
# (328,728) landing BFS'd off the south exit, 12 of 13 Fall mushroom trips.
_MOUNTAIN_LANDING_NORTH_PULL = Waypoint(
    tilemap=0x10, target_px=(328, 688), radius=16, run_direction="up", force_run=True
)
# The grape inbound with the same pull. Without it the first hop (328,728) pixel-stuck and the BFS
# walked back off the south exit: Spring D3 of ``y1new`` failed three trips in a row, 2 of 11 in ``fa``.
# The base list stays as is because the mushroom and spa routes slice it by index.
_MOUNTAIN_ENTRY_TO_FIRST_BERRY_PULLED: List[Waypoint] = (
    [_MOUNTAIN_ENTRY_TO_FIRST_BERRY[0], _MOUNTAIN_LANDING_NORTH_PULL] + list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY[1:])
)
_MOUNTAIN_ENTRY_TO_FIRST_MUSHROOM: List[Waypoint] = [
    _MOUNTAIN_ENTRY_TO_FIRST_BERRY[0],
    _MOUNTAIN_LANDING_NORTH_PULL,
    *_MOUNTAIN_ENTRY_TO_FIRST_BERRY[1:3],
    *[Waypoint(tilemap=0x10, target_px=px, radius=4) for px in _MUSHROOM_EAST_LADDER_PX],
]
# Fall poison mushroom: ROM tile 0x34 at (43,39) gives item 2 (200G, 50G over the mushroom) in Fall.
# Stand on (42,39), two tiles east of the mushroom stand's tile (41,39), and face right.
_POISON_MUSHROOM_STAND_PX = (678, 633)
_FIRST_MUSHROOM_TO_MOUNTAIN_EXIT: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=px, radius=16 if px == (616, 712) else 10)
    for px in reversed(_MUSHROOM_EAST_LADDER_PX)
] + list(_FIRST_BERRY_TO_MOUNTAIN_EXIT[-4:])


# Summer forage. ROM ``DATA8_82CFB4`` (item = table[tile*4 + season]) gives mountain tile
# 0x32 at (28,10) item 4, the green fruit (200G, ``Items_Price_Table``), in Summer only; grapes
# (tile 0x30) are Spring only and mushrooms (0x33/0x34) Fall only. The stand is the tile below,
# (28,11), facing up. The climb reuses the second-grape hops from the grape dirt up to (408,200)
# (tile 25,12), then two short hops east and north: about 330 px beyond the first grape.
_GREEN_FRUIT_STAND_PX = (454, 187)
_GREEN_FRUIT_CLIMB: List[Waypoint] = [
    Waypoint(0x10, px, radius=4, run_direction=face, force_run=True)
    for px, face in (((312, 328), "up"), ((408, 328), "right"),
                     ((408, 264), "up"), ((408, 200), "up"),
                     ((456, 200), "right"))
] + [Waypoint(0x10, _GREEN_FRUIT_STAND_PX, radius=3, run_direction="up", force_run=True)]
_MOUNTAIN_ENTRY_TO_GREEN_FRUIT: List[Waypoint] = (
    list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY_PULLED[:-1]) + _GREEN_FRUIT_CLIMB
)
# Back down the same hops to the grape dirt at (312, 360), then the grape's own descent.
_GREEN_FRUIT_TO_MOUNTAIN_EXIT: List[Waypoint] = [
    Waypoint(0x10, px, radius=8)
    for px in ((456, 200), (408, 200), (408, 264), (408, 328), (312, 328), (312, 360))
] + list(_FIRST_BERRY_TO_MOUNTAIN_EXIT)

_MOUNTAIN_ENTRY_TO_POISON_MUSHROOM: List[Waypoint] = list(_MOUNTAIN_ENTRY_TO_FIRST_MUSHROOM) + [
    Waypoint(0x10, _POISON_MUSHROOM_STAND_PX, radius=3, run_direction="right", force_run=True)
]

# Farm/path land → carpenter-gap dirt → west climb → east mid → lip.
# Same hops as first grape until (312, 360), plus a 40px north pull
# after land so BFS to (424,712) does not walk off the south exit.
# Never the east fish pond. Grape inbound stays without the pull.
_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA: List[Waypoint] = (
    [
        _MOUNTAIN_ENTRY_TO_FIRST_BERRY[0],
        _MOUNTAIN_LANDING_NORTH_PULL,
    ]
    + list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY[1:-1])
    + list(_WEST_MID_TO_OUTDOOR_SPA)
)

# Spa lip → reverse ridge to west-mid → reverse grape dirt → mid terrace
# ~(328, 568) → south exit. Do not jump the x=20 grape cliff (spawn still
# sits there) and do not reverse through camp/fish (620, 488 pins 38,28).
_OUTDOOR_SPA_TO_MOUNTAIN_EXIT: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(569, 201), radius=14, run_direction="left"),
    Waypoint(tilemap=0x10, target_px=(569, 214), radius=12),
    Waypoint(tilemap=0x10, target_px=(529, 246), radius=12),
    Waypoint(tilemap=0x10, target_px=(433, 255), radius=12),
    Waypoint(tilemap=0x10, target_px=(433, 300), radius=12),
    Waypoint(tilemap=0x10, target_px=(433, 345), radius=12),
    Waypoint(tilemap=0x10, target_px=(416, 345), radius=12),
    Waypoint(tilemap=0x10, target_px=(372, 347), radius=12),
    Waypoint(tilemap=0x10, target_px=(348, 361), radius=12),
    Waypoint(tilemap=0x10, target_px=(312, 360), radius=12),
    Waypoint(tilemap=0x10, target_px=(168, 360), radius=12, run_direction="left", force_run=True),
    Waypoint(tilemap=0x10, target_px=(72, 368), radius=10),
    Waypoint(tilemap=0x10, target_px=(80, 432), radius=10),
    Waypoint(tilemap=0x10, target_px=(144, 448), radius=10),
    Waypoint(tilemap=0x10, target_px=(192, 464), radius=10),
    Waypoint(tilemap=0x10, target_px=(240, 488), radius=10),
    Waypoint(tilemap=0x10, target_px=(328, 488), radius=8, run_direction="right", force_run=True),
    Waypoint(tilemap=0x10, target_px=(328, 568), radius=16),
] + list(_FIRST_BERRY_TO_MOUNTAIN_EXIT[2:])

# Bin is F2 pocket x=8–9/y=29–30. From the west-gate entry the north stand
# (8,28) facing down is reachable without clearing cargo.
_FARM_WEST_GATE_TO_SHIPPING_BIN: List[Waypoint] = [
    Waypoint(
        tilemap=0x00,
        target_px=(8 * 16 + 8, 28 * 16 + 8),
        radius=1,
        action_on_arrive="press_a",
        action_face="down",
        action_frames=28,
        action_cooldown=36,
    ),
]

# East-gate landing is ~(744, 424), already inside a (728, 424) radius 18, and
# the column below x=744 is wall: a forced run down from there stood still and
# its sidestep walked back out to the path (four Fall sale trips, a Winter tool
# trip). Run west along y=424 to the x=696 column the cow's return climbs.
_TOWN_EAST_GATE_TO_LOWER_ROAD: List[Waypoint] = [
    Waypoint(tilemap=0x04, target_px=(696, 424), radius=6, run_direction="left", force_run=True),
    Waypoint(tilemap=0x04, target_px=(696, 890), radius=6, run_direction="down", force_run=True),
]
_TOWN_TO_ANIMAL_SHOP_DOOR: List[Waypoint] = [
    *_TOWN_EAST_GATE_TO_LOWER_ROAD,
    Waypoint(tilemap=0x04, target_px=(601, 890), radius=8, run_direction="left"),
    Waypoint(tilemap=0x04, target_px=(601, 874), radius=10, is_exit=True, exit_direction="up"),
]
_ANIMAL_SHOP_STAGING = Waypoint(
    tilemap=0x24, target_px=(128, 200), radius=12, run_direction="up"
)
# General store (tilemap 0x22, door metatile 0xC6 at town tile (25,53) =
# (408,856); the two 0xFF tiles under it are the walkway). Same east road
# and y=890 lower road as the animal shop, then west to x=408 and up.
_TOWN_TO_TOOL_SHOP_DOOR: List[Waypoint] = [
    *_TOWN_EAST_GATE_TO_LOWER_ROAD,
    Waypoint(tilemap=0x04, target_px=(440, 890), radius=10, run_direction="left"),
    Waypoint(tilemap=0x04, target_px=(408, 880), radius=10, is_exit=True, exit_direction="up"),
]
# Interior spawn is (144,456). The shelves are in column 14, so the stands
# are x=217 (one px past the wall stop at 216) facing right: milker rows
# y=336..351 (tile 21), brush y=352..367 (tile 22). The hop ends 17px short
# of the wall; BuyShelfItemTask closes the last px and the row.
_TOOL_SHOP_TO_MILKER: List[Waypoint] = [
    Waypoint(tilemap=0x22, target_px=(150, 400), radius=10),
    Waypoint(tilemap=0x22, target_px=(190, 360), radius=10),
    Waypoint(tilemap=0x22, target_px=(196, 344), radius=3),
]
_TOOL_SHOP_TO_BRUSH: List[Waypoint] = [
    Waypoint(tilemap=0x22, target_px=(150, 400), radius=10),
    Waypoint(tilemap=0x22, target_px=(196, 358), radius=3),
]
_TOOL_SHOP_TO_TOWN: List[Waypoint] = [
    Waypoint(tilemap=0x22, target_px=(190, 360), radius=12),
    Waypoint(tilemap=0x22, target_px=(160, 420), radius=12),
    Waypoint(tilemap=0x22, target_px=(144, 440), radius=10),
    Waypoint(tilemap=0x22, target_px=(144, 452), radius=10, is_exit=True, exit_direction="down"),
]
# Exit lands at (408,872) after ~120f of leaked interior coords.
_TOWN_TOOL_SHOP_TO_PATH: List[Waypoint] = [
    Waypoint(tilemap=0x04, target_px=(440, 900), radius=10),
    Waypoint(tilemap=0x04, target_px=(696, 890), radius=6, run_direction="right", force_run=True),
    Waypoint(tilemap=0x04, target_px=(696, 424), radius=8, run_direction="up", force_run=True),
    Waypoint(tilemap=0x04, target_px=(728, 424), radius=12, is_exit=True, exit_direction="right"),
]
_HOUSE_L1: List[Waypoint] = [Waypoint(tilemap=0x00, target_px=(136, 376), radius=8)]
_FARM_TO_TOWN: List[Waypoint] = list(_FARM_TO_PATH) + list(_PATH_TO_TOWN)
# Plaza → farm gate. Town/shop return and leaked farm-west ~(10, 422).
# NEVER concatenate this after a mountain 0x10 exit: (7,7)/(8,7) are
# 0xFF, and a descent aimed at (132,128) is the run13 pin. Mountain
# gate → farm is `_PATH_MOUNTAIN_GATE_TO_FARM` / `path_return_to_farm`.
_PATH_PLAZA_TO_FARM: List[Waypoint] = [_PATH_CROSSROADS, _PATH_FARM_EXIT]
# Mountain gate → farm, the reverse of the proven climb (measured live:
# (137,19)→(137,52)→(137,92)→(150,110)→(148,128)). Column 8 carries rows
# 1–5, then the lane must step east to column 9: (7,7) and (8,7) are 0xFF,
# so aiming a descent straight at the crossroads (132,128) walks into a
# sealed cell — the run13 `return_to_bin` pin at (133,101), 6x identical,
# and ~200 frames of BFS flail on the days it did escape.
_PATH_MOUNTAIN_GATE_TO_FARM: List[Waypoint] = [
    Waypoint(tilemap=0x0C, target_px=(137, 40), radius=10),
    Waypoint(tilemap=0x0C, target_px=(137, 88), radius=10),
    Waypoint(tilemap=0x0C, target_px=(152, 108), radius=8),
    Waypoint(tilemap=0x0C, target_px=(152, 128), radius=10),
    _PATH_FARM_EXIT,
]
_SPA_TO_FARM: List[Waypoint] = (
    list(_OUTDOOR_SPA_TO_MOUNTAIN_EXIT)
    + list(_PATH_MOUNTAIN_GATE_TO_FARM)
    + [Waypoint(tilemap=0x00, target_px=(40, 424), radius=24)]
)

def compose_routes(*parts: Sequence[Waypoint]) -> List[Waypoint]:
    """Concatenate named hops. Callers own the live MultNav, not a tape."""
    out: List[Waypoint] = []
    for part in parts:
        out.extend(part)
    return out


def segment_waypoints(*names: str) -> List[Waypoint]:
    """Look up and concatenate reusable path segments by name."""
    missing = [name for name in names if name not in SEGMENTS]
    if missing:
        raise KeyError(f"unknown path segment(s): {', '.join(missing)}")
    return compose_routes(*(SEGMENTS[name] for name in names))


SEGMENTS: Dict[str, List[Waypoint]] = {
    "farm_to_path": list(_FARM_TO_PATH),
    "path_to_town": list(_PATH_TO_TOWN),
    "path_to_town_shop": list(_PATH_TO_TOWN_SHOP),
    "path_to_mountain": list(_PATH_TO_MOUNTAIN),
    "path_to_farm": list(_PATH_PLAZA_TO_FARM),
    "path_mountain_gate_to_farm": list(_PATH_MOUNTAIN_GATE_TO_FARM),
    "town_to_shop_door": list(_TOWN_TO_SHOP_DOOR),
    "shop_to_counter": list(_SHOP_TO_COUNTER),
    "shop_to_town": list(_SHOP_TO_TOWN),
    "town_shop_to_path": list(_TOWN_SHOP_TO_PATH),
    "town_to_tool_door": list(_TOWN_TO_TOOL_SHOP_DOOR),
    "town_to_animal_door": list(_TOWN_TO_ANIMAL_SHOP_DOOR),
    "animal_shop_to_medicine": [
        _ANIMAL_SHOP_STAGING,
        Waypoint(0x24, (88, 144), radius=3),
        Waypoint(0x24, (58, 144), radius=2),
    ],
    "animal_shelf_to_town": [
        Waypoint(0x24, (137, 200), radius=8),
        Waypoint(0x24, (137, 212), radius=8, is_exit=True, exit_direction="down"),
        Waypoint(0x04, (601, 904), radius=2, run_direction="down"),
    ],
    "town_animal_shop_to_path": [
        # Ellen stands on (648,904) on weekday mornings (sprite 0x023C-0x0241): a tight waypoint on her pixel never
        # arrives, and the replan walks back into the shop door (Y2 Winter D19, BUY_MEDICINE: the day was lost, four
        # sick cows died). Stop short, swing south of her, join the x=696 column from below, as animal_shop_town_to_farm.
        Waypoint(0x04, (632, 904), radius=10, run_direction="right", force_run=True),
        Waypoint(0x04, (656, 936), radius=10),
        Waypoint(0x04, (696, 920), radius=8),
        Waypoint(0x04, (696, 424), radius=6, run_direction="up", force_run=True),
        Waypoint(0x04, (756, 424), radius=8, is_exit=True, exit_direction="right"),
    ],
    "tool_shop_to_milker": list(_TOOL_SHOP_TO_MILKER),
    "tool_shop_to_brush": list(_TOOL_SHOP_TO_BRUSH),
    "tool_shop_to_town": list(_TOOL_SHOP_TO_TOWN),
    "town_tool_shop_to_path": list(_TOWN_TOOL_SHOP_TO_PATH),
    "town_south_to_path": [
        Waypoint(0x04, (688, 924), radius=12),
        Waypoint(0x04, (688, 700), radius=12, run_direction="up"),
        Waypoint(0x04, (688, 424), radius=12, run_direction="up"),
        Waypoint(0x04, (728, 424), radius=12, is_exit=True, exit_direction="right"),
    ],
    "mountain_entry_to_first_berry": list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY_PULLED),
    # ROM MapMountain ($A7A600): spring grape (2,4). The upper west
    # plateau is reached around the cliff at column 3, not up its south lip.
    "mountain_entry_to_second_grape": list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY[:-1]) + [
        Waypoint(0x10, px, radius=4, run_direction=face, force_run=True)
        for px, face in (((312,328), "up"), ((408,328), "right"),
                         ((408,264), "up"), ((408,200), "up"),
                         ((304,200), "left"), ((200,200), "left"),
                         ((200,136), "up"), ((104,136), "left"),
                         ((56,136), "left"), ((56,56), "up"))
    ] + [Waypoint(0x10, (40,57), radius=2, run_direction="left", force_run=True)],
    "mountain_entry_to_first_mushroom": list(_MOUNTAIN_ENTRY_TO_FIRST_MUSHROOM),
    "mountain_entry_to_green_fruit": list(_MOUNTAIN_ENTRY_TO_GREEN_FRUIT),
    "mountain_entry_to_poison_mushroom": list(_MOUNTAIN_ENTRY_TO_POISON_MUSHROOM),
    "first_berry_to_mountain_exit": list(_FIRST_BERRY_TO_MOUNTAIN_EXIT),
    "first_mushroom_to_mountain_exit": list(_FIRST_MUSHROOM_TO_MOUNTAIN_EXIT),
    "green_fruit_to_mountain_exit": list(_GREEN_FRUIT_TO_MOUNTAIN_EXIT),
    "farm_west_gate_to_shipping_bin": list(_FARM_WEST_GATE_TO_SHIPPING_BIN),
}


ROUTES: Dict[str, List[Waypoint]] = {
    "farm_to_house": [
        Waypoint(tilemap=0x00, target_px=(136, 424), radius=12),
    ],
    "farm_to_house_level1": list(_HOUSE_L1),
    "farm_to_house_level2": list(_HOUSE_L1),
    "farm_to_path": list(_FARM_TO_PATH),
    "path_to_town": list(_PATH_TO_TOWN),
    "path_to_mountain": list(_PATH_TO_MOUNTAIN),
    # The horse scene plays on entering the path (from Winter Y1) and leaves its
    # actors west of the farmer at (184,120); the walk home east is clear.
    "horse_event_loop": list(_FARM_TO_PATH[:-1]) + [_PATH_FARM_EXIT],
    "farm_to_town": list(_FARM_TO_TOWN),
    "go_to_town": list(_FARM_TO_TOWN),
    "farm_to_shop_door": list(_FARM_TO_PATH)
    + list(_PATH_TO_TOWN_SHOP)
    + list(_TOWN_TO_SHOP_DOOR),
    "mountain_entry_to_first_berry": list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY_PULLED),
    "mountain_entry_to_first_mushroom": list(SEGMENTS["mountain_entry_to_first_mushroom"]),
    "farm_to_first_mountain_berry": list(_FARM_TO_MOUNTAIN_GATE)
    + list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY_PULLED),
    "first_berry_to_mountain_exit": list(_FIRST_BERRY_TO_MOUNTAIN_EXIT),
    "first_mushroom_to_mountain_exit": list(SEGMENTS["first_mushroom_to_mountain_exit"]),
    "first_mountain_berry_to_shipping_bin": list(_FIRST_BERRY_TO_MOUNTAIN_EXIT)
    + list(_PATH_MOUNTAIN_GATE_TO_FARM)
    + [Waypoint(tilemap=0x00, target_px=(80, 424), radius=12)]
    + list(_FARM_WEST_GATE_TO_SHIPPING_BIN),
    "first_mushroom_to_shipping_bin": list(_FIRST_MUSHROOM_TO_MOUNTAIN_EXIT)
    + list(_PATH_MOUNTAIN_GATE_TO_FARM)
    + [Waypoint(tilemap=0x00, target_px=(80, 424), radius=12)]
    + list(_FARM_WEST_GATE_TO_SHIPPING_BIN),
    "mountain_entry_to_green_fruit": list(_MOUNTAIN_ENTRY_TO_GREEN_FRUIT),
    "green_fruit_to_mountain_exit": list(_GREEN_FRUIT_TO_MOUNTAIN_EXIT),
    "mountain_entry_to_poison_mushroom": list(_MOUNTAIN_ENTRY_TO_POISON_MUSHROOM),
    "poison_mushroom_to_shipping_bin": list(_FIRST_MUSHROOM_TO_MOUNTAIN_EXIT)
    + list(_PATH_MOUNTAIN_GATE_TO_FARM)
    + [Waypoint(tilemap=0x00, target_px=(80, 424), radius=12)]
    + list(_FARM_WEST_GATE_TO_SHIPPING_BIN),
    "green_fruit_to_shipping_bin": list(_GREEN_FRUIT_TO_MOUNTAIN_EXIT)
    + list(_PATH_MOUNTAIN_GATE_TO_FARM)
    + [Waypoint(tilemap=0x00, target_px=(80, 424), radius=12)]
    + list(_FARM_WEST_GATE_TO_SHIPPING_BIN),
    # Early-game town loop: shop + church fronts, then leave. Completing
    # this route is the planner's "ready to go home" signal on day 1.
    "town_explore": [
        *_FARM_WEST_TO_TOWN,
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=20),
        Waypoint(tilemap=0x04, target_px=(600, 230), radius=20),
        Waypoint(tilemap=0x04, target_px=(375, 200), radius=24),
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=20),
        Waypoint(tilemap=0x04, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
    # The old (232,128) hop pixel-stuck at (278,128) for 18k frames (ROM, Fall D3 Sunday pin): the north
    # town row is walled there. Same road as the D1 Maria route, then up into the church (07:10 from the house).
    "farm_to_church": list(_FARM_WEST_TO_TOWN)
    + [
        Waypoint(tilemap=0x04, target_px=(696, 424), radius=8),
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(500, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(376, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(376, 200), radius=14),
        Waypoint(tilemap=0x04, target_px=(358, 150), radius=8, is_exit=True, exit_direction="up"),
    ],
    "farm_to_animal_shop_staging": list(_FARM_WEST_TO_TOWN)
    + list(_TOWN_TO_ANIMAL_SHOP_DOOR)
    + [_ANIMAL_SHOP_STAGING],
    "farm_to_animal_shop_counter": list(_FARM_WEST_TO_TOWN)
    + list(_TOWN_TO_ANIMAL_SHOP_DOOR)
    + [_ANIMAL_SHOP_STAGING, Waypoint(tilemap=0x24, target_px=(201, 158), radius=4)],
    "farm_to_animal_shop_counter_sale": list(_FARM_WEST_TO_TOWN)
    + list(_TOWN_TO_ANIMAL_SHOP_DOOR)
    + [
        _ANIMAL_SHOP_STAGING,
        # Sale menu replay needs the recording pixel; a loose radius leaves
        # the player too low/left.
        Waypoint(tilemap=0x24, target_px=(201, 158), radius=1),
    ],
    "animal_shop_to_town": [
        Waypoint(tilemap=0x24, target_px=(137, 200), radius=12),
        Waypoint(tilemap=0x24, target_px=(137, 212), radius=12, is_exit=True, exit_direction="down"),
        # After the transition, step off the town door tile before idle wait.
        Waypoint(tilemap=0x04, target_px=(601, 904), radius=2, run_direction="down"),
    ],
    "animal_shop_town_to_farm": [
        # Leave the door column east before turning north. A diagonal from
        # (600,888) to the town gate steps straight back into the shop. Stop short of (648,904): Ellen
        # stands there on weekday mornings (sprite 0x023C), and a tight waypoint on her tile never
        # arrives (Y2 Spring D16: four pixel-stuck replans, then a wedged run).
        Waypoint(tilemap=0x04, target_px=(632, 904), radius=10, run_direction="right", force_run=True),
        # Swing south of her spot (rows 57-58 of the plaza are open) and join the x=696 column from below.
        Waypoint(tilemap=0x04, target_px=(656, 936), radius=10),
        Waypoint(tilemap=0x04, target_px=(696, 920), radius=8),
        Waypoint(tilemap=0x04, target_px=(696, 424), radius=6, run_direction="up", force_run=True),
        Waypoint(tilemap=0x04, target_px=(756, 424), radius=8, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=8, is_exit=True, exit_direction="right"),
        # Clear the west gate horizontally before descending: y=448 outside
        # the farm is fenced. The hawker's berry lands south of this entrance.
        Waypoint(tilemap=0x00, target_px=(80, 424), radius=4, run_direction="right", force_run=True),
        Waypoint(tilemap=0x00, target_px=(80, 448), radius=12),
    ],
    # Spring D1 town handoff (docs/town_day1_recon.md). Natural entry lands
    # at town gate ~(712,424). Routes assume start on 0x04.
    "d1_town_to_flower_shop": [
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(600, 262), radius=10, is_exit=True, exit_direction="up"),
        Waypoint(tilemap=0x1C, target_px=(144, 456), radius=20),
    ],
    "d1_flower_back_to_nina": [
        Waypoint(tilemap=0x1D, target_px=(104, 184), radius=16),
        Waypoint(tilemap=0x1D, target_px=(104, 120), radius=12),
        Waypoint(tilemap=0x1D, target_px=(101, 102), radius=6),  # bit 0x04 face left + A
    ],
    "d1_flower_back_exit_to_town": [
        Waypoint(tilemap=0x1D, target_px=(104, 184), radius=14),
        Waypoint(tilemap=0x1D, target_px=(104, 210), radius=12, is_exit=True, exit_direction="down"),
        Waypoint(tilemap=0x1C, target_px=(144, 456), radius=18),
        Waypoint(tilemap=0x1C, target_px=(144, 480), radius=12, is_exit=True, exit_direction="down"),
        # Town-space door remap is a hold-Down in the talk sequence; a
        # (600,280) nav waypoint re-enters the shop from leaked interior coords.
    ],
    # Stop at church door lip; scripted up enters 0x1B. Stay on y≈280 until
    # x≈376, then north — a direct (500,280)→(411,216) hop stalls.
    "d1_town_to_maria": [
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(500, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(376, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(376, 200), radius=14),
        Waypoint(tilemap=0x04, target_px=(358, 150), radius=8),
    ],
    "d1_church_to_maria": [
        Waypoint(tilemap=0x1B, target_px=(128, 456), radius=16),
        Waypoint(tilemap=0x1B, target_px=(103, 405), radius=6),  # bit 0x20 face up + A
    ],
    "d1_maria_to_town": [
        Waypoint(tilemap=0x1B, target_px=(128, 456), radius=14),
        Waypoint(tilemap=0x1B, target_px=(128, 470), radius=10, is_exit=True, exit_direction="down"),
        Waypoint(tilemap=0x04, target_px=(376, 200), radius=16),
    ],
    "d1_town_to_ann": [
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=18),
        Waypoint(tilemap=0x04, target_px=(688, 700), radius=18, run_direction="down"),
        Waypoint(tilemap=0x04, target_px=(688, 924), radius=16, run_direction="down"),
        Waypoint(tilemap=0x04, target_px=(500, 924), radius=16),
        Waypoint(tilemap=0x04, target_px=(392, 914), radius=18),
    ],
    "d1_town_to_eve": [
        Waypoint(tilemap=0x04, target_px=(388, 950), radius=14, run_direction="down"),
        Waypoint(tilemap=0x04, target_px=(300, 950), radius=14),
        Waypoint(tilemap=0x04, target_px=(152, 950), radius=12),
        Waypoint(tilemap=0x04, target_px=(152, 896), radius=18),
    ],
    "d1_town_to_livestock": [
        Waypoint(tilemap=0x04, target_px=(300, 950), radius=16),
        Waypoint(tilemap=0x04, target_px=(601, 950), radius=14),
        Waypoint(tilemap=0x04, target_px=(601, 888), radius=10, is_exit=True, exit_direction="up"),
    ],
    # Gift stand is NE of the buy-cow counter, face down + A. (201,157)
    # face-right is the later buy-cow menu and does NOT set 0x10.
    # run_direction skips BFS; the counter blocks a pure path.
    "d1_livestock_to_event_stand": [
        Waypoint(tilemap=0x24, target_px=(128, 200), radius=14, run_direction="up", force_run=True),
        Waypoint(tilemap=0x24, target_px=(128, 158), radius=8, run_direction="up", force_run=True),
        Waypoint(tilemap=0x24, target_px=(201, 158), radius=6, run_direction="right", force_run=True),
        Waypoint(tilemap=0x24, target_px=(201, 121), radius=8, run_direction="up", force_run=True),
        Waypoint(tilemap=0x24, target_px=(230, 121), radius=6, run_direction="right", force_run=True),
        Waypoint(tilemap=0x24, target_px=(230, 139), radius=4, run_direction="down", force_run=True),
    ],
    # Align with town_day1_rest truck slice start (f≈9200 @ ~(688,357)).
    "d1_town_to_truck": [
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(688, 360), radius=10),
    ],
    "d1_town_to_truck_stand": [
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(688, 360), radius=12),
        Waypoint(tilemap=0x04, target_px=(700, 400), radius=12),
        Waypoint(tilemap=0x04, target_px=(715, 421), radius=10),
    ],
    # Shipping-bin / south-field return. Prefer y=60 path west, stay south
    # of the long y=31 fence until x<=8, then (6,33)→(4,30) so MultNav never
    # seals against house/cliff tiles at (6,32).
    "farm_south_to_west_gate": [
        Waypoint(tilemap=0x00, target_px=(55 * 16 + 8, 60 * 16 + 8), radius=18),
        Waypoint(tilemap=0x00, target_px=(48 * 16 + 8, 60 * 16 + 8), radius=18),
        Waypoint(tilemap=0x00, target_px=(40 * 16 + 8, 60 * 16 + 8), radius=18),
        Waypoint(tilemap=0x00, target_px=(32 * 16 + 8, 60 * 16 + 8), radius=18),
        Waypoint(tilemap=0x00, target_px=(24 * 16 + 8, 58 * 16 + 8), radius=18),
        Waypoint(tilemap=0x00, target_px=(16 * 16 + 8, 52 * 16 + 8), radius=18),
        Waypoint(tilemap=0x00, target_px=(10 * 16 + 8, 42 * 16 + 8), radius=18),
        Waypoint(tilemap=0x00, target_px=(8 * 16 + 8, 35 * 16 + 8), radius=16),
        Waypoint(tilemap=0x00, target_px=(6 * 16 + 8, 33 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(4 * 16 + 8, 30 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(3 * 16 + 8, 27 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=16),
    ],
    "farm_to_shed": [
        # fix_rainy_day: upper-left path, drop along shed left, enter from
        # below. Right/top corner clips on the building after remodel.
        Waypoint(tilemap=0x00, target_px=(137, 375), radius=12),
        Waypoint(tilemap=0x00, target_px=(244, 375), radius=12),
        Waypoint(tilemap=0x00, target_px=(354, 377), radius=12),
        Waypoint(tilemap=0x00, target_px=(354, 489), radius=12),
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "pocket_to_shed": [
        # West plant pocket. farm_to_shed's (137,375) is west through the
        # shipping-ditch no-go and sealed live can-fetch.
        Waypoint(tilemap=0x00, target_px=(244, 375), radius=12),
        Waypoint(tilemap=0x00, target_px=(354, 377), radius=12),
        Waypoint(tilemap=0x00, target_px=(354, 489), radius=12),
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "upper_farm_to_shed": [
        Waypoint(tilemap=0x00, target_px=(456, 424), radius=16),
        Waypoint(tilemap=0x00, target_px=(456, 489), radius=12),
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "field_to_shed": [
        # From harvest shipping stand, stay south of the well/stump at y=504.
        # A y=489 run_direction=right B-charges the stump.
        Waypoint(tilemap=0x00, target_px=(344, 504), radius=12),
        Waypoint(tilemap=0x00, target_px=(400, 504), radius=12),
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "near_shed_to_shed": [
        # Stop on the threshold; DirectionalTransition owns the walk-in.
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "farm_to_coop": [
        Waypoint(tilemap=0x00, target_px=(137, 375), radius=12),
        Waypoint(tilemap=0x00, target_px=(454, 360), radius=12),
    ],
    "farm_to_coop_sale": [
        Waypoint(tilemap=0x00, target_px=(456, 424), radius=16),
        Waypoint(tilemap=0x00, target_px=(454, 360), radius=12),
    ],
    "barn_to_coop": [
        Waypoint(tilemap=0x00, target_px=(454, 360), radius=12),
    ],
    "farm_to_barn": [
        Waypoint(tilemap=0x00, target_px=(137, 375), radius=16),
        Waypoint(tilemap=0x00, target_px=(244, 375), radius=16),
        Waypoint(tilemap=0x00, target_px=(329, 360), radius=18),
    ],
    "path_to_farm": list(_PATH_PLAZA_TO_FARM),
    "path_mountain_gate_to_farm": list(_PATH_MOUNTAIN_GATE_TO_FARM),
    "farm_to_mountain": list(_FARM_TO_MOUNTAIN_GATE),
    "mountain_entry_to_outdoor_spa": list(_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA),
    "mountain_entry_to_spa": list(_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA),
    "fish_spot_to_outdoor_spa": list(_FISH_TO_OUTDOOR_SPA),
    "farm_to_spa": list(_FARM_TO_MOUNTAIN_GATE) + list(_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA),
    "mountain_entry_to_fish_power_berry_spots": [
        Waypoint(tilemap=0x10, target_px=(328, 718), radius=16),
        Waypoint(tilemap=0x10, target_px=(496, 708), radius=16),
        Waypoint(tilemap=0x10, target_px=(518, 558), radius=16),
        Waypoint(tilemap=0x10, target_px=(582, 414), radius=16),
        Waypoint(tilemap=0x10, target_px=(624, 371), radius=12),
        Waypoint(tilemap=0x10, target_px=(686, 411), radius=12),
    ],
    "outdoor_spa_to_farm": list(_SPA_TO_FARM),
    "mountain_to_farm": list(_SPA_TO_FARM),
    "church_sunday_talk_loop": [
        Waypoint(tilemap=0x1B, target_px=(128, 456), radius=16),
        Waypoint(tilemap=0x1B, target_px=(203, 409), radius=8, action_on_arrive="press_a", action_face="right"),
        Waypoint(tilemap=0x1B, target_px=(85, 409), radius=8, action_on_arrive="press_a", action_face="left"),
        Waypoint(tilemap=0x1B, target_px=(85, 342), radius=8, action_on_arrive="press_a", action_face="left"),
        Waypoint(tilemap=0x1B, target_px=(101, 278), radius=8, action_on_arrive="press_a", action_face="left"),
        Waypoint(tilemap=0x1B, target_px=(155, 278), radius=8, action_on_arrive="press_a", action_face="right"),
        Waypoint(tilemap=0x1B, target_px=(43, 281), radius=8, action_on_arrive="press_a", action_face="right"),
        Waypoint(tilemap=0x1B, target_px=(229, 280), radius=8, action_on_arrive="press_a", action_face="left"),
        Waypoint(tilemap=0x1B, target_px=(206, 134), radius=8, action_on_arrive="press_a", action_face="left"),
        Waypoint(tilemap=0x1B, target_px=(141, 139), radius=8, action_on_arrive="press_a", action_face="left"),
    ],
    # Weekday courting (tasks/relationships.TownCourtTask): Ellen stands at (648,904) beside the animal shop's
    # door, Nina at (520,264) beside the flower shop's; each walk stops below the door the loop reloads through.
    "farm_to_ellen_weekday": list(_FARM_WEST_TO_TOWN) + list(_TOWN_TO_ANIMAL_SHOP_DOOR[:-1]),
    "farm_to_nina_weekday": list(_FARM_WEST_TO_TOWN) + [
        Waypoint(tilemap=0x04, target_px=(696, 424), radius=8),
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=10),
    ],
    "nina_weekday_to_farm": [
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=10),
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(696, 424), radius=8),
        Waypoint(tilemap=0x04, target_px=(756, 424), radius=8, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x00, target_px=(80, 448), radius=12),
    ],
    # The church door lets out at ~(350,160) on the town map: the inbound road reversed.
    "church_to_farm": [
        Waypoint(tilemap=0x1B, target_px=(130, 468), radius=12, is_exit=True, exit_direction="down"),
        Waypoint(tilemap=0x04, target_px=(376, 200), radius=14),
        Waypoint(tilemap=0x04, target_px=(376, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(500, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(696, 424), radius=8),
        Waypoint(tilemap=0x04, target_px=(756, 424), radius=8, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x00, target_px=(80, 448), radius=12),
    ],
    "town_to_farm": [
        Waypoint(tilemap=0x04, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
    "event_town_to_farm": [
        Waypoint(tilemap=0x05, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
}

# The animal shop has a back door at interior tile (5, 3) into the fenced yard
# where a bought hen spawns (~(568, 792) on town map 0x04). The yard has no
# other way in. Walking down at x~582 from the hen stand re-enters the shop.
ROUTES["animal_shop_to_yard"] = [
    Waypoint(tilemap=0x24, target_px=(88, 112), radius=8),
    Waypoint(tilemap=0x24, target_px=(88, 72), radius=6, is_exit=True, exit_direction="up"),
    Waypoint(tilemap=0x04, target_px=(584, 808), radius=10),
]
ROUTES["animal_yard_to_farm"] = [
    Waypoint(tilemap=0x04, target_px=(582, 800), radius=3, is_exit=True, exit_direction="down"),
    *ROUTES["animal_shop_to_town"],
    *ROUTES["animal_shop_town_to_farm"],
]
