"""Named multi-map waypoint routes and route helpers.

Coordinate lists are hand-authored from recordings; do not "simplify" hops.
"""

from typing import Dict, List, Optional, Sequence

from harvest.core.tile_catalog import TILE_SIZE
from harvest.maps.farm_gate import farm_to_west_gate_waypoints
from harvest.maps.farm_pond import FARM_TILEMAP_IDS
from harvest.maps.map_types import Waypoint

# Path 0x0C is a ~16x10-tile fork. Farm (y~422) and mountain (y~740)
# pixels leak across the transition; Manhattan then prefers a later
# exit over the plaza center.
PATH_TILEMAP_ID = 0x0C
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
    if tilemap == PATH_TILEMAP_ID and path_coords_leaked(px, py):
        for i, wp in enumerate(waypoints):
            if wp.tilemap == PATH_TILEMAP_ID:
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
_FARM_WEST_EXIT = Waypoint(
    tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"
)
_PATH_FARM_GATE = Waypoint(tilemap=0x0C, target_px=(232, 128), radius=16)
_PATH_CROSSROADS = Waypoint(tilemap=0x0C, target_px=(132, 128), radius=16)
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
_FARM_WEST_TO_TOWN: List[Waypoint] = [_FARM_WEST_EXIT, _PATH_TOWN_EXIT]

# Pond A6 occupies y=25 x=0-6. Column x=8 is A0 y=23-25 then A8 y=26-28.
# Live After_Rocks on (7,25)/(7,26) sat at (127,420) against the pond
# face. Stay on the house column until y=27, then run west.
_FARM_GATE_PINCH_TO_EXIT: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(136, 440), radius=8, run_direction="down", force_run=True),
    Waypoint(tilemap=0x00, target_px=(72, 440), radius=8, run_direction="left"),
    _FARM_WEST_EXIT,
]

# L1 house front ~(136,344) BFS-cuts the NW ledge. Drop south first, then
# A0 above the pond and the x=7 pinch — not the A8 pond-edge row at y=424.
_FARM_TO_PATH: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(137, 375), radius=12),
    Waypoint(tilemap=0x00, target_px=(136, 392), radius=8),  # (8,24) A0
    *_FARM_GATE_PINCH_TO_EXIT,
    _PATH_FARM_GATE,
    _PATH_CROSSROADS,
]

# Sunday south crop ~(78,598). Viewport BFS to (136,424) cuts 0x5E beds and
# F2 ditch (9/11,26–28). Stay on untilled dirt, x=13 through y=31 fence,
# then north of the ditch onto the A8 gate road.
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
    Waypoint(tilemap=0x00, target_px=(200, 408), radius=8),  # (12,25) north of ditch
    Waypoint(tilemap=0x00, target_px=(136, 392), radius=8, run_direction="left"),
    *_FARM_GATE_PINCH_TO_EXIT,
    # Crossroads next — not PATH_FARM_GATE (232,128), which is east
    # toward the farm and pins leaked (10,422) landings.
    _PATH_CROSSROADS,
]
_PATH_TO_TOWN: List[Waypoint] = [_PATH_TOWN_EXIT]
# Already on 0x0C: plaza center first, then the north exit. Do not
# BFS-diagonal from the east landing / leaked farm y toward (132,30).
_PATH_TO_MOUNTAIN: List[Waypoint] = [_PATH_CROSSROADS, _PATH_MOUNTAIN_EXIT]
# Seed-shop town gate: stand on the open face, not (0,8)/(1,8) doorframe hugs.
_PATH_TO_TOWN_SHOP: List[Waypoint] = [
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
    Waypoint(tilemap=0x10, target_px=(520, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(424, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(328, 728), radius=20),
    Waypoint(tilemap=0x10, target_px=(312, 744), radius=16, is_exit=True, exit_direction="down"),
]

# Farm/path land → carpenter-gap dirt → west climb → east mid → lip.
# Same hops as first grape until (312, 360), plus a 40px north pull
# after land so BFS to (424,712) does not walk off the south exit.
# Never the east fish pond. Grape inbound stays without the pull.
_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA: List[Waypoint] = (
    [
        _MOUNTAIN_ENTRY_TO_FIRST_BERRY[0],
        Waypoint(
            tilemap=0x10,
            target_px=(328, 688),
            radius=16,
            run_direction="up",
            force_run=True,
        ),
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

_TOWN_TO_ANIMAL_SHOP_DOOR: List[Waypoint] = [
    Waypoint(tilemap=0x04, target_px=(688, 430), radius=16),
    Waypoint(tilemap=0x04, target_px=(688, 888), radius=16, run_direction="down"),
    Waypoint(tilemap=0x04, target_px=(601, 888), radius=12),
    Waypoint(tilemap=0x04, target_px=(601, 874), radius=12, is_exit=True, exit_direction="up"),
]
_ANIMAL_SHOP_STAGING = Waypoint(
    tilemap=0x24, target_px=(128, 200), radius=12, run_direction="up"
)
_HOUSE_L1: List[Waypoint] = [Waypoint(tilemap=0x00, target_px=(136, 344), radius=12)]
_FARM_TO_TOWN: List[Waypoint] = list(_FARM_TO_PATH) + list(_PATH_TO_TOWN)
_PATH_TO_FARM: List[Waypoint] = [_PATH_CROSSROADS, _PATH_FARM_EXIT]
_SPA_TO_FARM: List[Waypoint] = (
    list(_OUTDOOR_SPA_TO_MOUNTAIN_EXIT)
    + list(_PATH_TO_FARM)
    + [Waypoint(tilemap=0x00, target_px=(40, 424), radius=24)]
)

# Bush pick (37,57) face left; bin (62,60) stand one tile west face right.
_BERRY_PICK = Waypoint(
    tilemap=0x00,
    target_px=(37 * 16 + 8, 57 * 16 + 8),
    radius=8,
    action_on_arrive="press_a",
    action_face="left",
    action_frames=28,
    action_cooldown=36,
)
_BERRY_BUSH_TO_BIN: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 54 * 16 + 8), radius=10),
    Waypoint(tilemap=0x00, target_px=(40 * 16 + 8, 54 * 16 + 8), radius=12),
    Waypoint(tilemap=0x00, target_px=(48 * 16 + 8, 58 * 16 + 8), radius=14),
    Waypoint(tilemap=0x00, target_px=(55 * 16 + 8, 60 * 16 + 8), radius=14),
    Waypoint(
        tilemap=0x00,
        target_px=(61 * 16 + 8, 60 * 16 + 8),
        radius=10,
        action_on_arrive="press_a",
        action_face="right",
        action_frames=28,
        action_cooldown=36,
    ),
]


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
    "path_to_farm": list(_PATH_TO_FARM),
    "town_to_shop_door": list(_TOWN_TO_SHOP_DOOR),
    "shop_to_counter": list(_SHOP_TO_COUNTER),
    "shop_to_town": list(_SHOP_TO_TOWN),
    "town_shop_to_path": list(_TOWN_SHOP_TO_PATH),
    "mountain_entry_to_first_berry": list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY),
    "first_berry_to_mountain_exit": list(_FIRST_BERRY_TO_MOUNTAIN_EXIT),
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
    "farm_to_town": list(_FARM_TO_TOWN),
    "go_to_town": list(_FARM_TO_TOWN),
    "farm_to_shop_door": list(_FARM_TO_PATH)
    + list(_PATH_TO_TOWN_SHOP)
    + list(_TOWN_TO_SHOP_DOOR),
    "mountain_entry_to_first_berry": list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY),
    "farm_to_first_mountain_berry": list(_FARM_TO_MOUNTAIN_GATE)
    + list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY),
    "first_mountain_berry_to_shipping_bin": list(_FIRST_BERRY_TO_MOUNTAIN_EXIT)
    + list(_PATH_TO_FARM)
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
    "farm_to_church": list(_FARM_WEST_TO_TOWN)
    + [
        Waypoint(tilemap=0x04, target_px=(232, 128), radius=16),
        Waypoint(tilemap=0x04, target_px=(375, 139), radius=10, is_exit=True, exit_direction="up"),
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
    # OPEN_FENCE_GAP can finish on the soft-collision gap tile. BerryShipTask
    # first takes the verified east-past-wall/south charge. Escape commonly
    # lands ~(28,32), west of the pond — step down-left around live weeds at
    # (28,33)/(29,32), not east into pond. Live D2 has a 2x2 rock at
    # ~(27–28,45–46). North-of-bush approach: pocket (36–37,56–57) is sealed
    # south by weeds, west by stones, east by rock. Enter via (36,54)→(36,56).
    "berry_ship": [
        Waypoint(tilemap=0x00, target_px=(27 * 16 + 8, 35 * 16 + 8), radius=16),
        Waypoint(tilemap=0x00, target_px=(28 * 16 + 8, 39 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(25 * 16 + 8, 44 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(25 * 16 + 8, 50 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(32 * 16 + 8, 51 * 16 + 8), radius=12),
        Waypoint(tilemap=0x00, target_px=(33 * 16 + 8, 53 * 16 + 8), radius=10),
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 54 * 16 + 8), radius=10),
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 56 * 16 + 8), radius=8),
        _BERRY_PICK,
        *_BERRY_BUSH_TO_BIN,
    ],
    "berry_ship_repeat": [
        Waypoint(tilemap=0x00, target_px=(55 * 16 + 8, 60 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(48 * 16 + 8, 58 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(40 * 16 + 8, 54 * 16 + 8), radius=12),
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 54 * 16 + 8), radius=10),
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 56 * 16 + 8), radius=8),
        _BERRY_PICK,
        *_BERRY_BUSH_TO_BIN,
    ],
    # Shipping-bin / berry-field return. Prefer y=60 path west, stay south
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
    "path_to_farm": list(_PATH_TO_FARM),
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
    "church_to_farm": [
        Waypoint(tilemap=0x1B, target_px=(130, 468), radius=12, is_exit=True, exit_direction="down"),
        Waypoint(tilemap=0x04, target_px=(360, 468), radius=16),
        Waypoint(tilemap=0x04, target_px=(600, 430), radius=16),
        Waypoint(tilemap=0x04, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
    "town_to_farm": [
        Waypoint(tilemap=0x04, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
    "town_to_farm_west_gate_sale": [
        Waypoint(tilemap=0x04, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        # sell_chicken.json re-enters at the farm west gate along y=448.
        # Generic path-to-farm exit at y=128 lands on the upper path.
        Waypoint(tilemap=0x0C, target_px=(230, 118), radius=6),
        Waypoint(tilemap=0x0C, target_px=(244, 118), radius=4, is_exit=True, exit_direction="right"),
    ],
    "event_town_to_farm": [
        Waypoint(tilemap=0x05, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
}
