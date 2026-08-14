"""Named multi-map waypoint routes and route helpers.

Coordinate lists are hand-authored from recordings; do not "simplify" hops.
"""

from typing import Dict, List, Optional, Sequence

from harvest.core.tile_catalog import TILE_SIZE
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


def slice_route_from_position(
    waypoints: List[Waypoint],
    px: int,
    py: int,
    *,
    tilemap: Optional[int] = None,
) -> List[Waypoint]:
    """Return a suffix of ``waypoints`` starting at the nearest relevant hop.

    MultiMapNav always begins at index 0. When already mid-mountain (fish
    stand, spa lip, west climb), forcing the south entry first walks away
    from the goal. Pick the closest same-map waypoint and continue from there
    (or the previous hop if we are slightly past it).

    Leaked 0x0C coords must start at the first path hop (crossroads), not
    a later exit. Mountain north-edge y<80 is handled by callers (Gotz).
    """
    if not waypoints:
        return []
    if tilemap == PATH_TILEMAP_ID and path_coords_leaked(px, py):
        for i, wp in enumerate(waypoints):
            if wp.tilemap == PATH_TILEMAP_ID:
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
    # If already within arrival radius of a later waypoint, skip ahead.
    for i in range(len(waypoints) - 1, best_i - 1, -1):
        wp = waypoints[i]
        if tilemap is not None and wp.tilemap != tilemap:
            continue
        if (
            abs(wp.target_px[0] - px) <= wp.radius
            and abs(wp.target_px[1] - py) <= wp.radius
        ):
            return list(waypoints[i:])
    # Start one hop earlier so we still approach along the corridor.
    start = max(0, best_i - 1)
    return list(waypoints[start:])


def densify_waypoints(
    waypoints: Sequence[Waypoint],
    *,
    max_hop_tiles: int = 7,
    tile_size: int = TILE_SIZE,
) -> List[Waypoint]:
    """Insert intermediate hops so same-map targets stay within viewport BFS range.

    SNES only loads ~16x14 tiles around the player. BFS beyond ~7–10 tiles sees
    stale tile IDs. Hand-authored routes should still prefer known walkable
    corridors; this helper fills large pixel gaps with linear interpolants for
    the same tilemap (map transitions and exit/action waypoints are preserved).
    """
    if max_hop_tiles < 1:
        raise ValueError("max_hop_tiles must be >= 1")
    if not waypoints:
        return []

    max_hop_px = max_hop_tiles * tile_size
    result: List[Waypoint] = [waypoints[0]]
    for nxt in waypoints[1:]:
        prev = result[-1]
        # Never densify across map changes, exits, or scripted actions.
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


# Outdoor spa path on mountain 0x10 (hot_spring_bath + entry approach).
# Viewport-limited BFS needs hops ≤ ~15 tiles. Path tiles: 0xA0/0xA8 (clear
# of stumps/rocks along this corridor — validated on spring mountain RAM).
# Do NOT route via camp tent pond ~(697,406) or west cave door.
_MOUNTAIN_ENTRY_APPROACH: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(328, 718), radius=16),
    Waypoint(tilemap=0x10, target_px=(420, 713), radius=16),
    Waypoint(tilemap=0x10, target_px=(496, 708), radius=16),
    Waypoint(tilemap=0x10, target_px=(580, 680), radius=14),
    Waypoint(tilemap=0x10, target_px=(640, 620), radius=14),
    Waypoint(tilemap=0x10, target_px=(680, 520), radius=14),
    Waypoint(tilemap=0x10, target_px=(686, 430), radius=14),
]

# West mid corridor y~470 → west climb → east mid y~361 → NE lip ~(619,201).
# Climb hops use tight radius: a loose radius on (70,377) "arrives" at y~390
# still below the ridge and then BFS cannot cut NE into solid tiles.
_FISH_TO_OUTDOOR_SPA: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(640, 428), radius=14),
    Waypoint(tilemap=0x10, target_px=(560, 454), radius=14),
    Waypoint(tilemap=0x10, target_px=(480, 468), radius=14),
    Waypoint(tilemap=0x10, target_px=(380, 470), radius=14),
    Waypoint(tilemap=0x10, target_px=(280, 470), radius=14),
    Waypoint(tilemap=0x10, target_px=(185, 456), radius=14),
    Waypoint(tilemap=0x10, target_px=(103, 442), radius=12),
    # Full west climb (human: y 442 → 407 → 361 at x~70–78).
    Waypoint(tilemap=0x10, target_px=(70, 413), radius=10),
    Waypoint(tilemap=0x10, target_px=(70, 361), radius=10),
    # East mid corridor — run_direction skips stale-tile BFS on clear path.
    Waypoint(
        tilemap=0x10,
        target_px=(148, 361),
        radius=12,
        run_direction="right",
    ),
    Waypoint(
        tilemap=0x10,
        target_px=(248, 361),
        radius=12,
        run_direction="right",
    ),
    Waypoint(
        tilemap=0x10,
        target_px=(348, 361),
        radius=12,
        run_direction="right",
    ),
    Waypoint(tilemap=0x10, target_px=(430, 345), radius=12),
    # North then east to lip. Avoid D0 building at tile(36,13): approach west
    # at x=569 → y=201, then run east along the A0 lip (human bath).
    Waypoint(tilemap=0x10, target_px=(433, 255), radius=12),
    Waypoint(tilemap=0x10, target_px=(529, 246), radius=12),
    Waypoint(tilemap=0x10, target_px=(569, 214), radius=10),
    Waypoint(tilemap=0x10, target_px=(569, 201), radius=8),
    Waypoint(
        tilemap=0x10,
        target_px=(619, 201),
        radius=10,
        run_direction="right",
    ),
]

_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA: List[Waypoint] = (
    list(_MOUNTAIN_ENTRY_APPROACH) + list(_FISH_TO_OUTDOOR_SPA)
)

# Spa lip → reverse bath path → south exit (for return_to_farm).
# Drop run_direction on reverse (east runs become west walks via BFS/hops).
# Start west of lip so post-bath water stand does not block the first hop.
_OUTDOOR_SPA_TO_MOUNTAIN_EXIT: List[Waypoint] = [
    Waypoint(
        tilemap=0x10,
        target_px=(569, 201),
        radius=14,
        run_direction="left",
    ),
    Waypoint(tilemap=0x10, target_px=(569, 214), radius=12),
    Waypoint(tilemap=0x10, target_px=(529, 246), radius=12),
    # South on the x≈433 ridge (human climb column), then SW into mid corridor.
    Waypoint(tilemap=0x10, target_px=(433, 255), radius=12),
    Waypoint(tilemap=0x10, target_px=(433, 300), radius=12),
    Waypoint(tilemap=0x10, target_px=(433, 345), radius=12),
    Waypoint(tilemap=0x10, target_px=(416, 345), radius=12),
    Waypoint(tilemap=0x10, target_px=(372, 347), radius=12),
    Waypoint(tilemap=0x10, target_px=(348, 361), radius=12),
    Waypoint(
        tilemap=0x10,
        target_px=(280, 361),
        radius=12,
        run_direction="left",
    ),
    Waypoint(
        tilemap=0x10,
        target_px=(248, 361),
        radius=12,
        run_direction="left",
    ),
    Waypoint(
        tilemap=0x10,
        target_px=(148, 361),
        radius=12,
        run_direction="left",
    ),
    Waypoint(tilemap=0x10, target_px=(70, 361), radius=10),
    Waypoint(tilemap=0x10, target_px=(70, 413), radius=10),
    Waypoint(tilemap=0x10, target_px=(103, 442), radius=12),
    Waypoint(tilemap=0x10, target_px=(185, 456), radius=12),
    Waypoint(tilemap=0x10, target_px=(280, 470), radius=12),
    Waypoint(tilemap=0x10, target_px=(380, 470), radius=12),
    Waypoint(tilemap=0x10, target_px=(480, 468), radius=12),
    Waypoint(tilemap=0x10, target_px=(560, 454), radius=12),
    Waypoint(tilemap=0x10, target_px=(640, 428), radius=12),
    Waypoint(tilemap=0x10, target_px=(686, 430), radius=12),
    Waypoint(tilemap=0x10, target_px=(680, 520), radius=12),
    Waypoint(tilemap=0x10, target_px=(640, 620), radius=12),
    Waypoint(tilemap=0x10, target_px=(580, 680), radius=12),
    Waypoint(tilemap=0x10, target_px=(496, 708), radius=12),
    Waypoint(tilemap=0x10, target_px=(420, 713), radius=12),
    Waypoint(tilemap=0x10, target_px=(328, 718), radius=12),
    Waypoint(
        tilemap=0x10,
        target_px=(312, 744),
        radius=16,
        is_exit=True,
        exit_direction="down",
    ),
]

# Path 0x0C is the farm / town / mountain fork. Keep these hops named so
# town and mountain share farm→crossroads instead of forking the farm exit.
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
_PATH_FARM_EXIT = Waypoint(
    tilemap=0x0C,
    target_px=(244, 128),
    # Live return crosses the right boundary around y≈102 before reaching the
    # nominal y=128 center.  A wider arrival radius arms exit_walk first.
    radius=32,
    is_exit=True,
    exit_direction="right",
    exit_push_frames=18,
)

# L1 house front ~(136,344) BFS-cuts the NW ledge toward the west
# gate. Drop south first (farm_to_shed stand), then the F0-door dirt
# at gate y, then west — not the house NW corner or the paddock
# north fence at ~(80,392).
_FARM_TO_PATH: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(137, 375), radius=12),
    Waypoint(tilemap=0x00, target_px=(136, 424), radius=12),
    _FARM_WEST_EXIT,
    _PATH_FARM_GATE,
    _PATH_CROSSROADS,
]
_PATH_TO_TOWN: List[Waypoint] = [_PATH_TOWN_EXIT]
# Already on 0x0C: plaza center first, then the north exit. Do not
# BFS-diagonal from the east landing / leaked farm y toward (132,30).
_PATH_TO_MOUNTAIN: List[Waypoint] = [_PATH_CROSSROADS, _PATH_MOUNTAIN_EXIT]
# Seed-shop town gate: stand on the open face, not (0,8)/(1,8) doorframe hugs.
_PATH_TO_TOWN_SHOP: List[Waypoint] = [
    Waypoint(
        tilemap=0x0C,
        target_px=(40, 128),
        radius=8,
        is_exit=True,
        exit_direction="left",
    ),
]
# shop_door landmark tile (37,13) → (600,216). Do not stand on (37,14).
_SHOP_DOOR_PX = (37 * 16 + 8, 13 * 16 + 8)
_TOWN_TO_SHOP_DOOR: List[Waypoint] = [
    # buy_potato_seeds_d2: east road is sealed west of ~x=684. Open face
    # just inside the gate (not 46,26 / 47,26 hugs), north at x=684, west
    # on y=288, then the shop column to shop_door (37,13).
    Waypoint(tilemap=0x04, target_px=(728, 424), radius=18),
    Waypoint(tilemap=0x04, target_px=(692, 424), radius=14, run_direction="left"),
    Waypoint(tilemap=0x04, target_px=(684, 400), radius=14, run_direction="up"),
    Waypoint(tilemap=0x04, target_px=(684, 340), radius=14, run_direction="up"),
    Waypoint(tilemap=0x04, target_px=(684, 288), radius=10),
    Waypoint(tilemap=0x04, target_px=(640, 288), radius=12),
    Waypoint(tilemap=0x04, target_px=(608, 288), radius=10),
    # Tape stand south of the shop (37,17). Walk up through the door
    # from here — a (601,246) stand pins on the porch.
    Waypoint(
        tilemap=0x04,
        target_px=(602, 274),
        radius=10,
        is_exit=True,
        exit_direction="up",
    ),
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
_TOWN_SHOP_TO_PATH: List[Waypoint] = [
    # Shop exit lands on the doorface ~(601,232). Step south to the plaza
    # first — standing on (37,14) and pathing east seals.
    Waypoint(tilemap=0x04, target_px=(602, 274), radius=12),
    Waypoint(tilemap=0x04, target_px=(608, 288), radius=12),
    Waypoint(tilemap=0x04, target_px=(640, 288), radius=14, run_direction="right"),
    Waypoint(tilemap=0x04, target_px=(684, 288), radius=12),
    Waypoint(tilemap=0x04, target_px=(684, 340), radius=14, run_direction="down"),
    Waypoint(tilemap=0x04, target_px=(684, 400), radius=14, run_direction="down"),
    Waypoint(tilemap=0x04, target_px=(692, 424), radius=14),
    Waypoint(
        tilemap=0x04,
        target_px=(728, 424),
        radius=12,
        is_exit=True,
        exit_direction="right",
    ),
]
_FARM_TO_MOUNTAIN_GATE: List[Waypoint] = list(_FARM_TO_PATH) + list(_PATH_TO_MOUNTAIN)

# mountain_grape_stand tape (Y1_Inside_House): transition settles on the
# south dirt ~(328, 728) tile (20, 45). A cliff blocks due-north; the only
# recorded gap is east to (32, 44) then north to (32, 39) (carpenter
# corridor — walk through, never A / never step onto Gotz). Then west to
# x=20 and north to the stand (326, 409) tile (20, 25). A + Don't eat.
_MOUNTAIN_ENTRY_TO_FIRST_BERRY: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(328, 728), radius=20),
    Waypoint(tilemap=0x10, target_px=(424, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(520, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(520, 632), radius=16),
    Waypoint(tilemap=0x10, target_px=(472, 600), radius=16),
    Waypoint(tilemap=0x10, target_px=(392, 568), radius=16),
    Waypoint(tilemap=0x10, target_px=(328, 568), radius=16),
    Waypoint(tilemap=0x10, target_px=(328, 488), radius=8),
    # Short x=19 wrap is NPC-blocked. Tape's long west loop: (20,30) →
    # (4,22) corridor → east to (19,22) → south to stand.
    Waypoint(tilemap=0x10, target_px=(240, 488), radius=10),
    Waypoint(tilemap=0x10, target_px=(192, 464), radius=10),
    Waypoint(tilemap=0x10, target_px=(144, 448), radius=10),
    Waypoint(tilemap=0x10, target_px=(80, 432), radius=10),
    Waypoint(tilemap=0x10, target_px=(72, 368), radius=10),
    Waypoint(tilemap=0x10, target_px=(168, 360), radius=12),
    Waypoint(tilemap=0x10, target_px=(312, 360), radius=12),
    Waypoint(tilemap=0x10, target_px=(326, 409), radius=10),
]

# First ground grape → south mountain exit. The inbound corridor cannot
# climb the x=20 cliff under the grape, but the return can jump it onto
# the mid terrace ~(328, 568). A second cliff still blocks due-south from
# that ledge to the land tile, so finish on the recorded east/south dirt.
# Do not reuse the outdoor-spa return here (wrong ridge).
_FIRST_BERRY_TO_MOUNTAIN_EXIT: List[Waypoint] = [
    Waypoint(tilemap=0x10, target_px=(326, 409), radius=10),
    Waypoint(
        tilemap=0x10,
        target_px=(328, 568),
        radius=24,
        run_direction="down",
        force_run=True,
    ),
    Waypoint(tilemap=0x10, target_px=(392, 568), radius=16),
    Waypoint(tilemap=0x10, target_px=(472, 600), radius=16),
    Waypoint(tilemap=0x10, target_px=(520, 632), radius=16),
    Waypoint(tilemap=0x10, target_px=(520, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(424, 712), radius=16),
    Waypoint(tilemap=0x10, target_px=(328, 728), radius=20),
    Waypoint(
        tilemap=0x10,
        target_px=(312, 744),
        radius=16,
        is_exit=True,
        exit_direction="down",
    ),
]

# The bin is the F2 pocket at x=8–9/y=29–30.  From the west-gate entry the
# north stand (8,28), facing down, is reachable without clearing cargo.  The
# historical forage route's (61,60) is ordinary ground.
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
    # Plaza first so a mountain-north or leaked landing does not hug
    # the east fence on the way to the farm exit.
    "path_to_farm": [_PATH_CROSSROADS, _PATH_FARM_EXIT],
    "town_to_shop_door": list(_TOWN_TO_SHOP_DOOR),
    "shop_to_counter": list(_SHOP_TO_COUNTER),
    "shop_to_town": list(_SHOP_TO_TOWN),
    "town_shop_to_path": list(_TOWN_SHOP_TO_PATH),
    "mountain_entry_to_first_berry": list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY),
    "first_berry_to_mountain_exit": list(_FIRST_BERRY_TO_MOUNTAIN_EXIT),
    "farm_west_gate_to_shipping_bin": list(_FARM_WEST_GATE_TO_SHIPPING_BIN),
}


# Berry ship route: farm-only (discovered via ship_berry recording analysis)
# Berry bush at tile(36,57) ~px(585,920), shipping bin at tile(62,60) ~px(1001,969)
# The recording picks berry with A at (585,920) and ships at (1001,969).
ROUTES: Dict[str, List[Waypoint]] = {
    "farm_to_house": [
        # Base farmhouse entry recording starts here and transitions by walking
        # north. Keep this as map data so future remodel coordinates have one
        # place to change.
        Waypoint(tilemap=0x00, target_px=(136, 424), radius=12),
    ],
    "farm_to_house_level1": [
        # First remodel shifts the exterior threshold upward on the farm map.
        Waypoint(tilemap=0x00, target_px=(136, 344), radius=12),
    ],
    "farm_to_house_level2": [
        # Provisional until the second-remodel save is recorded.
        Waypoint(tilemap=0x00, target_px=(136, 344), radius=12),
    ],
    "farm_to_path": list(_FARM_TO_PATH),
    "path_to_town": list(_PATH_TO_TOWN),
    "path_to_mountain": list(_PATH_TO_MOUNTAIN),
    # Shared farm→crossroads + west exit. Same hops as farm_to_path + path_to_town.
    "farm_to_town": list(_FARM_TO_PATH) + list(_PATH_TO_TOWN),
    "go_to_town": list(_FARM_TO_PATH) + list(_PATH_TO_TOWN),
    "farm_to_shop_door": list(_FARM_TO_PATH)
    + list(_PATH_TO_TOWN_SHOP)
    + list(_TOWN_TO_SHOP_DOOR),
    "shop_to_farm": list(_SHOP_TO_TOWN)
    + list(_TOWN_SHOP_TO_PATH)
    + [_PATH_FARM_EXIT],
    "mountain_entry_to_first_berry": list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY),
    "farm_to_first_mountain_berry": list(_FARM_TO_MOUNTAIN_GATE)
    + list(_MOUNTAIN_ENTRY_TO_FIRST_BERRY),
    "first_mountain_berry_to_shipping_bin": list(_FIRST_BERRY_TO_MOUNTAIN_EXIT)
    + list(SEGMENTS["path_to_farm"])
    + [Waypoint(tilemap=0x00, target_px=(80, 424), radius=12)]
    + list(_FARM_WEST_GATE_TO_SHIPPING_BIN),
    # Early-game town loop: enter town, touch shop + church fronts, then leave.
    # Completing this route is the planner's "ready to go home" signal on day 1.
    "town_explore": [
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x0C, target_px=(10, 128), radius=8, is_exit=True, exit_direction="left"),
        # Town entry lands near the east gate; walk west toward the square.
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=20),
        # Seed shop front (shop_door landmark ~tile 37,13 → px ~600,216).
        Waypoint(tilemap=0x04, target_px=(600, 230), radius=20),
        # Church plaza approach (church_door ~tile 23,8 → px ~375,140).
        Waypoint(tilemap=0x04, target_px=(375, 200), radius=24),
        # Return to east gate for farm exit.
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=20),
        Waypoint(tilemap=0x04, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
    "town_to_church": [
        Waypoint(tilemap=0x04, target_px=(232, 128), radius=16),
        Waypoint(tilemap=0x04, target_px=(375, 139), radius=10, is_exit=True, exit_direction="up"),
    ],
    "farm_to_church": [
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x0C, target_px=(10, 128), radius=8, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x04, target_px=(232, 128), radius=16),
        Waypoint(tilemap=0x04, target_px=(375, 139), radius=10, is_exit=True, exit_direction="up"),
    ],
    "town_to_animal_shop": [
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=16),
        Waypoint(tilemap=0x04, target_px=(688, 888), radius=16, run_direction="down"),
        Waypoint(tilemap=0x04, target_px=(601, 888), radius=12),
        Waypoint(tilemap=0x04, target_px=(601, 874), radius=12, is_exit=True, exit_direction="up"),
    ],
    "animal_shop_to_counter": [
        Waypoint(tilemap=0x24, target_px=(128, 200), radius=12, run_direction="up"),
        Waypoint(tilemap=0x24, target_px=(201, 158), radius=4),
    ],
    "farm_to_animal_shop_staging": [
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x0C, target_px=(10, 128), radius=8, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=16),
        Waypoint(tilemap=0x04, target_px=(688, 888), radius=16, run_direction="down"),
        Waypoint(tilemap=0x04, target_px=(601, 888), radius=12),
        Waypoint(tilemap=0x04, target_px=(601, 874), radius=12, is_exit=True, exit_direction="up"),
        Waypoint(tilemap=0x24, target_px=(128, 200), radius=12, run_direction="up"),
    ],
    "farm_to_animal_shop_counter": [
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x0C, target_px=(10, 128), radius=8, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=16),
        Waypoint(tilemap=0x04, target_px=(688, 888), radius=16, run_direction="down"),
        Waypoint(tilemap=0x04, target_px=(601, 888), radius=12),
        Waypoint(tilemap=0x04, target_px=(601, 874), radius=12, is_exit=True, exit_direction="up"),
        Waypoint(tilemap=0x24, target_px=(128, 200), radius=12, run_direction="up"),
        Waypoint(tilemap=0x24, target_px=(201, 158), radius=4),
    ],
    "farm_to_animal_shop_counter_sale": [
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x0C, target_px=(10, 128), radius=8, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x04, target_px=(688, 430), radius=16),
        Waypoint(tilemap=0x04, target_px=(688, 888), radius=16, run_direction="down"),
        Waypoint(tilemap=0x04, target_px=(601, 888), radius=12),
        Waypoint(tilemap=0x04, target_px=(601, 874), radius=12, is_exit=True, exit_direction="up"),
        Waypoint(tilemap=0x24, target_px=(128, 200), radius=12, run_direction="up"),
        # The sale menu replay expects the same pixel alignment as the
        # recording; a loose radius can leave the player too low/left.
        Waypoint(tilemap=0x24, target_px=(201, 158), radius=1),
    ],
    "animal_shop_to_town": [
        Waypoint(tilemap=0x24, target_px=(137, 200), radius=12),
        Waypoint(tilemap=0x24, target_px=(137, 212), radius=12, is_exit=True, exit_direction="down"),
        # After the transition the player resolves at the animal-shop door on
        # town. Step off the door tile before any idle wait or follow-up route.
        Waypoint(tilemap=0x04, target_px=(601, 904), radius=2, run_direction="down"),
    ],
    # ── Spring D1 town handoff (docs/town_day1_recon.md) ──
    # Natural entry lands at town gate ~(712,424). Routes assume start on 0x04.
    "d1_town_to_flower_shop": [
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(600, 262), radius=10, is_exit=True, exit_direction="up"),
        # Front-room settle near spawn ~(144,456)
        Waypoint(tilemap=0x1C, target_px=(144, 456), radius=20),
    ],
    "d1_flower_back_to_nina": [
        Waypoint(tilemap=0x1D, target_px=(104, 184), radius=16),
        Waypoint(tilemap=0x1D, target_px=(104, 120), radius=12),
        # town_day1_rest: bit 0x04 at ~(101,102) face left + A.
        Waypoint(tilemap=0x1D, target_px=(101, 102), radius=6),
    ],
    "d1_flower_back_exit_to_town": [
        Waypoint(tilemap=0x1D, target_px=(104, 184), radius=14),
        Waypoint(tilemap=0x1D, target_px=(104, 210), radius=12, is_exit=True, exit_direction="down"),
        Waypoint(tilemap=0x1C, target_px=(144, 456), radius=18),
        Waypoint(tilemap=0x1C, target_px=(144, 480), radius=12, is_exit=True, exit_direction="down"),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=16),
    ],
    # Enter only — stop at church door lip; scripted up enters 0x1B.
    # Corridor: stay on y≈280 until x≈376, then north — a direct (500,280)→
    # (411,216) hop stalls on the mid-plaza hard block (power-on composed path).
    "d1_town_to_maria": [
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(500, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(376, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(376, 200), radius=14),
        # town_day1_rest church door approach ~(358,150); is_exit up is flaky
        # so talk sequence scripts the final up into 0x1B.
        Waypoint(tilemap=0x04, target_px=(358, 150), radius=8),
    ],
    "d1_church_to_maria": [
        Waypoint(tilemap=0x1B, target_px=(128, 456), radius=16),
        # town_day1_rest: bit 0x20 at ~(103,405) face up + A.
        Waypoint(tilemap=0x1B, target_px=(103, 405), radius=6),
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
        # Live-verified stand: face left at ~(388–392,914–924) sets bit 0x01.
        # Talk is done by PressAUntilBit after nav (not action_on_arrive) so
        # facing/mash can retry without multi_nav consuming the only A press.
        Waypoint(tilemap=0x04, target_px=(392, 914), radius=6),
    ],
    "d1_town_to_eve": [
        # From Ann stand ~(388,914): drop south, run west, stand below Eve.
        # Live Eve sprite ~ (152,872); stand below facing up.
        Waypoint(tilemap=0x04, target_px=(388, 950), radius=14, run_direction="down"),
        Waypoint(tilemap=0x04, target_px=(300, 950), radius=14),
        Waypoint(tilemap=0x04, target_px=(152, 950), radius=12),
        Waypoint(tilemap=0x04, target_px=(152, 896), radius=6),
    ],
    # Enter only — stop near dealer; scripted push to D1 event stand follows.
    "d1_town_to_livestock": [
        Waypoint(tilemap=0x04, target_px=(300, 950), radius=16),
        Waypoint(tilemap=0x04, target_px=(601, 950), radius=14),
        Waypoint(
            tilemap=0x04,
            target_px=(601, 888),
            radius=10,
            is_exit=True,
            exit_direction="up",
        ),
        Waypoint(tilemap=0x24, target_px=(128, 200), radius=14, run_direction="up"),
        # BFS-reachable approach. D1 bit needs ~(230,139) face down — scripted
        # after nav (counter blocks a pure BFS to that pixel).
        Waypoint(tilemap=0x24, target_px=(201, 154), radius=6),
    ],
    "d1_livestock_to_town": [
        Waypoint(tilemap=0x24, target_px=(137, 200), radius=12),
        Waypoint(tilemap=0x24, target_px=(137, 212), radius=10, is_exit=True, exit_direction="down"),
        # Clear the animal-shop door lip (door ~y888). Wider radius — south
        # road BFS often stalls past y~950.
        Waypoint(tilemap=0x04, target_px=(601, 940), radius=20, run_direction="down"),
    ],
    # Align with town_day1_rest truck slice start (f≈9200 @ ~(688,357)).
    # Slice then walks to stand (715,421), leave dialog, east path 0x0C → house.
    "d1_town_to_truck": [
        # Rest-slice lineup: recording f9200 walks the last yards to the shipper.
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(688, 360), radius=10),
    ],
    # Pure _TruckLeaveTask stand (town_day1_rest engage ~(715,421)).
    "d1_town_to_truck_stand": [
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(688, 360), radius=12),
        Waypoint(tilemap=0x04, target_px=(700, 400), radius=12),
        Waypoint(tilemap=0x04, target_px=(715, 421), radius=10),
    ],
    # Real opening gate is ~(712,424); east path exit still near x≈756.
    # Note: truck leave dialogue often cutscenes straight into the farmhouse
    # (path tilemap briefly shows house coords, then house 0x15) — see
    # tasks/town_day1_rest.json. d1_town_to_farm is for walking when no cutscene.
    "d1_town_to_farm": [
        Waypoint(tilemap=0x04, target_px=(728, 424), radius=16),
        Waypoint(tilemap=0x04, target_px=(756, 422), radius=12, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(200, 128), radius=14),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x00, target_px=(80, 424), radius=20),
    ],
    "berry_ship": [
        # OPEN_FENCE_GAP can finish on the soft-collision gap tile. BerryShipTask
        # first takes the verified east-past-wall/south charge, so this route
        # begins where that escape lands instead of walking back into the wall.
        # Escape commonly lands ~(28,32), west of the pond. Step down-left
        # around live weeds at (28,33)/(29,32), not east into pond water.
        Waypoint(tilemap=0x00, target_px=(27 * 16 + 8, 35 * 16 + 8), radius=16),
        Waypoint(tilemap=0x00, target_px=(28 * 16 + 8, 39 * 16 + 8), radius=14),
        # Live D2 has a 2x2 rock at ~(27–28,45–46). Detour west, descend,
        # then rejoin east instead of selecting the rock as a waypoint.
        Waypoint(tilemap=0x00, target_px=(25 * 16 + 8, 44 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(25 * 16 + 8, 50 * 16 + 8), radius=14),
        # North-of-bush approach (BFS-verified on Y1_Spring_D1_Farm). The bush
        # pocket at (36–37,56–57) is sealed south by weeds (36–37,58–59), west
        # by stones, east by a 2x2 rock. Enter via (36,54)→(36,56) — never
        # lift_throw the south weeds (that thrash-held debris and never picked).
        Waypoint(tilemap=0x00, target_px=(32 * 16 + 8, 51 * 16 + 8), radius=12),
        Waypoint(tilemap=0x00, target_px=(33 * 16 + 8, 53 * 16 + 8), radius=10),
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 54 * 16 + 8), radius=10),
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 56 * 16 + 8), radius=8),
        # Bush landmark (36,57); stand east and face left. Longer A hold for
        # forage interact (human ship_berry uses sustained A).
        Waypoint(
            tilemap=0x00,
            target_px=(37 * 16 + 8, 57 * 16 + 8),
            radius=8,
            action_on_arrive="press_a",
            action_face="left",
            action_frames=28,
            action_cooldown=36,
        ),
        # Exit north then east to the y=60 path → shipping bin.
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 54 * 16 + 8), radius=10),
        Waypoint(tilemap=0x00, target_px=(40 * 16 + 8, 54 * 16 + 8), radius=12),
        Waypoint(tilemap=0x00, target_px=(48 * 16 + 8, 58 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(55 * 16 + 8, 60 * 16 + 8), radius=14),
        Waypoint(
            tilemap=0x00,
            # Bin occupies (62,60); stand one tile west and face right.
            target_px=(61 * 16 + 8, 60 * 16 + 8),
            radius=10,
            action_on_arrive="press_a",
            action_face="right",
            action_frames=28,
            action_cooldown=36,
        ),
    ],
    "berry_ship_repeat": [
        # Second forage starts at the shipping bin. Retrace the north pocket
        # approach; never south-weed lift thrash.
        Waypoint(tilemap=0x00, target_px=(55 * 16 + 8, 60 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(48 * 16 + 8, 58 * 16 + 8), radius=14),
        Waypoint(tilemap=0x00, target_px=(40 * 16 + 8, 54 * 16 + 8), radius=12),
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 54 * 16 + 8), radius=10),
        Waypoint(tilemap=0x00, target_px=(36 * 16 + 8, 56 * 16 + 8), radius=8),
        Waypoint(
            tilemap=0x00,
            target_px=(37 * 16 + 8, 57 * 16 + 8),
            radius=8,
            action_on_arrive="press_a",
            action_face="left",
            action_frames=28,
            action_cooldown=36,
        ),
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
    ],
    "farm_south_to_west_gate": [
        # Shipping-bin / berry-field return. Prefer y=60 path west, stay south
        # of the long y=31 fence until x<=8, then step north via (6,33)→(4,30)
        # so MultNav never seals against house/cliff tiles at (6,32).
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
        # Morning route from the house frontage to the shed.  Match the
        # fix_rainy_day recording: run the upper-left path, drop along the shed
        # frontage's left side, then enter from below.  Approaching through the
        # shed's right/top corner clips on the building edge after the remodel.
        Waypoint(tilemap=0x00, target_px=(137, 375), radius=12),
        Waypoint(tilemap=0x00, target_px=(244, 375), radius=12),
        Waypoint(tilemap=0x00, target_px=(354, 377), radius=12),
        Waypoint(tilemap=0x00, target_px=(354, 489), radius=12),
        # Stop below the shed threshold; the ensure task owns the transition.
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "upper_farm_to_shed": [
        # From barn/coop frontage, stay on the right-side corridor. Reusing the
        # farmhouse route cuts left toward the well and can wedge on its body.
        Waypoint(tilemap=0x00, target_px=(456, 424), radius=16),
        Waypoint(tilemap=0x00, target_px=(456, 489), radius=12),
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "field_to_shed": [
        # From the harvest shipping stand, stay south of the well/stump at
        # y=504 then step to the door. A y=489 run_direction=right B-charges
        # the stump and left/right-thrashes (live D2 ENSURE_WATERING_CAN).
        Waypoint(tilemap=0x00, target_px=(344, 504), radius=12),
        Waypoint(tilemap=0x00, target_px=(400, 504), radius=12),
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "near_shed_to_shed": [
        # Stop on the threshold; DirectionalTransition owns the walk-in.
        # run_direction=up B-charged the jamb and clipped (live D2 shed door).
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12),
    ],
    "farm_to_coop": [
        # Route from the house frontage to the coop approach.  Use BFS here:
        # straight run shortcuts hit farm collision around the house/yard edge.
        Waypoint(tilemap=0x00, target_px=(137, 375), radius=12),
        # Coop door triggers at (454, 346); stop just south of it.
        Waypoint(tilemap=0x00, target_px=(454, 360), radius=12),
    ],
    "farm_to_coop_sale": [
        # Chicken sale starts from active farm work or the sale drop point, not
        # the farmhouse door. Stay on the right-side corridor to avoid a
        # leftward detour before entering the coop.
        Waypoint(tilemap=0x00, target_px=(456, 424), radius=16),
        Waypoint(tilemap=0x00, target_px=(454, 360), radius=12),
    ],
    "barn_to_coop": [
        # After exiting the barn the player is already aligned with the coop
        # approach corridor; skip the farmhouse waypoint but still let BFS steer
        # around the collision just east of the barn door.
        Waypoint(tilemap=0x00, target_px=(454, 360), radius=12),
    ],
    "farm_to_barn": [
        # Morning route from the house frontage to the barn door approach.
        # Keep hops short enough for viewport-limited BFS.
        # Start south of the farmhouse threshold; targeting north from the
        # stabilized exit can walk back into the house.
        Waypoint(tilemap=0x00, target_px=(137, 375), radius=16),
        Waypoint(tilemap=0x00, target_px=(244, 375), radius=16),
        # Stop just below the barn threshold.  ENTER_BARN handles the transition.
        Waypoint(tilemap=0x00, target_px=(329, 360), radius=18),
    ],
    "path_to_farm": [_PATH_CROSSROADS, _PATH_FARM_EXIT],
    "farm_to_mountain": list(_FARM_TO_MOUNTAIN_GATE),
    # Mountain entry (south) → upper outdoor hot spring (0xF7 pond).
    # Path: SE bottom → fish area → west mid y~470 → west climb → east mid
    # y~361 → lip ~(619,201) tile(38,12). Short hops for viewport BFS.
    "mountain_entry_to_outdoor_spa": list(_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA),
    # Alias — same upper pond (not west cave door).
    "mountain_entry_to_spa": list(_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA),
    # From fish/camp stand (mountain_fish_power_berry_end) into bath path.
    "fish_spot_to_outdoor_spa": list(_FISH_TO_OUTDOOR_SPA),
    # Historical west-cave approach (sunday blue-feather path). Not for soak.
    "mountain_entry_to_cave": [
        Waypoint(tilemap=0x10, target_px=(328, 720), radius=22),
        Waypoint(tilemap=0x10, target_px=(420, 713), radius=20),
        Waypoint(tilemap=0x10, target_px=(518, 690), radius=18),
        Waypoint(tilemap=0x10, target_px=(500, 600), radius=18),
        Waypoint(tilemap=0x10, target_px=(400, 540), radius=18),
        Waypoint(tilemap=0x10, target_px=(280, 490), radius=18),
        Waypoint(tilemap=0x10, target_px=(180, 460), radius=16),
        Waypoint(tilemap=0x10, target_px=(146, 430), radius=14),
        Waypoint(tilemap=0x10, target_px=(166, 411), radius=12),
    ],
    # Full farm → upper outdoor spa pond (map 0x10; season-stable tilemap).
    "farm_to_spa": list(_FARM_TO_MOUNTAIN_GATE) + list(_MOUNTAIN_ENTRY_TO_OUTDOOR_SPA),
    "mountain_entry_to_fish_power_berry_spots": [
        Waypoint(tilemap=0x10, target_px=(328, 718), radius=16),
        Waypoint(tilemap=0x10, target_px=(496, 708), radius=16),
        Waypoint(tilemap=0x10, target_px=(518, 558), radius=16),
        Waypoint(tilemap=0x10, target_px=(582, 414), radius=16),
        Waypoint(tilemap=0x10, target_px=(624, 371), radius=12),
        Waypoint(tilemap=0x10, target_px=(686, 411), radius=12),
    ],
    # Spa lip / mid-mountain → reverse corridor → south exit → farm.
    "outdoor_spa_to_farm": list(_OUTDOOR_SPA_TO_MOUNTAIN_EXIT)
    + [
        Waypoint(
            tilemap=0x0C,
            target_px=(244, 128),
            radius=16,
            is_exit=True,
            exit_direction="right",
        ),
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=24),
    ],
    "mountain_to_farm": list(_OUTDOOR_SPA_TO_MOUNTAIN_EXIT)
    + [
        # Path→farm: accept arrival on path stand or farm just past the door
        # (exit walk often lands already on 0x00).
        Waypoint(
            tilemap=0x0C,
            target_px=(244, 128),
            radius=16,
            is_exit=True,
            exit_direction="right",
        ),
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=24),
    ],
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
    "church_to_town": [
        Waypoint(tilemap=0x1B, target_px=(130, 468), radius=12, is_exit=True, exit_direction="down"),
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
        # The chicken seller pickup sequence in sell_chicken.json re-enters
        # at the farm west gate and approaches along y=448.  The generic
        # path-to-farm exit at y=128 lands on the upper path instead.
        Waypoint(tilemap=0x0C, target_px=(230, 118), radius=6),
        Waypoint(tilemap=0x0C, target_px=(244, 118), radius=4, is_exit=True, exit_direction="right"),
    ],
    "event_town_to_farm": [
        Waypoint(tilemap=0x05, target_px=(756, 422), radius=16, is_exit=True, exit_direction="right"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
}
