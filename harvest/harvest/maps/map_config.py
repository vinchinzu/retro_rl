"""Map configuration registry — tilemap IDs, walkable tiles, exits, and routes.

Pure-data module. No imports from farm_clearer (avoids circular deps).
Walkable tile sets are duplicated here intentionally.

Tilemap IDs discovered via map_discovery.py (buy_potato_seeds replay):
  0x00 = Farm
  0x0C = Path (crossroads between farm and town)
  0x04 = Town
  0x1C = Shop interior
  0x24 = Animal shop interior

Transition positions (pixel coords):
  Farm(0x00) --left--> Path(0x0C)  exit=(11,424)  entry=(11,424)
  Path(0x0C) --left--> Town(0x04)  exit=(10,128)  entry=(10,128)
  Town(0x04) --enter-> Shop(0x1C)  exit=(601,218) entry=(601,218)
  Shop(0x1C) --exit--> Town(0x04)  exit=(138,468) entry=(138,468)
  Town(0x04) --right-> Path(0x0C)  exit=(756,422) entry=(756,422)
  Path(0x0C) --right-> Farm(0x00)  exit=(244,128) entry=(244,128)
"""

from typing import Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple

from harvest.core.tile_catalog import (
    CHURCH_WALKABLE,
    COOP_WALKABLE,
    FARM_WALKABLE,
    MOUNTAIN_WALKABLE,
    PATH_WALKABLE,
    SHOP_WALKABLE,
    TILE_SIZE,
    TOWN_WALKABLE,
)

INTERIOR_WALKABLE = SHOP_WALKABLE | COOP_WALKABLE


class MapExit(NamedTuple):
    region: Tuple[int, int, int, int]  # tile bbox (x1, y1, x2, y2) near exit
    direction: str                      # walk direction to trigger transition
    dest_tilemap: int                   # tilemap you arrive at


class MapLandmark(NamedTuple):
    name: str
    tile: Tuple[int, int]
    kind: str
    face: Optional[str] = None
    action: Optional[str] = None
    source: str = "recorded"
    note: str = ""

    @property
    def target_px(self) -> Tuple[int, int]:
        return (self.tile[0] * 16 + 8, self.tile[1] * 16 + 8)


class MapConfig(NamedTuple):
    name: str
    walkable_tiles: FrozenSet[int]
    exits: List[MapExit]
    landmarks: Tuple[MapLandmark, ...] = ()
    source: str = "recorded"


class Waypoint(NamedTuple):
    tilemap: int                            # expected tilemap
    target_px: Tuple[int, int]              # pixel target on this map
    radius: int = 12                        # arrival tolerance
    action_on_arrive: Optional[str] = None  # "press_a" etc, or None
    action_face: Optional[str] = None       # direction to face for action
    action_frames: int = 30
    action_cooldown: int = 60
    is_exit: bool = False                   # True = walk off map after arriving
    exit_direction: Optional[str] = None
    run_direction: Optional[str] = None     # "up"/"down"/"left"/"right" — skip BFS, just run


# Farm coordinates whose visual/collision behavior is not represented by the
# metatile alone.  Several well-body tiles render as 0xA1, which is walkable in
# other farm contexts, so keep the coordinate-specific facts here.
FARM_NO_GO_TILES: FrozenSet[Tuple[int, int]] = frozenset({
    # Shipping-bin ditch / F2 water fringe (does NOT refill the can).
    (9, 26), (9, 27), (9, 28),
    (11, 26), (11, 27), (11, 28),
    # House frontage.
    (8, 12), (9, 12), (10, 12),
    # Well body.  These are visually solid even when the live tile ID is 0xA1.
    (15, 26), (16, 26), (17, 26),
    (15, 27), (16, 27), (17, 27),
})

# ── Farm water sources (ROM-mapped, spring farm tilemap 0x00) ──────────────
# CheckToolSuccess farm fill only when tile-in-front *property* is F0/F9–FD
# (can → 0x14). Raw F1/F2/F7/F8 look like water but do not fill.
#
# Main pond (F0) water cells ~(31–34, 31–33). Human refill stand (go_to_water_source):
# south lip (32,34)/(33,34) face up. North lip (33,30) face down also fills.
#
# Early-spring west plant pocket (y≤30, x≈10–25) is cut off from the pond by a
# solid 0x05 fence wall on y=31 (x=11–29). Clearing ≥1 fence opens full BFS to
# F0/FC/FD/FB. Until then only non-fill F1/F8 north stream is pathable.
FARM_MAIN_POND_WATER_BOUNDS: Tuple[int, int, int, int] = (31, 31, 34, 33)
FARM_MAIN_POND_STANDS: Tuple[Tuple[Tuple[int, int], str], ...] = (
    ((32, 34), "up"),    # human go_to_water_source_end
    ((33, 34), "up"),
    ((33, 30), "down"),  # Y1_Near_Pond (ROM fill 0→20)
    ((34, 30), "down"),
    ((32, 30), "down"),
)
# Fence row that walls west field off from the main pond / south farm.
FARM_POND_ACCESS_FENCE_ROW: int = 31
FARM_POND_ACCESS_FENCE_X_RANGE: Tuple[int, int] = (11, 29)

# Named water pockets: (name, water_tile_id, fills_can, sample_cells)
FARM_WATER_POCKETS: Tuple[Tuple[str, int, bool, Tuple[Tuple[int, int], ...]], ...] = (
    ("main_pond", 0xF0, True, ((32, 32), (33, 32), (31, 32), (34, 32))),
    ("north_stream_f1", 0xF1, False, ((13, 16), (14, 16))),
    ("shipping_ditch_f2", 0xF2, False, ((8, 29), (9, 29), (8, 30), (9, 30))),
    ("north_pool_f7", 0xF7, False, ((24, 5), (26, 5), (28, 6))),
    ("north_stream_f8", 0xF8, False, ((18, 22),)),
    ("north_spur_f9", 0xF9, True, ((26, 12), (26, 13))),
    ("east_spur_fa", 0xFA, True, ((46, 14), (46, 15))),
    ("east_spur_fb", 0xFB, True, ((49, 36), (49, 37))),
    ("south_stream_fc", 0xFC, True, ((14, 49), (14, 50))),
    ("southeast_fd", 0xFD, True, ((41, 54), (41, 55))),
)

FARM_TILEMAP_IDS: FrozenSet[int] = frozenset({0x00, 0x01, 0x02, 0x03})
FARM_TILEMAP_NAMES: Dict[int, str] = {
    0x00: "farm",
    0x01: "farm_summer",
    0x02: "farm_fall",
    0x03: "farm_winter",
}

NO_GO_TILES_BY_TILEMAP: Dict[int, FrozenSet[Tuple[int, int]]] = {
    tilemap_id: FARM_NO_GO_TILES for tilemap_id in FARM_TILEMAP_IDS
}


# ── Map Registry ──

MAP_REGISTRY: Dict[int, MapConfig] = {
    0x00: MapConfig(
        name="farm",
        walkable_tiles=FARM_WALKABLE,
        exits=[
            # Exit left → path crossroads. Player at ~px(11,424) tile(0,26)
            MapExit(region=(0, 24, 0, 28), direction="left", dest_tilemap=0x0C),
        ],
        landmarks=(
            MapLandmark("house_door", (8, 26), "door", face="up", source="recorded"),
            MapLandmark("shed_door", (26, 30), "door", face="up", source="recorded"),
            MapLandmark("coop_door", (28, 21), "door", face="up", source="recorded"),
            MapLandmark("barn_door", (19, 21), "door", face="up", source="state_name"),
            MapLandmark("shipping_bin", (62, 60), "shipping_bin", face="right", action="press_a", source="recorded"),
            MapLandmark("berry_bush", (36, 57), "forage", face="left", action="press_a", source="recorded"),
            # Primary can-refill stand (main F0 pond south lip). Verified:
            # go_to_water_source_end @ (32,34) can=20; Y1_Near_Pond north lip also fills.
            MapLandmark(
                "pond_edge",
                (32, 34),
                "water_source",
                face="up",
                action="use_tool",
                source="go_to_water_source_recording",
                note="Main F0 pond south lip; CheckToolSuccess fill property F0",
            ),
            MapLandmark(
                "pond_edge_north",
                (33, 30),
                "water_source",
                face="down",
                action="use_tool",
                source="rom_probe",
                note="Main F0 pond north lip; ROM fill 0→20",
            ),
            # Non-fill water (do not use for watering-can refill).
            MapLandmark(
                "shipping_ditch",
                (9, 28),
                "water_nonfill",
                face="down",
                source="rom_probe",
                note="F2 shipping pocket — never refills can",
            ),
            MapLandmark(
                "north_stream",
                (18, 23),
                "water_nonfill",
                face="up",
                source="rom_probe",
                note="F8 stream — pathable from west pocket but does not fill",
            ),
            MapLandmark("field_origin", (3, 34), "crop_field", source="state_diff"),
        ),
    ),
    0x0C: MapConfig(
        name="path",
        walkable_tiles=PATH_WALKABLE,
        exits=[
            # Exit left → town. Player at ~px(10,128) tile(0,8)
            MapExit(region=(0, 0, 2, 10), direction="left", dest_tilemap=0x04),
            # Exit up/right → mountain spring. Seen in get_berry replay at tile(8,0).
            MapExit(region=(6, 0, 10, 2), direction="up", dest_tilemap=0x10),
            # Exit right → farm. Player at ~px(244,128) tile(15,8)
            MapExit(region=(14, 6, 16, 10), direction="right", dest_tilemap=0x00),
        ],
        landmarks=(
            MapLandmark("farm_gate", (15, 8), "exit", source="recorded"),
            MapLandmark("town_gate", (0, 8), "exit", source="recorded"),
            MapLandmark("mountain_gate", (8, 0), "exit", source="recorded"),
        ),
    ),
    0x04: MapConfig(
        name="town",
        walkable_tiles=TOWN_WALKABLE,
        exits=[
            # Exit right → path. Player at ~px(756,422) tile(47,26)
            MapExit(region=(45, 24, 48, 28), direction="right", dest_tilemap=0x0C),
        ],
        landmarks=(
            MapLandmark("shop_door", (37, 13), "door", face="up", action="press_a", source="buy_potato_replay"),
            MapLandmark("church_door", (23, 8), "door", face="up", action="press_a", source="sunday_go_to_church_replay"),
            MapLandmark("animal_shop_door", (37, 54), "door", face="up", source="buy_cow_replay"),
        ),
    ),
    0x05: MapConfig(
        name="town_event",
        walkable_tiles=TOWN_WALKABLE,
        exits=[
            # Event variant seen when leaving the house on Summer 10.  The
            # church exterior layout matches town well enough to return home.
            MapExit(region=(45, 24, 48, 28), direction="right", dest_tilemap=0x0C),
        ],
        landmarks=(
            MapLandmark("church_door", (23, 8), "door", face="up", action="press_a", source="summer10_autoplay_diagnostic"),
            MapLandmark("animal_shop_door", (37, 54), "door", face="up", source="summer10_autoplay_diagnostic"),
        ),
        source="runtime_diagnostic",
    ),
    0x10: MapConfig(
        name="mountain_spring",
        walkable_tiles=MOUNTAIN_WALKABLE,
        exits=[
            # Recorded get_berry replay returns to path at tile(19,46).
            MapExit(region=(18, 44, 20, 48), direction="down", dest_tilemap=0x0C),
            # West cave hole → MapMountainCave 0x29 (NOT the hot spring).
            # Sunday replay enters at tile(10,25) walking up from px~(166,411).
            MapExit(region=(9, 23, 12, 26), direction="up", dest_tilemap=0x29),
        ],
        landmarks=(
            MapLandmark("mountain_entry", (19, 46), "exit", source="get_berry_replay"),
            MapLandmark("west_stump", (20, 25), "stump", face="down", action="use_axe", source="get_berry_replay"),
            MapLandmark("east_stump", (41, 13), "stump", face="right", action="use_axe", source="get_berry_replay"),
            # True outdoor hot spring (hot_spring_bath recording): upper pond
            # water tile 0xF7 at (39,12); stand A0 (38,12) ~(619,201); A+dir.
            # Stamina 100→130 verified. Not camp tent pond, not cave 0x29.
            MapLandmark(
                "spa_outdoor_pond",
                (38, 12),
                "hot_spring",
                face="right",
                action="press_a",
                source="hot_spring_bath_recording",
                note="A0 lip ~(619,201); water 0xF7 at (39,12); A+right into pond",
            ),
            MapLandmark(
                "spa_water",
                (39, 12),
                "hot_spring",
                face="right",
                action="press_a",
                source="hot_spring_bath_recording",
                note="tile id 0xF7; player_action=3 while crossing",
            ),
            MapLandmark(
                "spa_camp_pond",
                (43, 26),
                "pond",
                face="up",
                source="spa_outdoor_recon / spa_f0_probe",
                note="WRONG pond (tent/F0); no stam restore — do not use for soak",
            ),
            MapLandmark(
                "cave_door",
                (10, 25),
                "cave",
                face="up",
                source="sunday_go_to_mountain_replay",
                note="west hole → 0x29 MapMountainCave; NOT hot spring",
            ),
            # Legacy aliases kept so older routes/docs resolve.
            MapLandmark(
                "spa_door",
                (10, 25),
                "cave",
                face="up",
                source="sunday_go_to_mountain_replay",
                note="deprecated alias of cave_door; do not use for stamina soak",
            ),
            MapLandmark(
                "spa_area",
                (38, 12),
                "hot_spring",
                source="hot_spring_bath_recording",
                note="alias of spa_outdoor_pond (upper lip)",
            ),
            MapLandmark("fish_power_berry_cast_spot", (39, 23), "fishing_spot", face="right", action="use_tool", source="mountain_fish_power_berry_replay"),
            MapLandmark("fish_power_berry_throw_spot", (42, 25), "waterfall_pool", face="up", action="press_a", source="mountain_fish_power_berry_replay"),
        ),
        source="recorded_provisional",
    ),
    0x29: MapConfig(
        name="mountain_cave",
        walkable_tiles=INTERIOR_WALKABLE,
        exits=[
            # Walk down from cave stand to return to mountain.
            MapExit(region=(20, 8, 26, 12), direction="down", dest_tilemap=0x10),
        ],
        landmarks=(
            MapLandmark(
                "cave_stand",
                (23, 8),
                "cave",
                source="sunday_go_to_mountain_replay",
                note="auto-relocate ~90f after entry at ~(376,128); not spa",
            ),
            MapLandmark(
                "right_lip",
                (26, 7),
                "cave",
                face="right",
                source="sunday_go_to_mountain_replay + 2026-07-31 recon",
                note="A1→A0 lip ~(416,121); B runs not jumps; not hot spring",
            ),
        ),
        source="recorded_provisional",
    ),
    0x15: MapConfig(
        name="house",
        walkable_tiles=INTERIOR_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark("bed_stand", (4, 5), "bed", face="up", action="press_a", source="go_to_sleep_replay"),
            MapLandmark("house_exit_inside", (8, 12), "door", face="down", source="leave_house_replay"),
        ),
    ),
    0x16: MapConfig(
        name="house_level1",
        walkable_tiles=INTERIOR_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark("bed_stand", (4, 5), "bed", face="up", action="press_a", source="provisional_same_as_base"),
            MapLandmark("house_exit_inside", (8, 12), "door", face="down", source="verified_level1_exit"),
        ),
        source="recorded_provisional",
    ),
    0x17: MapConfig(
        name="house_level2",
        walkable_tiles=INTERIOR_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark("bed_stand", (18, 6), "bed", face="up", action="press_a", source="l2_house_wife_bed_replay"),
            MapLandmark("house_exit_inside", (8, 12), "door", face="down", source="l2_house_wife_bed_replay"),
        ),
        source="recorded_provisional",
    ),
    0x1B: MapConfig(
        name="church",
        walkable_tiles=CHURCH_WALKABLE,
        exits=[
            MapExit(region=(7, 27, 9, 30), direction="down", dest_tilemap=0x04),
        ],
        landmarks=(
            MapLandmark("church_exit_inside", (8, 29), "door", face="down", source="sunday_go_to_church_replay"),
            MapLandmark(
                "church_ann_question_stand",
                (12, 25),
                "npc_talk",
                face="right",
                action="press_a",
                source="sunday_go_to_church_replay",
                note="Text 0x00A2 -> 0x00A3, recorded Yes, Ann hearts +4.",
            ),
            MapLandmark("church_talk_southwest", (5, 25), "npc_talk", face="left", action="press_a", source="sunday_go_to_church_replay", note="Text 0x00D1."),
            MapLandmark("church_talk_west_middle", (5, 21), "npc_talk", face="left", action="press_a", source="sunday_go_to_church_replay", note="Text 0x004E."),
            MapLandmark("church_talk_west_front", (6, 17), "npc_talk", face="left", action="press_a", source="sunday_go_to_church_replay", note="Text 0x0037."),
            MapLandmark("church_talk_center_front", (9, 17), "npc_talk", face="right", action="press_a", source="sunday_go_to_church_replay", note="Text 0x0042."),
            MapLandmark("church_talk_left_front", (2, 17), "npc_talk", face="right", action="press_a", source="sunday_go_to_church_replay", note="Text 0x003C."),
            MapLandmark("church_talk_right_front", (14, 17), "npc_talk", face="left", action="press_a", source="sunday_go_to_church_replay", note="Text 0x0048."),
            MapLandmark(
                "church_maria_question_stand",
                (12, 8),
                "npc_talk",
                face="left",
                action="press_a",
                source="sunday_go_to_church_replay",
                note="Text 0x008E -> 0x008F, recorded Yes, Maria hearts +4.",
            ),
            MapLandmark("church_priest_sermon_stand", (8, 8), "npc_talk", face="left", action="press_a", source="sunday_go_to_church_replay", note="Text 0x0059."),
        ),
        source="recorded_provisional",
    ),
    0x1C: MapConfig(
        name="shop",
        walkable_tiles=SHOP_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark("seed_counter", (8, 10), "register", face="up", action="press_a", source="buy_potato_replay"),
            # Flower-shop front room on D1 (same tilemap as seed shop in this ROM).
            MapLandmark(
                "flower_owner_counter",
                (2, 21),
                "register",
                face="down",
                action="press_a",
                source="town_day1_rest",
                note="town_day1_rest: bit 0x08 at px(34,347) face down+A.",
            ),
        ),
    ),
    # Flower-shop back room (Nina). Walkable set provisional — same interior set.
    0x1D: MapConfig(
        name="flower_back",
        walkable_tiles=SHOP_WALKABLE | INTERIOR_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark(
                "nina_stand",
                (6, 6),
                "npc_talk",
                face="left",
                action="press_a",
                source="town_day1_rest",
                note="town_day1_rest: bit 0x04 at px(101,102) face left+A.",
            ),
        ),
        source="town_day1_recon",
    ),
    0x24: MapConfig(
        name="animal_shop",
        walkable_tiles=SHOP_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark("animal_counter", (12, 9), "register", face="right", action="press_a", source="buy_cow_replay"),
        ),
    ),
    0x26: MapConfig(
        name="shed",
        walkable_tiles=INTERIOR_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark("shed_exit_inside", (8, 12), "door", face="down", source="probe"),
        ),
        source="recorded_provisional",
    ),
    0x27: MapConfig(
        name="barn",
        walkable_tiles=INTERIOR_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark("barn_exit_inside", (8, 22), "door", face="down", source="buy_cow_replay"),
            MapLandmark("barn_shipping_bin", (2, 22), "shipping_bin", face="left", action="press_a", source="cow_chores_replay"),
            MapLandmark("cow_talk_stand", (10, 17), "animal_station", face="left", action="press_a", source="buy_cow_replay"),
            MapLandmark("fodder_dispenser", (13, 11), "animal_station", face="right", action="press_a", source="buy_cow_replay"),
            MapLandmark("cow_feed_trough", (7, 17), "animal_station", face="right", action="press_a", source="buy_cow_replay"),
        ),
        source="recorded_provisional",
    ),
    0x28: MapConfig(
        name="coop",
        walkable_tiles=COOP_WALKABLE,
        exits=[],
        landmarks=(
            MapLandmark("feed_trough", (6, 6), "animal_station", face="up", action="press_a", source="coop_chores_replay"),
            MapLandmark("incubator", (2, 4), "animal_station", face="up", action="press_a", source="coop_chores_replay"),
            MapLandmark("egg_shipping_bin", (1, 10), "shipping_bin", face="down", action="press_a", source="ram_verified"),
        ),
    ),
}


def _config_for_tilemap(tilemap_id: int) -> Optional[MapConfig]:
    if tilemap_id in FARM_TILEMAP_IDS:
        return MAP_REGISTRY.get(0x00)
    return MAP_REGISTRY.get(tilemap_id)


def get_walkable_tiles(tilemap_id: int) -> FrozenSet[int]:
    """Return walkable tiles for a map, falling back to farm tiles if unknown."""
    config = _config_for_tilemap(tilemap_id)
    if config is not None:
        return config.walkable_tiles
    return FARM_WALKABLE


def get_no_go_tiles(tilemap_id: int) -> FrozenSet[Tuple[int, int]]:
    """Return coordinate-specific blocked tiles for a map."""
    return NO_GO_TILES_BY_TILEMAP.get(tilemap_id, frozenset())


def get_map_name(tilemap_id: int) -> str:
    if tilemap_id in FARM_TILEMAP_NAMES:
        return FARM_TILEMAP_NAMES[tilemap_id]
    config = _config_for_tilemap(tilemap_id)
    return config.name if config is not None else f"tilemap_0x{tilemap_id:02X}"


def get_landmarks(tilemap_id: int) -> Tuple[MapLandmark, ...]:
    config = _config_for_tilemap(tilemap_id)
    return config.landmarks if config is not None else ()


def find_landmark(name: str, tilemap_id: Optional[int] = None) -> Optional[tuple[int, MapLandmark]]:
    if tilemap_id is not None:
        for landmark in get_landmarks(tilemap_id):
            if landmark.name == name:
                return tilemap_id, landmark
        return None
    for candidate_tilemap, config in MAP_REGISTRY.items():
        for landmark in config.landmarks:
            if landmark.name == name:
                return candidate_tilemap, landmark
    return None


# ── Named routes (waypoint lists for known multi-map trips) ──

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
    """
    if not waypoints:
        return []
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

_FARM_TO_MOUNTAIN_GATE: List[Waypoint] = [
    Waypoint(tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"),
    Waypoint(tilemap=0x0C, target_px=(232, 128), radius=16),
    Waypoint(tilemap=0x0C, target_px=(132, 30), radius=10, is_exit=True, exit_direction="up"),
]


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
    "farm_to_town": [
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x0C, target_px=(10, 128), radius=8, is_exit=True, exit_direction="left"),
    ],
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
    "d1_town_to_maria": [
        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(500, 280), radius=14),
        Waypoint(tilemap=0x04, target_px=(411, 216), radius=14),
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
    "d1_town_to_truck": [
        Waypoint(tilemap=0x04, target_px=(688, 888), radius=18),
        Waypoint(tilemap=0x04, target_px=(688, 500), radius=18, run_direction="up"),
        Waypoint(tilemap=0x04, target_px=(728, 424), radius=12),
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
        # 1. Navigate to berry bush area and pick berry
        Waypoint(tilemap=0x00, target_px=(585, 920), radius=16,
                 action_on_arrive="press_a", action_face="left",
                 action_frames=10, action_cooldown=30),
        # 2. Navigate to shipping bin and ship
        Waypoint(tilemap=0x00, target_px=(1001, 969), radius=16,
                 action_on_arrive="press_a", action_face="right",
                 action_frames=10, action_cooldown=30),
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
        # From the harvest shipping stand, avoid pushing left through the bin.
        Waypoint(tilemap=0x00, target_px=(344, 504), radius=12),
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12, run_direction="right"),
    ],
    "near_shed_to_shed": [
        Waypoint(tilemap=0x00, target_px=(424, 489), radius=12, run_direction="up"),
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
    "path_to_farm": [
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
    ],
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
