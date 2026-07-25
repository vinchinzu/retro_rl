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

from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

from harvest.core.tile_catalog import (
    CHURCH_WALKABLE,
    COOP_WALKABLE,
    FARM_WALKABLE,
    MOUNTAIN_WALKABLE,
    PATH_WALKABLE,
    SHOP_WALKABLE,
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
    # Pond edge / water fringe.
    (9, 26), (9, 27), (9, 28),
    (11, 26), (11, 27), (11, 28),
    # House frontage.
    (8, 12), (9, 12), (10, 12),
    # Well body.  These are visually solid even when the live tile ID is 0xA1.
    (15, 26), (16, 26), (17, 26),
    (15, 27), (16, 27), (17, 27),
})

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
            MapLandmark("pond_edge", (9, 28), "water_source", face="down", action="use_tool", source="recorded"),
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
        ],
        landmarks=(
            MapLandmark("mountain_entry", (19, 46), "exit", source="get_berry_replay"),
            MapLandmark("west_stump", (20, 25), "stump", face="down", action="use_axe", source="get_berry_replay"),
            MapLandmark("east_stump", (41, 13), "stump", face="right", action="use_axe", source="get_berry_replay"),
            MapLandmark("spa_area", (41, 13), "hot_spring", source="get_berry_replay", note="same approach area as east stump until map is fully decoded"),
            MapLandmark("fish_power_berry_cast_spot", (39, 23), "fishing_spot", face="right", action="use_tool", source="mountain_fish_power_berry_replay"),
            MapLandmark("fish_power_berry_throw_spot", (42, 25), "waterfall_pool", face="up", action="press_a", source="mountain_fish_power_berry_replay"),
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
        ),
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
    "farm_to_mountain": [
        Waypoint(tilemap=0x00, target_px=(40, 424), radius=16, is_exit=True, exit_direction="left"),
        Waypoint(tilemap=0x0C, target_px=(232, 128), radius=16),
        Waypoint(tilemap=0x0C, target_px=(132, 30), radius=10, is_exit=True, exit_direction="up"),
    ],
    "mountain_entry_to_fish_power_berry_spots": [
        Waypoint(tilemap=0x10, target_px=(328, 718), radius=16),
        Waypoint(tilemap=0x10, target_px=(496, 708), radius=16),
        Waypoint(tilemap=0x10, target_px=(518, 558), radius=16),
        Waypoint(tilemap=0x10, target_px=(582, 414), radius=16),
        Waypoint(tilemap=0x10, target_px=(624, 371), radius=12),
        Waypoint(tilemap=0x10, target_px=(686, 411), radius=12),
    ],
    "mountain_to_farm": [
        Waypoint(tilemap=0x10, target_px=(312, 744), radius=16, is_exit=True, exit_direction="down"),
        Waypoint(tilemap=0x0C, target_px=(244, 128), radius=12, is_exit=True, exit_direction="right"),
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
