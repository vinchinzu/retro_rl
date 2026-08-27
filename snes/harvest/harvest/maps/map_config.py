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

Public API is re-exported from submodules for back-compat::

    from harvest.maps.map_config import Waypoint, ROUTES, get_walkable_tiles, ...
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

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

from harvest.maps.map_types import (
    INTERIOR_WALKABLE,
    MapConfig,
    MapExit,
    MapLandmark,
    Waypoint,
)
from harvest.maps.farm_pond import (
    COW_BARN_EAST_FACE_TILES,
    EAST_SPUR_FA_FACE,
    EAST_SPUR_FA_STAND,
    FARM_MAIN_POND_STANDS,
    FARM_MAIN_POND_WATER_BOUNDS,
    FARM_NO_GO_TILES,
    HORSE_BARN_LEAVE_TILE,
    HORSE_BARN_WALL_TILES,
    FARM_POND_ACCESS_FENCE_ROW,
    FARM_POND_ACCESS_FENCE_X_RANGE,
    FARM_POND_ACCESS_STAGING_TILES,
    FARM_POND_MULTIHOP_WAYPOINTS,
    FARM_POND_POST_GAP_CORRIDOR,
    FARM_POND_REFILL_CORRIDOR,
    FARM_TILEMAP_IDS,
    FARM_TILEMAP_NAMES,
    FARM_WATER_POCKETS,
    NO_GO_TILES_BY_TILEMAP,
    farm_pond_refill_primary_stand,
    farm_pond_refill_stands,
    player_in_west_plant_pocket,
    WEST_PLANT_POCKET_BOUNDS,
    WEST_POCKET_PLANT_CENTER,
)
from harvest.maps.map_routes import (
    ROUTES,
    SEGMENTS,
    compose_routes,
    densify_waypoints,
    farm_to_spa_waypoints,
    farm_to_west_gate_waypoints,
    path_coords_leaked,
    farm_coords_look_like_path,
    segment_waypoints,
    slice_route_from_position,
    SOUTH_FIELD_MIN_Y_PX,
)

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
            MapLandmark(
                "first_berry",
                (20, 25),
                "forage",
                face="down",
                source="mountain_grape_stand",
                note=(
                    "Ground grape at ~(326,409); same tile as west_stump. "
                    "Face down, A, Don't eat. Not carpenter 2x2 / Gotz."
                ),
            ),
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
            MapLandmark(
                "seed_counter",
                (11, 21),
                "register",
                face="up",
                action="press_a",
                source="buy_potato_seeds_d2",
                note="Clerk stand (182,342). Old replay (8,10) was not the buy tile.",
            ),
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


__all__ = [
    # types
    "MapExit",
    "MapLandmark",
    "MapConfig",
    "Waypoint",
    "INTERIOR_WALKABLE",
    # tile catalog re-exports used by importers
    "FARM_WALKABLE",
    "CHURCH_WALKABLE",
    "COOP_WALKABLE",
    "MOUNTAIN_WALKABLE",
    "PATH_WALKABLE",
    "SHOP_WALKABLE",
    "TOWN_WALKABLE",
    "TILE_SIZE",
    # farm / pond
    "COW_BARN_EAST_FACE_TILES",
    "EAST_SPUR_FA_FACE",
    "EAST_SPUR_FA_STAND",
    "FARM_NO_GO_TILES",
    "HORSE_BARN_LEAVE_TILE",
    "HORSE_BARN_WALL_TILES",
    "FARM_MAIN_POND_WATER_BOUNDS",
    "FARM_MAIN_POND_STANDS",
    "FARM_POND_ACCESS_FENCE_ROW",
    "FARM_POND_ACCESS_FENCE_X_RANGE",
    "FARM_POND_ACCESS_STAGING_TILES",
    "FARM_POND_POST_GAP_CORRIDOR",
    "FARM_POND_MULTIHOP_WAYPOINTS",
    "FARM_POND_REFILL_CORRIDOR",
    "farm_pond_refill_primary_stand",
    "farm_pond_refill_stands",
    "player_in_west_plant_pocket",
    "WEST_PLANT_POCKET_BOUNDS",
    "WEST_POCKET_PLANT_CENTER",
    "FARM_WATER_POCKETS",
    "FARM_TILEMAP_IDS",
    "FARM_TILEMAP_NAMES",
    "NO_GO_TILES_BY_TILEMAP",
    # registry + helpers
    "MAP_REGISTRY",
    "get_walkable_tiles",
    "get_no_go_tiles",
    "get_map_name",
    "get_landmarks",
    "find_landmark",
    # routes
    "ROUTES",
    "SEGMENTS",
    "compose_routes",
    "segment_waypoints",
    "slice_route_from_position",
    "path_coords_leaked",
    "farm_coords_look_like_path",
    "densify_waypoints",
    "farm_to_spa_waypoints",
    "farm_to_west_gate_waypoints",
    "SOUTH_FIELD_MIN_Y_PX",
]
