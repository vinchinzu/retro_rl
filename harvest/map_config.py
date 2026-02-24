"""Map configuration registry — tilemap IDs, walkable tiles, exits, and routes.

Pure-data module. No imports from farm_clearer (avoids circular deps).
Walkable tile sets are duplicated here intentionally.

Tilemap IDs discovered via map_discovery.py (buy_potato_seeds replay):
  0x00 = Farm
  0x0C = Path (crossroads between farm and town)
  0x04 = Town
  0x1C = Shop interior

Transition positions (pixel coords):
  Farm(0x00) --left--> Path(0x0C)  exit=(11,424)  entry=(11,424)
  Path(0x0C) --up----> Town(0x04)  exit=(10,128)  entry=(10,128)
  Town(0x04) --enter-> Shop(0x1C)  exit=(601,218) entry=(601,218)
  Shop(0x1C) --exit--> Town(0x04)  exit=(138,468) entry=(138,468)
  Town(0x04) --right-> Path(0x0C)  exit=(756,422) entry=(756,422)
  Path(0x0C) --right-> Farm(0x00)  exit=(244,128) entry=(244,128)
"""

from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple


class MapExit(NamedTuple):
    region: Tuple[int, int, int, int]  # tile bbox (x1, y1, x2, y2) near exit
    direction: str                      # walk direction to trigger transition
    dest_tilemap: int                   # tilemap you arrive at


class MapConfig(NamedTuple):
    name: str
    walkable_tiles: FrozenSet[int]
    exits: List[MapExit]


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


# ── Walkable tile sets (duplicated from farm_clearer to avoid import cycle) ──

FARM_WALKABLE: FrozenSet[int] = frozenset({
    0x00, 0x01, 0x02, 0x03, 0x07, 0x08,
    0x70,  # Planted grass
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85,  # Grass variants
    0xA0, 0xA2, 0xA3, 0xA8,  # Paths, borders, empty tiles
})

# Path/crossroads (0x0C) walkable tiles — discovered via map_discovery.py
# Player walked on: 0xA0 (74%), 0xC0 (18%), 0xA1 (4%), 0xA2 (4%)
PATH_WALKABLE: FrozenSet[int] = frozenset({
    0xA0, 0xA1, 0xA2, 0xC0,
})

# Town (0x04) walkable tiles — discovered via map_discovery.py
# Player walked on: 0xA0 (60%), 0xA2 (8%), 0xA4 (8%), 0xD6 (7%),
#                   0xC3 (1%), 0xC0 (1%), 0xA1 (1%)
# Note: 0xFF (16%) excluded — likely unloaded tiles during transition
TOWN_WALKABLE: FrozenSet[int] = frozenset({
    0xA0, 0xA1, 0xA2, 0xA4, 0xC0, 0xC3, 0xD6,
})

# Shop interior (0x1C) walkable tiles — discovered via map_discovery.py
# Player walked on: 0xD4 (64%), 0xA1 (18%), 0xA0 (8%), 0xC3 (6%), 0xD6 (2%)
SHOP_WALKABLE: FrozenSet[int] = frozenset({
    0xA0, 0xA1, 0xC3, 0xD4, 0xD6,
})


# ── Map Registry ──

MAP_REGISTRY: Dict[int, MapConfig] = {
    0x00: MapConfig(
        name="farm",
        walkable_tiles=FARM_WALKABLE,
        exits=[
            # Exit left → path crossroads. Player at ~px(11,424) tile(0,26)
            MapExit(region=(0, 24, 0, 28), direction="left", dest_tilemap=0x0C),
        ],
    ),
    0x0C: MapConfig(
        name="path",
        walkable_tiles=PATH_WALKABLE,
        exits=[
            # Exit up → town. Player at ~px(10,128) tile(0,8)
            MapExit(region=(0, 0, 2, 10), direction="up", dest_tilemap=0x04),
            # Exit right → farm. Player at ~px(244,128) tile(15,8)
            MapExit(region=(14, 6, 16, 10), direction="right", dest_tilemap=0x00),
        ],
    ),
    0x04: MapConfig(
        name="town",
        walkable_tiles=TOWN_WALKABLE,
        exits=[
            # Exit right → path. Player at ~px(756,422) tile(47,26)
            MapExit(region=(45, 24, 48, 28), direction="right", dest_tilemap=0x0C),
        ],
    ),
    0x1C: MapConfig(
        name="shop",
        walkable_tiles=SHOP_WALKABLE,
        exits=[],
    ),
}


def get_walkable_tiles(tilemap_id: int) -> FrozenSet[int]:
    """Return walkable tiles for a map, falling back to farm tiles if unknown."""
    config = MAP_REGISTRY.get(tilemap_id)
    if config is not None:
        return config.walkable_tiles
    return FARM_WALKABLE


# ── Named routes (waypoint lists for known multi-map trips) ──

# Berry ship route: farm-only (discovered via ship_berry recording analysis)
# Berry bush at tile(36,57) ~px(585,920), shipping bin at tile(62,60) ~px(1001,969)
# The recording picks berry with A at (585,920) and ships at (1001,969).
ROUTES: Dict[str, List[Waypoint]] = {
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
}
