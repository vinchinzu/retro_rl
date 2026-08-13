"""Map configuration types — exits, landmarks, configs, waypoints.

Pure-data types for harvest maps. No imports from farm_clearer.
"""

from typing import FrozenSet, List, NamedTuple, Optional, Tuple

from harvest.core.tile_catalog import SHOP_WALKABLE, COOP_WALKABLE

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
    # Keep walking inward briefly after a tilemap flip.  Some outdoor exits
    # expose the destination tilemap before its coordinates/tile window settle.
    exit_push_frames: int = 0
