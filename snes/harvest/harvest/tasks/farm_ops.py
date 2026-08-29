"""
Generic farm tile-scan and tool helpers.

TileScanner, ToolManager, and use_tool sequences used by clears / crops /
planner tasks. FarmClearer FSM stays in farm_clearer; nav stays in nav.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Tuple
import os

import numpy as np

from retro_harness.actions import action_names

from harvest.core.carry import backpack_tool
from harvest.core.stamina import SWING_STAMINA_COST, stamina_cost_for_hits
from harvest.core.tile_catalog import (
    ADDR_MAP,
    ADDR_TOOL,
    CLEARABLE_DEBRIS_TYPES,
    DEBRIS_TOOL,
    LARGE_ROCK_DAMAGE_TILES,
    LARGE_ROCK_TL,
    LARGE_ROCK_TILES,
    LIFTABLE_TILES,
    MAP_WIDTH,
    ROCK as SMALL_ROCK_TILE,
    STUMP_TL,
    STUMP_TILES,
    TILE_SIZE,
    TILE_TO_DEBRIS,
    DebrisType,
    Tool,
    debris_footprint,
)
from harvest.tasks.nav import (
    Point,
    VIEWPORT_HOP_TILES,
    get_tile_at,
    make_action,
    manhattan,
)

__all__ = [
    "DAMAGE_ROCK_TL",
    "LOADED_FARM_STAND",
    "SHED_DOOR_TILE",
    "SWING_STAMINA_COST",
    "Target",
    "TileScanner",
    "ToolManager",
    "action_to_names",
    "cycle_tool",
    "choose_clear_target",
    "DEFAULT_PRIORITY",
    "drop_unarmed_debris",
    "find_blocker_opener",
    "find_pathable_approach",
    "find_unfailed_approach",
    "handle_navigating",
    "parse_priority_list",
    "shed_door_step_off_actions",
    "sort_targets_cluster",
    "snap_debris_anchor",
    "start_progress_watch",
    "use_tool",
    "use_tool_facing",
]


# Outdoor tool-shed door. Hammer fetch lands here on 0xFF and unloads the farm.
SHED_DOOR_TILE = (26, 30)
# a1 west-north of the door. Adjacent a8 still unloads distant metatiles.
LOADED_FARM_STAND = (25, 28)


# 2x2 damage-rock TL (0x11) matches intact LARGE_ROCK_TL (0x0D).
DAMAGE_ROCK_TL = min(LARGE_ROCK_DAMAGE_TILES)
_TWO_BY_TWO_IDS = STUMP_TILES | LARGE_ROCK_TILES | LARGE_ROCK_DAMAGE_TILES
_ANCHOR_IDS = frozenset({STUMP_TL, LARGE_ROCK_TL, DAMAGE_ROCK_TL})

# Hard obstacles first so pathing opens up, then cheap lifts.
DEFAULT_PRIORITY: List[DebrisType] = [
    DebrisType.ROCK,
    DebrisType.STUMP,
    DebrisType.STONE,
    DebrisType.WEED,
]
DEBRIS_NAMES = {
    "weed": DebrisType.WEED,
    "weeds": DebrisType.WEED,
    "bush": DebrisType.WEED,
    "stone": DebrisType.STONE,
    "stones": DebrisType.STONE,
    "rock": DebrisType.ROCK,
    "rocks": DebrisType.ROCK,
    "stump": DebrisType.STUMP,
    "stumps": DebrisType.STUMP,
    "fence": DebrisType.FENCE,
    "fences": DebrisType.FENCE,
}
_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_OPENER_TYPES = (DebrisType.STONE, DebrisType.FENCE)


# =============================================================================
# DATA
# =============================================================================

@dataclass
class Target:
    tile: Tuple[int, int]
    pos: Point
    debris_type: DebrisType
    tile_id: int

    @property
    def is_liftable(self) -> bool:
        return self.tile_id in LIFTABLE_TILES

    @property
    def required_tool(self) -> Optional[Tool]:
        return DEBRIS_TOOL.get(self.debris_type)

    @property
    def required_hits(self) -> int:
        # ROM: 2×2 stump/rock breaks at 6 registered hits ($096D CMP #6).
        # Small boulder 0x06 is a single hammer blow.
        if self.debris_type == DebrisType.STUMP:
            return 6
        if self.debris_type == DebrisType.ROCK and self.tile_id != SMALL_ROCK_TILE:
            return 6
        return 1

    def stamina_to_clear(self, *, lifting: bool, tool_hits: int = 0) -> int:
        if lifting:
            return 1
        return stamina_cost_for_hits(self.required_hits, tool_hits=tool_hits)

    @property
    def footprint(self) -> Tuple[Tuple[int, int], ...]:
        return debris_footprint(self.tile, self.tile_id)


ApproachChoice = Tuple[Target, Tuple[int, int], List[Tuple[int, int]]]


def sort_targets_cluster(targets: List[Target], player_pos: Point) -> List[Target]:
    """Nearest-neighbor with north bias so day-plan clear stays returnable.

    Prefer targets north of / near the y=31 fence; deep-south debris
    (y>38) is a softlock trap for return_home after water days (rr-5in).
    """
    remaining = list(targets)
    ordered: List[Target] = []
    cur = player_pos
    row_dir = 1
    while remaining:
        remaining.sort(
            key=lambda t: (
                2 if t.tile[1] > 40 else (1 if t.tile[1] > 32 else 0),
                manhattan(t.pos, cur),
                t.tile[1],
                t.tile[0] * row_dir,
            )
        )
        nxt = remaining.pop(0)
        ordered.append(nxt)
        if ordered and len(ordered) >= 2:
            prev_y = ordered[-2].tile[1]
            if nxt.tile[1] != prev_y:
                row_dir *= -1
        cur = nxt.pos
    return ordered


def shed_door_step_off_actions() -> List[np.ndarray]:
    """Hold west then north-west onto (25,28) a1. Door pin eats 1-frame taps."""
    run = dict(b=True)
    actions = [make_action(left=True, **run) for _ in range(16)]
    actions.extend(make_action(left=True, up=True, **run) for _ in range(16))
    return actions


def drop_unarmed_debris(
    priority: List[DebrisType], missing: List[int]
) -> List[DebrisType]:
    """Drop debris types whose required tool is actually missing.

    Stones stay (lift) when the hammer is gone; ROCK does not.
    """
    drop: Set[DebrisType] = set()
    if int(Tool.HAMMER) in missing:
        drop.add(DebrisType.ROCK)
    if int(Tool.AXE) in missing:
        drop.add(DebrisType.STUMP)
    kept = [dt for dt in priority if dt not in drop]
    return kept or [DebrisType.WEED, DebrisType.STONE]


def snap_debris_anchor(
    ram: np.ndarray, tx: int, ty: int, tile_id: int
) -> Optional[Tuple[int, int, int, DebrisType]]:
    """Map a 2x2 stump/rock cell to its top-left; None if the family is gone."""
    debris = TILE_TO_DEBRIS.get(tile_id)
    if debris is None:
        return None
    if tile_id not in _TWO_BY_TWO_IDS:
        return (tx, ty, tile_id, debris)
    if tile_id in _ANCHOR_IDS:
        return (tx, ty, tile_id, debris)
    want_stump = tile_id in STUMP_TILES
    for ox, oy in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
        ax, ay = tx + ox, ty + oy
        aid = int(get_tile_at(ram, ax, ay))
        if want_stump and aid == STUMP_TL:
            return (ax, ay, STUMP_TL, DebrisType.STUMP)
        if not want_stump and aid in (LARGE_ROCK_TL, DAMAGE_ROCK_TL):
            return (ax, ay, aid, DebrisType.ROCK)
    return None


# =============================================================================
# TOOL ACTION HELPERS
# =============================================================================

def action_to_names(action: np.ndarray) -> str:
    pressed = tuple(name.lower() for name in action_names(action))
    return "+".join(pressed) if pressed else "none"


def use_tool(frames: int = 20, cooldown: int = 10) -> List[np.ndarray]:
    """
    Use tool with proper timing.
    - frames: Number of frames to hold Y button
    - cooldown: Number of idle frames after tool use to let animation complete
    """
    actions = [make_action(y=True) for _ in range(frames)]
    actions.extend([make_action() for _ in range(cooldown)])
    return actions


def use_tool_facing(direction: str, frames: int = 20, cooldown: int = 10) -> List[np.ndarray]:
    """
    Use tool while keeping a facing direction without combining direction+Y.
    This avoids unintended movement if the target tile becomes walkable mid-sequence.
    """
    actions: List[np.ndarray] = []
    # Re-face briefly to stabilize direction, but never with Y held.
    actions.append(make_action(**{direction: True}))
    actions.append(make_action())
    actions.extend([make_action(y=True) for _ in range(frames)])
    actions.extend([make_action() for _ in range(cooldown)])
    return actions


def cycle_tool() -> List[np.ndarray]:
    return [make_action(x=True)] + [make_action() for _ in range(5)]


# =============================================================================
# SCANNER
# =============================================================================

class TileScanner:
    def __init__(self):
        self.debris_map = TILE_TO_DEBRIS.copy()
        self.frame_count = 0

    def scan(
        self,
        ram: np.ndarray,
        bounds: Optional[Tuple[int, int, int, int]] = None,
        *,
        types: Optional[Set[DebrisType]] = None,
    ) -> List[Target]:
        """Scan farm metatiles for debris.

        2x2 stump/large-rock objects emit a single target at the top-left
        cell so the clearer does not thrash four tiles of one boulder.
        """
        self.frame_count += 1
        if ADDR_MAP >= len(ram):
            return []

        # Save-state loaders may hand back ``bytes``; normalize for numpy ops.
        # ``np.asarray(bytes_slice)`` becomes a 0-d object in NumPy 2 — use
        # frombuffer on a memoryview instead.
        if isinstance(ram, np.ndarray):
            ram_arr = ram
        else:
            ram_arr = np.frombuffer(memoryview(ram), dtype=np.uint8)

        end = min(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, len(ram_arr))
        map_data = ram_arr[ADDR_MAP:end]
        if map_data.size == 0:
            return []

        mask = np.isin(map_data, list(self.debris_map.keys()))
        indices = np.flatnonzero(mask)

        targets: List[Target] = []
        for idx in indices:
            tile_id = int(map_data[idx])
            debris = self.debris_map.get(tile_id)
            if debris is None:
                continue
            if types is not None and debris not in types:
                continue

            ty, tx = divmod(int(idx), MAP_WIDTH)
            if bounds and not (
                bounds[0] <= tx <= bounds[2] and bounds[1] <= ty <= bounds[3]
            ):
                continue

            # Collapse 2x2 stump / large-rock / damage families to TL only.
            if tile_id in _TWO_BY_TWO_IDS and tile_id not in _ANCHOR_IDS:
                continue

            targets.append(
                Target(
                    tile=(tx, ty),
                    pos=Point(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8),
                    debris_type=debris,
                    tile_id=tile_id,
                )
            )

        if (
            os.getenv("FENCE_DEBUG") == "1"
            and targets
            and self.frame_count % 300 == 0
        ):
            top = targets[0]
            print(
                f"[SCANNER] Found {len(targets)} targets. "
                f"Top: {top.debris_type.name} at {top.tile}"
            )

        return targets

    def has_clearable_debris(
        self,
        ram: np.ndarray,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """True when any weed/stone/rock/stump remains in bounds."""
        return bool(
            self.scan(ram, bounds, types=set(CLEARABLE_DEBRIS_TYPES))
        )


def scan_typed_targets(
    ram,
    types,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    *,
    scanner: Optional[TileScanner] = None,
) -> List[Target]:
    """Scan one debris kind (or set) and optionally clip to farm_bounds."""
    scanner = scanner or TileScanner()
    wanted = tuple(types)
    if not wanted:
        return []
    return [
        target
        for target in scanner.scan(ram, bounds, types=set(wanted))
        if target.debris_type in wanted
    ]


# =============================================================================
# TOOL MANAGER
# =============================================================================

class ToolManager:
    """Selected + backpack two-slot carry.

    X swaps the pair (``d2_farm_plant``: after the seed bag spends, selected
    goes 0 and the can sits in the backpack — one X selects it). ``current``
    stays the selected slot so existing ``== wanted`` checks still work.
    """

    def __init__(self):
        self.current = 0
        self.backpack = 0
        self.seen: Set[int] = set()
        self.start_id: Optional[int] = None

    def update(self, ram: np.ndarray):
        self.current = int(ram[ADDR_TOOL]) if ADDR_TOOL < len(ram) else 0
        try:
            self.backpack = int(backpack_tool(ram))
        except Exception:
            self.backpack = 0

    def start_search(self):
        self.start_id = self.current
        self.seen = {self.current}

    def record(self):
        self.seen.add(self.current)

    def has(self, tool_id: int) -> bool:
        wanted = int(tool_id)
        return self.current == wanted or self.backpack == wanted

    def needs_swap(self, tool_id: int) -> bool:
        wanted = int(tool_id)
        return self.current != wanted and self.backpack == wanted

    def cycle_complete(self) -> bool:
        return self.start_id is not None and self.current == self.start_id and len(self.seen) > 1


def parse_priority_list(
    raw: Optional[str], priority_only: bool = False
) -> List[DebrisType]:
    if not raw:
        return list(DEFAULT_PRIORITY)

    parsed: List[DebrisType] = []
    for name in raw.split(","):
        debris = DEBRIS_NAMES.get(name.strip().lower())
        if debris and debris not in parsed:
            parsed.append(debris)

    if not parsed:
        return list(DEFAULT_PRIORITY)

    if not priority_only:
        for dt in DEFAULT_PRIORITY:
            if dt not in parsed:
                parsed.append(dt)

    return parsed


def _stands_for(target: Target) -> List[Tuple[int, int]]:
    occupied = set(target.footprint)
    stands: List[Tuple[int, int]] = []
    for tx, ty in target.footprint:
        for dx, dy in _CARDINALS:
            stand = (tx + dx, ty + dy)
            if stand in occupied:
                continue
            if not (0 <= stand[0] < MAP_WIDTH and 0 <= stand[1] < MAP_WIDTH):
                continue
            stands.append(stand)
    return stands


def find_unfailed_approach(
    clearer: Any, ram: np.ndarray, target: Target
) -> Optional[Tuple[int, int]]:
    """Choose the nearest viable stand not disproved for this target."""
    candidates: list[Tuple[int, Tuple[int, int]]] = []
    for stand in _stands_for(target):
        if (target.tile, stand) in clearer.failed_approaches:
            continue
        if not clearer.pathfinder.is_walkable(ram, *stand):
            continue
        distance = manhattan(
            Point(stand[0] * TILE_SIZE + 8, stand[1] * TILE_SIZE + 8),
            clearer.navigator.current_pos,
        )
        candidates.append((distance, stand))
    return min(candidates, default=(0, None))[1]


def _path_reaches(
    path: Optional[List[Tuple[int, int]]],
    start: Tuple[int, int],
    stand: Tuple[int, int],
) -> bool:
    if path is None:
        return False
    if start == stand:
        return True
    return bool(path) and path[-1] == stand


def find_pathable_approach(
    clearer: Any, ram: np.ndarray, target: Target
) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    """Prefer a stand BFS can actually reach over a nearer blocked neighbor."""
    start = clearer.navigator.current_tile
    full: list[Tuple[int, Tuple[int, int], List[Tuple[int, int]]]] = []
    hops: list[Tuple[int, Tuple[int, int], List[Tuple[int, int]]]] = []
    for stand in _stands_for(target):
        if (target.tile, stand) in clearer.failed_approaches:
            continue
        if not clearer.pathfinder.is_walkable(ram, *stand):
            continue
        path = clearer.pathfinder.find_path(
            ram, start, stand, max_steps=VIEWPORT_HOP_TILES
        )
        if path is None:
            continue
        distance = manhattan(
            Point(stand[0] * TILE_SIZE + 8, stand[1] * TILE_SIZE + 8),
            clearer.navigator.current_pos,
        )
        row = (distance, stand, path)
        if _path_reaches(path, start, stand):
            full.append(row)
        else:
            hops.append(row)
    pool = full or hops
    if not pool:
        return None, None
    _dist, stand, path = min(pool, key=lambda item: item[0])
    return stand, path


def _blocker_target(ram: np.ndarray, tile: Tuple[int, int]) -> Optional[Target]:
    tid = int(get_tile_at(ram, *tile))
    debris = TILE_TO_DEBRIS.get(tid)
    if debris not in _OPENER_TYPES:
        return None
    return Target(
        tile=tile,
        pos=Point(tile[0] * TILE_SIZE + 8, tile[1] * TILE_SIZE + 8),
        debris_type=debris,
        tile_id=tid,
    )


def find_blocker_opener(
    clearer: Any, ram: np.ndarray, weeds: Sequence[Target]
) -> Optional[ApproachChoice]:
    """Lift an adjacent stone/fence so a boxed weed gains a stand."""
    for weed in weeds:
        if find_unfailed_approach(clearer, ram, weed) is not None:
            continue
        blockers: list[Target] = []
        for dx, dy in _CARDINALS:
            blocker = _blocker_target(ram, (weed.tile[0] + dx, weed.tile[1] + dy))
            if blocker is not None:
                blockers.append(blocker)
        blockers.sort(key=lambda t: 0 if t.debris_type == DebrisType.STONE else 1)
        for blocker in blockers:
            approach, path = find_pathable_approach(clearer, ram, blocker)
            if approach is not None and path is not None:
                return blocker, approach, path
    return None


def choose_clear_target(
    clearer: Any, ram: np.ndarray, phase_targets: Sequence[Target]
) -> Optional[ApproachChoice]:
    """Pick the next debris stand, opening boxed weeds when needed."""
    for target in phase_targets:
        approach, path = find_pathable_approach(clearer, ram, target)
        if approach is not None and path is not None:
            return target, approach, path
    return find_blocker_opener(clearer, ram, phase_targets)


def start_progress_watch(clearer: Any, approach: Tuple[int, int]) -> None:
    clearer._nav_best_distance = manhattan(
        clearer.navigator.current_pos,
        Point(approach[0] * TILE_SIZE + 8, approach[1] * TILE_SIZE + 8),
    )
    clearer._nav_last_progress_frame = clearer.frame_count


def handle_navigating(clearer: Any, ram: np.ndarray) -> Optional[str]:
    """Navigate to a debris stand, rejecting cross-tile pixel oscillations."""
    if not clearer.current_target or not clearer.approach_tile:
        return "scanning"

    live_id = get_tile_at(ram, *clearer.current_target.tile)
    live_debris = TILE_TO_DEBRIS.get(live_id)
    if live_debris is None or live_debris != clearer.current_target.debris_type:
        clearer.current_target = None
        return "scanning"
    if live_id != clearer.current_target.tile_id:
        clearer.current_target = Target(
            tile=clearer.current_target.tile,
            pos=clearer.current_target.pos,
            debris_type=live_debris,
            tile_id=live_id,
        )

    if clearer.navigator.current_tile == clearer.approach_tile:
        return "clearing"

    approach_pos = Point(
        clearer.approach_tile[0] * TILE_SIZE + 8,
        clearer.approach_tile[1] * TILE_SIZE + 8,
    )
    distance = manhattan(clearer.navigator.current_pos, approach_pos)
    if clearer._nav_best_distance is None or distance < clearer._nav_best_distance:
        clearer._nav_best_distance = distance
        clearer._nav_last_progress_frame = clearer.frame_count
    elif clearer.frame_count - clearer._nav_last_progress_frame > clearer.max_nav_no_progress:
        failed = (clearer.current_target.tile, clearer.approach_tile)
        clearer.failed_approaches.add(failed)
        print(
            f"[NAV] No target progress at {clearer.navigator.current_pos}; "
            f"reject approach target={failed[0]} stand={failed[1]}"
        )
        clearer.navigator.path = []
        clearer.navigator.stasis = 0
        clearer.current_target = None
        clearer.approach_tile = None
        clearer._nav_best_distance = None
        return "scanning"

    if clearer.navigator.stasis > clearer.max_stasis:
        print(f"[NAV] Stuck at {clearer.navigator.current_tile}, trying alternate path")
        if clearer.navigator.path:
            clearer.pathfinder.temp_blocked.add(clearer.navigator.path[0])
        clearer.navigator.path = []
        clearer.navigator.stasis = 0
        return clearer._replan_nav_hop(ram)

    action = clearer.navigator.follow_path(ram)
    if action is not None:
        clearer.action_queue.append(action)
        return None
    if clearer.navigator.current_tile != clearer.approach_tile:
        return clearer._replan_nav_hop(ram)
    return "clearing"
