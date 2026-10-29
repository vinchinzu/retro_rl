"""
Grass planting module - Tills untilled ground and plants grass seeds.

Per-stand state machine:
  TILL:  hoe the untilled squares
  PLANT: stand on the centre and use the grass seeds (one bag = 3x3)
  CUT:   sickle mature pasture (hay)

Follows the Task protocol from fence_flow.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, List
from collections import deque


import numpy as np

from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import MATURE_GRASS_TILES
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.animal_status import read_held_item
from harvest.core.tile_catalog import (
    Tool,
    ADDR_INPUT_LOCK,
    LIFTABLE_TILES,
)
from harvest.planner.tasks.transitions import hands_are_clear, toss_held_actions
from harvest.tasks.nav import (
    Pathfinder,
    Navigator,
    Point,
    make_action,
    get_tile_at,
    manhattan,
    tile_dist,
    TILE_SIZE,
    MAP_WIDTH,
    WALKABLE_TILES,
)
from harvest.tasks.farm_clearer import (
    TileScanner,
    ToolManager,
    use_tool,
    cycle_tool,
)



# =============================================================================
# CONSTANTS (from Step 1 discovery)
# =============================================================================

# Tile IDs
UNTILLED_TILE = 0x01       # Cleared ground, not yet hoed
DRIED_TILLED = 0x02        # Previously hoed but dried out, needs re-hoeing
FRESH_TILLED = 0x07        # Freshly hoed, ready for planting
WATERED_TILLED = 0x08      # Rained-on 0x07; seed bit still set
PLANTED_GRASS_TILE = 0x70  # After planting grass seeds on tilled soil
GRASS_TILES = {0x1D, *range(0x70, 0x7D)}  # includes cut-regrowth $7C
# South field below the crop pocket (rings own y<=31); the well sits at
# (13..15, 49..50). Bags go on the best non-overlapping 3x3 stands inside it,
# nearest the anchor first, so stumps, rocks and boulders are skipped rather than
# cleared. The first stand sits at PASTURE_ANCHOR (the nav arrival). On the
# natural Summer D1 farm this holds ~21 stands (~180 seedable squares).
PASTURE_BOUNDS = (13, 33, 27, 48)
PASTURE_ANCHOR = (16, 34)
# Weeds, stones and lumber cost no stamina to lift (A) and toss, and the carry
# pair holds hoe + seeds, so lifting is the only clearing a pasture day has.
# Stumps and rocks need the axe or hammer and stay lost squares.
# ToolUsedGrassSeeds visits the nine squares around the farmer and seeds only
# the ones whose tile has the ROM's seed bit (hoed soil). Any other square is
# skipped without harm, so a bag over debris just plants fewer tiles.
_FRESH_SOIL = {FRESH_TILLED, WATERED_TILLED}
MIN_BAG_TILES = 6  # fewer seedable squares than this is not worth 500G
MAX_WALK_TILES = 30  # a longer walk to a swing stand means a weed wall is in the way
TILLED_TILES = {DRIED_TILLED, FRESH_TILLED}  # Both types of tilled

# Tiles the hoe can act on (both untilled AND dried-out tilled need re-hoeing)
TILLABLE_TILES = {UNTILLED_TILE, DRIED_TILLED}

# Tiles grass seeds can be planted on (only freshly hoed)
PLANTABLE_TILES = {FRESH_TILLED}

# Item ID for grass seeds when equipped
GRASS_SEED_ITEM_ID = 0x0C

# Power Berry detection
ADDR_POWER_BERRIES = 0x0976  # Count of collected berries (verified via state diff)

# Default bounds: full farmable area
DEFAULT_BOUNDS = (3, 3, 62, 60)

# Default no-go: entire left side of farm (house, shed, shipping bin, barn)
DEFAULT_NO_GO_RECTS = [(3, 3, 28, 60)]


# =============================================================================
# GRASS PLANT TASK
# =============================================================================

class Phase:
    CLEAR = "clear"
    TILL = "till"
    PLANT = "plant"
    CUT = "cut"


def reachable_tiles(ram: np.ndarray, pathfinder: Pathfinder, start: Tuple[int, int],
                    depth: int = MAX_WALK_TILES) -> Set[Tuple[int, int]]:
    """Tiles within ``depth`` steps of ``start``, walking through liftable debris.

    Debris is travel-solid until it is lifted, but a weed wall is not a reason
    to drop a stand: the clear phase works inward from the open side.
    """
    seen = {start: 0}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        if seen[(cx, cy)] >= depth:
            continue
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (nx, ny) not in seen and 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH and (
                    get_tile_at(ram, nx, ny) in LIFTABLE_TILES or pathfinder.is_walkable(ram, nx, ny)):
                seen[(nx, ny)] = seen[(cx, cy)] + 1
                queue.append((nx, ny))
    return set(seen)


def stand_squares(ram: np.ndarray, centre: Tuple[int, int],
                  reach: Optional[Set[Tuple[int, int]]]) -> Tuple[int, int]:
    """(seedable, liftable) squares of a bag at ``centre`` that the farmer can reach."""
    soil = lifts = 0
    for y in range(centre[1] - 1, centre[1] + 2):
        for x in range(centre[0] - 1, centre[0] + 2):
            if reach is not None and not any(
                    n in reach for n in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))):
                continue
            tid = get_tile_at(ram, x, y)
            soil += tid in (TILLABLE_TILES | _FRESH_SOIL)
            lifts += tid in LIFTABLE_TILES
    return soil, lifts


def plan_stands(ram: np.ndarray, bounds: Tuple[int, int, int, int], limit: int,
                reach: Optional[Set[Tuple[int, int]]],
                anchor: Tuple[int, int] = PASTURE_ANCHOR) -> List[Tuple[int, int]]:
    """Pick up to ``limit`` non-overlapping 3x3 stand centres inside ``bounds``.

    Greedy on walking-tile cost: 3 per square that can never be seeded (stump,
    rock, old pasture), 1 per debris square to lift, plus distance to the
    anchor. A stand needs MIN_BAG_TILES squares that are soil or liftable.
    Order: best first.
    """
    x0, y0, x1, y1 = bounds
    cost = {}
    for cy in range(y0 + 1, y1):
        for cx in range(x0 + 1, x1):
            if (reach is not None and (cx, cy) not in reach) or get_tile_at(ram, cx, cy) not in (
                    TILLABLE_TILES | _FRESH_SOIL | LIFTABLE_TILES):
                continue
            soil, lifts = stand_squares(ram, (cx, cy), reach)
            if soil + lifts >= MIN_BAG_TILES:
                cost[(cx, cy)] = 3 * (9 - soil - lifts) + lifts + abs(cx - anchor[0]) + abs(cy - anchor[1])
    stands: List[Tuple[int, int]] = []
    while cost and len(stands) < limit:
        best = min(cost, key=lambda c: (cost[c], c))
        stands.append(best)
        cost = {c: v for c, v in cost.items() if abs(c[0] - best[0]) > 2 or abs(c[1] - best[1]) > 2}
    return stands


@dataclass
class GrassPlantTask(Task):
    """Till ground and plant grass seeds within configurable bounds."""

    name: str = "grass_planter"
    bounds: Tuple[int, int, int, int] = DEFAULT_BOUNDS
    no_go_rects: List[Tuple[int, int, int, int]] = field(default_factory=lambda: list(DEFAULT_NO_GO_RECTS))
    till_only: bool = False
    cut_only: bool = False
    target_tiles: int = 0
    bag_limit: int = 0
    max_steps_per_target: int = 1200
    stasis_repath: int = 180
    max_failures: int = 50
    max_retry_passes: int = 2
    debug: bool = False
    debug_interval: int = 300

    chunk_size: int = 3  # Work in 3x3 chunks

    # Internal state (not constructor args)
    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _tool_mgr: ToolManager = field(default_factory=ToolManager, init=False)
    _no_go_tiles: Set[Tuple[int, int]] = field(default_factory=set, init=False)

    _phase: str = field(default=Phase.CLEAR, init=False)
    _state: str = field(default="scan", init=False)
    _target_tile: Optional[Tuple[int, int]] = field(default=None, init=False)
    _approach_tile: Optional[Tuple[int, int]] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _steps_on_target: int = field(default=0, init=False)
    _total_steps: int = field(default=0, init=False)
    _failures: int = field(default=0, init=False)
    _plant_failures: int = field(default=0, init=False)
    _failed_tiles: Set[Tuple[int, int]] = field(default_factory=set, init=False)
    _swing_fails: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False)
    _retry_pass: int = field(default=0, init=False)
    _bag_seeds: int = field(default=0, init=False)  # grass_seeds / planted_grass before this bag
    _bag_grass: int = field(default=0, init=False)
    _verify_wait: int = field(default=0, init=False)
    _toss_tries: int = field(default=0, init=False)
    cleared_count: int = field(default=0, init=False)
    _hay_before: int = field(default=0, init=False)
    _seeds_before: int = field(default=0, init=False)
    _grass_before: int = field(default=0, init=False)

    # Chunk tracking: current 3x3 chunk origin (top-left corner)
    _chunk_origin: Optional[Tuple[int, int]] = field(default=None, init=False)
    _chunk_phase_logged: str = field(default="", init=False)
    _chunk_order: List[Tuple[int, int]] = field(default_factory=list, init=False)
    _chunk_index: int = field(default=0, init=False)
    _act_count: int = field(default=0, init=False)  # total act attempts (for diagnostic logging)

    tilled_count: int = field(default=0, init=False)
    planted_count: int = field(default=0, init=False)
    berries_found: int = field(default=0, init=False)
    _prev_berry_count: int = field(default=0, init=False)
    _berry_dialog_frames: int = field(default=0, init=False)
    # One partial sow when the 32-tile gate is short and a bag is in hand.
    _short_sow_used: bool = field(default=False, init=False)

    def __post_init__(self):
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        if os.getenv("GRASS_DEBUG", "").lower() in ("1", "true", "yes"):
            self.debug = True
        self._phase = self._first_phase()
        self._state = "scan"
        self._target_tile = None
        self._approach_tile = None
        self._action_queue.clear()
        self._steps_on_target = 0
        self._total_steps = 0
        self._failures = 0
        self._plant_failures = 0
        self._failed_tiles.clear()
        self._chunk_origin = None
        self._chunk_phase_logged = ""
        self._retry_pass = 0
        self._hay_before = int(read_ram_value(world.ram, "stored_grass"))
        self._seeds_before = int(read_ram_value(world.ram, "grass_seeds"))
        self._grass_before = int(read_ram_value(world.ram, "planted_grass"))
        self._act_count = 0
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)
        self._init_no_go()
        if self.cut_only:
            self._chunk_order = self._build_chunk_order()
        else:
            seeds = self._seeds_before
            bags = min(seeds, self.bag_limit) if self.bag_limit else seeds
            self.cleared_count = 0
            reach = reachable_tiles(world.ram, self._pathfinder, self._navigator.current_tile)
            wanted = max(bags, 1) + 2
            # An odd start tile (indoors, off the map) leaves nothing reachable:
            # plan on the soil alone and let the walk fail loudly.
            stands = plan_stands(world.ram, self.bounds, wanted, reach) \
                or plan_stands(world.ram, self.bounds, wanted, None)
            self._chunk_order = [(cx - 1, cy - 1) for cx, cy in stands]
            print(f"[GRASS] stands={stands} bags={bags} planted={self._grass_before}")
        self._chunk_index = 0
        self._swing_fails.clear()
        self.tilled_count = 0
        self.planted_count = 0
        self.berries_found = 0
        pb = int(world.ram[ADDR_POWER_BERRIES]) if ADDR_POWER_BERRIES < len(world.ram) else 0
        self._prev_berry_count = pb
        self._berry_dialog_frames = 0
        self._short_sow_used = False

    def _init_no_go(self):
        """Build no-go tile set from rects + GRASS_NO_GO env var."""
        self._no_go_tiles.clear()
        all_rects = list(self.no_go_rects)
        env_val = os.getenv("GRASS_NO_GO", "")
        if env_val:
            for rect_str in env_val.split(";"):
                rect_str = rect_str.strip()
                if not rect_str:
                    continue
                parts = rect_str.split(",")
                if len(parts) == 4:
                    all_rects.append(tuple(int(p) for p in parts))
        for x1, y1, x2, y2 in all_rects:
            for ty in range(y1, y2 + 1):
                for tx in range(x1, x2 + 1):
                    self._no_go_tiles.add((tx, ty))
        if self._no_go_tiles:
            print(f"[GRASS] No-go: {len(self._no_go_tiles)} tiles from {len(all_rects)} rects")

    def _build_chunk_order(self) -> List[Tuple[int, int]]:
        """Build ordered list of chunk origins left-to-right, top-to-bottom."""
        x_min, y_min, x_max, y_max = self.bounds
        cs = self.chunk_size
        chunks = []
        for cy in range(y_min, y_max + 1, cs):
            for cx in range(x_min, x_max + 1, cs):
                chunks.append((cx, cy))
        return chunks

    def can_start(self, world: WorldState) -> bool:
        return int(read_ram_value(world.ram, "tilemap")) in range(4)

    def _target_ids(self):
        if self._phase == Phase.CUT:
            return MATURE_GRASS_TILES
        if self._phase == Phase.CLEAR:
            return LIFTABLE_TILES
        return TILLABLE_TILES if self._phase == Phase.TILL else PLANTABLE_TILES

    def _wanted_tool(self):
        if self._phase == Phase.CUT:
            return Tool.SICKLE
        if self._phase == Phase.CLEAR:
            return None  # A lifts with whatever is selected
        return Tool.HOE if self._phase == Phase.TILL else GRASS_SEED_ITEM_ID

    # ------------------------------------------------------------------
    # Scanning helpers
    # ------------------------------------------------------------------

    def _grass70(self, ram: np.ndarray) -> int:
        x0, y0, x1, y1 = self.bounds
        return sum(
            get_tile_at(ram, x, y) == PLANTED_GRASS_TILE
            for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
        )

    def _chunk_bag_ready(self, ram: np.ndarray, origin: Tuple[int, int]) -> bool:
        """True when the centre is hoed and at least MIN_BAG_TILES squares are.

        The bag is nine squares around the farmer. A square the ROM cannot seed
        (stone, stump, weed) is skipped, so it only shrinks the yield. Standing
        on a non-hoed centre would lose that square and the walk-off risk.
        """
        ox, oy = origin
        fresh = sum(get_tile_at(ram, x, y) in _FRESH_SOIL
                    for y in range(oy, oy + self.chunk_size)
                    for x in range(ox, ox + self.chunk_size))
        return fresh >= MIN_BAG_TILES and get_tile_at(ram, ox + 1, oy + 1) in _FRESH_SOIL

    def _scan_targets(self, ram: np.ndarray) -> List[Tuple[int, int]]:
        """Find tiles matching the current phase within full bounds (for tests/stats)."""
        target_ids = self._target_ids()
        x_min, y_min, x_max, y_max = self.bounds
        results = []
        for ty in range(y_min, y_max + 1):
            for tx in range(x_min, x_max + 1):
                tid = get_tile_at(ram, tx, ty)
                if tid in target_ids and (tx, ty) not in self._failed_tiles and (tx, ty) not in self._no_go_tiles:
                    results.append((tx, ty))
        return results

    def _scan_chunk(self, ram: np.ndarray, origin: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find target tiles within a single chunk."""
        target_ids = self._target_ids()
        x_min, y_min, x_max, y_max = self.bounds
        cs = self.chunk_size
        ox, oy = origin
        results = []
        for ty in range(oy, min(oy + cs, y_max + 1)):
            for tx in range(ox, min(ox + cs, x_max + 1)):
                if tx < x_min or ty < y_min:
                    continue
                tid = get_tile_at(ram, tx, ty)
                if tid in target_ids and (tx, ty) not in self._failed_tiles and (tx, ty) not in self._no_go_tiles:
                    results.append((tx, ty))
        return results

    def _face_dir(self, player: Tuple[int, int], target: Tuple[int, int]) -> str:
        dx = target[0] - player[0]
        dy = target[1] - player[1]
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _first_phase(self) -> str:
        return Phase.CUT if self.cut_only else Phase.CLEAR

    def _after(self, phase: str) -> Optional[str]:
        """Phase that follows ``phase`` in one chunk, or None when it is done."""
        if phase == Phase.CLEAR:
            return Phase.TILL
        if phase == Phase.TILL and not self.till_only:
            return Phase.PLANT
        return None

    def _next_chunk(self) -> None:
        self._chunk_index += 1
        self._phase = self._first_phase()
        self._chunk_origin = None

    def _approach(self, ram: np.ndarray, target: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
        """Nearest neighbour of ``target`` with a short walk, or None.

        ``find_approach`` takes the nearest neighbour even when a weed wall makes
        it a 100-step detour, which sends the farmer across the map.
        """
        tx, ty = target
        pos = self._navigator.current_pos
        ranked = sorted(
            ((tx + dx, ty + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))),
            key=lambda c: abs(c[0] * TILE_SIZE + 8 - pos.x) + abs(c[1] * TILE_SIZE + 8 - pos.y))
        for cand in ranked:
            if not self._pathfinder.is_walkable(ram, *cand):
                continue
            path = self._pathfinder.find_path(ram, self._navigator.current_tile, cand)
            if path is not None and len(path) <= MAX_WALK_TILES:
                return cand, path
        return None

    def _start_walk(self, target: Tuple[int, int], stand: Tuple[int, int],
                    path: List[Tuple[int, int]]) -> None:
        self._target_tile = target
        self._approach_tile = stand
        self._navigator.path = path
        self._navigator.stasis = 0
        self._steps_on_target = 0
        self._state = "navigate"

    def _handle_scan(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Per-chunk till, plant: finish one 3x3 stand, then advance."""
        while self._chunk_index < len(self._chunk_order):
            origin = self._chunk_order[self._chunk_index]
            targets = self._scan_chunk(ram, origin)

            if not targets:
                nxt = self._after(self._phase)
                if nxt is None:
                    self._next_chunk()
                else:
                    self._phase = nxt
                continue

            # Log chunk/phase transitions
            if self._chunk_origin != origin or self._chunk_phase_logged != self._phase:
                self._chunk_origin = origin
                self._chunk_phase_logged = self._phase
                cs = self.chunk_size
                retry = f" retry={self._retry_pass}" if self._retry_pass > 0 else ""
                print(f"[GRASS] {self._phase.upper()} chunk ({origin[0]},{origin[1]})-"
                      f"({origin[0]+cs-1},{origin[1]+cs-1})  {len(targets)} tiles{retry}")

            # One bag is the 3x3 around the chunk center, not a lone 0x07.
            if self._phase == Phase.PLANT:
                stand = (origin[0] + 1, origin[1] + 1)
                if not self._chunk_bag_ready(ram, origin):
                    ids = " ".join(f"{get_tile_at(ram, x, y):02X}" for y in range(origin[1], origin[1] + 3)
                                   for x in range(origin[0], origin[0] + 3))
                    print(f"[GRASS] skip chunk ({origin[0]},{origin[1]}) bag not ready tiles={ids}")
                    self._next_chunk()
                    continue
                path = self._pathfinder.find_path(ram, self._navigator.current_tile, stand)
                if path is None:
                    self._failures += 1
                    self._next_chunk()
                    continue
                self._start_walk(stand, stand, path)
                return None
            centre = (origin[0] + 1, origin[1] + 1)
            if self._phase != Phase.CUT and centre in self._failed_tiles:
                print(f"[GRASS] skip chunk ({origin[0]},{origin[1]}) centre {centre} would not clear/hoe")
                self._next_chunk()
                continue
            # Centre first (the farmer must end up standing on it); never the
            # tile under his feet while another target remains.
            here = self._navigator.current_tile
            targets.sort(key=lambda t: (t == here, t != centre))
            for target in targets:
                # CLEAR/TILL/CUT: navigate adjacent to target, face it, use the tool
                found = self._approach(ram, target)
                if found is None:
                    continue
                approach, path = found
                self._start_walk(target, approach, path)
                if self.debug:
                    print(f"[GRASS] -> ({target[0]},{target[1]}) via ({approach[0]},{approach[1]}) path={len(path)}")
                return None

            # All targets in this chunk are unreachable for current phase
            nxt = self._after(self._phase)
            if nxt is None:
                self._next_chunk()
            else:
                self._phase = nxt

        # All chunks exhausted - try retry pass
        if self._failed_tiles and self._retry_pass < self.max_retry_passes:
            self._retry_pass += 1
            n_failed = len(self._failed_tiles)
            self._failed_tiles.clear()
            self._swing_fails.clear()
            self._chunk_index = 0
            self._chunk_origin = None
            self._phase = self._first_phase()
            print(f"[GRASS] Pass done. Retry {self._retry_pass}/{self.max_retry_passes} ({n_failed} failed tiles)")
            return None

        msg = (f"cleared={self.cleared_count} tilled={self.tilled_count} planted={self.planted_count} "
               f"grass70={self._grass70(ram)}")
        if self.cut_only:
            hay = int(read_ram_value(ram, "stored_grass"))
            return TaskResult(status=TaskStatus.SUCCESS if hay > self._hay_before else TaskStatus.FAILURE,
                              reason=f"stored_grass={self._hay_before}->{hay}")
        if self.target_tiles and int(read_ram_value(ram, "planted_grass")) < self.target_tiles:
            if self._begin_short_sow(ram):
                return None
            return TaskResult(status=TaskStatus.FAILURE, reason=f"pasture short: {msg}")
        print(f"[GRASS] Complete: {msg}")
        return TaskResult(status=TaskStatus.SUCCESS, reason=msg)

    def _begin_short_sow(self, ram: np.ndarray) -> bool:
        """Sow the held bag when planted_grass is still under 32.

        A full 3x3 is preferred earlier in the scan. One fresh tile still
        spends the bag and is the only way 31 qualifies for the cow gate.
        """
        if self._short_sow_used or self.cut_only or self.till_only:
            return False
        planted = int(read_ram_value(ram, "planted_grass"))
        seeds = int(read_ram_value(ram, "grass_seeds"))
        if seeds <= 0 or planted >= 32:
            return False
        x0, y0, x1, y1 = self.bounds
        best: Optional[Tuple[int, int]] = None
        best_n = 0
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x, y) in self._no_go_tiles:
                    continue
                if get_tile_at(ram, x, y) not in _FRESH_SOIL:
                    continue
                fresh = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if get_tile_at(ram, x + dx, y + dy) in _FRESH_SOIL:
                            fresh += 1
                if fresh > best_n:
                    best_n = fresh
                    best = (x, y)
        if best is None:
            return False
        path = self._pathfinder.find_path(ram, self._navigator.current_tile, best)
        if path is None:
            return False
        self._short_sow_used = True
        self._phase = Phase.PLANT
        self._target_tile = best
        self._approach_tile = best
        self._navigator.path = path
        self._navigator.stasis = 0
        self._steps_on_target = 0
        self._state = "navigate"
        print(f"[GRASS] sow short bag at {best} planted={planted} seeds={seeds} fresh={best_n}")
        return True

    def _handle_navigate(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._target_tile is None or self._approach_tile is None:
            self._state = "scan"
            return None

        # Check target still valid
        tid = get_tile_at(ram, *self._target_tile)
        target_ids = self._target_ids()
        if tid not in target_ids:
            self._state = "scan"
            return None

        # Arrived?
        if self._navigator.current_tile == self._approach_tile:
            self._state = "center"
            return None

        # Stuck recovery
        if self._navigator.stasis > self.stasis_repath and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            path = self._pathfinder.find_path(ram, self._navigator.current_tile, self._approach_tile)
            if path:
                self._navigator.path = path
                self._navigator.stasis = 0
            else:
                self._failures += 1
                self._failed_tiles.add(self._target_tile)
                self._state = "scan"
                if self._failures >= self.max_failures:
                    return TaskResult(status=TaskStatus.FAILURE, reason="too many nav failures")
                return None

        action = self._navigator.follow_path(ram)
        if action is not None:
            self._action_queue.append(action)
        return None

    def _handle_center(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Center on approach tile before acting."""
        if self._approach_tile is None:
            self._state = "scan"
            return None
        tol = 1 if self._phase == Phase.PLANT else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is None:
            self._state = "act"
        else:
            self._action_queue.append(center_action)
        return None

    def _handle_act(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Use tool on target tile. Matches farm_clearer timing pattern."""
        if self._target_tile is None or self._approach_tile is None:
            self._state = "scan"
            return None

        tx, ty = self._target_tile

        # Re-validate target
        tid = get_tile_at(ram, tx, ty)
        target_ids = self._target_ids()
        if tid not in target_ids:
            if self._phase == Phase.TILL and tid == FRESH_TILLED:
                self.tilled_count += 1
            self._state = "scan"
            return None

        # Position check depends on phase
        player = self._navigator.current_tile
        if self._phase == Phase.PLANT:
            # Must be ON target tile (grass seeds plant 3x3 around player)
            if player != self._approach_tile:
                self._state = "navigate"
                return None
        else:
            # Must be adjacent to target tile (hoe acts on tile in front)
            if tile_dist(player, self._target_tile) > 1:
                self._state = "navigate"
                return None

        # Wait for queued actions to finish
        if self._action_queue:
            return None

        # Wait for input lock to clear and player to settle (same as farm_clearer)
        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1 or self._navigator.stasis < 6:
            return None

        # Re-center on approach tile before each use (drift correction)
        tol = 1 if self._phase == Phase.PLANT else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is not None:
            self._action_queue.append(center_action)
            return None

        # Check correct tool is equipped
        wanted_tool = self._wanted_tool()
        if wanted_tool is not None and self._tool_mgr.current != wanted_tool:
            self._tool_mgr.start_search()
            self._state = "tool_switch"
            return None

        self._act_count += 1
        phase = self._phase.upper()
        if self._phase == Phase.PLANT:
            self._bag_seeds = int(read_ram_value(ram, "grass_seeds"))
            self._bag_grass = int(read_ram_value(ram, "planted_grass"))
            self._verify_wait = 0
            stood = get_tile_at(ram, *player)
            if stood not in _FRESH_SOIL:
                self._state = "scan"
                return None
            # A direction hold walks off 0x07 and the bag misses the 3x3.
            direction = "still"
        else:
            # Adjacent to target - face it then Y to hoe
            direction = self._face_dir(player, self._target_tile)

        if self._act_count <= 20 or self.debug:
            print(f"[GRASS] {phase} ({tx},{ty}) [0x{tid:02X}] from ({player[0]},{player[1]}) facing {direction} tool=0x{self._tool_mgr.current:02X}")

        if self._phase == Phase.CLEAR:
            # Face, settle, hold A to lift (no stamina), then wait out the pickup.
            self._action_queue.extend([make_action(**{direction: True}) for _ in range(3)])
            self._action_queue.extend([make_action() for _ in range(4)])
            self._action_queue.extend([make_action(a=True) for _ in range(18)])
            self._action_queue.extend([make_action() for _ in range(20)])
            self._state = "verify"
            return None

        # Face direction → settle → Y press → cooldown
        # Tile change takes ~26 frames after Y ends, so cooldown must be >= 30
        if self._phase != Phase.PLANT:
            self._action_queue.extend([make_action(**{direction: True}) for _ in range(4)])
            self._action_queue.extend([make_action() for _ in range(6)])
        self._action_queue.extend(use_tool(frames=20, cooldown=30))
        self._state = "verify"
        return None

    def _handle_verify(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Check if tool use changed the target tile."""
        if self._action_queue:
            return None

        # Wait for input lock to clear (animation done)
        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1:
            return None

        if self._target_tile is None:
            self._state = "scan"
            return None

        tx, ty = self._target_tile
        tid = get_tile_at(ram, tx, ty)

        if self._phase == Phase.CLEAR:
            if tid in LIFTABLE_TILES:
                self._failures += 1
                self._swing_fails[(tx, ty)] = self._swing_fails.get((tx, ty), 0) + 1
                if self._swing_fails[(tx, ty)] >= 2:
                    self._failed_tiles.add((tx, ty))
            else:
                self.cleared_count += 1
            if not hands_are_clear(ram):
                self._toss_tries = 0
                self._state = "toss"
                return None
        elif self._phase == Phase.TILL:
            # Success = tile changed to freshly hoed (0x07)
            if tid == FRESH_TILLED:
                self.tilled_count += 1
                if self._act_count <= 20 or self.debug:
                    print(f"[GRASS] TILL OK ({tx},{ty}) -> 0x{tid:02X} tilled={self.tilled_count}")
            else:
                self._failures += 1
                # One unregistered swing is retried; a second miss gives up on
                # the square (the bag then skips it) instead of looping.
                self._swing_fails[(tx, ty)] = self._swing_fails.get((tx, ty), 0) + 1
                if self._swing_fails[(tx, ty)] >= 2:
                    self._failed_tiles.add((tx, ty))
                if self._act_count <= 20 or self.debug:
                    print(f"[GRASS] TILL FAIL ({tx},{ty}) still 0x{tid:02X} failures={self._failures}")
        elif self._phase == Phase.CUT:
            if tid in MATURE_GRASS_TILES:
                self._failures += 1
                self._failed_tiles.add((tx, ty))
        else:
            # The ROM visits one square per frame and drops the seed count on
            # the ninth, so a bag is spent exactly when grass_seeds falls.
            player = self._navigator.current_tile
            grass = int(read_ram_value(ram, "planted_grass"))
            seeds = int(read_ram_value(ram, "grass_seeds"))
            if seeds >= self._bag_seeds:
                if self._verify_wait < 240:  # the animation outlives the input lock
                    self._verify_wait += 1
                    return None
                self._failures += 1  # the press never registered; scan re-presses
            else:
                self.planted_count += grass - self._bag_grass
                print(f"[GRASS] PLANT @ ({player[0]},{player[1]}) +{grass - self._bag_grass} "
                      f"grass={self._bag_grass}->{grass} grass70={self._grass70(ram)} seeds={seeds}")
                if grass == self._bag_grass:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=f"bag spent, planted_grass stayed {grass}")
                self._next_chunk()  # one bag per stand, even if a square stayed bare

        if self._failures >= self.max_failures:
            return TaskResult(status=TaskStatus.FAILURE, reason="too many till failures")

        self._state = "scan"
        return None

    def _toss_faces(self, ram: np.ndarray) -> List[str]:
        """Throw directions, most open ground first (a blocked toss keeps the item)."""
        px, py = self._navigator.current_tile
        runs = []
        for order, (face, dx, dy) in enumerate((("down", 0, 1), ("left", -1, 0), ("right", 1, 0), ("up", 0, -1))):
            run = 0
            while run < 3 and get_tile_at(ram, px + dx * (run + 1), py + dy * (run + 1)) not in LIFTABLE_TILES \
                    and self._pathfinder.is_walkable(ram, px + dx * (run + 1), py + dy * (run + 1)):
                run += 1
            runs.append((-run, order, face))
        return [face for _, _, face in sorted(runs)]

    def _handle_toss(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Throw the lifted debris away from the stand until the hands are clear."""
        if hands_are_clear(ram):
            self._state = "scan"
            return None
        if self._toss_tries >= 4:
            return TaskResult(status=TaskStatus.FAILURE,
                              reason=f"held 0x{read_held_item(ram):02X} would not toss")
        face = self._toss_faces(ram)[self._toss_tries]
        self._toss_tries += 1
        self._action_queue.extend(toss_held_actions(face=face))
        return None

    def _handle_tool_switch(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Cycle tools until we find the one we need."""
        wanted = self._wanted_tool()
        self._tool_mgr.update(ram)
        current = self._tool_mgr.current

        if current == wanted:
            if self.debug:
                print(f"[GRASS] Found {self._phase} tool")
            # The first swing after an X cycle is swallowed by the switch animation.
            self._action_queue.extend(make_action() for _ in range(24))
            self._state = "center"  # re-center before acting
            return None

        self._tool_mgr.record()

        if self._tool_mgr.cycle_complete():
            # Wrapped around without finding tool
            name = {Phase.CUT: "SICKLE", Phase.TILL: "HOE"}.get(self._phase, "GRASS_SEEDS")
            print(f"[GRASS] Tool {name} not found in inventory")
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{name} not in inventory")

        self._action_queue.extend(cycle_tool())
        return None

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, world: WorldState) -> TaskResult:
        if self._total_steps == 0 and not (self.cut_only or self.till_only) and self._seeds_before <= 0:
            return TaskResult(status=TaskStatus.FAILURE, reason="no grass seeds in hand")
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)
        self._total_steps += 1
        self._steps_on_target += 1

        if self.debug and self._total_steps % self.debug_interval == 0:
            cur = self._navigator.current_tile
            phase = self._phase.upper()
            print(f"[GRASS] step={self._total_steps} phase={phase} state={self._state} pos={cur} "
                  f"tilled={self.tilled_count} planted={self.planted_count} failures={self._failures}")

        # ---- Power Berry detection ----
        cur_pb = int(world.ram[ADDR_POWER_BERRIES]) if ADDR_POWER_BERRIES < len(world.ram) else 0
        if cur_pb > self._prev_berry_count:
            gained = cur_pb - self._prev_berry_count
            self.berries_found += gained
            self._prev_berry_count = cur_pb
            self._action_queue.clear()
            self._berry_dialog_frames = 90
            print(f"[GRASS] POWER BERRY! total={cur_pb}")

        if self._berry_dialog_frames > 0:
            self._berry_dialog_frames -= 1
            action = make_action(a=True) if self._berry_dialog_frames % 2 == 0 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="berry_dialog")

        # ---- Timeout per target ----
        if self._steps_on_target > self.max_steps_per_target and self._target_tile is not None:
            self._failed_tiles.add(self._target_tile)
            self._failures += 1
            self._state = "scan"
            self._target_tile = None
            self._action_queue.clear()
            if self._failures >= self.max_failures:
                return TaskResult(status=TaskStatus.FAILURE, reason="too many target timeouts")

        # Drain action queue first - tool animations must complete even during input lock
        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        # Dialog dismissal (only after queue is empty so tool swings aren't interrupted)
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        seeds = int(read_ram_value(world.ram, "grass_seeds"))
        daily_limit = min(self.bag_limit, self._seeds_before)
        if input_lock == 1 and self.bag_limit and self._seeds_before - seeds >= daily_limit:
            grass = int(read_ram_value(world.ram, "planted_grass"))
            return TaskResult(status=TaskStatus.SUCCESS if grass > self._grass_before else TaskStatus.FAILURE,
                              reason=(f"pasture progress {self._grass_before}->{grass} "
                                      f"grass70={self._grass70(world.ram)}; remaining tomorrow"))
        # Between bags only: the gate count climbs square by square mid-bag.
        if input_lock == 1 and self._state == "scan" and self.target_tiles \
                and int(read_ram_value(world.ram, "planted_grass")) >= self.target_tiles:
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"planted_grass={int(read_ram_value(world.ram, 'planted_grass'))}")
        if input_lock != 1:
            action = make_action(a=True) if self._total_steps % 2 == 0 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="dialog")

        # State dispatch
        handlers = {
            "scan": self._handle_scan,
            "navigate": self._handle_navigate,
            "center": self._handle_center,
            "act": self._handle_act,
            "verify": self._handle_verify,
            "toss": self._handle_toss,
            "tool_switch": self._handle_tool_switch,
        }

        handler = handlers.get(self._state)
        if handler:
            result = handler(world.ram)
            if result is not None:
                return result

        # Drain any actions queued by handlers
        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def phase_text(self) -> str:
        return f"{self._phase}:{self._state}"

    @property
    def progress_text(self) -> str:
        s = f"cleared={self.cleared_count} tilled={self.tilled_count} planted={self.planted_count}"
        if self._retry_pass > 0:
            s += f" retry={self._retry_pass}"
        if self._failures:
            s += f" till_fail={self._failures}"
        if self.berries_found:
            s += f" berries={self.berries_found}"
        return s
