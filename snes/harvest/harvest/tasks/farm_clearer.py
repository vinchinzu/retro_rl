"""
Farm clearing module - Phase-based debris clearing with tool management.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Set
from collections import deque
import os
import json

import numpy as np

from retro_harness.actions import action_names, snes_action

from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_STAMINA,
    ADDR_TILEMAP,
    ADDR_TOOL,
    ADDR_X,
    ADDR_Y,
    CLEARABLE_DEBRIS_TYPES,
    DEBRIS_TOOL,
    FARM_WALKABLE,
    LARGE_ROCK_TL,
    LARGE_ROCK_TILES,
    LIFTABLE_TILES,
    MAP_WIDTH,
    POND_CHARACTERISTIC_TILES,
    STALE_TILE_IDS,
    STUMP_TL,
    STUMP_TILES,
    TILE_SIZE,
    TILE_TO_DEBRIS,
    DebrisType,
    Tool,
    debris_footprint,
    get_tile_at as _catalog_get_tile_at,
    is_multitile_debris_anchor,
)


# =============================================================================
# CONSTANTS
# =============================================================================

WALKABLE_TILES = FARM_WALKABLE

# Hard obstacles first so pathing opens up, then cheap lifts.
DEFAULT_PRIORITY: List[DebrisType] = [
    DebrisType.ROCK,
    DebrisType.STUMP,
    DebrisType.STONE,
    DebrisType.WEED,
]

# SNES only loads ~16x14 tiles; BFS beyond this sees stale 0x72/0xFF.
VIEWPORT_HOP_TILES = 7
# Hammer/axe hits cost 2 stamina; stop before a multi-hit cannot finish.
MIN_CLEAR_STAMINA = 4


# =============================================================================
# DATA
# =============================================================================

@dataclass
class Point:
    x: int
    y: int

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


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
        # Base tools need 6 consecutive hits on stump / large rock.
        if self.debris_type == DebrisType.ROCK or self.debris_type == DebrisType.STUMP:
            return 6
        return 1

    @property
    def footprint(self) -> Tuple[Tuple[int, int], ...]:
        return debris_footprint(self.tile, self.tile_id)


# =============================================================================
# UTILITIES
# =============================================================================

def make_action(**buttons) -> np.ndarray:
    """Compatibility wrapper around the shared named-button builder."""

    return snes_action(dtype=np.int32, **buttons)


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


def get_pos_from_ram(ram: np.ndarray) -> Point:
    if ADDR_X + 1 < len(ram) and ADDR_Y + 1 < len(ram):
        x = int(ram[ADDR_X]) + (int(ram[ADDR_X + 1]) << 8)
        y = int(ram[ADDR_Y]) + (int(ram[ADDR_Y + 1]) << 8)
        return Point(x, y)
    return Point(0, 0)


def get_tile_at(ram: np.ndarray, tx: int, ty: int) -> int:
    return _catalog_get_tile_at(ram, tx, ty)


def manhattan(p1: Point, p2: Point) -> int:
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


def tile_dist(t1: Tuple[int, int], t2: Tuple[int, int]) -> int:
    return abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])


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

            # Collapse 2x2 families to their top-left anchor only.
            if tile_id in STUMP_TILES | LARGE_ROCK_TILES:
                if not is_multitile_debris_anchor(tile_id):
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


# =============================================================================
# PATHFINDER
# =============================================================================

class Pathfinder:
    def __init__(self, scanner: TileScanner, walkable_tiles: Optional[Set[int]] = None):
        self.scanner = scanner
        self.walkable_tiles = walkable_tiles if walkable_tiles is not None else WALKABLE_TILES
        self._dynamic_walkable_tiles = walkable_tiles is None
        self.no_go_tiles: Set[Tuple[int, int]] = set()
        self.temp_blocked: Set[Tuple[int, int]] = set()
        self.extra_walkable: Set[Tuple[int, int]] = set()  # tiles treated as walkable (e.g. crop tiles)

    def base_walkable_tiles(self, ram: np.ndarray) -> Set[int]:
        if not self._dynamic_walkable_tiles:
            return self.walkable_tiles
        try:
            from harvest.maps.map_config import get_walkable_tiles

            tilemap_id = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0x00
            return get_walkable_tiles(tilemap_id)
        except Exception:
            return self.walkable_tiles

    def base_no_go_tiles(self, ram: np.ndarray) -> Set[Tuple[int, int]]:
        try:
            from harvest.maps.map_config import get_no_go_tiles

            tilemap_id = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0x00
            return set(get_no_go_tiles(tilemap_id))
        except Exception:
            return set()

    def is_walkable(self, ram: np.ndarray, tx: int, ty: int, walkable_override: Optional[Set[int]] = None, current_pos: Optional[Tuple[int, int]] = None) -> bool:
        # Always allow moving from current tile
        if current_pos and (tx, ty) == current_pos:
            return True

        if (tx, ty) in self.no_go_tiles or (tx, ty) in self.base_no_go_tiles(ram) or (tx, ty) in self.temp_blocked:
            return False
        tile_id = get_tile_at(ram, tx, ty)
        if tile_id in self.base_walkable_tiles(ram):
            return True
        if (tx, ty) in self.extra_walkable:
            return True
        if walkable_override and (tx, ty) in walkable_override:
            return True
        return False

    def find_path(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable_override: Optional[Set[int]] = None,
        *,
        max_steps: Optional[int] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        if start == goal:
            return []

        queue = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {
            start: None
        }

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                break

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < MAP_WIDTH
                    and 0 <= ny < MAP_WIDTH
                    and (nx, ny) not in came_from
                ):
                    is_goal = (nx, ny) == goal
                    if (
                        is_goal
                        and walkable_override
                        and (nx, ny) in walkable_override
                    ):
                        came_from[(nx, ny)] = (cx, cy)
                        continue

                    if self.is_walkable(
                        ram,
                        nx,
                        ny,
                        walkable_override=walkable_override,
                        current_pos=start,
                    ):
                        came_from[(nx, ny)] = (cx, cy)
                        queue.append((nx, ny))

        if goal not in came_from:
            # Live SNES maps go stale outside the viewport, so a full path to a
            # distant goal often cannot be closed. Fall back to a hop toward it.
            if max_steps is not None and max_steps > 0:
                return self.find_hop_toward(
                    ram,
                    start,
                    goal,
                    walkable_override=walkable_override,
                    max_steps=max_steps,
                )
            return None

        path: List[Tuple[int, int]] = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]  # type: ignore[assignment]
        path.reverse()
        if max_steps is not None and len(path) > max_steps:
            return path[:max_steps]
        return path

    def find_hop_toward(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable_override: Optional[Set[int]] = None,
        *,
        max_steps: int = VIEWPORT_HOP_TILES,
    ) -> Optional[List[Tuple[int, int]]]:
        """BFS within ``max_steps``; return a path to the closest reached tile."""
        if start == goal:
            return []
        if max_steps <= 0:
            return None

        queue: deque[Tuple[Tuple[int, int], int]] = deque([(start, 0)])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        best = start
        best_dist = manhattan(
            Point(start[0] * TILE_SIZE + 8, start[1] * TILE_SIZE + 8),
            Point(goal[0] * TILE_SIZE + 8, goal[1] * TILE_SIZE + 8),
        )

        while queue:
            (cx, cy), depth = queue.popleft()
            if depth >= max_steps:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH):
                    continue
                if (nx, ny) in came_from:
                    continue
                if not self.is_walkable(
                    ram,
                    nx,
                    ny,
                    walkable_override=walkable_override,
                    current_pos=start,
                ):
                    continue
                came_from[(nx, ny)] = (cx, cy)
                queue.append(((nx, ny), depth + 1))
                dist = manhattan(
                    Point(nx * TILE_SIZE + 8, ny * TILE_SIZE + 8),
                    Point(goal[0] * TILE_SIZE + 8, goal[1] * TILE_SIZE + 8),
                )
                if dist < best_dist:
                    best = (nx, ny)
                    best_dist = dist

        if best == start:
            return None

        path: List[Tuple[int, int]] = []
        cur: Optional[Tuple[int, int]] = best
        while cur is not None and cur != start:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path or None

    def find_approach(
        self,
        ram: np.ndarray,
        target: Tuple[int, int],
        player_pos: Point,
        walkable_override: Optional[Set[int]] = None,
        footprint: Optional[Tuple[Tuple[int, int], ...]] = None,
    ) -> Optional[Tuple[int, int]]:
        best, best_dist = None, None
        cells = footprint if footprint else (target,)
        occupied = set(cells)
        for tx, ty in cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ax, ay = tx + dx, ty + dy
                if (ax, ay) in occupied:
                    continue
                if (
                    0 <= ax < MAP_WIDTH
                    and 0 <= ay < MAP_WIDTH
                    and self.is_walkable(
                        ram, ax, ay, walkable_override=walkable_override
                    )
                ):
                    dist = manhattan(
                        Point(ax * TILE_SIZE + 8, ay * TILE_SIZE + 8),
                        player_pos,
                    )
                    if best_dist is None or dist < best_dist:
                        best, best_dist = (ax, ay), dist
        return best

    def find_nearest_walkable(self, ram: np.ndarray, target: Tuple[int, int], max_radius: int = 4, walkable_override: Optional[Set[int]] = None) -> Optional[Tuple[int, int]]:
        tx, ty = target
        if self.is_walkable(ram, tx, ty, walkable_override=walkable_override):
            return (tx, ty)
        for radius in range(1, max_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) != radius:
                        continue
                    ax, ay = tx + dx, ty + dy
                    if 0 <= ax < MAP_WIDTH and 0 <= ay < MAP_WIDTH and self.is_walkable(ram, ax, ay, walkable_override=walkable_override):
                        return (ax, ay)
        return None


# =============================================================================
# NAVIGATOR
# =============================================================================

class Navigator:
    def __init__(self, pathfinder: Pathfinder):
        self.pathfinder = pathfinder
        self.current_pos = Point(0, 0)
        self.path: List[Tuple[int, int]] = []
        self.stasis = 0

    def update(self, ram: np.ndarray):
        new_pos = get_pos_from_ram(ram)
        new_tile = (new_pos.x // TILE_SIZE, new_pos.y // TILE_SIZE)
        old_tile = (self.current_pos.x // TILE_SIZE, self.current_pos.y // TILE_SIZE)
        # Reset stasis only on tile-level movement, not 1px oscillation
        self.stasis = 0 if new_tile != old_tile else self.stasis + 1
        self.current_pos = new_pos

    @property
    def current_tile(self) -> Tuple[int, int]:
        return (self.current_pos.x // TILE_SIZE, self.current_pos.y // TILE_SIZE)

    def at_tile(self, tile: Tuple[int, int], tolerance: int = 2) -> bool:
        target = Point(tile[0] * TILE_SIZE + 8, tile[1] * TILE_SIZE + 8)
        return abs(self.current_pos.x - target.x) <= tolerance and abs(self.current_pos.y - target.y) <= tolerance

    def center_on_tile(self, tile: Tuple[int, int], tolerance: int = 1) -> Optional[np.ndarray]:
        """Micro-adjust to be centered on the given tile."""
        tgt_x = tile[0] * TILE_SIZE + 8
        tgt_y = tile[1] * TILE_SIZE + 8
        
        dx = tgt_x - self.current_pos.x
        dy = tgt_y - self.current_pos.y
        
        if abs(dx) < tolerance and abs(dy) < tolerance:
            return None
            
        action = np.zeros(12, dtype=np.int32)
        # We don't hold B (run) for micro-centering to avoid overshooting
        # Move along the dominant axis only to avoid diagonal drift.
        if abs(dx) >= abs(dy) and abs(dx) >= tolerance:
            action[7] = 1 if dx > 0 else 0  # Right
            action[6] = 1 if dx < 0 else 0  # Left
        elif abs(dy) >= tolerance:
            action[5] = 1 if dy > 0 else 0  # Down
            action[4] = 1 if dy < 0 else 0  # Up
            
        return action

    def follow_path(self, ram: np.ndarray) -> Optional[np.ndarray]:
        while self.path and self.current_tile == self.path[0]:
            self.path.pop(0)

        if not self.path:
            return None

        next_tile = self.path[0]
        curr_tile = self.current_tile

        # If first path tile is not adjacent (> 1 tile away on either axis),
        # the path is stale (e.g. centering overshoot moved us far away).
        # Clear and let BFS recompute from our actual position.
        dx_gap = abs(next_tile[0] - curr_tile[0])
        dy_gap = abs(next_tile[1] - curr_tile[1])
        if dx_gap > 1 or dy_gap > 1:
            self.path = []
            self.stasis = 0
            return None

        tgt_x = curr_tile[0] * TILE_SIZE + 8
        tgt_y = curr_tile[1] * TILE_SIZE + 8

        dx_next = next_tile[0] - curr_tile[0]
        dy_next = next_tile[1] - curr_tile[1]
        
        # PROACTIVE CENTERING:
        # If we are moving horizontally, we MUST be vertically centered (Y).
        # If we are moving vertically, we MUST be horizontally centered (X).
        # Use tolerance=3 to avoid 1px oscillation that causes infinite stasis loops.
        # If centering has been stuck (stasis > 30), skip it entirely.
        # PROACTIVE CENTERING:
        # If we are moving horizontally, we MUST be vertically centered (Y).
        # If we are moving vertically, we MUST be horizontally centered (X).
        # Tolerance=3 avoids 1px oscillation. Skip if stuck (stasis>30) to
        # let the bot push through rather than centering forever.
        CENTER_TOL = 3
        if self.stasis < 30:
            if dx_next != 0 and abs(self.current_pos.y - tgt_y) >= CENTER_TOL:
                return self.center_on_tile(curr_tile, tolerance=CENTER_TOL)
            if dy_next != 0 and abs(self.current_pos.x - tgt_x) >= CENTER_TOL:
                return self.center_on_tile(curr_tile, tolerance=CENTER_TOL)

        if not self.pathfinder.is_walkable(ram, *next_tile, current_pos=curr_tile):
            val = get_tile_at(ram, *next_tile)
            if val in STALE_TILE_IDS:
                self.path = []
                self.stasis = 0
                return None
            # Log more clearly why we are blocked
            print(f"[NAVIGATOR] Blocked! tile={next_tile} id=0x{val:02X} walkable={val in self.pathfinder.base_walkable_tiles(ram)} temp_blocked={next_tile in self.pathfinder.temp_blocked} no_go={next_tile in self.pathfinder.no_go_tiles}")
            self.pathfinder.temp_blocked.add(next_tile)
            self.path = []
            self.stasis = 0
            return None

        direction = 'right' if dx_next > 0 else 'left' if dx_next < 0 else 'down' if dy_next > 0 else 'up'
        action = make_action(**{direction: True, 'b': True})
        
        if os.getenv("FENCE_DEBUG") == "1" and self.stasis > 0 and self.stasis % 60 == 0:
            print(f"[NAV] pos=({self.current_pos.x},{self.current_pos.y}) next={next_tile} target=({tgt_x},{tgt_y}) dir={direction} stasis={self.stasis} path_len={len(self.path)}")
            import sys; sys.stdout.flush()
            
        return action


# =============================================================================
# TOOL MANAGER
# =============================================================================

class ToolManager:
    def __init__(self):
        self.current = 0
        self.seen: Set[int] = set()
        self.start_id: Optional[int] = None

    def update(self, ram: np.ndarray):
        self.current = int(ram[ADDR_TOOL]) if ADDR_TOOL < len(ram) else 0

    def start_search(self):
        self.start_id = self.current
        self.seen = {self.current}

    def record(self):
        self.seen.add(self.current)

    def cycle_complete(self) -> bool:
        return self.start_id is not None and self.current == self.start_id and len(self.seen) > 1


# =============================================================================
# FARM CLEARER
# =============================================================================

class FarmClearer:
    """Phase-based farm clearing: rock → stump → stone → weed."""

    def __init__(self, priority: Optional[List[DebrisType]] = None):
        self.priority = priority or DEFAULT_PRIORITY.copy()

        self.scanner = TileScanner()
        self.pathfinder = Pathfinder(self.scanner)
        self.navigator = Navigator(self.pathfinder)
        self.tool_manager = ToolManager()

        self.current_phase: Optional[DebrisType] = None
        self.current_target: Optional[Target] = None
        self.approach_tile: Optional[Tuple[int, int]] = None
        self.action_queue: deque = deque()
        self.state = "scanning"

        self.failed_tiles: Set[Tuple[int, int]] = set()
        self.cleared_count = 0
        self.tiles_cleared: Set[Tuple[int, int]] = set()
        self.tile_attempts: Dict[Tuple[int, int, int], int] = {}
        self.frame_count = 0
        self.farm_bounds: Optional[Tuple[int, int, int, int]] = None

        self.prefer_lift_for_weeds = True
        self.prefer_lift_for_stones = False
        self.max_stasis = 120
        self.debug_interval = 300
        self.min_stamina = MIN_CLEAR_STAMINA
        self.stamina_exhausted = False
        self.tools_missing = False
        self.scan_miss_streak = 0
        self.max_scan_misses = 90

        self.searching_tool: Optional[Tool] = None
        self.tool_search_frames = 0

        self.startup_tasks: List[Dict] = []
        self.startup_index = 0
        self.startup_done = False
        self.task_queue: deque = deque()
        self.tasks_dir: Optional[str] = None

        self.target_hits = 0
        self.clearing_start_frame = 0
        self.suppress_move_frames = 0
        self._pending_lift_verify: Optional[Tuple[int, int]] = None
        self._init_no_go()

    def _init_no_go(self):
        default = "9,26;9,27;9,28;11,26;11,27;11,28;8,12;9,12;10,12"
        for entry in os.getenv("NO_GO_TILES", default).replace("|", ";").split(";"):
            parts = [p.strip() for p in entry.split(",") if p.strip()]
            if len(parts) == 2:
                try:
                    self.pathfinder.no_go_tiles.add((int(parts[0]), int(parts[1])))
                except ValueError:
                    pass

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def add_startup_task(self, task_type: str, **kwargs):
        self.startup_tasks.append({"type": task_type, **kwargs})

    def _load_task(self, name: str) -> Optional[List[np.ndarray]]:
        if not self.tasks_dir:
            return None
        path = os.path.join(self.tasks_dir, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return [np.array(frame, dtype=np.int32) for frame in data.get("frames", [])]

    def _emit_action(self, action: np.ndarray, src: str) -> np.ndarray:
        if self.suppress_move_frames > 0:
            self.suppress_move_frames -= 1
            # Strip directional inputs on tool-swing frames to prevent drift.
            # Direction-only frames (the initial face tap) pass through so the
            # character actually turns toward the target before swinging.
            if action[1] == 1:  # Y button pressed (tool use)
                action = action.copy()
                action[4:8] = 0
                src = f"{src}+suppress"
        if os.getenv("ACTION_DEBUG") == "1":
            buttons = action_to_names(action)
            if buttons != "none" or os.getenv("ACTION_DEBUG_VERBOSE") == "1" and self.frame_count % 30 == 0:
                print(f"[ACTION] frame={self.frame_count} state={self.state} src={src} buttons={buttons}")
        return action

    def _requested_startup_tools(self) -> Set[int]:
        wanted: Set[int] = set()
        for step in self.startup_tasks:
            if step.get("type") != "task":
                continue
            name = str(step.get("name", ""))
            mapping = {
                "get_hammer": int(Tool.HAMMER),
                "get_axe": int(Tool.AXE),
                "get_sickle": int(Tool.SICKLE),
                "get_hoe": int(Tool.HOE),
            }
            tool_id = mapping.get(name)
            if tool_id is not None:
                wanted.add(tool_id)
        return wanted

    def _enable_lift_only_mode(self, missing: List[int]) -> None:
        """Continue clearing weeds/stones by hand when hammer/axe are unavailable."""
        self.prefer_lift_for_weeds = True
        self.prefer_lift_for_stones = True
        # Prefer weeds first: they lift reliably and open crop-field paths.
        self.priority = [DebrisType.WEED, DebrisType.STONE]
        names = ", ".join(f"0x{tool:02X}" for tool in missing)
        print(
            f"[CLEARER] Startup missing tools: {names}; "
            "continuing with lift-only weeds/stones"
        )

    def _finalize_startup_tools(self) -> None:
        """Re-scan carry tools after recordings and flag missing gear."""
        self.tool_manager.start_search()
        self.tool_manager.record()
        # Single-frame snapshot is enough: recordings should leave tools selected.
        have = set(self.tool_manager.seen)
        have.add(self.tool_manager.current)
        missing = sorted(self._requested_startup_tools() - have)

        # Even without startup recordings, rocks/stumps need hammer/axe.
        needs_hammer = DebrisType.ROCK in self.priority or DebrisType.STONE in self.priority
        needs_axe = DebrisType.STUMP in self.priority
        if needs_hammer and int(Tool.HAMMER) not in have:
            missing = sorted(set(missing) | {int(Tool.HAMMER)})
        if needs_axe and int(Tool.AXE) not in have:
            missing = sorted(set(missing) | {int(Tool.AXE)})

        if missing:
            self.tools_missing = True
            self._enable_lift_only_mode(missing)
        else:
            self.tools_missing = False

    def _run_startup(self, ram: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        if self.startup_done:
            return False, None

        # One-time tool inventory scan at the very beginning
        if not hasattr(self, '_tool_scan_done'):
            self._tool_scan_done = False
            self._tool_scan_frames = 0
            self.tool_manager.start_search()

        if not self._tool_scan_done:
            self._tool_scan_frames += 1
            self.tool_manager.record()

            # Scan complete after one full cycle or timeout
            if self.tool_manager.cycle_complete() or self._tool_scan_frames > 60:
                self._tool_scan_done = True
                tools_found = [f"0x{t:02X}" for t in sorted(self.tool_manager.seen)]
                print(f"[CLEARER] Tool inventory: {', '.join(tools_found)}")
            else:
                # Continue cycling
                if self._tool_scan_frames % 6 == 0:  # Cycle every 6 frames
                    self.action_queue.extend(cycle_tool())
                return True, self.action_queue.popleft() if self.action_queue else make_action()

        if self.task_queue:
            return True, self.task_queue.popleft()

        if self.startup_index >= len(self.startup_tasks):
            self._finalize_startup_tools()
            self.startup_done = True
            print("[CLEARER] Startup complete")
            return False, None

        step = self.startup_tasks[self.startup_index]
        step_type = step.get("type", "")

        if step_type == "task":
            task_name = step.get("name", "")

            # Check if we should skip tool acquisition tasks using pre-scanned inventory
            if task_name in ("get_hammer", "get_axe", "get_sickle", "get_hoe"):
                tool_map = {
                    "get_hammer": Tool.HAMMER,
                    "get_axe": Tool.AXE,
                    "get_sickle": Tool.SICKLE,
                    "get_hoe": Tool.HOE,
                }
                required_tool = tool_map.get(task_name)

                # Check if tool was found in inventory scan
                if required_tool and int(required_tool) in self.tool_manager.seen:
                    print(
                        f"[CLEARER] Skipping {task_name} "
                        f"(already have {required_tool.name})"
                    )
                    self.startup_index += 1
                    return True, make_action()

            # Execute the task
            frames = self._load_task(task_name)
            if frames:
                print(f"[CLEARER] Task: {task_name} ({len(frames)} frames)")
                self.task_queue.extend(frames)
            else:
                print(f"[CLEARER] Task not found: {task_name}")
            self.startup_index += 1
            return True, self.task_queue.popleft() if self.task_queue else make_action()

        elif step_type == "nav":
            target = step.get("target")
            radius = step.get("radius", 12)
            timeout = step.get("timeout", 0)
            if "start_frame" not in step:
                step["start_frame"] = self.frame_count

            if timeout and self.frame_count - step["start_frame"] >= timeout:
                print(f"[CLEARER] Nav timeout: {step.get('name')}")
                self.startup_index += 1
                self.navigator.path = []
                return True, make_action()

            if target and abs(target.x - self.navigator.current_pos.x) <= radius and abs(target.y - self.navigator.current_pos.y) <= radius:
                print(f"[CLEARER] Nav done: {step.get('name')}")
                self.startup_index += 1
                self.navigator.path = []
                return True, make_action()

            if self.navigator.stasis > self.max_stasis:
                if self.navigator.path:
                    self.pathfinder.temp_blocked.add(self.navigator.path[0])
                self.navigator.path = []
                self.navigator.stasis = 0

            if target and not self.navigator.path:
                target_tile = (target.x // TILE_SIZE, target.y // TILE_SIZE)
                approach = self.pathfinder.find_approach(ram, target_tile, self.navigator.current_pos)
                if not approach:
                    approach = self.pathfinder.find_nearest_walkable(ram, target_tile, max_radius=4)
                if approach:
                    path = self.pathfinder.find_path(ram, self.navigator.current_tile, approach)
                    if path:
                        self.navigator.path = path

            action = self.navigator.follow_path(ram)
            return True, action if action is not None else make_action()

        self.startup_index += 1
        return True, make_action()

    def _should_lift(self, target: Target) -> bool:
        if not target.is_liftable:
            return False
        if target.debris_type == DebrisType.WEED:
            return self.prefer_lift_for_weeds
        if target.debris_type == DebrisType.STONE:
            return self.prefer_lift_for_stones
        if target.debris_type == DebrisType.FENCE:
            return True
        return False

    def _face_dir(self, player: Tuple[int, int], target: Tuple[int, int]) -> str:
        dx, dy = target[0] - player[0], target[1] - player[1]
        return 'right' if abs(dx) >= abs(dy) and dx > 0 else 'left' if abs(dx) >= abs(dy) else 'down' if dy > 0 else 'up'

    def _stamina_too_low(self, ram: np.ndarray) -> bool:
        if ADDR_STAMINA >= len(ram):
            return False
        return int(ram[ADDR_STAMINA]) < self.min_stamina

    def _sort_targets_cluster(
        self, targets: List[Target], player_pos: Point
    ) -> List[Target]:
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

    def _try_adjacent_opportunity(
        self, ram: np.ndarray, player_tile: Tuple[int, int]
    ) -> Optional[str]:
        """Clear any priority debris already adjacent to the player."""
        best: Optional[Target] = None
        best_rank: Optional[int] = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = player_tile[0] + dx, player_tile[1] + dy
            if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH):
                continue
            if (nx, ny) in self.failed_tiles:
                continue
            tile_id = get_tile_at(ram, nx, ny)
            debris = TILE_TO_DEBRIS.get(tile_id)
            if debris is None or debris not in CLEARABLE_DEBRIS_TYPES:
                continue
            # Prefer multitile anchors; if standing next to a non-TL cell of a
            # 2x2, snap to that family's TL when present.
            if tile_id in STUMP_TILES and tile_id != STUMP_TL:
                for ox, oy in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
                    ax, ay = nx + ox, ny + oy
                    if get_tile_at(ram, ax, ay) == STUMP_TL:
                        nx, ny, tile_id, debris = ax, ay, STUMP_TL, DebrisType.STUMP
                        break
                else:
                    continue
            if tile_id in LARGE_ROCK_TILES and tile_id != LARGE_ROCK_TL:
                for ox, oy in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
                    ax, ay = nx + ox, ny + oy
                    if get_tile_at(ram, ax, ay) == LARGE_ROCK_TL:
                        nx, ny, tile_id, debris = (
                            ax,
                            ay,
                            LARGE_ROCK_TL,
                            DebrisType.ROCK,
                        )
                        break
                else:
                    continue
            try:
                rank = self.priority.index(debris)
            except ValueError:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = Target(
                    tile=(nx, ny),
                    pos=Point(nx * TILE_SIZE + 8, ny * TILE_SIZE + 8),
                    debris_type=debris,
                    tile_id=tile_id,
                )
        if best is None:
            return None
        self.current_target = best
        self.approach_tile = player_tile
        self.navigator.path = []
        self.navigator.stasis = 0
        self.target_hits = 0
        self.clearing_start_frame = 0
        print(
            f"[CLEARER] Adjacent {best.debris_type.name} at {best.tile} "
            "-> clear now"
        )
        return "clearing"

    def _handle_scanning(self, ram: np.ndarray) -> Optional[str]:
        if self._stamina_too_low(ram):
            self.stamina_exhausted = True
            print(
                f"[CLEARER] Stamina low "
                f"({int(ram[ADDR_STAMINA]) if ADDR_STAMINA < len(ram) else '?'});"
                " stopping clear"
            )
            return "complete"

        targets = self.scanner.scan(
            ram, self.farm_bounds, types=set(CLEARABLE_DEBRIS_TYPES)
        )
        if not targets:
            return "complete"

        player_tile = self.navigator.current_tile
        opportunity = self._try_adjacent_opportunity(ram, player_tile)
        if opportunity:
            return opportunity

        xs = [t.tile[0] for t in targets]
        ys = [t.tile[1] for t in targets]
        self.farm_bounds = (
            max(2, min(xs)),
            max(2, min(ys)),
            min(61, max(xs)),
            min(61, max(ys)),
        )

        counts: Dict[DebrisType, int] = {}
        for t in targets:
            counts[t.debris_type] = counts.get(t.debris_type, 0) + 1

        new_phase = None
        for dt in self.priority:
            if counts.get(dt, 0) > 0:
                new_phase = dt
                break

        if new_phase != self.current_phase:
            if new_phase:
                print(f"[CLEARER] Phase: {new_phase.name}")
            self.current_phase = new_phase

        if not self.current_phase:
            return "complete"

        phase_targets = [
            t
            for t in targets
            if t.debris_type == self.current_phase
            and t.tile not in self.failed_tiles
        ]
        phase_targets = self._sort_targets_cluster(
            phase_targets, self.navigator.current_pos
        )

        for target in phase_targets:
            approach = self.pathfinder.find_approach(
                ram,
                target.tile,
                self.navigator.current_pos,
                footprint=target.footprint,
            )
            if approach:
                path = self.pathfinder.find_path(
                    ram,
                    self.navigator.current_tile,
                    approach,
                    max_steps=VIEWPORT_HOP_TILES,
                )
                if path is not None:
                    self.scan_miss_streak = 0
                    self.current_target = target
                    self.approach_tile = approach
                    self.navigator.path = path
                    self.navigator.stasis = 0
                    self.target_hits = 0
                    self.clearing_start_frame = 0
                    tool = (
                        target.required_tool.name
                        if target.required_tool
                        else "HANDS"
                    )
                    print(
                        f"[CLEARER] Target: {target.debris_type.name} "
                        f"at {target.tile} ({tool})"
                    )
                    return "navigating"

        self.scan_miss_streak += 1
        if self.scan_miss_streak >= self.max_scan_misses:
            print(
                f"[CLEARER] No reachable {self.current_phase.name if self.current_phase else 'debris'} "
                f"after {self.scan_miss_streak} scans; stopping with "
                f"cleared={self.cleared_count}"
            )
            return "complete"
        return None

    def _replan_nav_hop(self, ram: np.ndarray) -> Optional[str]:
        """Plan a viewport-limited hop toward the current approach tile."""
        if not self.current_target or not self.approach_tile:
            return "scanning"
        path = self.pathfinder.find_path(
            ram,
            self.navigator.current_tile,
            self.approach_tile,
            max_steps=VIEWPORT_HOP_TILES,
        )
        if path is None:
            self.failed_tiles.add(self.current_target.tile)
            self.current_target = None
            return "scanning"
        self.navigator.path = path
        self.navigator.stasis = 0
        return None

    def _handle_navigating(self, ram: np.ndarray) -> Optional[str]:
        if not self.current_target or not self.approach_tile:
            return "scanning"

        live_id = get_tile_at(ram, *self.current_target.tile)
        live_debris = TILE_TO_DEBRIS.get(live_id)
        if live_debris is None:
            self.current_target = None
            return "scanning"
        if live_debris != self.current_target.debris_type:
            self.current_target = None
            return "scanning"
        if live_id != self.current_target.tile_id:
            self.current_target = Target(
                tile=self.current_target.tile,
                pos=self.current_target.pos,
                debris_type=live_debris,
                tile_id=live_id,
            )

        if self.navigator.current_tile == self.approach_tile:
            return "clearing"

        if self.navigator.stasis > self.max_stasis:
            print(
                f"[NAV] Stuck at {self.navigator.current_tile}, "
                "trying alternate path"
            )
            if self.navigator.path:
                self.pathfinder.temp_blocked.add(self.navigator.path[0])
            self.navigator.path = []
            self.navigator.stasis = 0
            return self._replan_nav_hop(ram)

        action = self.navigator.follow_path(ram)
        if action is not None:
            self.action_queue.append(action)
            return None

        # Hop segment finished short of the approach — replan next hop.
        if self.navigator.current_tile != self.approach_tile:
            return self._replan_nav_hop(ram)
        return "clearing"

    def _handle_clearing(self, ram: np.ndarray) -> Optional[str]:
        if not self.current_target:
            return "scanning"

        # Track when we entered clearing state for timeout
        if self.clearing_start_frame == 0:
            self.clearing_start_frame = self.frame_count
            self.action_queue.clear()
            self.task_queue.clear()
            self.navigator.path = []

        # Timeout (600 frames: per-hit clearing with centering between each)
        if self.frame_count - self.clearing_start_frame > 600:
            print(f"[CLEARER] Clearing timeout at {self.current_target.tile}, moving on")
            self.failed_tiles.add(self.current_target.tile)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        # Re-validate target tile.  Rocks change tile ID as they take damage;
        # keep hitting as long as the debris *type* is unchanged.
        current_tile_id = get_tile_at(ram, *self.current_target.tile)
        if current_tile_id != self.current_target.tile_id:
            new_debris = TILE_TO_DEBRIS.get(current_tile_id)
            if new_debris is None:
                # Tile fully cleared
                pos_key = self.current_target.tile
                if pos_key not in self.tiles_cleared:
                    self.tiles_cleared.add(pos_key)
                    self.cleared_count += 1
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"
            if new_debris != self.current_target.debris_type:
                # Changed to a different debris type, rescan
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"
            # Same debris type, different visual (rock taking damage) — continue
            self.current_target = Target(
                tile=self.current_target.tile,
                pos=self.current_target.pos,
                debris_type=new_debris,
                tile_id=current_tile_id,
            )

        player = self.navigator.current_tile
        target = self.current_target.tile

        if tile_dist(player, target) > 1:
            return "navigating"

        # Wait for any queued actions (current hit animation) to finish
        if self.action_queue:
            return None

        # Finish verifying a lift after the queued A presses drain.
        if self._pending_lift_verify is not None:
            verify_tile = self._pending_lift_verify
            self._pending_lift_verify = None
            if TILE_TO_DEBRIS.get(get_tile_at(ram, *verify_tile)) is None:
                if verify_tile not in self.tiles_cleared:
                    self.tiles_cleared.add(verify_tile)
                    self.cleared_count += 1
                # Toss immediately so we do not block house/shop doors later.
                self.action_queue.extend([make_action(down=True) for _ in range(2)])
                self.action_queue.extend([make_action() for _ in range(2)])
                self.action_queue.extend([make_action(a=True) for _ in range(12)])
                self.action_queue.extend([make_action() for _ in range(12)])
            else:
                print(f"[CLEARER] Lift did not clear {verify_tile}")
                self.failed_tiles.add(verify_tile)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        # Wait until inputs are accepted and player is stationary
        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        if input_lock != 1 or self.navigator.stasis < 6:
            return None

        # Re-center on approach tile before every hit to correct animation drift
        if self.approach_tile:
            center_action = self.navigator.center_on_tile(
                self.approach_tile, tolerance=2
            )
            if center_action is not None:
                self.action_queue.append(center_action)
                return None

        # Lift check
        if self._should_lift(self.current_target):
            print(f"[CLEARER] Lifting {self.current_target.debris_type.name}")
            direction = self._face_dir(player, target)
            self.action_queue.extend(
                [make_action(**{direction: True}) for _ in range(3)]
            )
            self.action_queue.extend([make_action() for _ in range(4)])
            self.action_queue.extend([make_action(a=True) for _ in range(18)])
            self.action_queue.extend([make_action() for _ in range(20)])
            self._pending_lift_verify = target
            return None

        # Tool check
        tool = self.current_target.required_tool
        if tool is None:
            self.failed_tiles.add(target)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        if self.tool_manager.current != tool:
            print(
                f"[CLEARER] Need {tool.name}, "
                f"have 0x{self.tool_manager.current:02X}"
            )
            self.searching_tool = tool
            self.tool_manager.start_search()
            self.tool_search_frames = 0
            return "tool_switch"

        if self._stamina_too_low(ram):
            self.stamina_exhausted = True
            self.current_target = None
            self.clearing_start_frame = 0
            return "complete"

        # First hit: attempt tracking and logging
        if self.target_hits == 0:
            tile_key = (target[0], target[1], self.current_target.tile_id)
            attempts = self.tile_attempts.get(tile_key, 0)
            if attempts >= 3:
                print(
                    f"[CLEARER] Giving up on "
                    f"{self.current_target.debris_type.name} at {target} "
                    f"tile=0x{self.current_target.tile_id:02X} "
                    "(3 failed attempts)"
                )
                self.failed_tiles.add(target)
                self.current_target = None
                return "scanning"
            self.tile_attempts[tile_key] = attempts + 1
            direction = self._face_dir(player, target)
            if attempts == 0:
                print(
                    f"[CLEARER] Clearing "
                    f"{self.current_target.debris_type.name} at {target} "
                    f"tile=0x{self.current_target.tile_id:02X} from {player} "
                    f"facing {direction} "
                    f"({self.current_target.required_hits} hits)"
                )
            else:
                print(
                    f"[CLEARER] Re-targeting "
                    f"{self.current_target.debris_type.name} at {target} "
                    f"tile=0x{self.current_target.tile_id:02X} "
                    f"(attempt {attempts + 1}/3)"
                )

        # Hits delivered — only count after the tile is actually gone.
        if self.target_hits >= self.current_target.required_hits:
            if TILE_TO_DEBRIS.get(current_tile_id) is None:
                pos_key = self.current_target.tile
                if pos_key not in self.tiles_cleared:
                    self.tiles_cleared.add(pos_key)
                    self.cleared_count += 1
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"
            # Still present after claimed hits — keep swinging a bit more,
            # then fail the tile.
            if self.target_hits >= self.current_target.required_hits + 3:
                print(
                    f"[CLEARER] Hits exhausted but tile remains at {target}"
                )
                self.failed_tiles.add(target)
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"

        # Queue a SINGLE hit: face → settle → swing → cooldown
        direction = self._face_dir(player, target)
        self.action_queue.append(make_action(**{direction: True}))
        self.action_queue.extend([make_action() for _ in range(8)])
        self.action_queue.extend(use_tool(frames=20, cooldown=20))
        self.target_hits += 1

        return None

    def _handle_tool_switch(self, ram: np.ndarray) -> Optional[str]:
        if not self.searching_tool:
            return "clearing"

        self.tool_search_frames += 1

        if self.tool_manager.current == self.searching_tool:
            print(f"[CLEARER] Found {self.searching_tool.name}")
            self.searching_tool = None
            return "clearing"

        self.tool_manager.record()

        if self.tool_manager.cycle_complete() or self.tool_search_frames > 300:
            print(f"[CLEARER] Can't find {self.searching_tool.name}")
            frames = self._load_task(f"get_{self.searching_tool.name.lower()}")
            if frames:
                print(f"[CLEARER] Running get_{self.searching_tool.name.lower()}")
                self.task_queue.extend(frames)
                self.searching_tool = None
                self.tool_manager.start_search()
                return None

            if self.current_target:
                self.failed_tiles.add(self.current_target.tile)
            self.current_target = None
            self.searching_tool = None
            self.clearing_start_frame = 0
            return "scanning"

        self.action_queue.extend(cycle_tool())
        return None

    def tick(self, ram: np.ndarray) -> Optional[np.ndarray]:
        self.frame_count += 1
        self.navigator.update(ram)
        self.tool_manager.update(ram)

        if self.frame_count % self.debug_interval == 0:
            stamina = ram[ADDR_STAMINA] if ADDR_STAMINA < len(ram) else 0
            targets = self.scanner.scan(ram, self.farm_bounds)
            print(
                f"[CLEARER] Debug @ {self.frame_count}f "
                f"pos={self.navigator.current_pos} "
                f"tool=0x{self.tool_manager.current:02X} "
                f"stamina={stamina} state={self.state} "
                f"targets={len(targets)} cleared={self.cleared_count} "
                f"failed={len(self.failed_tiles)}"
            )

        running, action = self._run_startup(ram)
        if running:
            return action if action is not None else make_action()

        if self.task_queue:
            return self._emit_action(self.task_queue.popleft(), "task")

        if self.action_queue:
            return self._emit_action(self.action_queue.popleft(), "queue")

        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        if input_lock != 1:
            action = (
                make_action(a=True)
                if self.frame_count % 2 == 0
                else make_action(b=True)
            )
            return self._emit_action(action, "unlock")

        if self.state == "complete":
            return None

        handlers = {
            "scanning": self._handle_scanning,
            "navigating": self._handle_navigating,
            "clearing": self._handle_clearing,
            "tool_switch": self._handle_tool_switch,
        }

        if self.state in handlers:
            next_state = handlers[self.state](ram)
            if next_state == "complete":
                self.state = "complete"
                return None
            if next_state:
                self.state = next_state

        if self.action_queue:
            return self._emit_action(self.action_queue.popleft(), "queue")

        return self._emit_action(make_action(), "idle")


# =============================================================================
# PRIORITY PARSING
# =============================================================================

DEBRIS_NAMES = {
    "weed": DebrisType.WEED, "weeds": DebrisType.WEED, "bush": DebrisType.WEED,
    "stone": DebrisType.STONE, "stones": DebrisType.STONE,
    "rock": DebrisType.ROCK, "rocks": DebrisType.ROCK,
    "stump": DebrisType.STUMP, "stumps": DebrisType.STUMP,
}


def parse_priority_list(raw: Optional[str], priority_only: bool = False) -> List[DebrisType]:
    if not raw:
        return list(DEFAULT_PRIORITY)

    parsed = []
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
