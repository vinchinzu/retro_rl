"""
Farm clearing module - Phase-based debris clearing with tool management.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Dict, Tuple, Set
from collections import deque
import os
import json

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

class DebrisType(IntEnum):
    WEED = 1
    STONE = 2
    ROCK = 3
    STUMP = 4
    FENCE = 5


class Tool(IntEnum):
    NONE = 0x00
    SICKLE = 0x01
    HOE = 0x02
    HAMMER = 0x03
    AXE = 0x04
    WATERING_CAN = 0x10


TILE_TO_DEBRIS: Dict[int, DebrisType] = {
    0x03: DebrisType.WEED,
    0x04: DebrisType.STONE,  # Small stone (1 hit with hammer or liftable)
    0x05: DebrisType.FENCE,
    0x06: DebrisType.ROCK,   # Big rock (6 hits with hammer)
    0x09: DebrisType.ROCK,   # Reclassified: stump tiles treated as rocks
    0x0A: DebrisType.ROCK,
    0x0B: DebrisType.ROCK,
    0x0C: DebrisType.ROCK,
    0x0D: DebrisType.ROCK,
    0x0E: DebrisType.ROCK,
    0x0F: DebrisType.ROCK,
    0x10: DebrisType.ROCK,
    0x11: DebrisType.ROCK,
    0x12: DebrisType.ROCK,
    0x13: DebrisType.ROCK,
    0x14: DebrisType.ROCK,
}

DEBRIS_TOOL: Dict[DebrisType, Optional[Tool]] = {
    DebrisType.WEED: Tool.SICKLE,
    DebrisType.STONE: Tool.HAMMER,
    DebrisType.ROCK: Tool.HAMMER,
    DebrisType.STUMP: Tool.AXE,
    DebrisType.FENCE: None,
}

LIFTABLE_TILES: Set[int] = {0x03, 0x04, 0x05}  # weeds, small stones, fence only
POND_CHARACTERISTIC_TILES: Set[int] = {0xA6, 0xF0}
WALKABLE_TILES: Set[int] = {
    0x00, 0x01, 0x02, 0x03, 0x07, 0x08, 
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, # Grass variants
    0xA0, 0xA2, 0xA3, 0xA8, # Paths, borders, and some empty tiles
}

DEFAULT_PRIORITY: List[DebrisType] = [DebrisType.ROCK, DebrisType.STONE, DebrisType.WEED]

# RAM addresses
ADDR_X = 0x00D6
ADDR_Y = 0x00D8
ADDR_TOOL = 0x0921
ADDR_STAMINA = 0x0918
ADDR_MAP = 0x09B6
ADDR_TILEMAP = 0x0022
ADDR_INPUT_LOCK = 0x019A

TILE_SIZE = 16
MAP_WIDTH = 64


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
        if self.debris_type == DebrisType.ROCK or self.debris_type == DebrisType.STUMP:
            return 7
        return 1


# =============================================================================
# UTILITIES
# =============================================================================

def make_action(**buttons) -> np.ndarray:
    action = np.zeros(12, dtype=np.int32)
    idx_map = {'up': 4, 'down': 5, 'left': 6, 'right': 7, 'a': 8, 'b': 0, 'y': 1, 'x': 9, 'l': 10, 'r': 11}
    for btn, pressed in buttons.items():
        if pressed and btn.lower() in idx_map:
            action[idx_map[btn.lower()]] = 1
    return action


def action_to_names(action: np.ndarray) -> str:
    idx_map = {4: 'up', 5: 'down', 6: 'left', 7: 'right', 8: 'a', 0: 'b', 1: 'y', 9: 'x', 10: 'l', 11: 'r'}
    pressed = [name for idx, name in idx_map.items() if idx < len(action) and action[idx] == 1]
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
    if tx < 0 or ty < 0 or tx >= MAP_WIDTH or ty >= MAP_WIDTH:
        return 0
    addr = ADDR_MAP + ty * MAP_WIDTH + tx
    return int(ram[addr]) if addr < len(ram) else 0


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

    def scan(self, ram: np.ndarray, bounds: Optional[Tuple[int, int, int, int]] = None) -> List[Target]:
        self.frame_count += 1
        if ADDR_MAP >= len(ram):
            return []

        end = min(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, len(ram))
        map_data = ram[ADDR_MAP:end]

        mask = np.isin(map_data, list(self.debris_map.keys()))
        indices = np.nonzero(mask)[0]

        targets = []
        for idx in indices:
            tile_id = int(map_data[idx])
            debris = self.debris_map.get(tile_id)
            if debris is None:
                continue

            ty, tx = divmod(int(idx), MAP_WIDTH)
            if bounds and not (bounds[0] <= tx <= bounds[2] and bounds[1] <= ty <= bounds[3]):
                continue

            targets.append(Target(
                tile=(tx, ty),
                pos=Point(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8),
                debris_type=debris,
                tile_id=tile_id,
            ))
        
        if os.getenv("FENCE_DEBUG") == "1" and len(targets) > 0 and self.frame_count % 300 == 0:
            print(f"[SCANNER] Found {len(targets)} targets. Top: {targets[0].debris_type.name} at {targets[0].tile}")
            
        return targets


# =============================================================================
# PATHFINDER
# =============================================================================

class Pathfinder:
    def __init__(self, scanner: TileScanner):
        self.scanner = scanner
        self.no_go_tiles: Set[Tuple[int, int]] = set()
        self.temp_blocked: Set[Tuple[int, int]] = set()

    def is_walkable(self, ram: np.ndarray, tx: int, ty: int, walkable_override: Optional[Set[int]] = None, current_pos: Optional[Tuple[int, int]] = None) -> bool:
        # Always allow moving from current tile
        if current_pos and (tx, ty) == current_pos:
            return True
            
        if (tx, ty) in self.no_go_tiles or (tx, ty) in self.temp_blocked:
            return False
        tile_id = get_tile_at(ram, tx, ty)
        if tile_id in WALKABLE_TILES:
            return True
        if walkable_override and (tx, ty) in walkable_override:
            return True
        return False

    def find_path(self, ram: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], walkable_override: Optional[Set[int]] = None) -> Optional[List[Tuple[int, int]]]:
        if start == goal:
            return []

        queue = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                break

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                # Check bounds and came_from
                if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH and (nx, ny) not in came_from:
                    is_goal = (nx, ny) == goal
                    # allow goal to be non-walkable if it's in override
                    if is_goal and walkable_override and (nx, ny) in walkable_override:
                        came_from[(nx, ny)] = (cx, cy)
                        continue
                    
                    if self.is_walkable(ram, nx, ny, walkable_override=walkable_override, current_pos=start):
                        came_from[(nx, ny)] = (cx, cy)
                        queue.append((nx, ny))

        if goal not in came_from:
            return None

        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path

    def find_approach(self, ram: np.ndarray, target: Tuple[int, int], player_pos: Point, walkable_override: Optional[Set[int]] = None) -> Optional[Tuple[int, int]]:
        best, best_dist = None, None
        tx, ty = target
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = tx + dx, ty + dy
            if 0 <= ax < MAP_WIDTH and 0 <= ay < MAP_WIDTH and self.is_walkable(ram, ax, ay, walkable_override=walkable_override):
                dist = manhattan(Point(ax * TILE_SIZE + 8, ay * TILE_SIZE + 8), player_pos)
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
        self.stasis = self.stasis + 1 if self.current_pos == new_pos else 0
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
        tgt_x = curr_tile[0] * TILE_SIZE + 8
        tgt_y = curr_tile[1] * TILE_SIZE + 8
        
        dx_next = next_tile[0] - curr_tile[0]
        dy_next = next_tile[1] - curr_tile[1]
        
        # PROACTIVE CENTERING:
        # If we are moving horizontally, we MUST be vertically centered (Y).
        # If we are moving vertically, we MUST be horizontally centered (X).
        # We use a very tight tolerance (1 pixel means exact center 8) to avoid clipping.
        # SNES characters are fickle.
        if dx_next != 0: # Moving horizontally
            if abs(self.current_pos.y - tgt_y) >= 1: # Tight center
                return self.center_on_tile(curr_tile, tolerance=1)
        if dy_next != 0: # Moving vertically
            if abs(self.current_pos.x - tgt_x) >= 1: # Tight center
                return self.center_on_tile(curr_tile, tolerance=1)

        if not self.pathfinder.is_walkable(ram, *next_tile, current_pos=curr_tile):
            val = get_tile_at(ram, *next_tile)
            # Log more clearly why we are blocked
            print(f"[NAVIGATOR] Blocked! tile={next_tile} id=0x{val:02X} walkable={val in WALKABLE_TILES} temp_blocked={next_tile in self.pathfinder.temp_blocked} no_go={next_tile in self.pathfinder.no_go_tiles}")

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
    """Phase-based farm clearing: weed → stone → rock → stump."""

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
        self.tiles_cleared: Set[Tuple[int, int, int]] = set()  # Track (x, y, tile_id) cleared
        self.tile_attempts: Dict[Tuple[int, int, int], int] = {}  # Track attempts per tile
        self.frame_count = 0
        self.farm_bounds: Optional[Tuple[int, int, int, int]] = None

        self.prefer_lift_for_weeds = True
        self.prefer_lift_for_stones = False
        self.max_stasis = 120  # Increased from 30 to give more time for navigation
        self.debug_interval = 300

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
        self.sequence_tool: Optional[Tool] = None
        self.sequence_frames = 0
        self.sequence_target: Optional[Tuple[int, int]] = None
        self.sequence_tile_id: Optional[int] = None
        self.sequence_debris: Optional[DebrisType] = None
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
            action = action.copy()
            # Strip directional inputs during sensitive hit sequences.
            action[4:8] = 0
            self.suppress_move_frames -= 1
            src = f"{src}+suppress"
        if os.getenv("ACTION_DEBUG") == "1":
            buttons = action_to_names(action)
            if buttons != "none" or os.getenv("ACTION_DEBUG_VERBOSE") == "1" and self.frame_count % 30 == 0:
                print(f"[ACTION] frame={self.frame_count} state={self.state} src={src} buttons={buttons}")
        return action

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
            self.startup_done = True
            print("[CLEARER] Startup complete")
            return False, None

        step = self.startup_tasks[self.startup_index]
        step_type = step.get("type", "")

        if step_type == "task":
            task_name = step.get("name", "")

            # Check if we should skip tool acquisition tasks using pre-scanned inventory
            if task_name in ("get_hammer", "get_axe", "get_sickle", "get_hoe"):
                tool_map = {"get_hammer": Tool.HAMMER, "get_axe": Tool.AXE, "get_sickle": Tool.SICKLE, "get_hoe": Tool.HOE}
                required_tool = tool_map.get(task_name)

                # Check if tool was found in inventory scan
                if required_tool and required_tool in self.tool_manager.seen:
                    print(f"[CLEARER] Skipping {task_name} (already have {required_tool.name})")
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

    def _handle_scanning(self, ram: np.ndarray) -> Optional[str]:
        targets = self.scanner.scan(ram, self.farm_bounds)
        if not targets:
            return "complete"

        # Opportunistic: if a rock is adjacent, clear it immediately with hammer.
        player_tile = self.navigator.current_tile
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = player_tile[0] + dx, player_tile[1] + dy
            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH:
                tile_id = get_tile_at(ram, nx, ny)
                debris = TILE_TO_DEBRIS.get(tile_id)
                if debris == DebrisType.ROCK:
                    self.current_target = Target(
                        tile=(nx, ny),
                        pos=Point(nx * TILE_SIZE + 8, ny * TILE_SIZE + 8),
                        debris_type=debris,
                        tile_id=tile_id,
                    )
                    self.approach_tile = player_tile
                    self.navigator.path = []
                    self.navigator.stasis = 0
                    self.target_hits = 0
                    self.clearing_start_frame = 0
                    print(f"[CLEARER] Adjacent ROCK at {(nx, ny)} -> clear now")
                    return "clearing"

        # Update bounds
        xs = [t.tile[0] for t in targets]
        ys = [t.tile[1] for t in targets]
        self.farm_bounds = (max(2, min(xs)), max(2, min(ys)), min(61, max(xs)), min(61, max(ys)))

        # Select phase
        counts = {}
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

        # Find target
        phase_targets = [t for t in targets if t.debris_type == self.current_phase and t.tile not in self.failed_tiles]
        phase_targets.sort(key=lambda t: manhattan(t.pos, self.navigator.current_pos))

        for target in phase_targets:
            approach = self.pathfinder.find_approach(ram, target.tile, self.navigator.current_pos)
            if approach:
                path = self.pathfinder.find_path(ram, self.navigator.current_tile, approach)
                if path is not None:
                    self.current_target = target
                    self.approach_tile = approach
                    self.navigator.path = path
                    self.navigator.stasis = 0  # Reset stasis when starting new target
                    self.target_hits = 0
                    self.clearing_start_frame = 0  # Reset clearing timeout
                    tool = target.required_tool.name if target.required_tool else "HANDS"
                    print(f"[CLEARER] Target: {target.debris_type.name} at {target.tile} ({tool})")
                    return "navigating"

        return None

    def _handle_navigating(self, ram: np.ndarray) -> Optional[str]:
        if not self.current_target or not self.approach_tile:
            return "scanning"

        if get_tile_at(ram, *self.current_target.tile) != self.current_target.tile_id:
            self.current_target = None
            return "scanning"

        # Check if we're actually on the approach tile (must be exact tile, not just nearby)
        # Using tile coordinates instead of pixel tolerance to avoid diagonal positioning
        if self.navigator.current_tile == self.approach_tile:
            return "clearing"

        if self.navigator.stasis > self.max_stasis:
            print(f"[NAV] Stuck at {self.navigator.current_tile}, trying alternate path")
            if self.navigator.path:
                self.pathfinder.temp_blocked.add(self.navigator.path[0])
            self.navigator.path = []
            self.navigator.stasis = 0  # Reset stasis after recovery attempt
            path = self.pathfinder.find_path(ram, self.navigator.current_tile, self.approach_tile)
            if path:
                self.navigator.path = path
            else:
                self.failed_tiles.add(self.current_target.tile)
                self.current_target = None
                return "scanning"

        action = self.navigator.follow_path(ram)
        if action is not None:
            self.action_queue.append(action)
        return None

    def _handle_clearing(self, ram: np.ndarray) -> Optional[str]:
        if not self.current_target:
            return "scanning"

        # Track when we entered clearing state for timeout
        if self.clearing_start_frame == 0:
            self.clearing_start_frame = self.frame_count
            # CRITICAL: Clear action queue to prevent navigation actions from interfering
            self.action_queue.clear()
            # Clear any pending scripted actions too (tool tasks, startup moves)
            self.task_queue.clear()
            # Stop any leftover navigation pathing
            self.navigator.path = []
            if os.getenv("FENCE_DEBUG") == "1":
                print(f"[CLEAR] Starting clear of {self.current_target.debris_type.name} at {self.current_target.tile}")

        # Timeout if we've been in clearing too long (need 300+ frames for 6-hit sequence)
        if self.frame_count - self.clearing_start_frame > 400:
            if os.getenv("FENCE_DEBUG") == "1":
                print(f"[CLEAR] Timeout after {self.frame_count - self.clearing_start_frame}f, queue={len(self.action_queue)}, hits={self.target_hits}")
            print(f"[CLEARER] Clearing timeout at {self.current_target.tile}, moving on")
            self.failed_tiles.add(self.current_target.tile)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        # Re-validate target tile each time we enter clearing; it may have changed.
        current_tile_id = get_tile_at(ram, *self.current_target.tile)
        if current_tile_id != self.current_target.tile_id:
            new_debris = TILE_TO_DEBRIS.get(current_tile_id)
            if new_debris is None:
                # Tile cleared or transformed into non-debris.
                self.current_target = None
                self.clearing_start_frame = 0
                return "scanning"
            # Update target to match current tile contents and restart selection.
            self.current_target = Target(
                tile=self.current_target.tile,
                pos=self.current_target.pos,
                debris_type=new_debris,
                tile_id=current_tile_id,
            )
            self.clearing_start_frame = 0
            return "scanning"

        player = self.navigator.current_tile
        target = self.current_target.tile

        if tile_dist(player, target) > 1:
            if os.getenv("FENCE_DEBUG") == "1" and self.frame_count % 30 == 0:
                print(f"[CLEAR] Too far: player={player} target={target} dist={tile_dist(player, target)}")
            return "navigating"

        # Ensure we're centered on the approach tile before beginning hits.
        # This prevents sliding/micro-movement that breaks multi-hit sequences.
        if self.approach_tile and not self.navigator.at_tile(self.approach_tile, tolerance=2):
            # Only try to center briefly; avoid getting stuck micro-adjusting.
            if self.frame_count - self.clearing_start_frame < 60:
                action = self.navigator.center_on_tile(self.approach_tile, tolerance=2)
                if action is not None:
                    self.action_queue.append(action)
                    return None

        # Wait until inputs are accepted and the player is fully stationary.
        # Some frames after transitions are input-locked; movement during hits resets the counter.
        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        if input_lock != 1 or self.navigator.stasis < 6:
            if os.getenv("FENCE_DEBUG") == "1" and self.frame_count % 15 == 0:
                print(f"[CLEAR] Waiting to settle: lock={input_lock} stasis={self.navigator.stasis}")
            return None

        if self._should_lift(self.current_target):
            print(f"[CLEARER] Lifting {self.current_target.debris_type.name}")
            direction = self._face_dir(player, target)
            self.action_queue.extend([make_action(**{direction: True, 'a': True}) for _ in range(4)])
            self.action_queue.extend([make_action() for _ in range(4)])
            self.action_queue.extend([make_action(a=True) for _ in range(12)])
            self.cleared_count += 1
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        tool = self.current_target.required_tool
        if tool is None:
            self.failed_tiles.add(target)
            self.current_target = None
            self.clearing_start_frame = 0
            return "scanning"

        if self.tool_manager.current != tool:
            print(f"[CLEARER] Need {tool.name}, have 0x{self.tool_manager.current:02X}")
            self.searching_tool = tool
            self.tool_manager.start_search()
            self.tool_search_frames = 0
            return "tool_switch"

        # Skip centering - just make sure we're on the right tile
        # The bot should already be close enough from navigation

        # Debug: log if we're stuck in clearing
        if os.getenv("FENCE_DEBUG") == "1" and (self.frame_count - self.clearing_start_frame) % 30 == 0:
            print(f"[CLEAR] In clearing for {self.frame_count - self.clearing_start_frame}f, target_hits={self.target_hits}, queue={len(self.action_queue)}")

        # Start clearing sequence: face direction and queue all hits at once
        if self.target_hits == 0:
            direction = self._face_dir(player, target)

            # Remember the tile ID before clearing
            tile_before = self.current_target.tile_id
            tile_key = (target[0], target[1], tile_before)

            # Check if we already attempted this specific tile
            attempts = self.tile_attempts.get(tile_key, 0)
            already_cleared = tile_key in self.tiles_cleared

            # If we've tried this tile 3+ times and it hasn't changed, mark as failed
            if attempts >= 3:
                print(f"[CLEARER] Giving up on {self.current_target.debris_type.name} at {target} tile=0x{tile_before:02X} (3 failed attempts)")
                self.failed_tiles.add(target)
                self.current_target = None
                return "scanning"

            if not already_cleared:
                print(f"[CLEARER] Clearing {self.current_target.debris_type.name} at {target} tile=0x{tile_before:02X} from {player} facing {direction} ({self.current_target.required_hits} hits)")
            else:
                print(f"[CLEARER] Re-targeting {self.current_target.debris_type.name} at {target} tile=0x{tile_before:02X} (attempt {attempts + 1}/3)")

            # Track this attempt
            self.tile_attempts[tile_key] = attempts + 1

            # 1. Face the target with a single tap (blocked by stump/rock so no movement).
            self.action_queue.append(make_action(**{direction: True}))

            # 2. CRITICAL: Stop ALL input and wait for character to become stationary
            # Reduced wait time since we're using looser centering tolerance
            self.action_queue.extend([make_action() for _ in range(20)])

            # 3. Queue all hits with ZERO directional input
            # Each hit must have NO movement keys pressed at all
            hit_actions: List[np.ndarray] = []
            for _ in range(self.current_target.required_hits):
                if self.current_target.required_hits > 1:
                    hit_actions.extend(use_tool_facing(direction, frames=20, cooldown=15))
                else:
                    hit_actions.extend(use_tool(frames=20, cooldown=15))

            # 4. Cooldown before scanning
            hit_actions.extend([make_action() for _ in range(20)])

            # Queue and suppress any directional inputs during the full sequence.
            self.action_queue.extend(hit_actions)
            self.suppress_move_frames = len(hit_actions)
            self.sequence_tool = tool
            self.sequence_frames = len(hit_actions)
            self.sequence_target = target
            self.sequence_tile_id = self.current_target.tile_id
            self.sequence_debris = self.current_target.debris_type

            # Mark this specific tile ID as cleared and increment counter
            if not already_cleared:
                self.tiles_cleared.add(tile_key)
                self.cleared_count += 1

            self.current_target = None
            self.clearing_start_frame = 0  # Reset timeout
            return "scanning"

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
            tilemap = ram[ADDR_TILEMAP] if ADDR_TILEMAP < len(ram) else 0
            stamina = ram[ADDR_STAMINA] if ADDR_STAMINA < len(ram) else 0
            targets = self.scanner.scan(ram, self.farm_bounds)
            print(f"[CLEARER] Debug @ {self.frame_count}f pos={self.navigator.current_pos} tool=0x{self.tool_manager.current:02X} stamina={stamina} state={self.state} targets={len(targets)} cleared={self.cleared_count} failed={len(self.failed_tiles)}")
            import sys; sys.stdout.flush()

        running, action = self._run_startup(ram)
        if running:
            return action if action is not None else make_action()

        if self.task_queue:
            action = self.task_queue.popleft()
            return self._emit_action(action, "task")

        if self.action_queue:
            # If we left clearing, abort any queued hit sequence to avoid swinging at air.
            if self.state != "clearing" and self.sequence_frames > 0:
                self.action_queue.clear()
                self.suppress_move_frames = 0
                self.sequence_frames = 0
                self.sequence_tool = None
                self.sequence_target = None
                self.sequence_tile_id = None
                self.sequence_debris = None
                return self._emit_action(make_action(), "queue_abort")
            if self.sequence_frames > 0 and self.sequence_tool is not None:
                # Abort if target tile changed mid-sequence (e.g., stump->rock or cleared).
                if self.sequence_target is not None:
                    live_tile_id = get_tile_at(ram, *self.sequence_target)
                    live_debris = TILE_TO_DEBRIS.get(live_tile_id)
                    if live_tile_id != self.sequence_tile_id or live_debris != self.sequence_debris:
                        self.action_queue.clear()
                        self.suppress_move_frames = 0
                        self.sequence_frames = 0
                        self.sequence_tool = None
                        self.sequence_target = None
                        self.sequence_tile_id = None
                        self.sequence_debris = None
                        return self._emit_action(make_action(), "queue_abort")
                if self.tool_manager.current != self.sequence_tool:
                    # Abort queued hit sequence if tool changed.
                    self.action_queue.clear()
                    self.suppress_move_frames = 0
                    self.sequence_frames = 0
                    self.sequence_tool = None
                    self.sequence_target = None
                    self.sequence_tile_id = None
                    self.sequence_debris = None
                    return self._emit_action(make_action(), "queue_abort")
            action = self.action_queue.popleft()
            if self.sequence_frames > 0:
                self.sequence_frames -= 1
                if self.sequence_frames == 0:
                    self.sequence_tool = None
                    self.sequence_target = None
                    self.sequence_tile_id = None
                    self.sequence_debris = None
            return self._emit_action(action, "queue")

        input_lock = ram[ADDR_INPUT_LOCK] if ADDR_INPUT_LOCK < len(ram) else 1
        if input_lock != 1:
            action = make_action(a=True) if self.frame_count % 2 == 0 else make_action(b=True)
            return self._emit_action(action, "unlock")

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
            if self.state != "clearing" and self.sequence_frames > 0:
                self.action_queue.clear()
                self.suppress_move_frames = 0
                self.sequence_frames = 0
                self.sequence_tool = None
                self.sequence_target = None
                self.sequence_tile_id = None
                self.sequence_debris = None
                return self._emit_action(make_action(), "queue_abort")
            if self.sequence_frames > 0 and self.sequence_tool is not None:
                if self.sequence_target is not None:
                    live_tile_id = get_tile_at(ram, *self.sequence_target)
                    live_debris = TILE_TO_DEBRIS.get(live_tile_id)
                    if live_tile_id != self.sequence_tile_id or live_debris != self.sequence_debris:
                        self.action_queue.clear()
                        self.suppress_move_frames = 0
                        self.sequence_frames = 0
                        self.sequence_tool = None
                        self.sequence_target = None
                        self.sequence_tile_id = None
                        self.sequence_debris = None
                        return self._emit_action(make_action(), "queue_abort")
                if self.tool_manager.current != self.sequence_tool:
                    self.action_queue.clear()
                    self.suppress_move_frames = 0
                    self.sequence_frames = 0
                    self.sequence_tool = None
                    self.sequence_target = None
                    self.sequence_tile_id = None
                    self.sequence_debris = None
                    return self._emit_action(make_action(), "queue_abort")
            action = self.action_queue.popleft()
            if self.sequence_frames > 0:
                self.sequence_frames -= 1
                if self.sequence_frames == 0:
                    self.sequence_tool = None
                    self.sequence_target = None
                    self.sequence_tile_id = None
                    self.sequence_debris = None
            return self._emit_action(action, "queue")

        action = make_action()
        return self._emit_action(action, "idle")


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
