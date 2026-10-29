"""
Shared navigation primitives for Harvest Moon SNES.

Pathfinder (viewport-limited BFS), Navigator (path following / centering),
Point, and the small RAM/action helpers that every map-walker needs.

Clears / crops / inventory import from here instead of farm_clearer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import os

import numpy as np

from retro_harness.actions import snes_action

from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
    FARM_WALKABLE,
    MAP_WIDTH,
    STALE_TILE_IDS,
    TILE_SIZE,
    get_tile_at as _catalog_get_tile_at,
)
from harvest.tasks.travel_walk import (
    PUSH_HOLD_FRAMES,
    block_push_facing,
    MAX_BUSY_SUPPRESS_FRAMES,
    MAX_IGNORED_HOLD_FRAMES,
    is_travel_occupied,
    is_travel_occupied_with,
    travel_occupied_grid,
    player_is_busy,
    read_player_action,
    read_player_direction,
)

# Re-export map constants used by nav consumers.
__all__ = [
    "Point",
    "Pathfinder",
    "Navigator",
    "WALKABLE_TILES",
    "VIEWPORT_HOP_TILES",
    "TILE_SIZE",
    "MAP_WIDTH",
    "make_action",
    "get_pos_from_ram",
    "get_tile_at",
    "manhattan",
    "tile_dist",
]


# =============================================================================
# CONSTANTS
# =============================================================================

WALKABLE_TILES = FARM_WALKABLE

# SNES only loads ~16x14 tiles; BFS beyond this sees stale 0x72/0xFF.
VIEWPORT_HOP_TILES = 7


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


# =============================================================================
# UTILITIES
# =============================================================================

def make_action(**buttons) -> np.ndarray:
    """Compatibility wrapper around the shared named-button builder."""

    return snes_action(dtype=np.int32, **buttons)


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
# PATHFINDER
# =============================================================================

class Pathfinder:
    """Viewport-aware BFS over farm metatiles.

    ``scanner`` is accepted for API compatibility with FarmClearer (which
    passes a TileScanner) but is not used by pathfinding itself. Typed as
    ``Any`` so this module does not import farm_clearer.
    """

    def __init__(self, scanner: Any = None, walkable_tiles: Optional[Set[int]] = None):
        self.scanner = scanner
        self.walkable_tiles = walkable_tiles if walkable_tiles is not None else WALKABLE_TILES
        self._dynamic_walkable_tiles = walkable_tiles is None
        self.no_go_tiles: Set[Tuple[int, int]] = set()
        self.temp_blocked: Set[Tuple[int, int]] = set()
        self.extra_walkable: Set[Tuple[int, int]] = set()  # tiles treated as walkable (e.g. crop tiles)
        # Static no-go tiles this caller has measured as walkable (e.g. the
        # shipping-bin lane); a tile still blocks when it is no_go/temp_blocked.
        self.allow_tiles: Set[Tuple[int, int]] = set()

    def base_walkable_tiles(self, ram: np.ndarray) -> Set[int]:
        if not self._dynamic_walkable_tiles:
            return self.walkable_tiles
        try:
            from harvest.maps.map_config import get_walkable_tiles

            tilemap_id = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0x00
            return get_walkable_tiles(tilemap_id)
        except Exception:
            return self.walkable_tiles

    def base_walkable_coords(self, ram: np.ndarray) -> Set[Tuple[int, int]]:
        try:
            from harvest.maps.map_config import get_walkable_coords

            tilemap_id = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0x00
            return set(get_walkable_coords(tilemap_id))
        except Exception:
            return set()

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

        if (tx, ty) in self.no_go_tiles or (tx, ty) in self.temp_blocked:
            return False
        if (tx, ty) in self.base_no_go_tiles(ram) and (tx, ty) not in self.allow_tiles:
            return False
        tile_id = get_tile_at(ram, tx, ty)
        if is_travel_occupied(ram, tx, ty):
            return False
        if tile_id in self.base_walkable_tiles(ram):
            return True
        if (tx, ty) in self.base_walkable_coords(ram):
            return True
        if (tx, ty) in self.extra_walkable:
            return True
        if walkable_override and (tx, ty) in walkable_override:
            return True
        return False

    def _walk_test(
        self,
        ram: np.ndarray,
        walkable_override: Optional[Set[int]] = None,
        current_pos: Optional[Tuple[int, int]] = None,
    ):
        """Closure equal to ``is_walkable(ram, x, y, ...)`` with state hoisted.

        BFS calls ``is_walkable`` thousands of times per search, and each call
        re-derived the tilemap sets and the grid base (a 4096-byte
        count_nonzero per tile read): ~96% of a Summer harvest day's CPU.
        Semantics are identical; ``test_walk_test_matches_is_walkable`` pins it.
        """
        from harvest.core.tile_catalog import MAP_HEIGHT, metatile_grid_base

        base = metatile_grid_base(ram)
        size = len(ram)
        no_go = self.no_go_tiles
        base_no_go = self.base_no_go_tiles(ram)
        if self.allow_tiles:
            base_no_go = set(base_no_go) - self.allow_tiles
        temp_blocked = self.temp_blocked
        walkable_tiles = self.base_walkable_tiles(ram)
        walkable_coords = self.base_walkable_coords(ram)
        extra = self.extra_walkable

        def tile_at(tx: int, ty: int) -> int:
            if tx < 0 or ty < 0 or tx >= MAP_WIDTH or ty >= MAP_HEIGHT:
                return 0
            addr = base + ty * MAP_WIDTH + tx
            return int(ram[addr]) if addr < size else 0

        # One vectorised pass over the whole grid replaces the per-cell
        # is_travel_occupied_with (6 tile reads + footprint tuples per call).
        count = MAP_WIDTH * MAP_HEIGHT
        flat = np.zeros(count, dtype=np.uint8)
        avail = max(0, min(count, size - base))
        if avail:
            flat[:avail] = ram[base:base + avail]
        grid = flat.reshape(MAP_HEIGHT, MAP_WIDTH)
        occupied = travel_occupied_grid(grid).ravel().tolist()
        tile_ok = np.isin(flat, tuple(walkable_tiles)).tolist()

        def walkable(tx: int, ty: int) -> bool:
            if current_pos and (tx, ty) == current_pos:
                return True
            if (tx, ty) in no_go or (tx, ty) in base_no_go or (tx, ty) in temp_blocked:
                return False
            if 0 <= tx < MAP_WIDTH and 0 <= ty < MAP_HEIGHT:
                idx = ty * MAP_WIDTH + tx
                if occupied[idx]:
                    return False
                tile_walkable = tile_ok[idx]
            else:
                if is_travel_occupied_with(tile_at, tx, ty):
                    return False
                tile_walkable = tile_at(tx, ty) in walkable_tiles
            if tile_walkable:
                return True
            if (tx, ty) in walkable_coords:
                return True
            if (tx, ty) in extra:
                return True
            if walkable_override and (tx, ty) in walkable_override:
                return True
            return False

        return walkable

    def block_push_facing(self, ram: np.ndarray, facing: Tuple[int, int], *, pixel_moved: bool = False) -> bool:
        """Mark a push-facing neighbor non-walkable (shared travel policy)."""
        return block_push_facing(
            self.temp_blocked, ram, facing, pixel_moved=pixel_moved
        )

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

        walkable = self._walk_test(ram, walkable_override, start)
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

                    if walkable(nx, ny):
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

    def bfs_tree(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
    ) -> Dict[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Full BFS parent map from ``start`` (no goal, no override).

        Parent pointers match ``find_path``'s: it stops once the goal is popped,
        and a node's parent is fixed when first discovered, so the path to any
        reachable goal is identical.  One tree replaces N failing ``find_path``
        calls that each flood the whole reachable area.
        """
        walkable = self._walk_test(ram, None, start)
        queue = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < MAP_WIDTH
                    and 0 <= ny < MAP_WIDTH
                    and (nx, ny) not in came_from
                    and walkable(nx, ny)
                ):
                    came_from[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))
        return came_from

    @staticmethod
    def path_from_tree(
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        if goal not in came_from:
            return None
        path: List[Tuple[int, int]] = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]  # type: ignore[assignment]
        path.reverse()
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

        walkable = self._walk_test(ram, walkable_override, start)
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
                if not walkable(nx, ny):
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
        self._center_dir: Optional[str] = None
        self._center_flips: int = 0
        self._push_tile: Optional[Tuple[int, int]] = None
        self._push_px: Optional[Tuple[int, int]] = None
        self._push_hold: int = 0
        self._busy_frames: int = 0
        self._ignored_hold: int = 0
        self.busy: bool = False

    def update(self, ram: np.ndarray):
        new_pos = get_pos_from_ram(ram)
        if player_is_busy(ram) and self._busy_frames < MAX_BUSY_SUPPRESS_FRAMES:
            # Buttons are being dropped (see travel_walk.player_is_busy): not
            # evidence of a wall, so neither stasis nor push-hold may grow.
            self._busy_frames += 1
            self.busy = True
            self.current_pos = new_pos
            return
        self.busy = False
        if not player_is_busy(ram):
            self._busy_frames = 0
        new_tile = (new_pos.x // TILE_SIZE, new_pos.y // TILE_SIZE)
        old_tile = (self.current_pos.x // TILE_SIZE, self.current_pos.y // TILE_SIZE)
        # Reset stasis only on tile-level movement, not 1px oscillation
        if new_tile != old_tile:
            self.stasis = 0
            self._center_dir = None
            self._center_flips = 0
        else:
            self.stasis += 1
        self.current_pos = new_pos

    def _input_ignored(self, ram: np.ndarray, facing: Tuple[int, int]) -> bool:
        """True when the farmer is not even turning toward ``facing``.

        A live farmer pushing a wall still turns to face it. If the facing byte
        disagrees with the direction we are holding, buttons are being dropped
        (scripted event pending, fade-in): no evidence about the tile. Summer
        D1: holding UP at the door front for 150f left player_direction on
        LEFT, then a dialogue opened; the stall had already blacklisted (8,25)
        and (7,26) and NAV_CROP never recovered.
        """
        cur = self.current_tile
        dx, dy = facing[0] - cur[0], facing[1] - cur[1]
        want = {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3}.get((dx, dy))
        if want is None:
            return False
        return read_player_direction(ram) != want

    def note_push_facing(self, ram: np.ndarray, facing: Tuple[int, int]) -> bool:
        """True when ``facing`` is (now) a push no-go. Call while charging it."""
        if facing in self.pathfinder.temp_blocked:
            return True
        if self.busy or self._input_ignored(ram, facing):
            # Bounded: a farmer that never turns for this long is not waiting
            # on an event, so fall back to treating the stall as a wall.
            self._ignored_hold += 1
            if self._ignored_hold <= MAX_IGNORED_HOLD_FRAMES:
                self._push_hold = 0
                return False
        else:
            self._ignored_hold = 0
        px = (self.current_pos.x, self.current_pos.y)
        if facing != self._push_tile or px != self._push_px:
            self._push_tile = facing
            self._push_px = px
            self._push_hold = 0
            return False
        self._push_hold += 1
        if self._push_hold < PUSH_HOLD_FRAMES:
            return False
        return self.pathfinder.block_push_facing(ram, facing, pixel_moved=False)

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
            direction = "right" if dx > 0 else "left"
            action[7] = 1 if dx > 0 else 0  # Right
            action[6] = 1 if dx < 0 else 0  # Left
        elif abs(dy) >= tolerance:
            direction = "down" if dy > 0 else "up"
            action[5] = 1 if dy > 0 else 0  # Down
            action[4] = 1 if dy < 0 else 0  # Up
        else:
            return None
        self._note_center_dir(direction)
        return action

    def _note_center_dir(self, direction: str) -> None:
        opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
        if self._center_dir is not None and opposite.get(self._center_dir) == direction:
            self._center_flips += 1
        elif self._center_dir != direction:
            self._center_flips = 0
        self._center_dir = direction

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
        # Tolerance=3 avoids 1px oscillation. Skip if stuck (stasis>30) or if
        # left/right (or up/down) centering has already flipped — that is the
        # in-place L/R glitch against a fence/stump.
        CENTER_TOL = 3
        CENTER_FLIP_LIMIT = 4
        if self.stasis < 30 and self._center_flips < CENTER_FLIP_LIMIT:
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
        if self.note_push_facing(ram, next_tile):
            print(
                f"[NAVIGATOR] Push-facing block tile={next_tile} "
                f"action={read_player_action(ram)}"
            )
            self.path = []
            self.stasis = 0
            return None
        action = make_action(**{direction: True, 'b': True})

        if os.getenv("FENCE_DEBUG") == "1" and self.stasis > 0 and self.stasis % 60 == 0:
            print(f"[NAV] pos=({self.current_pos.x},{self.current_pos.y}) next={next_tile} target=({tgt_x},{tgt_y}) dir={direction} stasis={self.stasis} path_len={len(self.path)}")
            import sys; sys.stdout.flush()

        return action
