"""
Generic farm tile-scan and tool helpers.

TileScanner, ToolManager, and use_tool sequences used by clears / crops /
planner tasks. FarmClearer FSM stays in farm_clearer; nav stays in nav.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import os

import numpy as np

from retro_harness.actions import action_names

from harvest.core.tile_catalog import (
    ADDR_MAP,
    ADDR_TOOL,
    CLEARABLE_DEBRIS_TYPES,
    DEBRIS_TOOL,
    LARGE_ROCK_TILES,
    LIFTABLE_TILES,
    MAP_WIDTH,
    STUMP_TILES,
    TILE_SIZE,
    TILE_TO_DEBRIS,
    DebrisType,
    Tool,
    debris_footprint,
    is_multitile_debris_anchor,
)
from harvest.tasks.nav import Point, make_action

__all__ = [
    "Target",
    "TileScanner",
    "ToolManager",
    "action_to_names",
    "use_tool",
    "use_tool_facing",
    "cycle_tool",
]


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
        # Base tools need 6 consecutive hits on stump / large rock.
        if self.debris_type == DebrisType.ROCK or self.debris_type == DebrisType.STUMP:
            return 6
        return 1

    @property
    def footprint(self) -> Tuple[Tuple[int, int], ...]:
        return debris_footprint(self.tile, self.tile_id)


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
