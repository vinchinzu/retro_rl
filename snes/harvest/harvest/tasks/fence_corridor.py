"""Fence corridor / y=31 wall policy.

Pure helpers: staging tiles, south-charge tapes, local-drop faces, and
wall-first target pick. FenceClearLoopTask stays the FSM.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from harvest.core.tile_catalog import DebrisType
from harvest.maps.map_config import (
    FARM_POND_ACCESS_FENCE_ROW,
    FARM_POND_ACCESS_FENCE_X_RANGE,
    FARM_POND_ACCESS_STAGING_TILES,
)
from harvest.tasks.farm_ops import Target, scan_typed_targets
from harvest.tasks.nav import VIEWPORT_HOP_TILES, get_tile_at, make_action, manhattan

# Main F0 pond south lip (same as map_config pond_edge / go_to_water_source).
POND_TILES = [(32, 34), (33, 34)]
ADDR_PLAYER_STATE = 0xD2
ACTION_CARRYING_BIT = 0x02
ADDR_PLAYER_ACTION = 0xD4
ACTION_DROPPING = 0x05

# Pond-side barriers (6-tile model). Leave south lip approach open:
# (30,34)/(31,34) are the west approach to POND_TILES (32,34)/(33,34).
POND_NO_GO_TILES = frozenset(
    {
        (30, 29),
        (31, 29),
        (32, 29),
        (33, 29),
        (34, 29),
        (35, 29),  # Top
        (30, 31),
        (30, 32),
        (30, 33),  # Far Left (not y=30/34)
        (31, 31),
        (31, 32),
        (31, 33),  # Near Left (not y=30/34)
        (34, 30),
        (34, 31),
        (34, 32),
        (34, 33),
        (34, 34),  # Near Right
        (35, 30),
        (35, 31),
        (35, 32),
        (35, 33),
        (35, 34),  # Far Right
        (32, 31),
        (32, 32),
        (32, 33),
        (33, 31),
        (33, 32),
        (33, 33),
    }
)

Tile = Tuple[int, int]
FencePick = Tuple[Target, Tile, List[Tile]]


def is_access_wall(tile: Tile) -> bool:
    x, y = int(tile[0]), int(tile[1])
    x0, x1 = FARM_POND_ACCESS_FENCE_X_RANGE
    return y == FARM_POND_ACCESS_FENCE_ROW and x0 <= x <= x1


def pond_dump_key(
    tile: Tile, skip_tiles: Iterable[Tile], pond_tiles: Sequence[Tile] = POND_TILES
) -> Tuple[int, int, int]:
    tile = (int(tile[0]), int(tile[1]))
    skip = 1 if tile in skip_tiles else 0
    x, y = tile
    wall = 0 if is_access_wall(tile) else 1
    pond = min(abs(x - p[0]) + abs(y - p[1]) for p in pond_tiles)
    return (skip, wall, pond)


def nearest_pond(tile: Tile, pond_tiles: Sequence[Tile] = POND_TILES) -> Tile:
    return min(pond_tiles, key=lambda p: abs(p[0] - tile[0]) + abs(p[1] - tile[1]))


def south_charge_actions() -> List[np.ndarray]:
    """B-run south through a just-lifted y=31 gap (ROM soft-blocks BFS)."""
    actions = [make_action(down=True) for _ in range(12)]
    actions.extend([make_action(down=True, b=True) for _ in range(160)])
    for _ in range(4):
        actions.extend([make_action(down=True, b=True) for _ in range(36)])
        actions.extend([make_action(left=True) for _ in range(5)])
        actions.extend([make_action(down=True, b=True) for _ in range(36)])
        actions.extend([make_action(right=True) for _ in range(5)])
    actions.extend([make_action() for _ in range(12)])
    return actions


def local_drop_faces(
    *, corridor_only: bool, tile_y: int, attempts: int
) -> Tuple[str, ...]:
    if corridor_only and tile_y <= 31:
        # At the gap, dropping south reseals the only exit cell.
        return ("left", "right", "up")
    if corridor_only and attempts == 0:
        return ("down", "left", "right")
    return ("down", "left", "right", "up")


def local_drop_actions(faces: Sequence[str]) -> List[np.ndarray]:
    actions: List[np.ndarray] = []
    for face in faces:
        actions.extend([make_action(**{face: True}) for _ in range(10)])
        actions.extend([make_action() for _ in range(4)])
        actions.extend([make_action(**{face: True, "a": True}) for _ in range(20)])
        actions.extend([make_action(a=True) for _ in range(8)])
        actions.extend([make_action() for _ in range(16)])
    return actions


def lift_actions(direction: str) -> List[np.ndarray]:
    actions = [make_action(**{direction: True}) for _ in range(10)]
    actions.extend([make_action(**{direction: True, "a": True}) for _ in range(25)])
    actions.extend([make_action() for _ in range(30)])
    return actions


def corridor_after_lift(tile: Tile, *, charge_done: bool) -> str:
    """Next corridor carry step: drop_south, south_charge, or local_drop."""
    if tile[1] >= 32:
        return "drop_south"
    if tile[1] <= 31 and not charge_done:
        return "south_charge"
    return "local_drop"


def filter_wall_targets(targets: Sequence[Target]) -> List[Target]:
    row = FARM_POND_ACCESS_FENCE_ROW
    x_min, x_max = FARM_POND_ACCESS_FENCE_X_RANGE
    return [
        target
        for target in targets
        if target.tile[1] == row and x_min <= target.tile[0] <= x_max
    ]


def weed_tiles(scanner, ram) -> set:
    return {
        target.tile
        for target in scanner.scan(ram, types={DebrisType.WEED})
    }


def choose_corridor_stage(
    pathfinder, ram, player: Tile, current_stage: Optional[Tile] = None
) -> Tuple[Optional[Tile], Optional[List[Tile]]]:
    if current_stage is not None:
        return current_stage, pathfinder.find_path(ram, player, current_stage)
    candidates = sorted(
        FARM_POND_ACCESS_STAGING_TILES,
        key=lambda tile: abs(tile[0] - player[0]) + abs(tile[1] - player[1]),
    )
    for candidate in candidates:
        path = pathfinder.find_path(ram, player, candidate)
        if path is not None:
            return candidate, path
    return None, None


def sort_fence_targets(
    targets: List[Target],
    *,
    pond_dump: bool,
    skip_tiles: Iterable[Tile],
    player_pos,
) -> List[Target]:
    if pond_dump:
        targets.sort(key=lambda t: pond_dump_key(t.tile, skip_tiles))
    else:
        targets.sort(key=lambda t: manhattan(t.pos, player_pos))
    return targets


def pick_fence_target(
    targets: Sequence[Target],
    *,
    ram,
    pathfinder,
    player_tile: Tile,
    player_pos,
    skip_tiles: Iterable[Tile],
    corridor_only: bool,
    pond_dump: bool,
    debug: bool = False,
) -> Optional[FencePick]:
    """Wall-first stand: full path beats a nearer hop; pond dump prefers wall hops."""
    reached_wall = None
    reached_other = None
    hop_wall = None
    hop_wall_dist = None
    hop_other = None
    hop_other_dist = None
    for target in targets:
        tile = (int(target.tile[0]), int(target.tile[1]))
        if tile in skip_tiles:
            continue
        if corridor_only:
            approach = (tile[0], tile[1] - 1)
            if not pathfinder.is_walkable(ram, *approach):
                approach = None
        else:
            approach = pathfinder.find_approach(ram, target.tile, player_pos)
        if approach is None:
            if debug:
                print(f"[FENCE] skip target {target.tile}: no approach")
            continue
        path = pathfinder.find_path(
            ram, player_tile, approach, max_steps=VIEWPORT_HOP_TILES
        )
        if path is None:
            if debug:
                print(f"[FENCE] skip target {target.tile}: no path")
            continue
        wall = is_access_wall(tile)
        reached = (
            not path or path[-1] == approach or player_tile == approach
        )
        row = (target, approach, path)
        if reached:
            if wall:
                return row
            if reached_other is None:
                reached_other = row
            continue
        dist = abs(player_tile[0] - approach[0]) + abs(
            player_tile[1] - approach[1]
        )
        if wall:
            if hop_wall is None or dist < hop_wall_dist:
                hop_wall = row
                hop_wall_dist = dist
        elif hop_other is None or dist < hop_other_dist:
            hop_other = row
            hop_other_dist = dist
    if pond_dump:
        return reached_wall or hop_wall or reached_other or hop_other
    return reached_wall or reached_other or hop_wall or hop_other


def scan_fence_targets(
    ram,
    debris_types,
    farm_bounds,
    *,
    scanner,
    corridor_only: bool,
):
    wanted = tuple(debris_types) or (DebrisType.FENCE,)
    targets = scan_typed_targets(ram, wanted, farm_bounds, scanner=scanner)
    if corridor_only:
        targets = filter_wall_targets(targets)
    return targets


def debug_fence_map(ram, total_steps, state, pos, target, approach) -> None:
    print(
        f"[FENCE] step={total_steps} state={state} pos={pos} "
        f"target={target} approach={approach}"
    )
    print("--- Map Area Dump (X:25-39, Y:25-39) ---")
    for y in range(25, 40):
        row = [f"{get_tile_at(ram, x, y):02x}" for x in range(25, 40)]
        print(f"Y={y:2d}: {' '.join(row)}")
    print("-----------------------------------------")


__all__ = [
    "ACTION_CARRYING_BIT",
    "ACTION_DROPPING",
    "ADDR_PLAYER_ACTION",
    "ADDR_PLAYER_STATE",
    "POND_NO_GO_TILES",
    "POND_TILES",
    "choose_corridor_stage",
    "corridor_after_lift",
    "debug_fence_map",
    "filter_wall_targets",
    "is_access_wall",
    "lift_actions",
    "local_drop_actions",
    "local_drop_faces",
    "nearest_pond",
    "pick_fence_target",
    "pond_dump_key",
    "scan_fence_targets",
    "sort_fence_targets",
    "south_charge_actions",
    "weed_tiles",
]
