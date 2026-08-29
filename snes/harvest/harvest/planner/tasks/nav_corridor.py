"""Travel corridor policy for MultiMapNavTask.

Soft solids (weed/stone/fence), live entity no-go, hop clamp, safe B-charge,
and lift-throw sequences. The waypoint FSM stays on MultiMapNavTask.
"""

from __future__ import annotations

from typing import Deque, List, Optional, Set, Tuple

import numpy as np

from harvest.core.animal_status import read_held_item
from harvest.core.npc_catalog import game_objects
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    DebrisType,
    FENCE,
    LIFTABLE_TILES,
    WEED,
)
from harvest.maps.map_config import Waypoint
from harvest.planner.day_plan_status import tilemaps_match
from harvest.planner.tasks.navigation import (
    MAX_HOP,
    _DIR_DELTA,
    _OPPOSITE_FACE,
    _neighbor_tile,
)
from harvest.tasks.farm_ops import TileScanner
from harvest.tasks.nav import (
    Navigator,
    Pathfinder,
    get_tile_at,
    make_action,
    TILE_SIZE,
)
from harvest.tasks.primitives import press_a_sequence


Tile = Tuple[int, int]


def dirs_toward(dx: int, dy: int) -> Tuple[str, str]:
    """Primary/secondary cardinals toward a pixel or tile delta."""
    if abs(dx) >= abs(dy):
        return ("right" if dx > 0 else "left", "down" if dy > 0 else "up")
    return ("down" if dy > 0 else "up", "right" if dx > 0 else "left")


def hop_target(cur: Tile, target_px: Tuple[int, int]) -> Tile:
    """BFS target clamped so each axis stays inside the loaded viewport."""
    final = (target_px[0] // TILE_SIZE, target_px[1] // TILE_SIZE)
    dx = final[0] - cur[0]
    dy = final[1] - cur[1]
    if abs(dx) <= MAX_HOP and abs(dy) <= MAX_HOP:
        return final
    cx = max(-MAX_HOP, min(MAX_HOP, dx))
    cy = max(-MAX_HOP, min(MAX_HOP, dy))
    limit = 7
    if abs(cx) > limit or abs(cy) > limit:
        scale = limit / max(abs(cx), abs(cy))
        cx = int(cx * scale)
        cy = int(cy * scale)
    return (cur[0] + cx, cur[1] + cy)


def tile_blocks_charge(pathfinder: Pathfinder, ram: np.ndarray, tx: int, ty: int) -> bool:
    """True when charging into this tile wastes frames (fence/solid/bush)."""
    if not (0 <= tx < 64 and 0 <= ty < 64):
        return True
    if not pathfinder.is_walkable(ram, tx, ty):
        return True
    tid = int(get_tile_at(ram, tx, ty))
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    return bool(tilemaps_match(tilemap, 0x00) and tid in {FENCE, WEED})


def safe_walk_action(
    pathfinder: Pathfinder,
    navigator: Navigator,
    ram: np.ndarray,
    preferred: str,
    *,
    secondary: Optional[str] = None,
    allow_detour: bool = False,
) -> Optional[np.ndarray]:
    """Hold B+dir only if the next tile is not a solid/bush thrash cell."""
    cur = navigator.current_tile
    order: List[str] = []
    candidates = (
        (preferred, secondary, "down", "right", "left", "up")
        if allow_detour
        else (preferred, secondary)
    )
    for direction in candidates:
        if direction and direction not in order:
            order.append(direction)
    for direction in order:
        nx, ny = _neighbor_tile(cur[0], cur[1], direction)
        if tile_blocks_charge(pathfinder, ram, nx, ny):
            continue
        if navigator.note_push_facing(ram, (nx, ny)):
            continue
        return make_action(**{direction: True, "b": True})
    return None


def farm_soft_blocks(
    scanner: TileScanner, ram: np.ndarray, tilemap: int
) -> Set[Tile]:
    """Weed/stone/fence cells that travel BFS must treat as no-go."""
    if not tilemaps_match(tilemap, 0x00):
        return set()
    return {
        target.tile
        for target in scanner.scan(
            ram,
            types={DebrisType.WEED, DebrisType.STONE, DebrisType.FENCE},
        )
    }


def entity_blocks(ram: np.ndarray, player_tile: Tile) -> Set[Tile]:
    """Reroute around live dog / NPC / animal sprites (not the player)."""
    blocked: Set[Tile] = set()
    try:
        objects = game_objects(ram)
    except Exception:
        objects = []
    for obj in objects:
        if getattr(obj, "is_player", False):
            continue
        tile = getattr(obj, "tile", None)
        if not tile:
            continue
        tx, ty = int(tile[0]), int(tile[1])
        if (tx, ty) == player_tile:
            continue
        if abs(tx - player_tile[0]) > 10 or abs(ty - player_tile[1]) > 10:
            continue
        kind = str(getattr(obj, "kind", "") or "")
        label = str(getattr(obj, "label", "") or "")
        if kind in {"animal", "npc_candidate"} or label in {"dog", "chicken", "cow"}:
            blocked.add((tx, ty))
        elif getattr(obj, "is_npc_candidate", False):
            blocked.add((tx, ty))
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap == 0x10 and blocked:
        padded = set(blocked)
        for bx, by in blocked:
            if bx < 28:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    padded.add((bx + dx, by + dy))
        padded.discard(player_tile)
        blocked = padded
    return blocked


def replace_no_go(
    pathfinder: Pathfinder, previous: Set[Tile], nxt: Set[Tile]
) -> None:
    pathfinder.no_go_tiles.difference_update(previous)
    pathfinder.no_go_tiles.update(nxt)


def liftable_gate_toward(
    current_tile: Tile, ram: np.ndarray, wp: Waypoint
) -> Optional[Tuple[str, Tile, int]]:
    """If a liftable soft solid blocks progress toward wp, return face/tile/id."""
    goal = (wp.target_px[0] // TILE_SIZE, wp.target_px[1] // TILE_SIZE)
    dx = goal[0] - current_tile[0]
    dy = goal[1] - current_tile[1]
    faces: List[str] = []
    if abs(dx) >= abs(dy):
        faces.append("right" if dx > 0 else "left")
        if dy != 0:
            faces.append("down" if dy > 0 else "up")
    else:
        faces.append("down" if dy > 0 else "up")
        if dx != 0:
            faces.append("right" if dx > 0 else "left")
    for face in faces:
        nx, ny = _neighbor_tile(current_tile[0], current_tile[1], face)
        if not (0 <= nx < 64 and 0 <= ny < 64):
            continue
        tid = int(get_tile_at(ram, nx, ny))
        if tid in LIFTABLE_TILES:
            return face, (nx, ny), tid
    return None


def _lift_face_toward(current_tile: Tile, cand: Tile) -> str:
    if cand[1] < current_tile[1]:
        return "up"
    if cand[1] > current_tile[1]:
        return "down"
    if cand[0] < current_tile[0]:
        return "left"
    if cand[0] > current_tile[0]:
        return "right"
    return "up"


def queue_lift_throw(
    action_queue: Deque[np.ndarray],
    current_tile: Tile,
    ram: np.ndarray,
    wp: Waypoint,
) -> Optional[str]:
    """Queue lift then throw, or skip when the gate is already open."""
    face = wp.action_face or "up"
    throw_face = _OPPOSITE_FACE.get(face, "down")
    if tilemaps_match(
        int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0, 0x00
    ):
        throw_face = "down"
    held = int(read_held_item(ram))
    face_tile = _neighbor_tile(current_tile[0], current_tile[1], face)
    dx, dy = _DIR_DELTA.get(face, (0, 0))
    candidates = [face_tile, (face_tile[0] + dx, face_tile[1] + dy)]
    target: Optional[Tile] = None
    tid = 0
    lift_face = face
    for cand in candidates:
        if not (0 <= cand[0] < 64 and 0 <= cand[1] < 64):
            continue
        cand_tid = int(get_tile_at(ram, *cand))
        if cand_tid in LIFTABLE_TILES:
            target = cand
            tid = cand_tid
            lift_face = _lift_face_toward(current_tile, cand)
            break
    hold = max(12, int(wp.action_frames))
    settle = max(12, int(wp.action_cooldown))

    if held != 0:
        action_queue.extend(
            press_a_sequence(
                throw_face,
                face_frames=6,
                pre_press_settle_frames=4,
                hold_frames=hold,
                settle_frames=settle,
            )
        )
        return f"throw held=0x{held:02X} face={throw_face}"

    if target is not None:
        action_queue.extend(
            press_a_sequence(
                lift_face,
                face_frames=8,
                pre_press_settle_frames=4,
                hold_frames=hold,
                settle_frames=settle,
            )
        )
        action_queue.extend(
            press_a_sequence(
                throw_face,
                face_frames=6,
                pre_press_settle_frames=4,
                hold_frames=hold,
                settle_frames=settle,
            )
        )
        return (
            f"lift_throw stand={current_tile} "
            f"target={target} tid=0x{tid:02X} face={lift_face}"
        )
    return None


def opportunistic_clear_waypoint(wp: Waypoint, face: str) -> Waypoint:
    """Waypoint-shaped lift_throw used when a weed seals a travel corridor."""
    return Waypoint(
        tilemap=wp.tilemap,
        target_px=wp.target_px,
        radius=wp.radius,
        action_on_arrive="lift_throw",
        action_face=face,
        action_frames=22,
        action_cooldown=24,
    )


def micro_center_action(cur_x: int, cur_y: int, target_px: Tuple[int, int]) -> np.ndarray:
    """Walk without B to center inside the current tile (no neighbor step)."""
    dx = target_px[0] - cur_x
    dy = target_px[1] - cur_y
    if abs(dx) >= abs(dy) and abs(dx) > 0:
        return make_action(right=dx > 0, left=dx < 0)
    if abs(dy) > 0:
        return make_action(down=dy > 0, up=dy < 0)
    return make_action()


__all__ = [
    "dirs_toward",
    "entity_blocks",
    "farm_soft_blocks",
    "hop_target",
    "liftable_gate_toward",
    "micro_center_action",
    "opportunistic_clear_waypoint",
    "queue_lift_throw",
    "replace_no_go",
    "safe_walk_action",
    "tile_blocks_charge",
]
