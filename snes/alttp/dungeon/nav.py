"""Fewest-turn tile BFS on one room attribute grid.

``start`` and ``goal`` are room-local pixels of Link's coordinate. Tile
``(col, row)`` covers ``[col*8, col*8+8)`` by ``[row*8, row*8+8)``. Each
step is 8 px, so a goal off that phase is unreachable.

The search minimises turns, then steps. A length-only path through an open
room staircases and Link flutters; an L does not.

``body`` is the inclusive pixel box (``walk_physics.LINK_BODY``). A pose is
legal when every tile under that box is walkable. Pits (``0x20``), deep
water (``0x08``) and the spike byte (``0x44``, not sampled on an indoor pin)
are walls unless named in ``allow``. Floor, shallow water, doors, stairs
and ice are walkable. Any other byte is a wall.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from alttp.dungeon.tilemap import TILE_COLS, TILE_ROWS, tile_kind
from alttp.walk_physics import LINK_BODY

__all__ = [
    "COMMIT_FRAMES",
    "DOOR_ALIGN_TOLERANCE",
    "DoorLine",
    "WalkCommit",
    "door_line",
    "find_path",
    "heading_for",
    "room_origin",
    "room_step",
]

# Ledger flutter window is 8 frames, inclusive. Hold one heading longer
# than that, and bleed a reverse with empty frames, so a 1-2 px overshoot
# cannot tug back inside the window.
COMMIT_FRAMES = 9
_OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}

_TILE = 8
# dx, dy. Order is not the turn count; the heap is.
_HEADINGS = ((8, 0), (-8, 0), (0, 8), (0, -8))
_OPTIONAL = {
    "pit": 0x20,
    "deep_water": 0x08,
    "spike": 0x44,
}
# Raised peg 0x67 is legend "wall". Down peg 0x66 is "floor". The live
# attribute grid is the peg state; this search does not write $C172.
_WALKABLE_KINDS = frozenset({"floor", "shallow_water", "door", "stairs", "ice"})


def find_path(
    tiles: object,
    start: tuple[int, int],
    goal: tuple[int, int],
    body: tuple[int, int, int, int],
    allow: Iterable[str] = (),
) -> list[tuple[int, int]] | None:
    """Room-local pixels from ``start`` to ``goal``, fewest turns then steps.

    ``None`` when no legal path exists. ``allow`` may name ``pit``,
    ``deep_water``, and ``spike``.
    """
    grid = np.asarray(tiles, dtype=np.uint8)
    opened = frozenset(allow)
    if not _legal(grid, start[0], start[1], body, opened):
        return None
    if start == goal:
        return [start]
    # (turns, steps) keyed by (x, y, heading). Heading -1 is the start.
    best: dict[tuple[int, int, int], tuple[int, int]] = {(start[0], start[1], -1): (0, 0)}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[int, int, int, int, int, int]] = []
    seq = 0
    heapq.heappush(heap, (0, 0, seq, start[0], start[1], -1))
    end: tuple[int, int, int] | None = None
    while heap:
        turns, steps, _, x, y, heading = heapq.heappop(heap)
        if best.get((x, y, heading)) != (turns, steps):
            continue
        if (x, y) == goal:
            end = (x, y, heading)
            break
        for index, (dx, dy) in enumerate(_HEADINGS):
            nx, ny = x + dx, y + dy
            if not _legal(grid, nx, ny, body, opened):
                continue
            nturns = turns if heading in (-1, index) else turns + 1
            nsteps = steps + 1
            key = (nx, ny, index)
            cand = (nturns, nsteps)
            if key in best and best[key] <= cand:
                continue
            best[key] = cand
            parent[key] = (x, y, heading)
            seq += 1
            heapq.heappush(heap, (nturns, nsteps, seq, nx, ny, index))
    if end is None:
        return None
    path = [goal]
    cursor = end
    while cursor[2] != -1:
        cursor = parent[cursor]
        path.append((cursor[0], cursor[1]))
    path.reverse()
    return path


def _legal(
    grid: np.ndarray,
    x: int,
    y: int,
    body: tuple[int, int, int, int],
    allow: frozenset[str],
) -> bool:
    rows, cols = grid.shape
    x0, y0, x1, y1 = body
    left, top, right, bottom = x + x0, y + y0, x + x1, y + y1
    if left < 0 or top < 0:
        return False
    c0, r0 = left // _TILE, top // _TILE
    c1, r1 = right // _TILE, bottom // _TILE
    if r1 >= rows or c1 >= cols:
        return False
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            if not _walkable(int(grid[row, col]), allow):
                return False
    return True


@dataclass(frozen=True)
class WalkCommit:
    """Heading held so far, and whether the pad is up to bleed a reverse."""

    direction: str | None = None
    frames: int = 0
    releasing: bool = False


def heading_for(x: int, y: int, path: Iterable[tuple[int, int]]) -> str | None:
    """Cardinal from ``(x, y)`` toward the next point on ``path``."""
    for px, py in path:
        dx, dy = px - x, py - y
        if dx == 0 and dy == 0:
            continue
        if abs(dx) >= abs(dy) and dx != 0:
            return "RIGHT" if dx > 0 else "LEFT"
        if dy != 0:
            return "DOWN" if dy > 0 else "UP"
    return None


def room_step(
    commit: WalkCommit, desired: str | None
) -> tuple[str | None, WalkCommit]:
    """Button to hold this frame, and the commit after it.

    ``desired`` may change every frame. The returned button stays on
    ``commit.direction`` until ``COMMIT_FRAMES``. A reverse releases the
    pad for ``COMMIT_FRAMES`` first.
    """
    if desired is None:
        # Remember the heading. Forgetting it lets the next waypoint reverse
        # on the following frame.
        idle = commit.frames + 1 if commit.releasing else 1
        return None, WalkCommit(commit.direction, idle, True)
    if commit.direction is None or commit.direction == desired:
        frames = commit.frames + 1 if commit.direction == desired and not commit.releasing else 1
        return desired, WalkCommit(desired, frames, False)
    reverse = _OPPOSITE.get(commit.direction) == desired
    if reverse:
        released = commit.frames + 1 if commit.releasing else 1
        if released <= COMMIT_FRAMES:
            return None, WalkCommit(commit.direction, released, True)
        return desired, WalkCommit(desired, 1, False)
    if commit.frames < COMMIT_FRAMES and not commit.releasing:
        return commit.direction, WalkCommit(commit.direction, commit.frames + 1, False)
    return desired, WalkCommit(desired, 1, False)


# CastleMain 2026-09-30, no RAM writes. Type 0x02 regular.
# West pos $0F8A centre link_y 3320, from x=599 ($6C=0): y 3305..3335
# entered room 0x60; y 3304 and 3336 stuck at x=576 with $6C=0.
# South pos $1BBC centre link_x 760, from y=3460: x 745 and 775 set
# $6C=1; x 744 and 776 stuck at y=3496 with $6C=0.
DOOR_ALIGN_TOLERANCE = {0x02: 15}

# CastleMain layer 0 bytes >= this are door mouths only (0x81, 0x85, 0x8E, 0x95).
_DOOR_MOUTH_MIN = 0x80
# Mouth is offset from the slot anchor. West (5, 31) uses rows 32-33;
# south (30, 55) uses cols 31-32. The window does not reach the next door.
_MOUTH_DC = (-2, 6)
_MOUTH_DR = (-1, 6)


@dataclass(frozen=True)
class DoorLine:
    """Cross-axis lineup before a push. ``centre`` is Link's coordinate."""

    axis: str  # "x" column for N/S, "y" row for E/W
    centre: int
    tolerance: int
    door_type: int


def room_origin(x: int, y: int) -> tuple[int, int]:
    """World pixel of tile (0, 0).

    A room is 512px. ``$EC`` (CastleMain ``$01F8``) also clears the low
    bits, so ``coord & ~mask`` still moves when Link is off an 8px
    boundary. CastleMain ``(760, 3520)`` and ``(700, 3326)`` share
    ``(512, 3072)``.
    """
    return (int(x) & ~0x1FF, int(y) & ~0x1FF)


def door_line(
    tiles: object,
    pos: int,
    direction: int,
    door_type: int,
    origin: tuple[int, int],
) -> DoorLine | None:
    """Centre column or row of one live door slot.

    ``origin`` is :func:`room_origin`, not the ``$EC`` residue.

    ``None`` when ``door_type`` has no measured tolerance, ``direction``
    is not a cardinal slot, or the anchor window has no mouth. The
    tolerance is the ROM capture window for that type (see
    ``DOOR_ALIGN_TOLERANCE``).
    """
    code = int(door_type) & 0xFF
    tolerance = DOOR_ALIGN_TOLERANCE.get(code)
    direc = int(direction) & 0xFF
    if tolerance is None or direc not in (0, 1, 2, 3):
        return None
    grid = np.asarray(tiles, dtype=np.uint8)
    anchor_c = (int(pos) >> 1) & 0x3F
    anchor_r = (int(pos) >> 7) & 0x3F
    c0 = max(0, anchor_c + _MOUTH_DC[0])
    c1 = min(TILE_COLS, anchor_c + _MOUTH_DC[1] + 1)
    r0 = max(0, anchor_r + _MOUTH_DR[0])
    r1 = min(TILE_ROWS, anchor_r + _MOUTH_DR[1] + 1)
    cols: list[int] = []
    rows: list[int] = []
    for row in range(r0, r1):
        for col in range(c0, c1):
            if int(grid[row, col]) >= _DOOR_MOUTH_MIN:
                cols.append(col)
                rows.append(row)
    if not cols:
        return None
    ox, oy = origin
    if direc in (0, 1):
        left = ox + min(cols) * _TILE
        width = (max(cols) - min(cols) + 1) * _TILE
        body = LINK_BODY[2] - LINK_BODY[0] + 1
        centre = left - LINK_BODY[0] + max(0, width - body) // 2
        return DoorLine("x", centre, tolerance, code)
    top = oy + min(rows) * _TILE
    height = (max(rows) - min(rows) + 1) * _TILE
    body = LINK_BODY[3] - LINK_BODY[1] + 1
    centre = top - LINK_BODY[1] + max(0, height - body) // 2
    return DoorLine("y", centre, tolerance, code)


def _walkable(value: int, allow: frozenset[str]) -> bool:
    for name, byte in _OPTIONAL.items():
        if value == byte or tile_kind(value) == name:
            return name in allow
    return tile_kind(value) in _WALKABLE_KINDS
