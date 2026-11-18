"""Barn map and the walker: where the farmer can stand, and how he gets there among the cows.

Measured on the ROM (2026-11-05): 24 farmers ran an empty barn at random for 400k frames each with no RAM
writes, and every pixel they stood on is in ``BARN_WALK_ROWS`` (4 px cells). A map poked cell by cell is
wrong: tile collision only stops a step into a solid, so a poked farmer walks freely inside the rails.

- The two wings are wide aisles (x 12..52, x 180..219), joined by the top row, the crossing rows y 180..187
  and the bottom hall. The centre aisle is 8 px wide (x 108..115). Each stall is an 8 px lane off a wing.
- A cow body stops a walk for good at 13 px. A run (B held) stands at that edge about 40 frames and then
  shoves her ahead of it, unless a wall is behind her. She can be shoved aside, never walked through.
- A press reaches the point 16 px ahead of the farmer, +-12 px around it (``COW_REACH_PX``). Talk reads that
  at the press. A tool reads the contact left by the last step, so a swing needs a real step towards her
  from the stand (``COW_STAND_PX``) and misses if she walks off before it lands, about 65 frames later.
- A cow takes a 32-frame step about every 256 frames, and entering the barn puts every cow in its stall.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional, Sequence, Tuple

import numpy as np

from harvest.core.animal_probe import BARN_TILEMAP
from harvest.core.animal_status import existing_cow_slots
from harvest.core.ram_catalog import read_animal_slot_field
from harvest.maps.map_config import find_landmark
from harvest.tasks.nav import make_action

Tile = Tuple[int, int]
Pixel = Tuple[int, int]

BLOCKED_REASON = "cow walk blocked"  # CowChoresTask failure the day plan answers with a barn re-entry
# A cow stepping on the door tile ($C2) leaves the barn and stops getting care. Entering resets every cow
# to her stall, so a chore pass walks out and back in before the quickest cow can reach the door.
BARN_SHIFT_REASON = "barn shift over"
# Cows wander from the first frame (a 16 px step every few hundred frames) and the door is the far end of the hall. Eight idle
# runs of 30,000 frames from a fresh entry saw nobody leave, the first cow within 72 px of the door at 1,870 and the median at
# 14,500, within 32 px at 13,490 and later. With the farmer working it was faster: fuzz seed 78 had a cow walk from 57 px to
# the door tile in 350 frames, between two job boundaries, 6,000 frames after entry; the old 6,000 timer and `y >= 308` rule
# hid that at ~3.7 resets a day. A reset is ~450 frames of exit and entry plus the walk to the door, so it is spent when a cow
# is within ``BARN_DOOR_DANGER_PX`` of the door (checked every job boundary and every walking frame with empty hands), or when
# the pass runs ``BARN_SHIFT_FRAMES``. A cow in the hall rows above (y 308..329) only slows the route to the bin.
BARN_SHIFT_FRAMES = 14000
BARN_DOOR_CENTER_PX: Pixel = (136, 360)  # tile (8, 22), the door tile $C2
BARN_DOOR_DANGER_PX = 72

_BARN_BIN = find_landmark("barn_shipping_bin", tilemap_id=BARN_TILEMAP)
_FODDER = find_landmark("fodder_dispenser", tilemap_id=BARN_TILEMAP)
FODDER_STAND: Tile = _FODDER[1].tile if _FODDER else (13, 11)
FODDER_FACE: str = _FODDER[1].face if _FODDER and _FODDER[1].face else "right"
FODDER_STAND_PX: Pixel = (FODDER_STAND[0] * 16 + 8, FODDER_STAND[1] * 16 + 8)
# The dispenser answers from the tile above the stand as well (ROM probe: A at (217,169) takes the
# fodder); a cow standing on one stand would otherwise eat the press as a talk.
FODDER_NORTH_STAND_PX: Pixel = (FODDER_STAND_PX[0], FODDER_STAND_PX[1] - 16)
BARN_SHIP_BIN_STAND: Tile = _BARN_BIN[1].tile if _BARN_BIN else (2, 22)
BARN_SHIP_BIN_FACE: str = _BARN_BIN[1].face if _BARN_BIN and _BARN_BIN[1].face else "left"
BARN_SHIP_BIN_PX: Pixel = (56, 348)  # ROM-paid left throw; avoids the lower-left corner's cow trap
BARN_DOOR_PX: Pixel = (128, 356)
# Where a farmer with a bottle waits for a cow to leave the bin stand: in the south hall, off her edge (seed 20 of the fuzz lost a
# bottle to a reset after nudging her for 1,300 frames).
BIN_HOLD_PX: Pixel = (BARN_SHIP_BIN_PX[0] + 40, BARN_SHIP_BIN_PX[1] + 4)

# Stand four pixels off a cow's edge, in line with her, so the step towards her ends at the edge.
COW_STAND_PX = 17
COW_EDGE_PX = 13
COW_REACH_PX = 12
PRESS_AHEAD_PX = 16
FACE_UNITS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}


@dataclass(frozen=True)
class CowFeedSpot:
    stand: Tile
    face: str
    interact_px: Pixel
    flag: int


# Feed trough coordinates decoded from the barn replacement table
# (DATA16_81B0ED) and Cow_Feed_Flags in the decomp.
COW_FEED_SPOTS: Tuple[CowFeedSpot, ...] = (
    CowFeedSpot((7, 17), "right", (113, 277), 0x0001),
    CowFeedSpot((7, 15), "right", (113, 245), 0x0002),
    CowFeedSpot((7, 13), "right", (113, 213), 0x0004),
    CowFeedSpot((7, 9), "right", (113, 149), 0x0008),
    CowFeedSpot((7, 7), "right", (113, 117), 0x0010),
    CowFeedSpot((7, 5), "right", (113, 85), 0x0020),
    CowFeedSpot((6, 17), "left", (111, 277), 0x0040),
    CowFeedSpot((6, 15), "left", (111, 245), 0x0080),
    CowFeedSpot((6, 13), "left", (111, 213), 0x0100),
    CowFeedSpot((6, 9), "left", (111, 149), 0x0200),
    CowFeedSpot((6, 7), "left", (111, 117), 0x0400),
    CowFeedSpot((6, 5), "left", (111, 85), 0x0800),
)
PREGNANT_COW_FEED_SPOT = CowFeedSpot((11, 22), "right", (177, 357), 0x1000)


# ── measured walk map ─────────────────────────────────────────────────

BARN_GRID = 4
BARN_MASK_Y0 = 52
BARN_WALK_ROWS: Tuple[str, ...] = (
    "...##########################################################...",  # y=52
    "...##########################################################...",  # y=56
    "...########................##................##########.........",  # y=60
    "...########................##................##########.........",  # y=64
    "...########................##................##########.........",  # y=68
    "...########................##................##########.........",  # y=72
    "...########................##................##########.........",  # y=76
    "...########................##................##########.........",  # y=80
    "...################........##........##################.........",  # y=84
    "...################........##........##################.........",  # y=88
    "...########................##................##########.........",  # y=92
    "...########................##................##########.........",  # y=96
    "...########................##................##########.........",  # y=100
    "...########................##................##########.........",  # y=104
    "...########................##................##########.........",  # y=108
    "...########................##................##########.........",  # y=112
    "...################........##........########################...",  # y=116
    "...################........##........########################...",  # y=120
    "...########................##................################...",  # y=124
    "...########................##................################...",  # y=128
    "...########................##................################...",  # y=132
    "...########................##................################...",  # y=136
    "...########................##................################...",  # y=140
    "...########................##................################...",  # y=144
    "...################........##........##################.........",  # y=148
    "...################........##........##################.........",  # y=152
    "...########................##................##########.........",  # y=156
    "...########................##................##########.........",  # y=160
    "...########................##................##########.........",  # y=164
    "...########................##................##########.........",  # y=168
    "...########................##................##########.........",  # y=172
    "...########................##................##########.........",  # y=176
    "...####################################################.........",  # y=180
    "...####################################################.........",  # y=184
    "...########................##................##########.........",  # y=188
    "...########................##................##########.........",  # y=192
    "...########................##................##########.........",  # y=196
    "...########................##................##########.........",  # y=200
    "...########................##................##########.........",  # y=204
    "...########................##................##########.........",  # y=208
    "...################........##........########################...",  # y=212
    "...################........##........########################...",  # y=216
    "...########................##................################...",  # y=220
    "...########................##................################...",  # y=224
    "...########................##................################...",  # y=228
    "...########................##................################...",  # y=232
    "...########................##................######......####...",  # y=236
    "...########................##................######......####...",  # y=240
    "...################........##........##############......####...",  # y=244
    "...################........##........##############......####...",  # y=248
    "...########................##................######......####...",  # y=252
    "...########................##................######......####...",  # y=256
    "...########................##................######......####...",  # y=260
    "...########................##................######......####...",  # y=264
    "...########................##................######......####...",  # y=268
    "...########................##................######......####...",  # y=272
    "...################........##........########################...",  # y=276
    "...################........##........########################...",  # y=280
    "...########................##................######.............",  # y=284
    "...########................##................######.............",  # y=288
    "...########................##................######.............",  # y=292
    "...########................##................######.............",  # y=296
    "...########................##................######.............",  # y=300
    "...########................##................######.............",  # y=304
    "...##########################################################...",  # y=308
    "...##########################################################...",  # y=312
    ".....########################################################...",  # y=316
    ".....########################################################...",  # y=320
    ".....########################################################...",  # y=324
    ".....########################################################...",  # y=328
    ".............################################........########...",  # y=332
    ".............################################........########...",  # y=336
    ".............################################........########...",  # y=340
    ".............################################........########...",  # y=344
    ".............################################........########...",  # y=348
    ".............################################........########...",  # y=352
    ".........####################################........########...",  # y=356
    ".........####################################........########...",  # y=360
)
BARN_CELLS: frozenset[Pixel] = frozenset(
    (col * BARN_GRID, BARN_MASK_Y0 + row * BARN_GRID)
    for row, text in enumerate(BARN_WALK_ROWS)
    for col, mark in enumerate(text)
    if mark == "#"
)
# Grid-following keeps the farmer up to two pixels off a cell centre, so a route along a wall scrapes.
_NEIGHBOURS = ((BARN_GRID, 0), (-BARN_GRID, 0), (0, BARN_GRID), (0, -BARN_GRID))
_WALL_COST: dict[Pixel, int] = {
    cell: 3 if any((cell[0] + dx, cell[1] + dy) not in BARN_CELLS for dx, dy in _NEIGHBOURS) else 0
    for cell in BARN_CELLS
}
COW_SOFT_COST = 6  # a route skirts a cow when that is cheap, and goes through her when it is not
COW_SOFT_BOX_PX = 14


def nearest_cell(pixel: Pixel) -> Pixel:
    """The walkable cell closest to ``pixel`` (a cow or a farmer can sit a few pixels off the grid)."""
    x = int(pixel[0]) // BARN_GRID * BARN_GRID
    y = int(pixel[1]) // BARN_GRID * BARN_GRID
    if (x, y) in BARN_CELLS:
        return x, y
    for radius in range(1, 24):
        ring = [
            (x + dx * BARN_GRID, y + dy * BARN_GRID)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if max(abs(dx), abs(dy)) == radius
        ]
        hits = [cell for cell in ring if cell in BARN_CELLS]
        if hits:
            return min(hits, key=lambda cell: abs(cell[0] - pixel[0]) + abs(cell[1] - pixel[1]))
    return x, y


def walk_distances(start: Pixel) -> dict[Pixel, int]:
    """Cells of walking from ``start`` to every cell of the walk map (breadth first, cows ignored)."""
    source = nearest_cell(start)
    steps = {source: 0}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for dx, dy in _NEIGHBOURS:
            step = (cell[0] + dx, cell[1] + dy)
            if step in BARN_CELLS and step not in steps:
                steps[step] = steps[cell] + 1
                queue.append(step)
    return steps


def walk_cost(steps: dict[Pixel, int], pixel: Pixel) -> int:
    """Pixels of walking to ``pixel`` from the origin of ``steps``; far when the map never reaches it."""
    return steps.get(nearest_cell(pixel), 1 << 12) * BARN_GRID


def plan_route(start: Pixel, goal: Pixel, cows: Iterable[Pixel] = (), *, clear: bool = False) -> list[Pixel]:
    """Corner waypoints from ``start`` to ``goal`` on the walk map (Dijkstra, four neighbours).

    ``clear`` closes the cells around every cow, so a walk never shoves one: a cow pushed along the hall
    ends up on the door tile and out of the barn. An empty list means no such route exists.
    """
    source, target = nearest_cell(start), nearest_cell(goal)
    soft: set[Pixel] = set()
    for cx, cy in cows:
        reach = COW_SOFT_BOX_PX // BARN_GRID + 1
        bx, by = int(cx) // BARN_GRID * BARN_GRID, int(cy) // BARN_GRID * BARN_GRID
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                soft.add((bx + dx * BARN_GRID, by + dy * BARN_GRID))
    soft -= {source, target}
    best = {source: 0}
    came: dict[Pixel, Pixel] = {}
    queue: list[Tuple[int, Pixel]] = [(0, source)]
    while queue:
        cost, cell = heapq.heappop(queue)
        if cell == target:
            break
        if cost > best[cell]:
            continue
        for dx, dy in _NEIGHBOURS:
            step = (cell[0] + dx, cell[1] + dy)
            if step not in BARN_CELLS or (clear and step in soft):
                continue
            total = cost + 1 + _WALL_COST[step] + (COW_SOFT_COST if step in soft else 0)
            if total < best.get(step, 1 << 30):
                best[step] = total
                came[step] = cell
                heapq.heappush(queue, (total, step))
    if target not in best:
        return []
    cells = [target]
    while cells[-1] != source:
        cells.append(came[cells[-1]])
    cells.reverse()
    corners = [
        cells[index]
        for index in range(1, len(cells) - 1)
        if (cells[index][0] - cells[index - 1][0], cells[index][1] - cells[index - 1][1])
        != (cells[index + 1][0] - cells[index][0], cells[index + 1][1] - cells[index][1])
    ]
    route = [*corners, cells[-1]] if len(cells) > 1 else [cells[-1]]
    return [(x + BARN_GRID // 2, y + BARN_GRID // 2) for x, y in route]


# ── walker ────────────────────────────────────────────────────────────

WALK_STILL_FRAMES = 8
COW_PUSH_FRAMES = 24  # retreat before the ~40-frame run hold starts shoving her
COW_AHEAD_PX = 17
MAX_WALK_NUDGES = 16
GOAL_DRIFT_PX = 12
_NUDGE_PATTERN = (("back", 5), ("side", 7), ("back", 9), ("other", 7), ("back", 14), ("side", 14), ("other", 14))
OPPOSITE_FACE = {"left": "right", "right": "left", "up": "down", "down": "up"}
_SIDES = {"left": ("up", "down"), "right": ("up", "down"), "up": ("left", "right"), "down": ("left", "right")}


def axis_step(pos: Pixel, target: Pixel) -> Tuple[Optional[str], int]:
    """Direction and remaining pixels along the longer axis from ``pos`` to ``target``."""
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if abs(dx) >= abs(dy):
        return ("right" if dx > 0 else "left" if dx < 0 else None), abs(dx)
    return ("down" if dy > 0 else "up"), abs(dy)


@dataclass
class BarnWalker:
    """Walk the map to a pixel. A held-still farmer steps back and tries again, which also clears a cow."""

    _path: list[Pixel] = field(default_factory=list)
    _goal: Optional[Pixel] = None
    _last: Pixel = (-1, -1)
    _still: int = 0
    _heading: str = "down"
    _queue: Deque[np.ndarray] = field(default_factory=deque)
    nudges: int = 0

    def clear(self) -> None:
        self._path, self._goal, self._last, self._still = [], None, (-1, -1), 0
        self._queue.clear()
        self.nudges = 0

    @property
    def blocked(self) -> bool:
        return self.nudges >= MAX_WALK_NUDGES

    def _nudge(self, pos: Pixel) -> None:
        kind, frames = _NUDGE_PATTERN[self.nudges % len(_NUDGE_PATTERN)]
        self.nudges += 1
        first, second = _SIDES[self._heading]
        direction = {"back": OPPOSITE_FACE[self._heading], "side": first, "other": second}[kind]
        if direction == "down" and pos[1] >= BARN_DOOR_PX[1] - 8 and abs(pos[0] - BARN_DOOR_PX[0]) <= 24:
            direction = "up"  # a step down here is the door, and the farm
        self._queue.extend(make_action(**{direction: True}) for _ in range(frames))
        self._path, self._goal, self._still = [], None, 0

    def _cow_ahead(self, pos: Pixel, cows: Sequence[Pixel]) -> bool:
        ux, uy = FACE_UNITS[self._heading]
        for cx, cy in cows:
            along = (cx - pos[0]) * ux + (cy - pos[1]) * uy
            across = abs((cx - pos[0]) * uy) + abs((cy - pos[1]) * ux)
            if 0 <= along <= COW_AHEAD_PX and across <= COW_AHEAD_PX:
                return True
        return False

    def step(self, pos: Pixel, goal: Pixel, *, tolerance: int = 2, cows: Sequence[Pixel] = ()) -> Optional[np.ndarray]:
        """The next pad frame towards ``goal``, or None once within ``tolerance`` on both axes."""
        if abs(goal[0] - pos[0]) <= tolerance and abs(goal[1] - pos[1]) <= tolerance:
            self.clear()
            return None
        if self._queue:
            self._last = pos
            return self._queue.popleft()
        self._still = self._still + 1 if pos == self._last else 0
        self._last = pos
        limit = COW_PUSH_FRAMES if self._cow_ahead(pos, cows) else WALK_STILL_FRAMES
        if self._still >= limit:
            self._nudge(pos)
            return self._queue.popleft()
        if self._goal is None or abs(goal[0] - self._goal[0]) + abs(goal[1] - self._goal[1]) > GOAL_DRIFT_PX:
            self._goal = goal
            self._path = plan_route(pos, goal, cows, clear=True) or plan_route(pos, goal, cows)
        elif cows and self._still >= 2 and self._cow_ahead(pos, cows):
            clear = plan_route(pos, goal, cows, clear=True)
            if clear:
                self._path, self._still = clear, 0
        while self._path and abs(self._path[0][0] - pos[0]) <= 2 and abs(self._path[0][1] - pos[1]) <= 2:
            self._path.pop(0)
        direction, remaining = axis_step(pos, self._path[0] if self._path else goal)
        if direction is None:
            return make_action()
        self._heading = direction
        return make_action(**{direction: True, "b": remaining > 2 and not self._cow_ahead(pos, cows)})


def cow_stands(cow: Pixel) -> list[Tuple[Pixel, str]]:
    """The four stands in line with a cow, each with the facing that reaches her. Walkable ones only."""
    cx, cy = cow
    rows = (
        ((cx + COW_STAND_PX, cy), "left"),
        ((cx - COW_STAND_PX, cy), "right"),
        ((cx, cy + COW_STAND_PX), "up"),
        ((cx, cy - COW_STAND_PX), "down"),
    )
    return [row for row in rows if (row[0][0] // BARN_GRID * BARN_GRID, row[0][1] // BARN_GRID * BARN_GRID) in BARN_CELLS]


def press_reaches(stand: Pixel, face: str, cow: Pixel) -> bool:
    """A press from ``stand`` facing ``face`` lands on a cow at ``cow`` (the ROM's probe box)."""
    ux, uy = FACE_UNITS[face]
    return (
        abs(stand[0] + ux * PRESS_AHEAD_PX - cow[0]) <= COW_REACH_PX
        and abs(stand[1] + uy * PRESS_AHEAD_PX - cow[1]) <= COW_REACH_PX
    )


def barn_cow_pixels(ram: np.ndarray) -> list[Pixel]:
    """Where every barn cow stands right now (the slot mirrors her sprite each frame)."""
    rows = []
    for slot in existing_cow_slots(ram):
        if read_animal_slot_field(ram, "cow", slot, "home_map_raw") in (0, BARN_TILEMAP):
            rows.append((
                int(read_animal_slot_field(ram, "cow", slot, "position_x")),
                int(read_animal_slot_field(ram, "cow", slot, "position_y")),
            ))
    return rows


def next_unfed_spot(flags: int, spots: Sequence[CowFeedSpot], *, near_y: int = FODDER_STAND_PX[1]) -> Optional[CowFeedSpot]:
    """The empty trough closest to the crossing row, or None when every required trough is filled."""
    empty = [spot for spot in spots if not flags & spot.flag]
    return min(empty, key=lambda spot: abs(spot.interact_px[1] - near_y)) if empty else None
