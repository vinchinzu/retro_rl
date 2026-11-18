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
    "free_chest_stand",
    "heading_for",
    "open_free_chests",
    "reach_cannonball_hall_clear",
    "reach_cannonball_hall_exit",
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


# Room 0x71 chest 0 is a 16x16 graphic. Facing up from the left column,
# one body south of that graphic, is the press that takes the boomerang.
# Other faces were not the open. A chest with no path to this stand stays shut.
_CHEST_FACE = "UP"


def free_chest_stand(
    tiles: object,
    link: tuple[int, int],
    col: int,
    row: int,
) -> tuple[int, int] | None:
    """Room-local stand south of a 16x16 chest, or None when no path exists.

    ``link`` is Link's room-local pixel. The stand is the left column of
    the chest, with the body on the floor row under the graphic. The
    caller faces up. No path is a reason to leave the chest closed.
    """
    goal = (int(col) * _TILE, (int(row) + 1) * _TILE)
    start = (int(link[0]) & ~7, int(link[1]) & ~7)
    if find_path(tiles, start, goal, LINK_BODY) is None:
        return None
    return goal


def _aim(x: int, y: int, tx: int, ty: int) -> str | None:
    dx, dy = tx - x, ty - y
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _walk_stand(env: object, x: int, y: int, room: int, limit: int) -> tuple[bool, int]:
    """One-frame steps along a fresh path. Stops on the stand pixel."""
    from alttp.dungeon.tilemap import read_room_tiles
    from alttp.startup import action_for, no_action, snapshot_env

    frames = 0
    still = 0
    previous: tuple[int, int] | None = None
    while frames < limit:
        snap = snapshot_env(env)
        if snap.room_base_id != room or snap.game_mode == 0x12:
            return False, frames
        if abs(snap.link_x - x) <= 1 and abs(snap.link_y - y) <= 1:
            return True, frames
        if abs(snap.link_x - x) <= 8 and abs(snap.link_y - y) <= 8:
            button = _aim(snap.link_x, snap.link_y, x, y)
        else:
            origin = room_origin(snap.link_x, snap.link_y)
            local = (snap.link_x - origin[0], snap.link_y - origin[1])
            goal = ((x - origin[0]) & ~7, (y - origin[1]) & ~7)
            path = find_path(
                read_room_tiles(env.get_ram(), snap.layer),  # type: ignore[attr-defined]
                (local[0] & ~7, local[1] & ~7),
                goal,
                LINK_BODY,
            )
            if not path or len(path) < 2:
                return False, frames
            button = _aim(
                snap.link_x,
                snap.link_y,
                origin[0] + path[1][0],
                origin[1] + path[1][1],
            )
        action = no_action() if button is None else action_for(button)
        env.step(action)  # type: ignore[attr-defined]
        frames += 1
        xy = (snap.link_x, snap.link_y)
        if xy == previous:
            still += 1
        else:
            still = 0
            previous = xy
        if still >= 24:
            return False, frames
    return False, frames


def open_free_chests(env: object, *, max_frames: int = 900):
    """Open each closed chest whose south stand has a floor path.

    Skip a chest when it is already open, on the other layer, the stand
    has no path, the walk cannot reach the stand, or one A press does
    not set the opened bit. The last of those is a missing ability bit,
    a missing big key, or a face this stand does not cover. This does
    not write RAM and does not spend a key.
    """
    from alttp.dungeon.rom_scan import closed_chests
    from alttp.dungeon.tilemap import read_room_tiles
    from alttp.primitives import PrimitiveResult, settle_control
    from alttp.startup import action_for, no_action, snapshot_env, step_frames

    snap = snapshot_env(env)
    if not snap.indoors:
        return PrimitiveResult(True, "outdoors", 0, snap)
    room = snap.room_base_id
    frames = 0
    opened: list[int] = []
    skipped: list[str] = []
    for index, col, row, layer in closed_chests(env):
        snap = snapshot_env(env)
        if snap.room_base_id != room:
            return PrimitiveResult(False, "left the room", frames, snap)
        if snap.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snap)
        if snap.layer != layer:
            skipped.append(f"chest {index} other layer")
            continue
        if frames >= max_frames:
            skipped.append(f"chest {index} frame cap")
            break
        origin = room_origin(snap.link_x, snap.link_y)
        stand = free_chest_stand(
            read_room_tiles(env.get_ram(), snap.layer),  # type: ignore[attr-defined]
            (snap.link_x - origin[0], snap.link_y - origin[1]),
            col,
            row,
        )
        if stand is None:
            skipped.append(f"chest {index} no path")
            continue
        reached, used = _walk_stand(
            env,
            origin[0] + stand[0],
            origin[1] + stand[1],
            room,
            max_frames - frames,
        )
        frames += used
        if not reached:
            skipped.append(f"chest {index} stand not reached")
            continue
        step_frames(env, action_for(_CHEST_FACE), 4)
        step_frames(env, action_for("A"), 1)
        frames += 5
        # The opened bit and inventory can update before the item receipt
        # freezes Link. In room 0x71, ordinary module/action bytes persist
        # for roughly 60 frames before the boomerang message appears.
        # Await that handoff before settle_control can claim movement ready.
        for waited in range(480):
            snap = snapshot_env(env)
            if snap.is_hold_up_item or snap.is_text_mode or snap.game_mode == 0x12:
                break
            if waited >= 40 and not _opened_bit(env, index):
                break
            step_frames(env, no_action(), 1)
            frames += 1
        else:
            return PrimitiveResult(
                False, f"chest {index} receipt timeout", frames, snapshot_env(env),
            )
        settled = settle_control(env, max_frames=480)
        frames += settled.frames
        snap = settled.snapshot
        if snap.game_mode == 0x12:
            return PrimitiveResult(False, "Link died", frames, snap)
        if not settled.ok and (snap.is_hold_up_item or snap.is_text_mode or not snap.has_control):
            return PrimitiveResult(False, settled.reason, frames, snap)
        if _opened_bit(env, index):
            opened.append(index)
        else:
            skipped.append(f"chest {index} press did not open")
    snap = snapshot_env(env)
    parts: list[str] = []
    if opened:
        parts.append("opened " + ",".join(str(i) for i in opened))
    if skipped:
        parts.append("skip " + "; ".join(skipped))
    if not parts:
        parts.append("no closed chest")
    return PrimitiveResult(True, "; ".join(parts), frames, snap)


def _opened_bit(env: object, index: int) -> bool:
    from alttp.dungeon.rom_scan import closed_chests

    return all(slot[0] != index for slot in closed_chests(env))


def reach_cannonball_hall_clear(env: object):
    """Development B9 crossing from its south landing, with ROM lookahead.

    Ends on the lower floor's clear area, before the unproved north exit.
    Forecast restores must be disclosed. No item interaction or RAM writes.
    ``stage_replay Play_eog9_7 alttp.dungeon.nav:reach_cannonball_hall_clear``
    exercises this tactic; it is not a continuous-spine row yet.
    """
    from alttp.primitives import PrimitiveResult
    from alttp.room_map import load_room_map
    from alttp.startup import action_for, no_action, snapshot_env

    before = snapshot_env(env)
    frames = 0
    geometry = load_room_map("room_b9")

    def checked():
        snap = snapshot_env(env)
        if (not snap.indoors or snap.room_base_id != 0xB9 or snap.hp != before.hp
                or snap.hp <= 0 or snap.sword_level != before.sword_level
                or snap.lamp_level != before.lamp_level
                or snap.follower != before.follower or snap.num_keys != before.num_keys):
            raise ValueError("B9 entry invariant miss")
        return snap

    def step(action):
        nonlocal frames
        env.step(action)
        frames += 1

    def walk(label):
        point = geometry.point(label)
        if point is None:
            raise ValueError(f"missing B9 map point {label}")
        for _ in range(180):
            snap = checked()
            dx, dy = point.x - snap.link_x, point.y - snap.link_y
            if max(abs(dx), abs(dy)) <= 1:
                return
            direction = (("RIGHT" if dx > 0 else "LEFT") if abs(dx) > abs(dy)
                         else ("DOWN" if dy > 0 else "UP"))
            step(action_for(direction))
        raise ValueError(f"B9 entry waypoint timeout: {label}")

    try:
        checked()
        landing = geometry.point("south_landing")
        if (landing is None or before.layer != 0 or not before.has_control
                or max(abs(before.link_x - landing.x), abs(before.link_y - landing.y)) > 3):
            raise ValueError("B9 south landing required")
        for label in ("drop_lineup", "drop_cross", "west_drop_mouth"):
            walk(label)
        for _ in range(180):
            snap = checked()
            if snap.layer == 1 and snap.has_control:
                break
            step(action_for("UP") if snap.has_control else no_action())
        else:
            raise ValueError("B9 west drop timeout")
        for label in ("lower_cross", "channel_start"):
            walk(label)
        point = geometry.point("channel_clear")
        if point is None:
            raise ValueError("missing B9 channel_clear")
        result = _forecast_walk(env, (point.x, point.y), max_frames=800)
        frames += result["frames"]
        return PrimitiveResult(result["ok"], result.get("blocker", "B9 lower channel cleared"),
                               frames, snapshot_env(env))
    except ValueError as exc:
        return PrimitiveResult(False, str(exc), frames, snapshot_env(env))


def reach_cannonball_hall_exit(env: object):
    """B9 lower clear area → east stair → upper north door, development.

    Also accepts the real south landing and composes the earned crossing.
    Geometry lives in room_b9; stop at the first prediction miss.
    """
    from alttp.primitives import PrimitiveResult
    from alttp.room_map import load_room_map
    from alttp.startup import action_for, no_action, snapshot_env

    before = snapshot_env(env)
    geometry = load_room_map("room_b9")
    frames = 0

    def checked():
        snap = snapshot_env(env)
        if (not snap.indoors or snap.room_base_id not in (0xB9, 0xA9)
                or snap.hp != before.hp or snap.hp <= 0
                or snap.sword_level != before.sword_level
                or snap.lamp_level != before.lamp_level
                or snap.follower != before.follower or snap.num_keys != before.num_keys):
            raise ValueError("B9 exit invariant miss")
        return snap

    def step(direction):
        nonlocal frames
        env.step(no_action() if direction is None else action_for(direction))
        frames += 1

    def walk(label, layer):
        point = geometry.point(label)
        if point is None:
            raise ValueError(f"missing B9 map point {label}")
        still = 0
        previous = None
        for _ in range(180):
            snap = checked()
            if snap.room_base_id != 0xB9:
                raise ValueError(f"unexpected room before {label}")
            dx, dy = point.x - snap.link_x, point.y - snap.link_y
            if max(abs(dx), abs(dy)) <= 1 and snap.layer == layer and snap.has_control:
                return
            pose = (snap.link_x, snap.link_y, snap.layer, snap.submodule)
            still = still + 1 if pose == previous else 0
            previous = pose
            if still >= 24:
                raise ValueError(f"B9 first movement miss: {label} at {pose}")
            direction = (("RIGHT" if dx > 0 else "LEFT") if abs(dx) > abs(dy)
                         else ("DOWN" if dy > 0 else "UP"))
            step(direction if snap.has_control else None)
        raise ValueError(f"B9 waypoint timeout: {label}")

    try:
        checked()
        if before.room_base_id != 0xB9 or not before.has_control:
            raise ValueError("B9 control-ready predecessor required")
        if before.layer == 0:
            result = reach_cannonball_hall_clear(env)
            frames += result.frames
            if not result.ok:
                return result
        clear = geometry.point("channel_clear")
        snap = checked()
        if (clear is None or snap.layer != 1
                or max(abs(snap.link_x - clear.x), abs(snap.link_y - clear.y)) > 3):
            raise ValueError("B9 channel_clear required")
        walk("north_east_clear", 1)
        walk("north_east_bypass", 1)
        walk("north_east_stair_foot", 1)
        walk("north_east_stair_top", 0)
        walk("north_upper_lineup", 0)
        for _ in range(480):
            snap = checked()
            if snap.room_base_id == 0xA9 and snap.has_control:
                return PrimitiveResult(True, "B9 north exit to A9", frames, snap)
            step("UP" if snap.has_control else None)
        raise ValueError("B9 north transition timeout")
    except ValueError as exc:
        return PrimitiveResult(False, str(exc), frames, snapshot_env(env))


def _forecast_walk(env, goal, *, max_frames=1800, horizon=64, width=18):
    """Beam over eight-frame cardinal holds; commit one checked hold."""
    from alttp.startup import action_for, no_action, snapshot_env
    from alttp.rollout import _restore
    from alttp.spine.tape import ram_crc

    start = snapshot_env(env)
    room, layer, hp = start.room_base_id, start.layer, start.hp
    actions = [action_for(d) for d in ('UP', 'LEFT', 'RIGHT', 'DOWN')] + [no_action()]
    committed = 0
    decisions = []
    visits = {}
    while committed < max_frames:
        s = snapshot_env(env)
        if max(abs(s.link_x-goal[0]), abs(s.link_y-goal[1])) <= 3:
            return dict(ok=True, frames=committed, decisions=decisions)
        live = bytes(env.em.get_state())
        # (ranking, state, first input, first CRCs, endpoint)
        beam = [(0, live, None, None, (s.link_x, s.link_y))]
        try:
            for depth in range(horizon // 8):
                choices = {}
                for _, blob, first, crcs, _ in beam:
                    for ai, mask in enumerate(actions):
                        _restore(env.em, blob)
                        trace = []
                        valid = True
                        for _ in range(8):
                            env.em.set_button_mask(mask, 0)
                            env.em.step()
                            end = snapshot_env(env)
                            if (end.hp != hp or end.room_base_id != room or end.layer != layer
                                    or not end.has_control or end.follower != start.follower
                                    or end.sword_level != start.sword_level or end.lamp_level != start.lamp_level
                                    or end.num_keys != start.num_keys):
                                valid = False
                                break
                            if depth == 0:
                                trace.append(ram_crc(env.get_ram()))
                        if not valid:
                            continue
                        xy = (end.link_x, end.link_y)
                        distance = abs(xy[0]-goal[0]) + abs(xy[1]-goal[1])
                        score = distance + 0.15 * visits.get((xy[0]//4, xy[1]//4), 0)
                        key = (xy[0]//2, xy[1]//2)
                        node = (score, bytes(env.em.get_state()), ai if first is None else first,
                                trace if first is None else crcs, xy)
                        if key not in choices or score < choices[key][0]:
                            choices[key] = node
                if not choices:
                    break
                beam = sorted(choices.values(), key=lambda n:n[0])[:width]
        finally:
            _restore(env.em, live)
        viable = [n for n in beam if n[2] is not None]
        if not viable:
            return dict(ok=False, frames=committed, blocker='no safe forecast', decisions=decisions)
        best = min(viable, key=lambda n:n[0])
        ai = best[2]
        for crc in best[3]:
            env.step(actions[ai])
            committed += 1
            if ram_crc(env.get_ram()) != crc:
                return dict(ok=False, frames=committed, blocker='prediction CRC mismatch', decisions=decisions)
        s = snapshot_env(env)
        cell = (s.link_x//4, s.link_y//4)
        visits[cell] = visits.get(cell,0)+1
        decisions.append(dict(frame=committed, x=s.link_x, y=s.link_y, hp=s.hp, action=ai))
    return dict(ok=False, frames=committed, blocker='frame cap', decisions=decisions)
