"""Light-world overworld screen graph, tile BFS, and castle route helpers."""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from alttp.dungeon.nav import WalkCommit, room_step
from alttp.ram import (
    HYRULE_CASTLE_SCREEN,
    LINKS_HOUSE_ROOM,
    LINKS_HOUSE_SCREEN,
    SANCTUARY_SCREEN,
    SECRET_HOLE_WORLD_X,
    SECRET_HOLE_WORLD_Y,
    WRAM_IDX,
    AlttpSnapshot,
    wram_index,
)
from alttp.walk_physics import LINK_BODY

SCREEN_LABELS = {
    0x13: "sanctuary",
    0x1B: "hyrule_castle",
    0x1C: "castle_approach",
    0x24: "north_field",
    0x2C: "links_house",
}

# Escape Link's House porch before heading north.
LINKS_HOUSE_ESCAPE_CLEAR_Y = 2936
LINKS_HOUSE_ESCAPE_X_MIN = 2386
LINKS_HOUSE_ESCAPE_X_MAX = 2402


def screen_to_grid(screen_id: int) -> tuple[int, int]:
    return divmod(int(screen_id) & 0x3F, 8)


def _neighbor_screens(screen_id: int) -> dict[str, int | None]:
    local = int(screen_id) & 0x3F
    base = int(screen_id) & ~0x3F
    row, col = divmod(local, 8)
    return {
        "north": (base + local - 8) if row > 0 else None,
        "south": (base + local + 8) if row < 7 else None,
        "west": (base + local - 1) if col > 0 else None,
        "east": (base + local + 1) if col < 7 else None,
    }


def shortest_screen_path(start_screen: int, target_screen: int) -> list[int]:
    """BFS on the 8×8 light/dark overworld grid (same world only)."""
    start = int(start_screen)
    target = int(target_screen)
    if (start & ~0x3F) != (target & ~0x3F):
        raise ValueError("Cross-world paths are not supported")
    if start == target:
        return [start]

    queue: deque[int] = deque([start])
    parents: dict[int, int | None] = {start: None}
    while queue:
        current = queue.popleft()
        for neighbor in _neighbor_screens(current).values():
            if neighbor is None or neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == target:
                path = [target]
                node = target
                while parents[node] is not None:
                    node = int(parents[node])
                    path.append(node)
                path.reverse()
                return path
            queue.append(neighbor)
    raise ValueError(f"No path from {start:#04x} to {target:#04x}")


def next_screen_in_path(current_screen: int, target_screen: int) -> int:
    path = shortest_screen_path(current_screen, target_screen)
    return path[0] if len(path) < 2 else path[1]


def direction_to_screen(current_screen: int, target_screen: int) -> str | None:
    current_row, current_col = screen_to_grid(current_screen)
    target_row, target_col = screen_to_grid(target_screen)
    if current_row > target_row:
        return "UP"
    if current_row < target_row:
        return "DOWN"
    if current_col > target_col:
        return "LEFT"
    if current_col < target_col:
        return "RIGHT"
    return None


def is_links_house_room(snapshot: AlttpSnapshot) -> bool:
    return snapshot.room_base_id == LINKS_HOUSE_ROOM


def next_direction_to_hyrule_castle(snapshot: AlttpSnapshot) -> str | None:
    """Return one cardinal direction toward castle grounds, or None."""
    if snapshot.indoors and is_links_house_room(snapshot):
        return "DOWN"
    if snapshot.indoors or snapshot.dark_world:
        return None
    if snapshot.screen_id == LINKS_HOUSE_SCREEN:
        if snapshot.link_x < LINKS_HOUSE_ESCAPE_X_MIN:
            if snapshot.link_y < LINKS_HOUSE_ESCAPE_CLEAR_Y:
                return "DOWN"
            return "RIGHT"
        if snapshot.link_x > LINKS_HOUSE_ESCAPE_X_MAX:
            return "LEFT"
        return "UP"
    next_screen = next_screen_in_path(snapshot.screen_id, HYRULE_CASTLE_SCREEN)
    return direction_to_screen(snapshot.screen_id, next_screen)


def on_hyrule_castle_screen(snapshot: AlttpSnapshot) -> bool:
    return (
        (not snapshot.indoors)
        and (not snapshot.dark_world)
        and snapshot.screen_id == HYRULE_CASTLE_SCREEN
    )


# snesrev Overworld_GetTileAttributeAtLocation. The live map16 words are
# $7E2000. $70C/$70E are already in 8px units (Overworld_LoadGFXAndScreenSize
# stores kOverworld_OffsetBaseX >> 3). Map8 words are ROM $8F8000 (3752×4);
# the attribute byte is ROM $8E9459. Both checked on HyruleCastleGrounds:
# base ($0600, $00C0), masks ($03F0, $007E), bush hole map16 $36 attr $50.
_MAP16_ROM = 0x8F8000
_MAP16_COUNT = 3752
_ATTR_ROM = 0x8E9459
_OW_MAP16 = 0x2000
_OW_BASE_Y = 0x708
_OW_MASK_Y = 0x70A
_OW_BASE_X = 0x70C
_OW_MASK_X = 0x70E
_OW_BIG = 0x712
_TILE = 8

# TileDetect_ExecuteInner outdoors, no R14. Slopes, ledges, pits, deep
# water, spikes and doors stay walls. Outdoor 0x22 stairs are walked;
# they do not load an indoor room. 0x48/0x4A are diggable ground.
_WALKABLE = frozenset(
    {
        0x00, 0x04, 0x05, 0x06, 0x07, 0x09, 0x0A, 0x14, 0x15, 0x16, 0x17,
        0x21, 0x22, 0x23, 0x24, 0x25, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x40, 0x41,
        0x45, 0x47, 0x48, 0x49, 0x4A, 0x5E, 0x5F, 0x60, 0x61, 0x62, 0x64,
        0x65, 0x66, 0x6C, 0x6D, 0x6E, 0x6F, 0xA6, 0xA7, 0xBE, 0xBF,
    }
    | set(range(0xD0, 0xF0))
)
# TileBehavior_Liftable, plus map16 ids from Overworld_HandleLiftableTiles.
_LIFTABLE_ATTR = frozenset(range(0x50, 0x57))
_LIFTABLE_MAP16 = frozenset(
    {
        0x36, 0x72A, 0x20F, 0x239, 0x101,
        0x36D, 0x36E, 0x374, 0x375, 0x23B, 0x23C, 0x23D, 0x23E,
    }
)
_DELTAS = {"RIGHT": (8, 0), "LEFT": (-8, 0), "DOWN": (0, 8), "UP": (0, -8)}
_HEADINGS = ("RIGHT", "LEFT", "DOWN", "UP")
# screen_1b_grounds secret_hole_approach. The standable 8px pose is one
# pixel east of this; the search snaps to it.
_BUSH_APPROACH = (2430, 1704)


def _snes_file_offset(ea: int) -> int:
    if (ea & 0x8000) == 0:
        raise ValueError(f"snes address {ea:#x} is not in a ROM window")
    return ((ea >> 16) & 0x7F) * 0x8000 + (ea & 0x7FFF)


def _u16(ram: np.ndarray, offset: int) -> int:
    index = wram_index(offset)
    return int(ram[index]) | (int(ram[index + 1]) << 8)


@lru_cache(maxsize=1)
def _ow_tables() -> tuple[np.ndarray, np.ndarray]:
    """Map16→four map8 words, and the 512-byte map8 attribute table."""
    from alttp.paths import INTEGRATION_DIR, REPO_ROOT

    path = REPO_ROOT / "roms" / "zelda3.sfc"
    if not path.is_file():
        path = INTEGRATION_DIR / "rom.sfc"
    data = path.read_bytes()
    if (len(data) & 0xFFFFF) == 0x200:
        data = data[0x200:]
    map16_at = _snes_file_offset(_MAP16_ROM)
    attr_at = _snes_file_offset(_ATTR_ROM)
    map16 = np.frombuffer(data[map16_at : map16_at + _MAP16_COUNT * 8], dtype="<u2").copy()
    attrs = np.frombuffer(data[attr_at : attr_at + 512], dtype=np.uint8).copy()
    return map16, attrs


@dataclass(frozen=True)
class ScreenEdge:
    """A standable pose whose next step leaves the loaded overworld area."""

    xy: tuple[int, int]
    direction: str
    to_screen: int | None


@dataclass(frozen=True)
class OwTiles:
    """One loaded overworld area at 8px resolution. No ROM writes."""

    origin_x: int
    origin_y: int
    cols: int
    rows: int
    screen: int
    big: bool
    attr: np.ndarray
    map16: np.ndarray
    liftable: np.ndarray

    def pixel_size(self) -> tuple[int, int]:
        return self.cols * _TILE, self.rows * _TILE

    def _cell(self, px: int, py: int) -> tuple[int, int] | None:
        col = (int(px) - self.origin_x) // _TILE
        row = (int(py) - self.origin_y) // _TILE
        if col < 0 or row < 0 or col >= self.cols or row >= self.rows:
            return None
        return col, row

    def sample(self, px: int, py: int) -> tuple[int, int, bool]:
        """Attribute, map16 and liftable flag under one pixel. OOB is a wall."""
        cell = self._cell(px, py)
        if cell is None:
            return 1, 0, False
        col, row = cell
        return int(self.attr[row, col]), int(self.map16[row, col]), bool(self.liftable[row, col])


def read_overworld_tiles(ram: object) -> OwTiles:
    """Decode the loaded map16 buffer into 8px attribute cells.

    Indoor ``$7F2000`` is not this grid. Raises if the mask pair is not a
    small (64) or big (128) area.
    """
    buf = np.asarray(ram, dtype=np.uint8)
    base_y = _u16(buf, _OW_BASE_Y)
    mask_y = _u16(buf, _OW_MASK_Y)
    base_x = _u16(buf, _OW_BASE_X)
    mask_x = _u16(buf, _OW_MASK_X)
    cols = (mask_x | 1) + 1
    rows = ((mask_y | 0xF) + 1) // _TILE
    if (cols, rows) not in ((64, 64), (128, 128)):
        raise ValueError(f"overworld mask {mask_x:#06x}/{mask_y:#06x} is not 64 or 128")
    words = np.frombuffer(
        buf[WRAM_IDX + _OW_MAP16 : WRAM_IDX + _OW_MAP16 + 0x2000].tobytes(),
        dtype="<u2",
    ).copy()
    map16_tab, attr_tab = _ow_tables()
    attr = np.ones((rows, cols), dtype=np.uint8)
    map16 = np.zeros((rows, cols), dtype=np.uint16)
    liftable = np.zeros((rows, cols), dtype=np.bool_)
    origin_x = base_x << 3
    for row in range(rows):
        y = base_y + row * _TILE
        y_part = ((y - base_y) & mask_y) * _TILE
        y_bit = (y & 8) >> 2
        for col in range(cols):
            x_unit = base_x + col
            index = (y_part | ((x_unit - base_x) & mask_x)) >> 1
            tile = int(words[index])
            map16[row, col] = tile
            if tile >= _MAP16_COUNT:
                continue
            word = int(map16_tab[tile * 4 + (y_bit | (x_unit & 1))])
            value = int(attr_tab[word & 0x1FF])
            if 0x10 <= value < 0x1C:
                value |= (word >> 14) & 1
            attr[row, col] = value
            liftable[row, col] = value in _LIFTABLE_ATTR or tile in _LIFTABLE_MAP16
    return OwTiles(
        origin_x=origin_x,
        origin_y=base_y,
        cols=cols,
        rows=rows,
        screen=int(buf[0x8A]),
        big=_u16(buf, _OW_BIG) != 0,
        attr=attr,
        map16=map16,
        liftable=liftable,
    )


def _body_inside(tiles: OwTiles, x: int, y: int, body: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = body
    width, height = tiles.pixel_size()
    left, top = x + x0, y + y0
    right, bottom = x + x1, y + y1
    return (
        left >= tiles.origin_x
        and top >= tiles.origin_y
        and right < tiles.origin_x + width
        and bottom < tiles.origin_y + height
    )


def pose_walkable(
    tiles: OwTiles,
    x: int,
    y: int,
    body: tuple[int, int, int, int] = LINK_BODY,
    *,
    allow_lift: bool = False,
) -> bool:
    """True when every tile under ``body`` is walkable and the body is inside."""
    if not _body_inside(tiles, x, y, body):
        return False
    x0, y0, x1, y1 = body
    left, top, right, bottom = x + x0, y + y0, x + x1, y + y1
    c0 = (left - tiles.origin_x) // _TILE
    r0 = (top - tiles.origin_y) // _TILE
    c1 = (right - tiles.origin_x) // _TILE
    r1 = (bottom - tiles.origin_y) // _TILE
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            if bool(tiles.liftable[row, col]) and not allow_lift:
                return False
            if int(tiles.attr[row, col]) not in _WALKABLE:
                return False
    return True


def _exit_screen(tiles: OwTiles, x: int, y: int, direction: str) -> int | None:
    """Screen reached by leaving the loaded area. None when that step stays in it
    or the neighbor is off the 8×8 world.
    """
    local = tiles.screen & 0x3F
    base = tiles.screen & ~0x3F
    row, col = divmod(local, 8)
    qcol = 1 if tiles.big and (x - tiles.origin_x) >= 512 else 0
    qrow = 1 if tiles.big and (y - tiles.origin_y) >= 512 else 0
    span = 2 if tiles.big else 1
    if direction == "UP":
        if tiles.big and qrow != 0:
            return None
        row, col = row - 1, col + qcol
    elif direction == "DOWN":
        if tiles.big and qrow == 0:
            return None
        row, col = row + span, col + qcol
    elif direction == "LEFT":
        if tiles.big and qcol != 0:
            return None
        row, col = row + qrow, col - 1
    elif direction == "RIGHT":
        if tiles.big and qcol == 0:
            return None
        row, col = row + qrow, col + span
    else:
        return None
    if not (0 <= row <= 7 and 0 <= col <= 7):
        return None
    return base + row * 8 + col


def leaves_area(
    tiles: OwTiles,
    x: int,
    y: int,
    direction: str,
    body: tuple[int, int, int, int] = LINK_BODY,
) -> bool:
    """The pose is inside and one 8px step that way puts the body outside."""
    if direction not in _DELTAS or not _body_inside(tiles, x, y, body):
        return False
    dx, dy = _DELTAS[direction]
    return not _body_inside(tiles, x + dx, y + dy, body)


def screen_edge_exits(
    tiles: OwTiles,
    body: tuple[int, int, int, int] = LINK_BODY,
) -> tuple[ScreenEdge, ...]:
    """Walkable poses that step off the loaded area.

    Only the border band can leave, so the interior is not scanned.
    """
    width, height = tiles.pixel_size()
    found: list[ScreenEdge] = []
    x0 = tiles.origin_x - _TILE
    y0 = tiles.origin_y - _TILE
    x1 = tiles.origin_x + width + _TILE
    y1 = tiles.origin_y + height + _TILE
    band = 32
    y = y0
    while y < y1:
        near_y = y < tiles.origin_y + band or y > tiles.origin_y + height - band
        x = x0
        while x < x1:
            near_x = x < tiles.origin_x + band or x > tiles.origin_x + width - band
            if (near_x or near_y) and pose_walkable(tiles, x, y, body):
                for direction in _HEADINGS:
                    if leaves_area(tiles, x, y, direction, body):
                        found.append(
                            ScreenEdge((x, y), direction, _exit_screen(tiles, x, y, direction))
                        )
            x += _TILE
        y += _TILE
    return tuple(found)


def _nearest_pose(
    tiles: OwTiles,
    x: int,
    y: int,
    body: tuple[int, int, int, int],
    *,
    radius: int,
    allow_lift: bool,
    blocked: Callable[[int, int], bool] | None,
) -> tuple[int, int] | None:
    snapped_x, snapped_y = int(x) & ~7, int(y) & ~7
    best: tuple[int, int] | None = None
    best_dist = 10**9
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = snapped_x + dx * _TILE, snapped_y + dy * _TILE
            if blocked is not None and blocked(px, py):
                continue
            if not pose_walkable(tiles, px, py, body, allow_lift=allow_lift):
                continue
            dist = abs(px - x) + abs(py - y)
            if dist < best_dist:
                best_dist = dist
                best = (px, py)
    return best


def find_overworld_path(
    tiles: OwTiles,
    start: tuple[int, int],
    goal: tuple[int, int] | None = None,
    body: tuple[int, int, int, int] = LINK_BODY,
    *,
    allow_lift: bool = False,
    edge: str | None = None,
    screen: int | None = None,
    blocked: Callable[[int, int], bool] | None = None,
) -> list[tuple[int, int]] | None:
    """Fewest-turn 8px path on real overworld attributes.

    ``edge`` names a loaded-area exit (``UP``/``DOWN``/``LEFT``/``RIGHT``).
    The goal is then the nearest such pose, not ``goal``. ``screen`` finds
    a standing quadrant inside the loaded area, without crossing its border.
    Liftable bushes and rocks are walls unless ``allow_lift``.
    ``None`` when no pose exists.
    """
    if edge is not None and edge not in _DELTAS:
        raise ValueError(f"edge must be a cardinal, got {edge!r}")
    if sum(value is not None for value in (goal, edge, screen)) != 1:
        raise ValueError("exactly one of goal, edge or screen is required")
    origin = _nearest_pose(
        tiles, start[0], start[1], body, radius=4, allow_lift=allow_lift, blocked=blocked
    )
    if origin is None:
        return None
    target: tuple[int, int] | None = None
    if goal is not None:
        assert goal is not None
        target = _nearest_pose(
            tiles, goal[0], goal[1], body, radius=6, allow_lift=allow_lift, blocked=blocked
        )
        if target is None:
            return None
        if origin == target:
            return [origin]
    elif edge is not None and leaves_area(tiles, origin[0], origin[1], edge, body):
        return [origin]

    def legal(px: int, py: int) -> bool:
        if blocked is not None and blocked(px, py):
            return False
        return pose_walkable(tiles, px, py, body, allow_lift=allow_lift)

    best: dict[tuple[int, int, int], tuple[int, int]] = {(origin[0], origin[1], -1): (0, 0)}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[int, int, int, int, int, int]] = []
    seq = 0
    heapq.heappush(heap, (0, 0, seq, origin[0], origin[1], -1))
    end: tuple[int, int, int] | None = None
    while heap:
        turns, steps, _, x, y, heading = heapq.heappop(heap)
        if best.get((x, y, heading)) != (turns, steps):
            continue
        if (x, y) == target or (
            screen is not None and standing_screen(tiles, x, y) == screen
        ):
            end = (x, y, heading)
            break
        if edge is not None and leaves_area(tiles, x, y, edge, body):
            end = (x, y, heading)
            break
        for index, direction in enumerate(_HEADINGS):
            dx, dy = _DELTAS[direction]
            nx, ny = x + dx, y + dy
            if not legal(nx, ny):
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
    path = [(end[0], end[1])]
    cursor = end
    while cursor[2] != -1:
        cursor = parent[cursor]
        path.append((cursor[0], cursor[1]))
    path.reverse()
    return path


def _segment_button(ax: int, ay: int, bx: int, by: int) -> str | None:
    dx, dy = bx - ax, by - ay
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    if dy != 0:
        return "DOWN" if dy > 0 else "UP"
    return None


def _short_of(x: int, y: int, corner: tuple[int, int], button: str) -> bool:
    if button == "LEFT":
        return x > corner[0]
    if button == "RIGHT":
        return x < corner[0]
    if button == "UP":
        return y > corner[1]
    if button == "DOWN":
        return y < corner[1]
    return False


def _path_segments(points: list[tuple[int, int]]) -> list[tuple[str, tuple[int, int]]]:
    segments: list[tuple[str, tuple[int, int]]] = []
    for left, right in zip(points, points[1:]):
        button = _segment_button(left[0], left[1], right[0], right[1])
        if button is None:
            continue
        if segments and segments[-1][0] == button:
            segments[-1] = (button, right)
        else:
            segments.append((button, right))
    return segments


def standing_screen(tiles: OwTiles, x: int, y: int) -> int:
    """Screen under ``(x, y)``. A big area stores only its northwest id."""
    local = int(tiles.screen) & 0x3F
    base = int(tiles.screen) & ~0x3F
    row, col = divmod(local, 8)
    if tiles.big:
        if int(x) - tiles.origin_x >= 512:
            col += 1
        if int(y) - tiles.origin_y >= 512:
            row += 1
    if not (0 <= row <= 7 and 0 <= col <= 7):
        return int(tiles.screen) & 0xFF
    return base + row * 8 + col


def _screen_manhattan(left: int, right: int) -> int:
    if (int(left) & ~0x3F) != (int(right) & ~0x3F):
        return 64
    left_row, left_col = divmod(int(left) & 0x3F, 8)
    right_row, right_col = divmod(int(right) & 0x3F, 8)
    return abs(left_row - right_row) + abs(left_col - right_col)


def _exit_probes(
    tiles: OwTiles,
    xy: tuple[int, int],
    target: int,
    skip: set[tuple[str, int]],
    avoid: set[int],
) -> list[ScreenEdge]:
    """Border exits, closer screens first. A few poses per direction.

    ``avoid`` is the screen we just left, plus screens already backed out
    of. Those exits sort last so a west-edge arrival does not step back.
    """
    here = _screen_manhattan(standing_screen(tiles, xy[0], xy[1]), target)
    groups: dict[tuple[str, int], list[ScreenEdge]] = {}
    for edge in screen_edge_exits(tiles):
        if edge.to_screen is None:
            continue
        key = (edge.direction, int(edge.to_screen))
        if key in skip:
            continue
        groups.setdefault(key, []).append(edge)
    ranked = sorted(
        groups,
        key=lambda key: (
            key[1] in avoid,
            _screen_manhattan(key[1], target) > here,
            _screen_manhattan(key[1], target),
            key[0],
        ),
    )
    probes: list[ScreenEdge] = []
    for key in ranked:
        poses = sorted(
            groups[key],
            key=lambda edge: abs(edge.xy[0] - xy[0]) + abs(edge.xy[1] - xy[1]),
        )
        if len(poses) <= 5:
            probes.extend(poses)
            continue
        step = max(1, len(poses) // 5)
        probes.extend(poses[index] for index in range(0, len(poses), step)[:5])
    return probes


def _path_to_exit(
    tiles: OwTiles,
    xy: tuple[int, int],
    edge: ScreenEdge,
) -> list[tuple[int, int]] | None:
    path = find_overworld_path(tiles, xy, edge.xy)
    if not path:
        return None
    end = path[-1]
    if abs(end[0] - edge.xy[0]) > 8 or abs(end[1] - edge.xy[1]) > 8:
        return None
    return path


def _dismiss_text(env: object, budget: int) -> tuple[bool, int]:
    """Advance a walked-into message. A closes one settled page. No RAM writes."""
    from alttp.startup import action_for, no_action, snapshot_env, step_frames

    used = 0
    for _ in range(6):
        waited = 0
        ready = False
        while used < budget and waited < 400:
            pose = snapshot_env(env)
            if pose.game_mode == 0x12:
                return False, used
            if not pose.is_text_mode:
                return True, used
            ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
            state = int(ram[wram_index(0x1CD4)])
            count = int(ram[wram_index(0x1CE9)])
            if state == 0x03 and count == 0:
                ready = True
                break
            step_frames(env, no_action(), 1)
            used += 1
            waited += 1
        if not ready:
            return False, used
        step_frames(env, action_for("A"), 1)
        used += 1
        step_frames(env, no_action(), 6)
        used += 6
        if not snapshot_env(env).is_text_mode:
            return True, used
    return (not snapshot_env(env).is_text_mode), used


def walk_to_screen(
    env: object,
    target_screen: int,
    *,
    max_frames: int = 20000,
    guard: bool = False,
) -> dict[str, object]:
    """Walk tile paths and edge exits until Link stands on ``target_screen``.

    Buttons only. Does not write RAM and does not load a state. A caller
    that starts from a pin has not made a continuous run. Internal quadrants
    of a big area use a tile path, not a screen-border exit. A guard message
    is closed with A; it is not a fight. ``guard`` dodges through
    ``PolicyGuard`` and does not swing.
    """
    from alttp.startup import action_for, no_action, snapshot_env, step_frames

    target = int(target_screen) & 0xFF
    holder = None
    policy = None
    if guard:
        from alttp.rollout import guard_press

        holder, policy = guard_press(env, max_frames)
    snap = snapshot_env(env)
    start_hp = snap.hp
    frames = 0
    trace: list[int] = []
    plan: tuple[ScreenEdge, list[tuple[str, tuple[int, int]]], int] | None = None
    skip: set[tuple[str, int]] = set()
    dead: set[int] = set()
    tiles: OwTiles | None = None
    loaded: int | None = None
    unchanged = 0
    replans = 0
    texts = 0
    hits = 0
    previous: tuple[int, int] | None = None

    def finish(ok: bool, reason: str, pose: AlttpSnapshot) -> dict[str, object]:
        standing = pose.screen_id
        if tiles is not None and not pose.indoors:
            standing = standing_screen(tiles, pose.link_x, pose.link_y)
        shown = ",".join(f"{screen:02X}" for screen in trace) or f"{pose.screen_id:02X}"
        guard_text = ""
        if policy is not None:
            guard_text = f" checks={policy.checks} detours={policy.detours_taken}"
        line = (
            f"AP-084 {'ok' if ok else 'fail'} {reason} frames={frames} "
            f"xy=({pose.link_x},{pose.link_y}) screen=0x{pose.screen_id:02X} "
            f"standing=0x{standing:02X} indoors={int(pose.indoors)} "
            f"hp={start_hp}->{pose.hp} hits=0->{hits} trace={shown} "
            f"continuous=0{guard_text}"
        )
        return {
            "ok": ok,
            "reason": reason,
            "frames": frames,
            "screen": pose.screen_id,
            "standing": standing,
            "hits": hits,
            "trace": list(trace),
            "policy_guard": policy.report()["policy_guard"] if policy else None,
            "line": line,
        }

    while frames < max_frames:
        snap = snapshot_env(env)
        if snap.game_mode == 0x12:
            return finish(False, "dead", snap)
        if snap.is_text_mode:
            texts += 1
            closed, used = _dismiss_text(env, max_frames - frames)
            frames += used
            if texts > 6 or not closed:
                return finish(False, "text", snapshot_env(env))
            previous = None
            unchanged = 0
            continue
        if not snap.has_control:
            step_frames(env, no_action(), 1)
            frames += 1
            previous = None
            unchanged = 0
            continue
        if snap.indoors:
            return finish(False, "indoors", snap)
        if tiles is None or snap.screen_id != loaded:
            ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
            tiles = read_overworld_tiles(ram)
            loaded = snap.screen_id
            plan = None
            skip = set()
            replans = 0
            texts = 0
            unchanged = 0
            previous = None
            here = standing_screen(tiles, snap.link_x, snap.link_y)
            if not trace or trace[-1] != here:
                trace.append(here)
        assert tiles is not None
        here = standing_screen(tiles, snap.link_x, snap.link_y)
        if not trace or trace[-1] != here:
            trace.append(here)
        if here == target:
            return finish(True, "screen", snap)
        if plan is None:
            avoid = set(dead)
            if len(trace) >= 2:
                avoid.add(trace[-2])
            chosen: tuple[ScreenEdge, list[tuple[str, tuple[int, int]]]] | None = None
            quadrants = {tiles.screen, tiles.screen + 1, tiles.screen + 8, tiles.screen + 9}
            if tiles.big and target in quadrants:
                path = find_overworld_path(tiles, (snap.link_x, snap.link_y), screen=target)
                if not path:
                    return finish(False, "no quadrant path", snap)
                chosen = (ScreenEdge(path[-1], "NONE", target), _path_segments(path))
            exits = () if chosen else _exit_probes(
                tiles, (snap.link_x, snap.link_y), target, skip, avoid
            )
            for edge in exits:
                path = _path_to_exit(tiles, (snap.link_x, snap.link_y), edge)
                if path is None:
                    continue
                chosen = (edge, _path_segments(path))
                break
            if chosen is None:
                return finish(False, "no exit", snap)
            if chosen[0].to_screen is not None and int(chosen[0].to_screen) in avoid:
                dead.add(here)
            plan = (chosen[0], chosen[1], 0)
        edge, segments, index = plan
        near_exit = (
            abs(snap.link_x - edge.xy[0]) <= 6 and abs(snap.link_y - edge.xy[1]) <= 6
        )
        if near_exit or index >= len(segments):
            button: str | None = edge.direction
        else:
            while index < len(segments) and not _short_of(
                snap.link_x, snap.link_y, segments[index][1], segments[index][0]
            ):
                index += 1
            plan = (edge, segments, index)
            button = segments[index][0] if index < len(segments) else edge.direction
        if policy is None or holder is None:
            step_frames(env, no_action() if button is None else action_for(button), 1)
        else:
            holder.button = button if button else "NONE"
            chosen = policy.step(snap)
            step_frames(env, np.asarray(chosen.action, dtype=np.int8), 1)
        frames += 1
        pose = snapshot_env(env)
        if pose.hp < snap.hp:
            hits += 1
        xy = (pose.link_x, pose.link_y)
        if (
            previous is not None
            and pose.has_control
            and pose.screen_id == loaded
            and (abs(xy[0] - previous[0]) > 48 or abs(xy[1] - previous[1]) > 48)
        ):
            return finish(False, "jumped", pose)
        if xy == previous and pose.screen_id == loaded and pose.has_control:
            unchanged += 1
        else:
            unchanged = 0
            previous = xy
        if unchanged < 45 or pose.screen_id != loaded:
            continue
        if edge.to_screen is None:
            return finish(False, "stuck", pose)
        skip.add((edge.direction, int(edge.to_screen)))
        plan = None
        replans += 1
        unchanged = 0
        if replans > 8:
            return finish(False, "stuck", pose)
    return finish(False, "timeout", snapshot_env(env))


def walk_sanctuary_exit_to_links_house(
    env: object,
    *,
    max_frames: int = 20000,
) -> dict[str, object]:
    """From an outdoor Sanctuary pin, walk to Link's house screen.

    Refuses to start unless the pin is already outdoors on screen ``0x13``.
    The walk is not continuous.
    """
    from alttp.startup import snapshot_env

    snap = snapshot_env(env)
    if snap.indoors or snap.dark_world or snap.screen_id != SANCTUARY_SCREEN:
        line = (
            f"AP-084 fail landing screen=0x{snap.screen_id:02X} "
            f"indoors={int(snap.indoors)} xy=({snap.link_x},{snap.link_y}) continuous=0"
        )
        return {
            "ok": False,
            "reason": "landing",
            "frames": 0,
            "screen": snap.screen_id,
            "standing": snap.screen_id,
            "line": line,
        }
    return walk_to_screen(env, LINKS_HOUSE_SCREEN, max_frames=max_frames)


def _at_bush_hole(snap: AlttpSnapshot, goal: tuple[int, int]) -> bool:
    """Map approach tolerance is 6. The planned pose is the standable cell."""
    if snap.indoors:
        return snap.room_base_id == 0x55
    ax, ay = _BUSH_APPROACH
    near_approach = abs(snap.link_x - ax) <= 6 and abs(snap.link_y - ay) <= 6
    near_goal = abs(snap.link_x - goal[0]) <= 4 and abs(snap.link_y - goal[1]) <= 4
    return near_approach or near_goal


def walk_castle_grounds_to_bush_hole(env: object, *, max_frames: int = 5000) -> dict[str, object]:
    """Follow ``find_overworld_path`` from the castle-grounds pin to the bush.

    Does not write RAM and does not load another state. The hole cell must
    be a liftable bush; screen-edge exits are counted on the same grid.
    """
    from alttp.startup import action_for, no_action, snapshot_env, step_frames

    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    tiles = read_overworld_tiles(ram)
    hole_attr, hole_map16, hole_lift = tiles.sample(SECRET_HOLE_WORLD_X, SECRET_HOLE_WORLD_Y)
    edges = screen_edge_exits(tiles)
    snap = snapshot_env(env)
    spawn = (snap.link_x, snap.link_y)
    up_path = find_overworld_path(tiles, spawn, edge="UP")
    path = find_overworld_path(tiles, spawn, _BUSH_APPROACH)
    samples = {}
    for direction in _HEADINGS:
        for edge in edges:
            if edge.direction == direction:
                samples[direction] = f"{edge.xy[0]},{edge.xy[1]}->{edge.to_screen}"
                break

    def finish(ok: bool, reason: str, frames: int, pose: AlttpSnapshot) -> dict[str, object]:
        xs = [p[0] for p in path] if path else []
        ys = [p[1] for p in path] if path else []
        line = (
            f"AP-053 {'ok' if ok else 'fail'} {reason} frames={frames} "
            f"xy=({pose.link_x},{pose.link_y}) room={pose.room_base_id:#x} "
            f"screen={pose.screen_id:#x} hole_attr={hole_attr:#04x} "
            f"hole_map16={hole_map16:#x} liftable={int(hole_lift)} "
            f"path={0 if path is None else len(path)} "
            f"bbox={min(xs) if xs else 0},{min(ys) if ys else 0},"
            f"{max(xs) if xs else 0},{max(ys) if ys else 0} "
            f"edges={len(edges)} up_exit={0 if up_path is None else len(up_path)} "
            f"samples={samples}"
        )
        return {"ok": ok, "reason": reason, "frames": frames, "line": line}

    if not hole_lift:
        return finish(False, "hole not liftable", 0, snap)
    if path is None:
        return finish(False, "no tile path", 0, snap)
    segments = _path_segments(path)
    commit = WalkCommit()
    index = 0
    frames = 0
    unchanged = 0
    replans = 0
    previous: tuple[int, int] | None = None
    while frames < max_frames:
        snap = snapshot_env(env)
        if _at_bush_hole(snap, path[-1]):
            return finish(True, "bush hole", frames, snap)
        if snap.is_text_mode:
            return finish(False, "text", frames, snap)
        if snap.game_mode == 0x12:
            return finish(False, "dead", frames, snap)
        if not snap.has_control:
            step_frames(env, no_action(), 1)
            frames += 1
            if frames > 180 and not snap.has_control:
                return finish(False, "no control", frames, snapshot_env(env))
            continue
        while index < len(segments) and not _short_of(
            snap.link_x, snap.link_y, segments[index][1], segments[index][0]
        ):
            index += 1
        desired = segments[index][0] if index < len(segments) else None
        button, commit = room_step(commit, desired)
        step_frames(env, no_action() if button is None else action_for(button), 1)
        frames += 1
        pose = snapshot_env(env)
        xy = (pose.link_x, pose.link_y)
        if previous is not None and (abs(xy[0] - previous[0]) > 24 or abs(xy[1] - previous[1]) > 24):
            return finish(False, "jumped", frames, pose)
        if xy == previous:
            unchanged += 1
        else:
            unchanged = 0
            previous = xy
        if unchanged == 40 and replans < 2:
            fresh = find_overworld_path(tiles, xy, _BUSH_APPROACH)
            replans += 1
            unchanged = 0
            commit = WalkCommit()
            if fresh:
                path = fresh
                segments = _path_segments(path)
                index = 0
        elif unchanged >= 80:
            return finish(False, "stuck", frames, pose)
    return finish(False, "timeout", frames, snapshot_env(env))
