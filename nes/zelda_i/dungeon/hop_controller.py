"""Shared hop lifecycle: timeout, death, scroll wait, then policy.

Dungeon dest hops subclass ``HopController`` and implement ``policy``.
Do not copy the frames/success/death preamble into each room file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.walk import live_env
from zelda_i.dungeon.postmortem import DamageLog
from zelda_i.dungeon.tracking import ObjectTracker, TrackedObject
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

WAIT_SCROLL = (2, 3, 4, 6, 7)
# Boxed: Link within BOXED_PX (Manhattan) of one spot for BOXED_FRAMES play
# frames. A walk covers ~36 px in 24 frames; a wall bounce stays inside 8.
BOXED_PX = 8
BOXED_FRAMES = 24
WAIT_SCROLL_B = (2, 3, 4, 6, 7, 10, 16)
DEATH_MODE = 17
CELLAR_MODE = 9


def axis_dir(
    xy: tuple[int, int], dest: tuple[int, int], *, y_first: bool, tol: int = 4
) -> str | None:
    """One cardinal toward dest. None when both axes are inside ``tol``."""
    x, y = xy
    tx, ty = dest
    dy, dx = ty - y, tx - x
    axes = (("DOWN", "UP", dy), ("RIGHT", "LEFT", dx))
    if not y_first:
        axes = tuple(reversed(axes))
    for pos, neg, delta in axes:
        if abs(delta) > tol:
            return pos if delta > 0 else neg
    return None


def dungeon_align_then_push(
    snap: ZeldaSnapshot,
    *,
    push_dir: str,
    target_x: int | None = None,
    target_y: int | None = None,
    x_tol: int = 2,
    y_tol: int = 2,
    door_plane: int | None = None,
    reason: str = "door",
) -> FrameAction:
    """Align to a door band, then hold the cardinal. No sword."""
    if target_y is not None and abs(snap.link_y - target_y) > y_tol:
        btn = "UP" if snap.link_y > target_y else "DOWN"
        return FrameAction(nes_action(btn), f"{reason}_align_y")
    if door_plane is not None and push_dir in ("LEFT", "RIGHT"):
        if push_dir == "LEFT" and snap.link_x > door_plane:
            return FrameAction(nes_action("LEFT"), f"{reason}_approach")
        if push_dir == "RIGHT" and snap.link_x < door_plane:
            return FrameAction(nes_action("RIGHT"), f"{reason}_approach")
        return FrameAction(nes_action(push_dir), f"{reason}_push")
    if target_x is not None and abs(snap.link_x - target_x) > x_tol:
        btn = "LEFT" if snap.link_x > target_x else "RIGHT"
        return FrameAction(nes_action(btn), f"{reason}_align_x")
    return FrameAction(nes_action(push_dir), f"{reason}_push")


# A dungeon door is always centred on its wall: x=120 for N/S, y=141 for E/W.
DUNGEON_DOOR_X = 120
DUNGEON_DOOR_Y = 141
DUNGEON_TOP_ROW_Y = 93
# Push depth past a dungeon door node: the goal row (y=93 for UP) is a node
# inside the room, and the push walks Link through the whole doorway before
# the scroll (0x38: 93 -> 76 and on). Anywhere on the door line past the
# node is still the push; routing back from it swapped 76 <-> 77 forever.
DUNGEON_DOOR_SLACK = 48


def door_nodes(nodes, direction: str, *, dungeon: bool = False) -> set[tuple[int, int]]:
    """Lattice nodes on the room edge ``direction`` faces (the door mouth).

    ``dungeon`` keeps only the edge nodes nearest the centred door: a room
    whose east column is open end to end (L2 0x6e, L3 0x5c) otherwise walks
    to (208,133) and pushes the wall beside the door.
    """
    if not nodes:
        return set()
    if direction in ("UP", "DOWN"):
        ys = [y for _, y in nodes]
        if dungeon and direction == "UP":
            # y=85 reads walkable (feet on floor) but Link's head is in the
            # top wall everywhere except the door gap (L7 0x38 pushed UP at
            # (112,93) for 11848f). The first interior row is the goal.
            ys = [y for y in ys if y >= DUNGEON_TOP_ROW_Y] or ys
        edge = min(ys) if direction == "UP" else max(ys)
        out = {n for n in nodes if n[1] == edge}
        axis, centre = 0, DUNGEON_DOOR_X
    else:
        xs = [x for x, _ in nodes]
        edge = min(xs) if direction == "LEFT" else max(xs)
        out = {n for n in nodes if n[0] == edge}
        axis, centre = 1, DUNGEON_DOOR_Y
    if dungeon and out:
        best = min(abs(n[axis] - centre) for n in out)
        out = {n for n in out if abs(n[axis] - centre) == best}
    return out


# Room secrets (``Z_05.asm`` ``CheckUnderworldSecrets``): the low three bits
# of ``LevelBlockAttrsByteF`` pick the trigger. 4 opens the shutters when the
# push block has moved, 5 reveals stairs. The block is object ``$68`` in slot
# 11, placed on the first ``$B0`` tile in play-area row ``$A`` (y=$90).
ADDR_LEVEL_BLOCK_ATTR_F = 0x04CD
ADDR_BLOCK_PUSH_COMPLETE = 0x04CF
SECRET_BLOCK_DOOR = 4
SECRET_BLOCK_STAIRS = 5
ANY_BLOCK_TRIGGER = tuple(range(8))
BLOCK_OBJECT_TYPE = 0x68
BLOCK_SLOT = 11
# ``UpdateBlock0Idle``: Link's (y + 3) is compared with the block's y, and a
# push needs 16 frames of the input facing the block (``BlockPushDirections``).
BLOCK_Y_OFFSET = 3


def pending_block_push(
    ram: Any,
    snap: ZeldaSnapshot,
    triggers: tuple[int, ...] = (SECRET_BLOCK_DOOR, SECRET_BLOCK_STAIRS),
) -> tuple[int, int] | None:
    """The push block's ``(x, y)`` when this room's secret (one of ``triggers``) waits on it."""
    trigger = int(ram[ADDR_LEVEL_BLOCK_ATTR_F]) & 0x07
    if trigger not in triggers:
        return None
    if int(ram[ADDR_BLOCK_PUSH_COMPLETE]) != 0:
        return None
    block = snap.object_in_slot(BLOCK_SLOT)
    if block is None or int(block.type_id) != BLOCK_OBJECT_TYPE:
        return None
    return int(block.x), int(block.y)


def _between_stand_and_block(
    x: int, y: int, stand: tuple[int, int], dx: int, dy: int, slack: int = 16
) -> bool:
    """On the push line, up to ``slack`` px from ``stand`` toward the block.

    The ROM lets Link walk into the block sprite while the 16-frame push
    counter runs, so the whole block depth is still "pushing".

    ``(dx, dy)`` is the stand's offset from the block, so the block lies at
    ``-sign`` of it.
    """
    sx, sy = stand
    if dx == 0:
        toward = sy - y if dy > 0 else y - sy
        return x == sx and 0 < toward <= slack
    toward = sx - x if dx > 0 else x - sx
    return y == sy and 0 < toward <= slack


def block_push_step(
    env: Any,
    snap: ZeldaSnapshot,
    triggers: tuple[int, ...] = (SECRET_BLOCK_DOOR, SECRET_BLOCK_STAIRS),
) -> str | None:
    """Walk to a face of the pending push block and push it, on the lattice.

    ``None`` when the room has no pending block secret, the tile map is not
    bound, or no face is reachable. L3 0x6B (gathered spine, 2026-09-22):
    the north and south shutters are trigger 4, so every door walk stood on
    the shut leaf; the old hand policy only ever opened it by stumbling on
    the block during an 2700-frame wiggle.
    """
    env = env if env is not None else live_env.current()
    if env is None or snap.mode != PLAY_MODE:
        return None
    ram = env.get_ram()
    block = pending_block_push(ram, snap, triggers)
    if block is None:
        return None
    from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
    from zelda_i.walk.physics import lattice_route, lattice_step

    if not has_room_tile_map(ram):
        return None
    nodes = ow_walkable_nodes(ram, overworld=False)
    bx, by = block
    ly = by - BLOCK_Y_OFFSET
    x, y = int(snap.link_x), int(snap.link_y)
    # First reachable face in a fixed order. "Nearest face" flipped between
    # two equal routes as Link moved 2 px (L6 0x09: 160,141 <-> 160,143).
    for dx, dy, push in ((0, 16, "UP"), (0, -16, "DOWN"), (16, 0, "LEFT"), (-16, 0, "RIGHT")):
        stand = (bx + dx, ly + dy)
        landing = (bx - dx, ly - dy)
        if stand not in nodes or landing not in nodes:
            continue
        # The push walks Link past the stand into the block sprite (L4 0x32:
        # 157 -> 152 before it moves). Still on the push line is still
        # pushing; routing back to the stand reset the 16-frame counter.
        if (x, y) == stand or _between_stand_and_block(x, y, stand, dx, dy):
            return push
        route = lattice_route(nodes, (x, y), {stand})
        if route is None:
            continue
        return lattice_step(x, y, route[0]) if route else push
    return None


def stairs_step(env: Any, snap: ZeldaSnapshot) -> str | None:
    """Push a pending block secret, then walk onto the room's stair tile.

    ``None`` when there is neither a pending block nor a visible staircase,
    or no tile map. Link stands on a stair cell at ``(x, y - 3)`` like a
    block face (``BLOCK_Y_OFFSET``); the last few pixels are a direct press.
    """
    env = env if env is not None else live_env.current()
    if env is None or snap.mode != PLAY_MODE:
        return None
    push = block_push_step(env, snap, (SECRET_BLOCK_STAIRS,))
    if push is not None:
        return push
    from zelda_i.dungeon.tilemap import has_room_tile_map, stair_cells

    ram = env.get_ram()
    if not has_room_tile_map(ram):
        return None
    cells = stair_cells(ram)
    if not cells:
        return None
    x, y = int(snap.link_x), int(snap.link_y)
    sx, sy = min(cells, key=lambda c: abs(c[0] - x) + abs(c[1] - BLOCK_Y_OFFSET - y))
    goal = (sx, sy - BLOCK_Y_OFFSET)
    # Slack 0: the stair cell is exact. With the default 8 the neighbour
    # node one step short is a goal too, and the last-pixels press swapped
    # with a route back to it (L9 0x61: 120 <-> 121 beside the stairs).
    route = lattice_goto_route(env, snap, goal, slack=0)
    if route is None:
        # Visible stairs sealed in by blocks with no block secret (L9 0x61:
        # trigger 0, stairs inside the diamond). The free 0x68 still slides;
        # push it from a reachable face, then the route opens.
        push = block_push_step(env, snap, ANY_BLOCK_TRIGGER)
        if push is not None:
            return push
    if route:
        from zelda_i.walk.physics import lattice_step

        return lattice_step(x, y, route[0])
    dx, dy = goal[0] - x, goal[1] - y
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    if dy:
        return "DOWN" if dy > 0 else "UP"
    return "UP"


def lattice_door_step(env: Any, snap: ZeldaSnapshot, direction: str) -> str | None:
    """First step of the ROM-collision route to the ``direction`` door.

    ``None`` without a bound env or a room tile map, or when no door node is
    reachable. On a door node it is ``direction`` itself (the push).
    """
    env = env if env is not None else live_env.current()
    if env is None:
        return None
    from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
    from zelda_i.walk.physics import lattice_route, lattice_step

    if int(snap.level) != 0:
        push = block_push_step(env, snap, (SECRET_BLOCK_DOOR,))
        if push is not None:
            return push
    ram = env.get_ram()
    if not has_room_tile_map(ram):
        return None
    nodes = ow_walkable_nodes(ram, overworld=int(snap.level) == 0)
    x, y = int(snap.link_x), int(snap.link_y)
    dungeon = int(snap.level) != 0
    goals = door_nodes(nodes, direction, dungeon=dungeon)
    if past_door_node(x, y, goals, direction, slack=DUNGEON_DOOR_SLACK if dungeon else 8):
        return direction
    route = lattice_route(nodes, (x, y), goals)
    if route is None:
        return None
    return lattice_step(x, y, route[0]) if route else direction


def ow_edge_band_step(
    env: Any, snap: ZeldaSnapshot, direction: str, lo: int, hi: int
) -> str | None:
    """Lattice step to the ``direction`` screen edge inside a cross-axis band.

    The band is the hop's exit row (LEFT/RIGHT: y) or column (UP/DOWN: x),
    so a scroll lands where the next hop expects. On an edge node in the
    band it is ``direction`` (the scroll push). ``None`` without tiles or
    when no band node is reachable; callers keep their hand walk behind it.
    """
    env = env if env is not None else live_env.current()
    if env is None or snap.mode != PLAY_MODE:
        return None
    from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
    from zelda_i.walk.physics import lattice_route, lattice_step

    ram = env.get_ram()
    if not has_room_tile_map(ram):
        return None
    nodes = ow_walkable_nodes(ram, overworld=int(snap.level) == 0)
    edge = door_nodes(nodes, direction)
    cross = 1 if direction in ("LEFT", "RIGHT") else 0
    goals = {n for n in edge if lo <= n[cross] <= hi}
    x, y = int(snap.link_x), int(snap.link_y)
    if past_door_node(x, y, goals, direction):
        return direction
    route = lattice_route(nodes, (x, y), goals)
    if route is None:
        return None
    return lattice_step(x, y, route[0]) if route else direction


_CARDINAL_STEP = {"UP": (0, -8), "DOWN": (0, 8), "LEFT": (-8, 0), "RIGHT": (8, 0)}


def inland_lattice_step(
    x: int, y: int, direction: str, inland_x: tuple[int, int], inland_y: tuple[int, int]
) -> str | None:
    """Lattice route into an inland box, only when the cardinal is into a block.

    Wall-leaving rules press one cardinal toward the interior. L7 0x38
    pressed UP at (200,181) for 14000f (x=200 is no open column) and L6 0x39
    DOWN at (120,108) for 12515f. An open cardinal keeps the old step, so
    rooms that already leave the wall walk it unchanged. ``None`` = keep it.
    """
    from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
    from zelda_i.walk.physics import lattice_route, lattice_starts, lattice_step

    env = live_env.current()
    if env is None or not has_room_tile_map(env.get_ram()):
        return None
    nodes = ow_walkable_nodes(env.get_ram(), overworld=False)
    starts = [n for n in lattice_starts(x, y) if n in nodes]
    if not starts:
        return None
    sx, sy = min(starts, key=lambda n: abs(n[0] - x) + abs(n[1] - y))
    dx, dy = _CARDINAL_STEP[direction]
    if (sx + dx, sy + dy) in nodes:
        return None
    goals = {
        n for n in nodes
        if inland_x[0] <= n[0] <= inland_x[1] and inland_y[0] <= n[1] <= inland_y[1]
    }
    if not goals:
        return None
    near = min(abs(n[0] - x) + abs(n[1] - y) for n in goals)
    goals = {n for n in goals if abs(n[0] - x) + abs(n[1] - y) <= near + 16}
    route = lattice_route(nodes, (x, y), goals)
    if not route:
        return None
    return lattice_step(x, y, route[0])


def lattice_goto(
    env: Any, snap: ZeldaSnapshot, goal: tuple[int, int], *, slack: int = 8
) -> str | None:
    """First lattice step toward the nodes nearest ``goal``.

    ``None`` without tiles, when unreachable, or when Link is already on one
    of those nodes (the caller's own fine alignment takes over there).
    """
    route = lattice_goto_route(env, snap, goal, slack=slack)
    if not route:
        return None
    from zelda_i.walk.physics import lattice_step

    return lattice_step(int(snap.link_x), int(snap.link_y), route[0])


def lattice_goto_route(
    env: Any, snap: ZeldaSnapshot, goal: tuple[int, int], *, slack: int = 8
) -> list[tuple[int, int]] | None:
    """Lattice corners to the nodes nearest ``goal``: ``[]`` on one, ``None`` if none."""
    env = env if env is not None else live_env.current()
    if env is None:
        return None
    from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes
    from zelda_i.walk.physics import lattice_route

    ram = env.get_ram()
    if not has_room_tile_map(ram):
        return None
    nodes = ow_walkable_nodes(ram, overworld=int(snap.level) == 0)
    if not nodes:
        return None
    gx, gy = int(goal[0]), int(goal[1])
    near = sorted(nodes, key=lambda n: abs(n[0] - gx) + abs(n[1] - gy))[:8]
    best = abs(near[0][0] - gx) + abs(near[0][1] - gy)
    goals = {n for n in near if abs(n[0] - gx) + abs(n[1] - gy) <= best + slack}
    return lattice_route(nodes, (int(snap.link_x), int(snap.link_y)), goals)


def past_door_node(x: int, y: int, goals, direction: str, slack: int = 8) -> bool:
    """Link is on a door node's line and at or beyond it toward the door.

    The push moves him off the lattice (y=68 past the 69 node on L3 0x6B),
    and a route back to the node is a one-pixel tug-of-war with the push.
    """
    for gx, gy in goals:
        if direction == "UP" and abs(x - gx) <= 2 and 0 <= gy - y <= slack:
            return True
        if direction == "DOWN" and abs(x - gx) <= 2 and 0 <= y - gy <= slack:
            return True
        if direction == "LEFT" and abs(y - gy) <= 2 and 0 <= gx - x <= slack:
            return True
        if direction == "RIGHT" and abs(y - gy) <= 2 and 0 <= x - gx <= slack:
            return True
    return False


# At the door node: frames of held push before a release, and its length.
# L3 0x6B (gathered spine): UP held at (120,68) for 3500 frames never
# passed; 120 idle frames then UP scrolled on the first try.
DOOR_PUSH_FRAMES = 48
DOOR_RELEASE_FRAMES = 32


@dataclass
class LatticeDoorWalker:
    """``lattice_door_step`` plus a release-and-repush at a stuck door."""

    pushed: int = 0
    released: int = 0
    frames: int = 0

    def action(
        self, env: Any, snap: ZeldaSnapshot, direction: str, reason: str
    ) -> FrameAction | None:
        step = lattice_door_step(env, snap, direction)
        if step is None:
            return None
        self.frames += 1
        at_door = step == direction and at_door_node(env, snap, direction)
        if not at_door:
            self.pushed = self.released = 0
            return FrameAction(nes_action(step), reason)
        if self.released:
            self.released -= 1
            if not self.released:
                self.pushed = 0
            return FrameAction(nes_idle_action(), f"{reason}_release")
        self.pushed += 1
        if self.pushed >= DOOR_PUSH_FRAMES:
            self.released = DOOR_RELEASE_FRAMES
        return FrameAction(nes_action(step), f"{reason}_push")


def at_door_node(env: Any, snap: ZeldaSnapshot, direction: str) -> bool:
    from zelda_i.dungeon.tilemap import has_room_tile_map, ow_walkable_nodes

    env = env if env is not None else live_env.current()
    if env is None:
        return False
    ram = env.get_ram()
    if not has_room_tile_map(ram):
        return False
    nodes = ow_walkable_nodes(ram, overworld=int(snap.level) == 0)
    return past_door_node(
        int(snap.link_x),
        int(snap.link_y),
        door_nodes(nodes, direction, dungeon=int(snap.level) != 0),
        direction,
        slack=DUNGEON_DOOR_SLACK if int(snap.level) != 0 else 8,
    )


@dataclass(frozen=True)
class CellarCross:
    """Two-ladder cellar: drop to floor, cross, climb."""

    west_x: int
    east_x: int
    floor_y: int
    mouth_y: int
    tol: int = 4


def cellar_cross_dir(xy: tuple[int, int], spec: CellarCross, *, on_floor: bool) -> str:
    """DOWN to floor, then to east ladder, then UP. Caller tracks on_floor."""
    x, y = xy
    if not on_floor and y < spec.floor_y - spec.tol:
        return "DOWN"
    if abs(x - spec.east_x) > spec.tol:
        return "LEFT" if x > spec.east_x else "RIGHT"
    return "UP"


@dataclass(kw_only=True)
class HopController:
    """Timeout / death / wait-scroll guard. Subclass ``policy`` and ``arrived``."""

    max_frames: int = 4000
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    wait_modes: tuple[int, ...] = WAIT_SCROLL
    spec_id: str = ""
    require_level: int | None = None
    done_reason: str = "done"
    # Motion + damage attribution for every dest hop. ``tracked`` is this
    # frame's velocity view; ``damage`` names what took each heart, so a red
    # hop reports a cause instead of only a death tile.
    tracker: ObjectTracker = field(default_factory=ObjectTracker)
    damage: DamageLog = field(default_factory=DamageLog)
    tracked: tuple[TrackedObject, ...] = ()
    last_reason: str = ""
    # The door this hop leaves by. When set and Link is boxed (a hand
    # policy walking into a block), the frame goes to the ROM-collision
    # lattice route to that door instead. ``bind_env`` supplies the tiles.
    exit_dir: str | None = None
    _env: Any = field(default=None, repr=False)
    _box_anchor: tuple[int, int] | None = field(default=None, repr=False)
    _box_frames: int = field(default=0, repr=False)
    _lattice_room: tuple[int, int] | None = field(default=None, repr=False)
    _door_walker: "LatticeDoorWalker | None" = field(default=None, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return False

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_idle_action(), "idle")

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_{snap.screen:02x}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_idle_action(), "wait_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        del snap, force
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "damage": self.damage.report(),
            "lattice_frames": self.lattice_frames,
        }

    def _boxed(self, snap: ZeldaSnapshot) -> bool:
        xy = (int(snap.link_x), int(snap.link_y))
        anchor = self._box_anchor
        if anchor is None or abs(xy[0] - anchor[0]) + abs(xy[1] - anchor[1]) > BOXED_PX:
            self._box_anchor = xy
            self._box_frames = 0
            return False
        self._box_frames += 1
        return self._box_frames >= BOXED_FRAMES

    def lattice_exit(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Lattice step to the ``exit_dir`` door once the hand policy is boxed.

        Latched per room: once boxed, the rest of this room is walked on the
        lattice, so the first step off the block does not hand the frame
        back to the policy that walked into it.
        """
        if self.exit_dir is None or snap.mode != PLAY_MODE:
            return None
        room = (int(snap.level), int(snap.screen))
        if self._lattice_room != room:
            if not self._boxed(snap):
                return None
            self._lattice_room = room
        if self._door_walker is None:
            self._door_walker = LatticeDoorWalker()
        return self._door_walker.action(
            self._env, snap, self.exit_dir, f"{self.done_reason}_lattice"
        )

    @property
    def lattice_frames(self) -> int:
        return 0 if self._door_walker is None else self._door_walker.frames

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def mark_fail(self, note: str, reason: str | None = None) -> FrameAction:
        self.failed = True
        self._note(note)
        return FrameAction(nes_idle_action(), reason or note)

    def mark_done(self, snap: ZeldaSnapshot, note: str | None = None) -> FrameAction:
        self.success = True
        self._note(note or self.on_arrive(snap))
        return FrameAction(nes_idle_action(), self.done_reason)

    def wait_not_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Idle while not play. Scroll/death/timeout stay in ``guard``."""
        if snap.mode == PLAY_MODE:
            return None
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            self._note(self.timeout_note(snap))
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == DEATH_MODE:
            return self.mark_fail("link_death")
        if snap.transitioning or snap.mode in self.wait_modes:
            return self.scroll_action(snap)
        if self.require_level is not None and snap.level != self.require_level:
            if snap.mode == PLAY_MODE and not snap.transitioning:
                return self.mark_fail(f"left_level_{snap.level}")
        return None

    def observe(self, snap: ZeldaSnapshot) -> tuple[TrackedObject, ...]:
        """Sample velocity and attribute damage. Idempotent per snapshot."""
        self.tracked = self.tracker.observe(snap)
        self.damage.observe(
            snap, self.tracked, action=self.last_reason, phase=self.spec_id
        )
        return self.tracked

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.observe(snap)
        self.frames += 1
        blocked = self.guard(snap)
        if blocked is not None:
            action = self.emit(
                snap, blocked, force=self.success or self.failed
            )
        elif self.arrived(snap):
            action = self.emit(snap, self.mark_done(snap), force=True)
        else:
            action = self.emit(snap, self.lattice_exit(snap) or self.policy(snap))
        self.last_reason = action.reason
        return action
