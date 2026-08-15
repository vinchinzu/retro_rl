"""Dungeon door-graph template for Zelda I (L2–L9 pathfinding primitive).

Encodes per-room exits with gate kinds (open / kill-clear / key / bomb / sealed)
so planners can BFS reachable rooms from inventory caps without the emulator.
Feeds future ``retro_harness.adventure.RouteGraph`` edges (epic rr-9n6); this
module does **not** promote full RouteGraph coverage.

Door bit layout matches live ``cur_opened_doors`` / ``open_doorway_mask``:

- RIGHT = 0x01, LEFT = 0x02, DOWN = 0x04, UP = 0x08

Prefer pure geometry over combat: OPEN and BOMB/KEY stand points are first-class;
KILL_CLEAR is a gate flag, not a combat policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Door bits (cur_opened_doors / open_doorway_mask)
# ---------------------------------------------------------------------------


class DoorDir(IntEnum):
    """Cardinal door direction with NES opened-door bit value."""

    RIGHT = 0x01
    LEFT = 0x02
    DOWN = 0x04
    UP = 0x08

    @property
    def bit(self) -> int:
        return int(self)

    @property
    def opposite(self) -> DoorDir:
        return _OPPOSITE[self]

    @property
    def label(self) -> str:
        return self.name.lower()


_OPPOSITE: dict[DoorDir, DoorDir] = {
    DoorDir.RIGHT: DoorDir.LEFT,
    DoorDir.LEFT: DoorDir.RIGHT,
    DoorDir.DOWN: DoorDir.UP,
    DoorDir.UP: DoorDir.DOWN,
}

_LABEL_TO_DIR: dict[str, DoorDir] = {
    "right": DoorDir.RIGHT,
    "r": DoorDir.RIGHT,
    "left": DoorDir.LEFT,
    "l": DoorDir.LEFT,
    "down": DoorDir.DOWN,
    "d": DoorDir.DOWN,
    "up": DoorDir.UP,
    "u": DoorDir.UP,
}


def door_dir_from_label(label: str) -> DoorDir:
    """Parse a direction label (``right`` / ``R`` / ``LEFT`` …)."""
    key = label.strip().lower()
    try:
        return _LABEL_TO_DIR[key]
    except KeyError as exc:
        raise ValueError(f"unknown door direction: {label!r}") from exc


def dirs_from_mask(mask: int) -> frozenset[DoorDir]:
    """Return DoorDir members set in an opened-door bitmask."""
    return frozenset(d for d in DoorDir if mask & d.bit)


# ---------------------------------------------------------------------------
# Gate + exit model
# ---------------------------------------------------------------------------


class GateKind(str, Enum):
    """How an exit becomes traversable."""

    OPEN = "open"
    """Walkable doorway (bit may already be set, or never needed — e.g. L2 entry N)."""

    KILL_CLEAR = "kill_clear"
    """Opens after room clear (``cur_opened_doors`` bit)."""

    KEY = "key"
    """Key door; consumes ``key_cost`` keys when planning."""

    BOMB = "bomb"
    """Bombable wall; needs bombs + optional ``bomb_stand`` geometry."""

    SEALED = "sealed"
    """Permanently closed / not a pathfinding edge."""


@dataclass(frozen=True)
class RoomExit:
    """One directed exit from a dungeon room."""

    direction: DoorDir
    target_room: int | None
    """Destination room id, or ``None`` for overworld / leave-dungeon."""

    gate: GateKind = GateKind.OPEN
    bomb_stand: tuple[int, int] | None = None
    """Link (x, y) stand point for bomb walls (e.g. L2 0x6f N @ (120, 101))."""

    approach_xy: tuple[int, int] | None = None
    """Optional approach align before the exit (door y / diamond band)."""

    key_cost: int = 0
    """Keys consumed when ``gate is KEY`` (default 1 if KEY and cost left 0)."""

    notes: str = ""
    verification: str = "planned"
    """``planned`` | ``probe_geometry`` | ``observed`` — not a STATUS promote."""

    def __post_init__(self) -> None:
        if self.gate is GateKind.KEY and self.key_cost <= 0:
            object.__setattr__(self, "key_cost", 1)
        if self.gate is not GateKind.KEY and self.key_cost:
            object.__setattr__(self, "key_cost", 0)
        if self.gate is GateKind.BOMB and self.bomb_stand is None:
            # Allowed (unknown stand) but callers should seed when known.
            pass

    @property
    def is_pathfinding(self) -> bool:
        return self.gate is not GateKind.SEALED and self.target_room is not None

    def effective_key_cost(self) -> int:
        if self.gate is GateKind.KEY:
            return max(1, self.key_cost)
        return 0


@dataclass(frozen=True)
class InventoryCaps:
    """Resource caps for door-graph BFS (consumable keys/bombs)."""

    keys: int = 0
    bombs: int = 0
    can_clear: bool = True
    """If False, KILL_CLEAR exits are blocked (geometry-only pathfinding)."""

    @classmethod
    def from_mapping(cls, caps: Mapping[str, object] | InventoryCaps) -> InventoryCaps:
        if isinstance(caps, InventoryCaps):
            return caps
        return cls(
            keys=int(caps.get("keys", 0) or 0),
            bombs=int(caps.get("bombs", 0) or 0),
            can_clear=bool(caps.get("can_clear", True)),
        )


def _exit_allowed(exit_: RoomExit, caps: InventoryCaps) -> bool:
    if exit_.gate is GateKind.SEALED:
        return False
    if exit_.target_room is None:
        return False
    if exit_.gate is GateKind.OPEN:
        return True
    if exit_.gate is GateKind.KILL_CLEAR:
        return caps.can_clear
    if exit_.gate is GateKind.KEY:
        return caps.keys >= exit_.effective_key_cost()
    if exit_.gate is GateKind.BOMB:
        return caps.bombs >= 1
    return False


def _consume(exit_: RoomExit, caps: InventoryCaps) -> InventoryCaps:
    if exit_.gate is GateKind.KEY:
        return InventoryCaps(
            keys=caps.keys - exit_.effective_key_cost(),
            bombs=caps.bombs,
            can_clear=caps.can_clear,
        )
    if exit_.gate is GateKind.BOMB:
        return InventoryCaps(
            keys=caps.keys,
            bombs=caps.bombs - 1,
            can_clear=caps.can_clear,
        )
    return caps


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


@dataclass
class DungeonDoorGraph:
    """Directed room → exits map with inventory-aware reachability BFS."""

    rooms: dict[int, tuple[RoomExit, ...]] = field(default_factory=dict)
    level: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        # Normalize values to tuples for immutability of membership.
        self.rooms = {
            int(rid): tuple(exits) for rid, exits in self.rooms.items()
        }

    @classmethod
    def from_exits(
        cls,
        exits: Mapping[int, Sequence[RoomExit]],
        *,
        level: int | None = None,
        name: str = "",
    ) -> DungeonDoorGraph:
        return cls(
            rooms={int(rid): tuple(exs) for rid, exs in exits.items()},
            level=level,
            name=name,
        )

    def room_ids(self) -> frozenset[int]:
        return frozenset(self.rooms)

    def edges_from(self, room: int) -> tuple[RoomExit, ...]:
        """All recorded exits from ``room`` (including SEALED / OW)."""
        return self.rooms.get(int(room), ())

    def pathfinding_edges_from(self, room: int) -> tuple[RoomExit, ...]:
        """Exits that participate in BFS (non-sealed with a room target)."""
        return tuple(e for e in self.edges_from(room) if e.is_pathfinding)

    def bfs_reachable(
        self,
        start: int,
        inventory_caps: InventoryCaps | Mapping[str, object] = InventoryCaps(),
    ) -> frozenset[int]:
        """Rooms reachable from ``start`` under inventory caps.

        Keys and bombs are **consumed** along paths (stateful BFS). OPEN and
        KILL_CLEAR (when ``can_clear``) do not spend resources. SEALED and
        overworld targets are skipped.
        """
        caps0 = InventoryCaps.from_mapping(inventory_caps)
        start = int(start)

        # State: (room, keys, bombs) — can_clear is fixed for the search.
        seen_states: set[tuple[int, int, int]] = set()
        reached: set[int] = set()
        queue: deque[tuple[int, InventoryCaps]] = deque()

        init = (start, caps0.keys, caps0.bombs)
        seen_states.add(init)
        reached.add(start)
        queue.append((start, caps0))

        while queue:
            room, caps = queue.popleft()
            for exit_ in self.edges_from(room):
                if not _exit_allowed(exit_, caps):
                    continue
                assert exit_.target_room is not None
                nxt_caps = _consume(exit_, caps)
                state = (exit_.target_room, nxt_caps.keys, nxt_caps.bombs)
                if state in seen_states:
                    continue
                seen_states.add(state)
                reached.add(exit_.target_room)
                queue.append((exit_.target_room, nxt_caps))
        return frozenset(reached)

    def bfs_path(
        self,
        start: int,
        goal: int,
        inventory_caps: InventoryCaps | Mapping[str, object] = InventoryCaps(),
    ) -> tuple[RoomExit, ...] | None:
        """Shortest exit-count path from ``start`` to ``goal``, or None."""
        caps0 = InventoryCaps.from_mapping(inventory_caps)
        start, goal = int(start), int(goal)
        if start == goal:
            return ()

        seen_states: set[tuple[int, int, int]] = {(start, caps0.keys, caps0.bombs)}
        # parent[state] = (prev_state, exit_used)
        parent: dict[
            tuple[int, int, int],
            tuple[tuple[int, int, int], RoomExit],
        ] = {}
        queue: deque[tuple[int, InventoryCaps]] = deque([(start, caps0)])

        goal_state: tuple[int, int, int] | None = None
        while queue:
            room, caps = queue.popleft()
            for exit_ in self.edges_from(room):
                if not _exit_allowed(exit_, caps):
                    continue
                assert exit_.target_room is not None
                nxt_caps = _consume(exit_, caps)
                state = (exit_.target_room, nxt_caps.keys, nxt_caps.bombs)
                if state in seen_states:
                    continue
                seen_states.add(state)
                parent[state] = ((room, caps.keys, caps.bombs), exit_)
                if exit_.target_room == goal:
                    goal_state = state
                    queue.clear()
                    break
                queue.append((exit_.target_room, nxt_caps))

        if goal_state is None:
            return None

        path: list[RoomExit] = []
        cursor = goal_state
        start_state = (start, caps0.keys, caps0.bombs)
        while cursor != start_state:
            prev, used = parent[cursor]
            path.append(used)
            cursor = prev
        return tuple(reversed(path))

    def exit_between(
        self,
        source: int,
        target: int,
        *,
        direction: DoorDir | None = None,
    ) -> RoomExit | None:
        """First matching pathfinding exit from source to target."""
        for exit_ in self.pathfinding_edges_from(source):
            if exit_.target_room != int(target):
                continue
            if direction is not None and exit_.direction is not direction:
                continue
            return exit_
        return None


def copy_exit(exit_: RoomExit) -> RoomExit:
    """Deep-enough copy of a frozen RoomExit (safe to stash on a new graph)."""
    return RoomExit(
        direction=exit_.direction,
        target_room=exit_.target_room,
        gate=exit_.gate,
        bomb_stand=exit_.bomb_stand,
        approach_xy=exit_.approach_xy,
        key_cost=exit_.key_cost,
        notes=exit_.notes,
        verification=exit_.verification,
    )


def clone_graph(graph: DungeonDoorGraph) -> DungeonDoorGraph:
    """Return a fresh DungeonDoorGraph with copied exits (safe to mutate rooms)."""
    raw = {
        rid: tuple(copy_exit(e) for e in exits)
        for rid, exits in graph.rooms.items()
    }
    return DungeonDoorGraph.from_exits(raw, level=graph.level, name=graph.name)
