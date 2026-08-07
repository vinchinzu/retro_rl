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


# ---------------------------------------------------------------------------
# Level 2 seed (Moon) — live recon LEVEL2_ROUTE.md 2026-08-06
# Rooms: 0x7d, 0x6d, 0x6c, 0x7e, 0x6e, 0x6f, 0x5f, 0x5e
# ---------------------------------------------------------------------------

# Room id aliases (match level2_dungeon constants without importing combat).
L2_ENTRY = 0x7D
L2_ROPES = 0x6D
L2_WEST_KEY = 0x6C
L2_EAST_KEY = 0x7E
L2_EAST_OF_ROPES = 0x6E
L2_COMPASS = 0x6F
L2_BOMB_N = 0x5F
L2_GORIYA_WEST = 0x5E
L2_ROPES_NORTH = 0x4E  # free UP from Goriya; 5× Rope + key
L2_BOOM = 0x4F
L2_BOOM_CANDIDATE = L2_BOOM  # alias  # Magical Boomerang RoomItemId 0x1e
L2_TRAPS_KEESE = 0x3F  # bomb-N of boom
L2_MOLDORM = 0x3E  # LEFT of 0x3f
L2_ROPES_UNLOCK = 0x2E  # N of Moldorm; kill→UP
L2_GORIYA_BOMBS = 0x1E  # N of 0x2e; bomb-N→boss
L2_DODONGO = 0x0E  # boss type 0x32
L2_WEST_OF_BOSS = 0x0D  # LEFT after kill; TF residual
L2_OW_DOOR_SCREEN = 0x3C  # overworld Moon door (leave-dungeon target)

# Geometry anchors from live recon / pure controllers.
_BOMB_STAND_6F_N = (120, 101)
_BOMB_STAND_5F_N = (120, 101)  # same stand → boom 0x4f
_BOMB_STAND_4F_N = (120, 101)  # → 0x3f
_BOMB_STAND_1E_N = (120, 101)  # → Dodongo 0x0e (walk-UP solid)
_APPROACH_6D_LEFT = (48, 141)  # mid-height LEFT door after clear
_APPROACH_7D_RIGHT = (208, 141)  # diamond pure push y=141
_APPROACH_6E_RIGHT = (208, 141)  # key door; band y≈113 then push y≥137
_APPROACH_5F_LEFT = (48, 141)
_APPROACH_4E_RIGHT = (208, 141)


def _l2_exits() -> dict[int, tuple[RoomExit, ...]]:
    """Seed edges for the verified L2 interior subgraph."""
    return {
        # Entry (south mouth): no combat at ready; N walkable without open bit;
        # E open via diamond-nav; W sealed; S → OW.
        L2_ENTRY: (
            RoomExit(
                DoorDir.UP,
                L2_ROPES,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="north walkable without open bit",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L2_EAST_KEY,
                GateKind.OPEN,
                approach_xy=_APPROACH_7D_RIGHT,
                notes="diamond-nav east (band y≈157 → wall → pure y=141)",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                None,
                GateKind.SEALED,
                notes="entry west sealed",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                None,
                GateKind.OPEN,
                approach_xy=(120, 205),
                notes=f"leave dungeon → OW 0x{L2_OW_DOOR_SCREEN:02x}",
                verification="observed",
            ),
        ),
        # Ropes: clear opens LEFT bit 0x02 → west key; R → 0x6e; D → entry; U sealed.
        L2_ROPES: (
            RoomExit(
                DoorDir.LEFT,
                L2_WEST_KEY,
                GateKind.KILL_CLEAR,
                approach_xy=_APPROACH_6D_LEFT,
                notes="cur_opened_doors bit 0x02 after clear",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                L2_ENTRY,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L2_EAST_OF_ROPES,
                GateKind.OPEN,
                notes="3× Rope room; residual east",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                None,
                GateKind.SEALED,
                notes="0x6d UP sealed",
                verification="observed",
            ),
        ),
        # West key: only return RIGHT to ropes; other dirs sealed.
        L2_WEST_KEY: (
            RoomExit(
                DoorDir.RIGHT,
                L2_ROPES,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(DoorDir.LEFT, None, GateKind.SEALED, verification="observed"),
            RoomExit(DoorDir.UP, None, GateKind.SEALED, verification="observed"),
            RoomExit(DoorDir.DOWN, None, GateKind.SEALED, verification="observed"),
        ),
        # East key (0x7e): LEFT → entry, UP → 0x6e.
        L2_EAST_KEY: (
            RoomExit(
                DoorDir.LEFT,
                L2_ENTRY,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L2_EAST_OF_ROPES,
                GateKind.OPEN,
                verification="observed",
            ),
        ),
        # 0x6e: D → 0x7e, L → 0x6d, R key → compass 0x6f.
        L2_EAST_OF_ROPES: (
            RoomExit(
                DoorDir.DOWN,
                L2_EAST_KEY,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L2_ROPES,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L2_COMPASS,
                GateKind.KEY,
                key_cost=1,
                approach_xy=_APPROACH_6E_RIGHT,
                notes="key door; diamond band y≈113; door opens y≥137",
                verification="observed",
            ),
        ),
        # Compass 0x6f: L return → 0x6e; bomb N @ (120,101) → 0x5f.
        L2_COMPASS: (
            RoomExit(
                DoorDir.LEFT,
                L2_EAST_OF_ROPES,
                GateKind.OPEN,
                notes="return after key door",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L2_BOMB_N,
                GateKind.BOMB,
                bomb_stand=_BOMB_STAND_6F_N,
                notes="bomb N stand (120,101) UP+B; R/D/L bomb walls no-open",
                verification="observed",
            ),
        ),
        # Bomb-N room 0x5f: 5× Gel + map 0x17; D hole → 0x6f; L key → 0x5e;
        # bomb UP → boom 0x4f; RIGHT sealed (not boom path).
        L2_BOMB_N: (
            RoomExit(
                DoorDir.DOWN,
                L2_COMPASS,
                GateKind.OPEN,
                notes="hole after bomb entry; doors bit often only DOWN=4",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L2_GORIYA_WEST,
                GateKind.KEY,
                key_cost=1,
                approach_xy=_APPROACH_5F_LEFT,
                notes="key-LEFT → Goriya 0x06; no kill-gate (rr-fvt)",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                None,
                GateKind.SEALED,
                notes="sealed walk+bomb; boom is bomb-UP → 0x4f (rr-cjf)",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L2_BOOM,
                GateKind.BOMB,
                bomb_stand=_BOMB_STAND_5F_N,
                notes="bomb N stand (120,101) UP+B → Magical Boomerang 0x4f",
                verification="observed",
            ),
        ),
        # Goriya west 0x5e: RIGHT → 0x5f (or bomb-R); free UP → 0x4e.
        L2_GORIYA_WEST: (
            RoomExit(
                DoorDir.RIGHT,
                L2_BOMB_N,
                GateKind.OPEN,
                notes="walk-RIGHT max_x≈160; bomb-R @(176,141) also opens",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L2_ROPES_NORTH,
                GateKind.OPEN,
                notes="free UP → 0x4e 5× Rope + key (rr-cjf)",
                verification="observed",
            ),
        ),
        # 0x4e ropes north: key RIGHT → boom; UP → 0x3e residual; DOWN → 0x5e.
        L2_ROPES_NORTH: (
            RoomExit(
                DoorDir.DOWN,
                L2_GORIYA_WEST,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L2_BOOM,
                GateKind.KEY,
                key_cost=1,
                approach_xy=_APPROACH_4E_RIGHT,
                notes="key-RIGHT → Magical Boomerang room",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                0x3E,
                GateKind.OPEN,
                notes="free UP → 0x3e residual",
                verification="observed",
            ),
        ),
        # Boom 0x4f: Magical Boomerang pure; D → 0x5f hole; L → 0x4e; bomb-N → 0x3f.
        L2_BOOM: (
            RoomExit(
                DoorDir.DOWN,
                L2_BOMB_N,
                GateKind.OPEN,
                notes="bomb hole after 0x5f bomb-N entry",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L2_ROPES_NORTH,
                GateKind.OPEN,
                notes="return after key-RIGHT from 0x4e",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L2_TRAPS_KEESE,
                GateKind.BOMB,
                bomb_stand=_BOMB_STAND_4F_N,
                notes="bomb-N @(120,101) → traps+Keese 0x3f",
                verification="observed",
            ),
        ),
        L2_TRAPS_KEESE: (
            RoomExit(
                DoorDir.DOWN,
                L2_BOOM,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L2_MOLDORM,
                GateKind.OPEN,
                notes="LEFT → Moldorm 0x3e",
                verification="observed",
            ),
        ),
        L2_MOLDORM: (
            RoomExit(
                DoorDir.RIGHT,
                L2_TRAPS_KEESE,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L2_ROPES_UNLOCK,
                GateKind.OPEN,
                notes="UP → 8× Rope unlock 0x2e",
                verification="observed",
            ),
        ),
        L2_ROPES_UNLOCK: (
            RoomExit(
                DoorDir.DOWN,
                L2_MOLDORM,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L2_GORIYA_BOMBS,
                GateKind.KILL_CLEAR,
                notes="clear 8 ropes → UP 0x1e (south-band x=120 align)",
                verification="observed",
            ),
        ),
        L2_GORIYA_BOMBS: (
            RoomExit(
                DoorDir.DOWN,
                L2_ROPES_UNLOCK,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L2_DODONGO,
                GateKind.BOMB,
                bomb_stand=_BOMB_STAND_1E_N,
                notes=(
                    "walk-UP solid after clear (doors=12 red herring); "
                    "bomb-N @(120,101) → Dodongo 0x0e"
                ),
                verification="observed",
            ),
        ),
        L2_DODONGO: (
            RoomExit(
                DoorDir.DOWN,
                L2_GORIYA_BOMBS,
                GateKind.OPEN,
                notes="return south after bomb entry",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L2_WEST_OF_BOSS,
                GateKind.KILL_CLEAR,
                notes=(
                    "after kill doors LEFT only → 0x0d TF (WEST of boss); "
                    "south-band maze (208,141)→(208,189)→(128,189)→(128,149)"
                ),
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                None,
                GateKind.SEALED,
                notes=(
                    "walkthrough 'TF east of boss' is wrong live — RIGHT sealed "
                    "(key/bomb/push fail); TF is LEFT→0x0d"
                ),
                verification="observed",
            ),
        ),
        # West of boss (0x0d): TF collect LIVE assisted (south-band maze).
        # Free: east column x≈208, south band y≈189, diamond corridors to (128,149).
        L2_WEST_OF_BOSS: (
            RoomExit(
                DoorDir.RIGHT,
                L2_DODONGO,
                GateKind.OPEN,
                notes="return east to boss room 0x0e",
                verification="observed",
            ),
        ),
    }


LEVEL_2_DOOR_GRAPH = DungeonDoorGraph.from_exits(
    _l2_exits(),
    level=2,
    name="level_2_moon",
)


def level_2_door_graph() -> DungeonDoorGraph:
    """Return a fresh copy of the L2 seed graph (safe to mutate rooms)."""
    raw = {
        rid: tuple(
            RoomExit(
                direction=e.direction,
                target_room=e.target_room,
                gate=e.gate,
                bomb_stand=e.bomb_stand,
                approach_xy=e.approach_xy,
                key_cost=e.key_cost,
                notes=e.notes,
                verification=e.verification,
            )
            for e in exits
        )
        for rid, exits in LEVEL_2_DOOR_GRAPH.rooms.items()
    }
    return DungeonDoorGraph.from_exits(raw, level=2, name="level_2_moon")


# ---------------------------------------------------------------------------
# Level 3 (Manji) seed — additive LIVE edges (rr-vpl / l3_past_5b 2026-08-07)
# ---------------------------------------------------------------------------

L3_ENTRY = 0x7C
L3_WEST_KEY = 0x7B
L3_NORTH_ZOLS = 0x6B
L3_DARKNUTS = 0x5B
L3_ZOL_KEY_4B = 0x4B
L3_COMPASS = 0x5A
L3_WEST_DARKNUTS = 0x59
L3_SOUTH_DARKNUTS = 0x69
L3_RAFT_PASSAGE = 0x0F
L3_MAP_4C = 0x4C
L3_BOMB_SHORTCUT = 0x5C
L3_KEESE_4A = 0x4A

_BOMB_STAND_5B_R = (192, 141)  # bomb-RIGHT → 0x5c boss shortcut
_STAIRS_69_RIGHT_Y = 141


def _l3_exits() -> dict[int, tuple[RoomExit, ...]]:
    """L3 edges observed assisted LIVE (not all pure-runner encoded)."""
    return {
        L3_ENTRY: (
            RoomExit(
                DoorDir.LEFT,
                L3_WEST_KEY,
                GateKind.OPEN,
                approach_xy=(48, 149),
                notes="LEFT+UP diagonal residual @ y≈149 (pure LEFT sticks x≈32)",
                verification="observed",
            ),
        ),
        L3_WEST_KEY: (
            RoomExit(
                DoorDir.UP,
                L3_NORTH_ZOLS,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="UP @ x≈120 (|dx|≤4); wider align sticks x≈112",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L3_ENTRY,
                GateKind.OPEN,
                notes="return east to entry",
                verification="observed",
            ),
        ),
        L3_NORTH_ZOLS: (
            RoomExit(
                DoorDir.UP,
                L3_DARKNUTS,
                GateKind.KILL_CLEAR,
                approach_xy=(120, 93),
                notes="after type-0x13 clear; free-explore + UP @ x≈120",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                L3_WEST_KEY,
                GateKind.OPEN,
                notes="return south",
                verification="observed",
            ),
        ),
        L3_DARKNUTS: (
            RoomExit(
                DoorDir.UP,
                L3_ZOL_KEY_4B,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="open without Darknut clear; 3× Zol + key",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                L3_NORTH_ZOLS,
                GateKind.OPEN,
                notes="return south",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L3_COMPASS,
                GateKind.OPEN,
                approach_xy=(32, 141),
                notes="4× Keese + traps + Compass 0x16 — Raft path",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L3_BOMB_SHORTCUT,
                GateKind.BOMB,
                bomb_stand=_BOMB_STAND_5B_R,
                notes="bomb-RIGHT @(192,141) → 0x5c boss shortcut (recon)",
                verification="observed",
            ),
        ),
        L3_ZOL_KEY_4B: (
            RoomExit(
                DoorDir.DOWN,
                L3_DARKNUTS,
                GateKind.OPEN,
                notes="return south",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L3_KEESE_4A,
                GateKind.KEY,
                approach_xy=(32, 141),
                notes="key door → 5× Keese 0x4a",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L3_MAP_4C,
                GateKind.KEY,
                approach_xy=(208, 141),
                notes="key door → map room item 0x17",
                verification="observed",
            ),
        ),
        L3_COMPASS: (
            RoomExit(
                DoorDir.RIGHT,
                L3_DARKNUTS,
                GateKind.OPEN,
                notes="return east",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L3_WEST_DARKNUTS,
                GateKind.KEY,
                approach_xy=(32, 141),
                notes="key door → 5× Darknut; long y=141 push (key-waste trap if misaligned)",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L3_KEESE_4A,
                GateKind.OPEN,
                notes="free north → 5× Keese",
                verification="observed",
            ),
        ),
        L3_WEST_DARKNUTS: (
            RoomExit(
                DoorDir.RIGHT,
                L3_COMPASS,
                GateKind.OPEN,
                notes="return east after key entry",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                L3_SOUTH_DARKNUTS,
                GateKind.KILL_CLEAR,
                approach_xy=(120, 205),
                notes="kill 5 Darknuts opens DOWN; settle spawn ~80–100f",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                0x49,
                GateKind.OPEN,
                notes="north mixed room (probe)",
                verification="observed",
            ),
        ),
        L3_SOUTH_DARKNUTS: (
            RoomExit(
                DoorDir.UP,
                L3_WEST_DARKNUTS,
                GateKind.OPEN,
                notes="return north",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L3_RAFT_PASSAGE,
                GateKind.KILL_CLEAR,
                approach_xy=(208, _STAIRS_69_RIGHT_Y),
                notes=(
                    "stairs RIGHT only @ y≈141 after 8× Darknut clear → "
                    "mode-9 passage 0x0f (Raft)"
                ),
                verification="observed",
            ),
        ),
        L3_RAFT_PASSAGE: (
            RoomExit(
                DoorDir.LEFT,
                None,
                GateKind.OPEN,
                notes=(
                    "underworld mode 9: DOWN y189 → RIGHT x≈176 → UP channel → "
                    "LEFT x≈136 touch Raft (ADDR_RAFT); not a cardinal door"
                ),
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L3_SOUTH_DARKNUTS,
                GateKind.OPEN,
                approach_xy=(48, 77),
                notes=(
                    "EXIT reverse: channel x≈176 south → west → NW stairs hold UP "
                    "→ mode10 → play 0x69 (Level3Raft backtrack)"
                ),
                verification="observed",
            ),
        ),
        L3_BOMB_SHORTCUT: (
            RoomExit(
                DoorDir.LEFT,
                L3_DARKNUTS,
                GateKind.OPEN,
                notes="return west to 0x5b",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L3_MAP_4C,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="north to map room (LIVE post-Raft shortcut)",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                0x5D,
                GateKind.KILL_CLEAR,
                approach_xy=(208, 141),
                notes="RIGHT only @ y≈141 after 3× Darknut clear → 0x5d boss prep",
                verification="observed",
            ),
        ),
        0x5D: (
            RoomExit(
                DoorDir.LEFT,
                L3_BOMB_SHORTCUT,
                GateKind.OPEN,
                notes="return west to 0x5c",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                0x4D,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes=(
                    "Manhandla candidate 0x4d type 0x3c (assisted glimpse; "
                    "UP gate residual / flaky)"
                ),
                verification="observed",
            ),
        ),
    }


LEVEL_3_DOOR_GRAPH = DungeonDoorGraph.from_exits(
    _l3_exits(),
    level=3,
    name="level_3_manji",
)


def level_3_door_graph() -> DungeonDoorGraph:
    """Return a fresh copy of the L3 seed graph (safe to mutate rooms)."""
    raw = {
        rid: tuple(
            RoomExit(
                direction=e.direction,
                target_room=e.target_room,
                gate=e.gate,
                bomb_stand=e.bomb_stand,
                approach_xy=e.approach_xy,
                key_cost=e.key_cost,
                notes=e.notes,
                verification=e.verification,
            )
            for e in exits
        )
        for rid, exits in LEVEL_3_DOOR_GRAPH.rooms.items()
    }
    return DungeonDoorGraph.from_exits(raw, level=3, name="level_3_manji")


__all__ = [
    "DoorDir",
    "GateKind",
    "RoomExit",
    "InventoryCaps",
    "DungeonDoorGraph",
    "LEVEL_2_DOOR_GRAPH",
    "level_2_door_graph",
    "LEVEL_3_DOOR_GRAPH",
    "level_3_door_graph",
    "door_dir_from_label",
    "dirs_from_mask",
    "L2_ENTRY",
    "L2_ROPES",
    "L2_WEST_KEY",
    "L2_EAST_KEY",
    "L2_EAST_OF_ROPES",
    "L2_COMPASS",
    "L2_BOMB_N",
    "L2_GORIYA_WEST",
    "L2_ROPES_NORTH",
    "L2_BOOM",
    "L2_TRAPS_KEESE",
    "L2_MOLDORM",
    "L2_ROPES_UNLOCK",
    "L2_GORIYA_BOMBS",
    "L2_DODONGO",
    "L2_WEST_OF_BOSS",
    "L2_OW_DOOR_SCREEN",
    "L3_ENTRY",
    "L3_WEST_KEY",
    "L3_NORTH_ZOLS",
    "L3_DARKNUTS",
    "L3_ZOL_KEY_4B",
    "L3_COMPASS",
    "L3_WEST_DARKNUTS",
    "L3_SOUTH_DARKNUTS",
    "L3_RAFT_PASSAGE",
    "L3_MAP_4C",
    "L3_BOMB_SHORTCUT",
    "L3_KEESE_4A",
]
