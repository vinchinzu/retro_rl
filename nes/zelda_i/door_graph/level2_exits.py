"""Level 2 (Moon) door-graph seed edges and room-id constants."""

from __future__ import annotations

from zelda_i.door_graph.core import (
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    RoomExit,
)
# Stands from puzzle catalog (single geometry authority).
from zelda_i.level2_puzzles import (
    BOMB_WALL_1E_NORTH,
    BOMB_WALL_4F_NORTH,
    BOMB_WALL_5F_NORTH,
    BOMB_WALL_6F_NORTH,
)


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
_BOMB_STAND_6F_N = BOMB_WALL_6F_NORTH.stand
_BOMB_STAND_5F_N = BOMB_WALL_5F_NORTH.stand  # same stand → boom 0x4f
_BOMB_STAND_4F_N = BOMB_WALL_4F_NORTH.stand  # → 0x3f
_BOMB_STAND_1E_N = BOMB_WALL_1E_NORTH.stand  # → Dodongo 0x0e (walk-UP solid)
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
