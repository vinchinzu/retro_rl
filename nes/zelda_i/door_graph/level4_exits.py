"""Level 4 (Snake) door-graph seed edges and room-id constants."""

from __future__ import annotations

from zelda_i.door_graph.core import (
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    RoomExit,
    clone_graph,
)
from zelda_i.level4.dungeon import (
    BOMB_21_NORTH_STAND,
    BOMB_61_NORTH_STAND,
    ROOM_L4_COMPASS_62,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_ENTRY,
    ROOM_L4_GLEEOK_13,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_MAP_21,
    ROOM_L4_MID_11,
    ROOM_L4_NORTH_30,
    ROOM_L4_STEPLADDER,
    ROOM_L4_TRIFORCE,
    ROOM_L4_VIRES_12,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    ROOM_L4_WATER_NORTH_20,
    ROOM_L4_ZOLS_40,
)

L4_ENTRY = ROOM_L4_ENTRY
L4_VIRES_61 = ROOM_L4_VIRES_61
L4_KEESE_KEY_51 = ROOM_L4_KEESE_KEY_51
L4_VIRES_50 = ROOM_L4_VIRES_50
L4_COMPASS_62 = ROOM_L4_COMPASS_62
L4_ZOLS_40 = ROOM_L4_ZOLS_40
L4_NORTH_30 = ROOM_L4_NORTH_30
L4_EAST_31 = ROOM_L4_EAST_31
L4_EAST_32 = ROOM_L4_EAST_32
L4_STEPLADDER = ROOM_L4_STEPLADDER
L4_WATER_20 = ROOM_L4_WATER_NORTH_20
L4_MAP_21 = ROOM_L4_MAP_21
L4_MID_11 = ROOM_L4_MID_11
L4_VIRES_12 = ROOM_L4_VIRES_12
L4_GLEEOK = ROOM_L4_GLEEOK_13
L4_TRIFORCE = ROOM_L4_TRIFORCE


def _l4_exits() -> dict[int, tuple[RoomExit, ...]]:
    """L4 edges from LEVEL4_ROUTE.md (live; not a Clean STATUS claim)."""
    return {
        L4_ENTRY: (
            RoomExit(
                DoorDir.UP,
                L4_VIRES_61,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="free UP @ x≈120",
                verification="observed",
            ),
        ),
        L4_VIRES_61: (
            RoomExit(
                DoorDir.DOWN,
                L4_ENTRY,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L4_KEESE_KEY_51,
                GateKind.BOMB,
                bomb_stand=BOMB_61_NORTH_STAND,
                notes="bomb-N face UP → 0x51 first key",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L4_COMPASS_62,
                GateKind.KEY,
                approach_xy=(208, 141),
                notes="KEY-RIGHT @ y≈141 compass maze",
                verification="observed",
            ),
        ),
        L4_KEESE_KEY_51: (
            RoomExit(
                DoorDir.DOWN,
                L4_VIRES_61,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L4_VIRES_50,
                GateKind.OPEN,
                approach_xy=(48, 141),
                notes="free LEFT @ y≈141",
                verification="observed",
            ),
            RoomExit(DoorDir.UP, None, GateKind.SEALED, verification="observed"),
            RoomExit(DoorDir.RIGHT, None, GateKind.SEALED, verification="observed"),
        ),
        L4_VIRES_50: (
            RoomExit(
                DoorDir.RIGHT,
                L4_KEESE_KEY_51,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L4_ZOLS_40,
                GateKind.OPEN,
                notes="scripted north maze (MAZE_50_TO_NORTH); not a dead-end",
                verification="observed",
            ),
        ),
        L4_COMPASS_62: (
            RoomExit(
                DoorDir.LEFT,
                L4_VIRES_61,
                GateKind.OPEN,
                notes="only durable exit; compass ADDR_COMPASS|0x08",
                verification="observed",
            ),
        ),
        L4_ZOLS_40: (
            RoomExit(
                DoorDir.DOWN,
                L4_VIRES_50,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L4_NORTH_30,
                GateKind.KILL_CLEAR,
                approach_xy=(120, 93),
                notes="free UP after Zol+key; L/R sealed",
                verification="observed",
            ),
            RoomExit(DoorDir.LEFT, None, GateKind.SEALED, verification="observed"),
            RoomExit(DoorDir.RIGHT, None, GateKind.SEALED, verification="observed"),
        ),
        L4_NORTH_30: (
            RoomExit(
                DoorDir.DOWN,
                L4_ZOLS_40,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L4_EAST_31,
                GateKind.KEY,
                approach_xy=(208, 141),
                notes="KEY-RIGHT @ y141 after Vire clear (ignore 0x2b)",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L4_WATER_20,
                GateKind.KEY,
                approach_xy=(120, 93),
                notes="KEY-UP needs stepladder + key (post-ladder)",
                verification="observed",
            ),
            RoomExit(DoorDir.LEFT, None, GateKind.SEALED, verification="observed"),
        ),
        L4_EAST_31: (
            RoomExit(
                DoorDir.LEFT,
                L4_NORTH_30,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L4_EAST_32,
                GateKind.KILL_CLEAR,
                notes="clear opens R; free RIGHT → 0x32",
                verification="observed",
            ),
        ),
        L4_EAST_32: (
            RoomExit(
                DoorDir.LEFT,
                L4_EAST_31,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L4_STEPLADDER,
                GateKind.KILL_CLEAR,
                notes="push left block → stairs 0x60 ADDR_LADDER / stepladder",
                verification="observed",
            ),
        ),
        L4_STEPLADDER: (
            RoomExit(
                DoorDir.DOWN,
                L4_EAST_32,
                GateKind.OPEN,
                notes="mode-9 exit after Keese clear; acquires stepladder",
                verification="observed",
            ),
        ),
        L4_WATER_20: (
            RoomExit(
                DoorDir.DOWN,
                L4_NORTH_30,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L4_MAP_21,
                GateKind.KILL_CLEAR,
                notes="state-BFS RIGHT after Vire clear → map 0x21",
                verification="observed",
            ),
        ),
        L4_MAP_21: (
            RoomExit(
                DoorDir.LEFT,
                L4_WATER_20,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L4_MID_11,
                GateKind.BOMB,
                bomb_stand=BOMB_21_NORTH_STAND,
                notes="BOMB_UP face UP → 0x11",
                verification="observed",
            ),
        ),
        L4_MID_11: (
            RoomExit(
                DoorDir.DOWN,
                L4_MAP_21,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L4_VIRES_12,
                GateKind.OPEN,
                notes="RIGHT → 5× Vire + block 0x68",
                verification="observed",
            ),
        ),
        L4_VIRES_12: (
            RoomExit(
                DoorDir.LEFT,
                L4_MID_11,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L4_GLEEOK,
                GateKind.KILL_CLEAR,
                notes="push 0x68 LEFT then maze RIGHT → Gleeok 0x13",
                verification="observed",
            ),
        ),
        L4_GLEEOK: (
            RoomExit(
                DoorDir.LEFT,
                L4_VIRES_12,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L4_TRIFORCE,
                GateKind.KILL_CLEAR,
                notes="after Gleeok kill → TF room 0x03 bit 0x08",
                verification="observed",
            ),
        ),
        L4_TRIFORCE: (
            RoomExit(
                DoorDir.DOWN,
                L4_GLEEOK,
                GateKind.OPEN,
                notes="TF 0x08",
                verification="observed",
            ),
        ),
    }


LEVEL_4_DOOR_GRAPH = DungeonDoorGraph.from_exits(
    _l4_exits(),
    level=4,
    name="level_4_snake",
)


def level_4_door_graph() -> DungeonDoorGraph:
    """Return a fresh copy of the L4 seed graph (safe to mutate rooms)."""
    return clone_graph(LEVEL_4_DOOR_GRAPH)
