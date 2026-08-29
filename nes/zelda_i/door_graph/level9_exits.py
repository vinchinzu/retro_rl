"""Level 9 fixture-suffix door-graph (route_eligible=false)."""

from __future__ import annotations

from zelda_i.door_graph.core import (
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    RoomExit,
    clone_graph,
)
from zelda_i.level9.ganon import ROOM_BEFORE_GANON, ROOM_GANON, ROOM_ZELDA
from zelda_i.level9.stairs import (
    BOMB_WALL_04_WEST,
    BOMB_WALL_31_WEST,
    CELLAR_67,
    ROOM03,
    ROOM04,
    ROOM30,
    ROOM31,
    ROOM41,
)

L9_ROOM_41 = ROOM41
L9_ROOM_31 = ROOM31
L9_ROOM_30 = ROOM30
L9_CELLAR_67 = CELLAR_67
L9_ROOM_04 = ROOM04
L9_ROOM_03 = ROOM03
L9_PATRA = ROOM_BEFORE_GANON
L9_GANON = ROOM_GANON
L9_ZELDA = ROOM_ZELDA

_FIXTURE = "fixture_only"


def _l9_exits() -> dict[int, tuple[RoomExit, ...]]:
    """Observed fixture suffix 0x41 → … → 0x32. Not a natural route."""
    return {
        L9_ROOM_41: (
            RoomExit(
                DoorDir.UP,
                L9_ROOM_31,
                GateKind.KILL_CLEAR,
                notes=f"{_FIXTURE}; north after Like-Like clear",
                verification="observed",
            ),
        ),
        L9_ROOM_31: (
            RoomExit(
                DoorDir.DOWN,
                L9_ROOM_41,
                GateKind.OPEN,
                notes=_FIXTURE,
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L9_ROOM_30,
                GateKind.BOMB,
                bomb_stand=BOMB_WALL_31_WEST.stand,
                notes=f"{_FIXTURE}; bomb-west",
                verification="observed",
            ),
        ),
        L9_ROOM_30: (
            RoomExit(
                DoorDir.RIGHT,
                L9_ROOM_31,
                GateKind.OPEN,
                notes=_FIXTURE,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L9_CELLAR_67,
                GateKind.OPEN,
                notes=f"{_FIXTURE}; block-stairs → cellar 0x67",
                verification="observed",
            ),
        ),
        L9_CELLAR_67: (
            RoomExit(
                DoorDir.DOWN,
                L9_ROOM_30,
                GateKind.OPEN,
                notes=_FIXTURE,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L9_ROOM_04,
                GateKind.OPEN,
                notes=f"{_FIXTURE}; cellar right mouth → 0x04",
                verification="observed",
            ),
        ),
        L9_ROOM_04: (
            RoomExit(
                DoorDir.LEFT,
                L9_ROOM_03,
                GateKind.BOMB,
                bomb_stand=BOMB_WALL_04_WEST.stand,
                notes=f"{_FIXTURE}; bomb-west → 0x03",
                verification="observed",
            ),
        ),
        L9_ROOM_03: (
            RoomExit(
                DoorDir.RIGHT,
                L9_ROOM_04,
                GateKind.OPEN,
                notes=_FIXTURE,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L9_PATRA,
                GateKind.OPEN,
                notes=f"{_FIXTURE}; stairs → cellar 0x77 left → Patra 0x52",
                verification="observed",
            ),
        ),
        L9_PATRA: (
            RoomExit(
                DoorDir.UP,
                L9_GANON,
                GateKind.KILL_CLEAR,
                notes=f"{_FIXTURE}; after Patra north bit → Ganon 0x42",
                verification="observed",
            ),
        ),
        L9_GANON: (
            RoomExit(
                DoorDir.UP,
                L9_ZELDA,
                GateKind.KILL_CLEAR,
                notes=f"{_FIXTURE}; after Ganon + Power TF → Zelda 0x32",
                verification="observed",
            ),
        ),
        L9_ZELDA: (
            RoomExit(
                DoorDir.DOWN,
                L9_GANON,
                GateKind.OPEN,
                notes=_FIXTURE,
                verification="observed",
            ),
        ),
    }


LEVEL_9_DOOR_GRAPH = DungeonDoorGraph.from_exits(
    _l9_exits(),
    level=9,
    name="level_9_fixture_suffix",
)


def level_9_door_graph() -> DungeonDoorGraph:
    """Return a fresh copy of the L9 fixture suffix graph."""
    return clone_graph(LEVEL_9_DOOR_GRAPH)
