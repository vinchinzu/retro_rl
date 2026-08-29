"""Level 5 (Lizard) door-graph seed edges and room-id constants."""

from __future__ import annotations

from zelda_i.door_graph.core import (
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    RoomExit,
    clone_graph,
)
from zelda_i.level5.dungeon import (
    ROOM_L5_EAST_67,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_NORTH_27,
    ROOM_L5_NORTH_55,
    ROOM_L5_NORTH_56,
    ROOM_L5_POLS_77,
    ROOM_L5_WEST_24,
    ROOM_L5_WEST_25,
    ROOM_L5_WEST_26,
    ROOM_L5_WEST_65,
)
from zelda_i.level5.tf_path import ROOM_L5_EAST_ZOLS, ROOM_L5_NORTH_GIBDOS
from zelda_i.level5.whistle_path import (
    BOMB_EAST_STAND,
    BOMB_WEST_STAND,
    ROOM_L5_BLUE_64,
    ROOM_L5_CELLAR_07,
    ROOM_L5_PASSAGE_06,
    ROOM_L5_WHISTLE_05,
    ROOM_L5_WHISTLE_ITEM,
)

L5_ENTRY = ROOM_L5_ENTRY
L5_GIBDO_66 = ROOM_L5_GIBDO_66
L5_EAST_67 = ROOM_L5_EAST_67
L5_POLS_77 = ROOM_L5_POLS_77
L5_WEST_65 = ROOM_L5_WEST_65
L5_BLUE_64 = ROOM_L5_BLUE_64
L5_NORTH_55 = ROOM_L5_NORTH_55
L5_NORTH_56 = ROOM_L5_NORTH_56
L5_EAST_57 = ROOM_L5_EAST_ZOLS
L5_NORTH_47 = ROOM_L5_NORTH_GIBDOS
L5_COMPASS_37 = 0x37  # ROOM_27_SPEC.source_room (LEVEL5_ROUTE / dungeon)
L5_NORTH_27 = ROOM_L5_NORTH_27
L5_WEST_26 = ROOM_L5_WEST_26
L5_WEST_25 = ROOM_L5_WEST_25
L5_DIGDOGGER = ROOM_L5_WEST_24
L5_TRIFORCE = 0x14  # LEVEL5_ROUTE TF room
L5_CELLAR_07 = ROOM_L5_CELLAR_07
L5_PASSAGE_06 = ROOM_L5_PASSAGE_06
L5_WHISTLE_05 = ROOM_L5_WHISTLE_05
L5_WHISTLE_ITEM = ROOM_L5_WHISTLE_ITEM


def _l5_exits() -> dict[int, tuple[RoomExit, ...]]:
    """L5 edges from LEVEL5_ROUTE.md (assisted / observed)."""
    return {
        L5_ENTRY: (
            RoomExit(
                DoorDir.UP,
                L5_GIBDO_66,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="north permanent archway",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L5_POLS_77,
                GateKind.KEY,
                approach_xy=(208, 141),
                notes="east key door after 0x66 key",
                verification="observed",
            ),
        ),
        L5_GIBDO_66: (
            RoomExit(
                DoorDir.DOWN,
                L5_ENTRY,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L5_EAST_67,
                GateKind.KILL_CLEAR,
                notes="doors=0x08 after Gibdo clear; Bubble dead-end",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L5_NORTH_56,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="free UP after east-key return",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                None,
                GateKind.SEALED,
                notes="natural west blocked; reach 0x65 via 0x56→0x55",
                verification="observed",
            ),
        ),
        L5_EAST_67: (
            RoomExit(
                DoorDir.LEFT,
                L5_GIBDO_66,
                GateKind.OPEN,
                notes="dead-end Bubbles",
                verification="observed",
            ),
        ),
        L5_POLS_77: (
            RoomExit(
                DoorDir.LEFT,
                L5_ENTRY,
                GateKind.OPEN,
                notes="replacement key 0x19 after Pols Voice",
                verification="observed",
            ),
        ),
        L5_NORTH_56: (
            RoomExit(
                DoorDir.DOWN,
                L5_GIBDO_66,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L5_EAST_57,
                GateKind.OPEN,
                notes="east Zols; do not clear (statue 0x5f)",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L5_NORTH_55,
                GateKind.OPEN,
                notes="west to 0x55 (whistle approach)",
                verification="observed",
            ),
        ),
        L5_EAST_57: (
            RoomExit(
                DoorDir.LEFT,
                L5_NORTH_56,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L5_NORTH_47,
                GateKind.OPEN,
                notes="ROM N=open; do not clear Zols first",
                verification="observed",
            ),
        ),
        L5_NORTH_47: (
            RoomExit(
                DoorDir.DOWN,
                L5_EAST_57,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L5_COMPASS_37,
                GateKind.KILL_CLEAR,
                notes="UP after Gibdo clear → Darknuts + compass",
                verification="observed",
            ),
        ),
        L5_COMPASS_37: (
            RoomExit(
                DoorDir.DOWN,
                L5_NORTH_47,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L5_NORTH_27,
                GateKind.KILL_CLEAR,
                notes="free UP after Darknut/compass",
                verification="observed",
            ),
        ),
        L5_NORTH_27: (
            RoomExit(
                DoorDir.DOWN,
                L5_COMPASS_37,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L5_WEST_26,
                GateKind.KEY,
                approach_xy=(32, 141),
                notes="west key door after mixed clear",
                verification="observed",
            ),
        ),
        L5_WEST_26: (
            RoomExit(
                DoorDir.RIGHT,
                L5_NORTH_27,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L5_WEST_25,
                GateKind.OPEN,
                approach_xy=(32, 141),
                notes="free WEST y=141",
                verification="observed",
            ),
        ),
        L5_WEST_25: (
            RoomExit(
                DoorDir.RIGHT,
                L5_WEST_26,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L5_DIGDOGGER,
                GateKind.KEY,
                approach_xy=(32, 141),
                notes="west key → Digdogger 0x24",
                verification="observed",
            ),
        ),
        L5_DIGDOGGER: (
            RoomExit(
                DoorDir.RIGHT,
                L5_WEST_25,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L5_TRIFORCE,
                GateKind.KILL_CLEAR,
                notes="after Digdogger (whistle shrink) → TF 0x14 bit 0x10",
                verification="observed",
            ),
        ),
        L5_TRIFORCE: (
            RoomExit(
                DoorDir.DOWN,
                L5_DIGDOGGER,
                GateKind.OPEN,
                notes="TF 0x10",
                verification="observed",
            ),
        ),
        L5_NORTH_55: (
            RoomExit(
                DoorDir.RIGHT,
                L5_NORTH_56,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                L5_WEST_65,
                GateKind.OPEN,
                notes="ROOM_65_SPEC source is 0x55 DOWN",
                verification="observed",
            ),
        ),
        L5_WEST_65: (
            RoomExit(
                DoorDir.UP,
                L5_NORTH_55,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L5_GIBDO_66,
                GateKind.BOMB,
                bomb_stand=BOMB_EAST_STAND,
                notes="bomb-east → 0x66",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L5_BLUE_64,
                GateKind.BOMB,
                bomb_stand=BOMB_WEST_STAND,
                notes="bomb-west → 0x64 stairs",
                verification="observed",
            ),
        ),
        L5_BLUE_64: (
            RoomExit(
                DoorDir.RIGHT,
                L5_WEST_65,
                GateKind.OPEN,
                notes="east bomb hole return",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L5_CELLAR_07,
                GateKind.OPEN,
                notes="center stairs → cellar 0x07",
                verification="observed",
            ),
        ),
        L5_CELLAR_07: (
            RoomExit(
                DoorDir.DOWN,
                L5_BLUE_64,
                GateKind.OPEN,
                notes="left mouth return",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L5_PASSAGE_06,
                GateKind.OPEN,
                notes="other mouth → 0x06",
                verification="observed",
            ),
        ),
        L5_PASSAGE_06: (
            RoomExit(
                DoorDir.RIGHT,
                L5_CELLAR_07,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L5_WHISTLE_05,
                GateKind.KEY,
                notes="key-west → 0x05",
                verification="observed",
            ),
        ),
        L5_WHISTLE_05: (
            RoomExit(
                DoorDir.RIGHT,
                L5_PASSAGE_06,
                GateKind.OPEN,
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L5_WHISTLE_ITEM,
                GateKind.KILL_CLEAR,
                notes="clear+block stairs → Recorder 0x04 / whistle",
                verification="observed",
            ),
        ),
        L5_WHISTLE_ITEM: (
            RoomExit(
                DoorDir.DOWN,
                L5_WHISTLE_05,
                GateKind.OPEN,
                notes="left mouth back; acquires whistle",
                verification="observed",
            ),
        ),
    }


LEVEL_5_DOOR_GRAPH = DungeonDoorGraph.from_exits(
    _l5_exits(),
    level=5,
    name="level_5_lizard",
)


def level_5_door_graph() -> DungeonDoorGraph:
    """Return a fresh copy of the L5 seed graph (safe to mutate rooms)."""
    return clone_graph(LEVEL_5_DOOR_GRAPH)
