"""Level 9 door graphs with fixture and natural evidence kept separate."""

from __future__ import annotations

from zelda_i.door_graph.core import (
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    InventoryCaps,
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
    ROOM51,
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
L9_ENTRY = 0x76
L9_OLD_MAN_TF = 0x66
L9_SILVER_ARROWS = 0x10
L9_ROOM_51 = ROOM51
L9_ROOM_61 = 0x61
L9_ROOM_62 = 0x62
L9_SUFFIX_JOIN = 0x41
# Keep hex here so door_graph does not import level9.dungeon.
_SELECTED_PREFIX_ROOMS: tuple[int, ...] = (
    0x76, 0x66, 0x65, 0x55, 0x60, 0x14, 0x15, 0x16, 0x06, 0x05,
    0x70, 0x63, 0x62, 0x61, 0x75, 0x20, 0x10,
)
_SELECTED_JOIN_ROOMS: tuple[int, ...] = (
    0x10, 0x20, 0x61, 0x51, 0x41, 0x31, 0x30, 0x67, 0x04, 0x03, 0x77, 0x52,
)
L9_CELLAR_60 = 0x60
L9_CELLAR_70 = 0x70
L9_CELLAR_75 = 0x75
L9_CELLAR_77 = 0x77
_HYP = "hypothesis"
_FIXTURE = "fixture_only"
_MAGIC = InventoryCaps(keys=99, bombs=16, can_clear=True)


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


def _e(
    direction: DoorDir,
    dest: int,
    gate: GateKind = GateKind.OPEN,
    *,
    notes: str,
    verification: str,
    bomb_stand: tuple[int, int] | None = None,
) -> RoomExit:
    return RoomExit(
        direction,
        dest,
        gate,
        bomb_stand=bomb_stand,
        notes=notes,
        verification=verification,
    )


def _natural_exits() -> dict[int, tuple[RoomExit, ...]]:
    """Magical Key 0x76 → Silver 0x10 → join 0x41. Fixture-live dest hops marked observed."""
    hyp = _HYP
    obs = "observed; route_eligible=false"
    return {
        L9_ENTRY: (
            _e(
                DoorDir.UP,
                L9_OLD_MAN_TF,
                notes=f"{obs}; dest 2/2 leftover (120,205) hold UP",
                verification="observed",
            ),
        ),
        L9_OLD_MAN_TF: (
            _e(DoorDir.DOWN, L9_ENTRY, notes=hyp, verification="planned"),
            _e(
                DoorDir.LEFT,
                0x65,
                notes=f"{obs}; dest 1/1 leftover (120,205) LEFT after west shutter census",
                verification="observed",
            ),
        ),
        0x65: (
            _e(DoorDir.RIGHT, L9_OLD_MAN_TF, notes=hyp, verification="planned"),
            _e(DoorDir.UP, 0x55, GateKind.BOMB, notes=f"{hyp}; bomb north to Lanmola", verification="planned"),
        ),
        0x55: (
            _e(DoorDir.DOWN, 0x65, notes=hyp, verification="planned"),
            _e(DoorDir.UP, L9_CELLAR_60, notes=f"{hyp}; Lanmola block-stairs → cellar 0x60", verification="planned"),
        ),
        L9_CELLAR_60: (
            _e(DoorDir.RIGHT, 0x55, notes=f"{hyp}; cellar right dest live", verification="observed"),
            _e(DoorDir.LEFT, 0x14, notes=f"{hyp}; cellar left dest live", verification="observed"),
        ),
        0x14: (
            _e(DoorDir.RIGHT, 0x15, GateKind.KEY, notes=f"{hyp}; Magic Key assumed", verification="planned"),
        ),
        0x15: (
            _e(DoorDir.LEFT, 0x14, notes=hyp, verification="planned"),
            _e(DoorDir.RIGHT, 0x16, notes=f"{hyp}; first Patra skippable", verification="planned"),
        ),
        0x16: (
            _e(DoorDir.LEFT, 0x15, notes=hyp, verification="planned"),
            _e(DoorDir.UP, 0x06, GateKind.KEY, notes=f"{hyp}; Old Man bomb-left hint", verification="planned"),
        ),
        0x06: (
            _e(DoorDir.DOWN, 0x16, notes=hyp, verification="planned"),
            _e(DoorDir.LEFT, 0x05, GateKind.BOMB, notes=f"{hyp}; bomb west to stairs 0x05", verification="planned"),
        ),
        0x05: (
            _e(DoorDir.RIGHT, 0x06, notes=hyp, verification="planned"),
            _e(DoorDir.UP, L9_CELLAR_70, notes=f"{hyp}; block-stairs → cellar 0x70", verification="planned"),
        ),
        L9_CELLAR_70: (
            _e(DoorDir.RIGHT, 0x05, notes=f"{hyp}; cellar right dest live", verification="observed"),
            _e(DoorDir.LEFT, 0x63, notes=f"{hyp}; cellar left dest live 5 Zols", verification="observed"),
        ),
        0x63: (
            _e(DoorDir.LEFT, L9_ROOM_62, GateKind.KEY, notes=f"{hyp}; west to 8-Keese corridor", verification="planned"),
        ),
        L9_ROOM_62: (
            _e(DoorDir.RIGHT, 0x63, GateKind.KEY, notes=f"{obs}; 8 Keese; E key", verification="observed"),
            _e(DoorDir.LEFT, L9_ROOM_61, notes=f"{obs}; W open to other Patra", verification="observed"),
            RoomExit(
                DoorDir.UP, None, GateKind.SEALED,
                notes="dead belief: 0x62 north is wall, not Patra 0x52",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN, None, GateKind.SEALED,
                notes="dead belief: 0x62 south is wall",
                verification="observed",
            ),
        ),
        L9_ROOM_61: (
            _e(DoorDir.RIGHT, L9_ROOM_62, notes=obs, verification="observed"),
            _e(
                DoorDir.DOWN,
                L9_CELLAR_75,
                notes=f"{hyp}; Patra kill + left block-stairs (floor, not south door)",
                verification="planned",
            ),
            _e(
                DoorDir.UP,
                L9_ROOM_51,
                notes=f"{hyp}; ROM north open toward 0x41 join",
                verification="planned",
            ),
        ),
        L9_CELLAR_75: (
            _e(DoorDir.RIGHT, L9_ROOM_61, notes=f"{obs}; cellar right dest other Patra", verification="observed"),
            _e(DoorDir.LEFT, 0x20, notes=f"{obs}; cellar left dest", verification="observed"),
        ),
        0x20: (
            _e(DoorDir.UP, L9_SILVER_ARROWS, GateKind.BOMB, notes=f"{hyp}; ROM N bomb to Silver Arrow room", verification="planned"),
            _e(DoorDir.DOWN, L9_CELLAR_75, notes=f"{hyp}; block-stairs return through cellar 0x75", verification="planned"),
        ),
        L9_SILVER_ARROWS: (
            _e(DoorDir.DOWN, 0x20, notes=f"{hyp}; item cellar returns here; ADDR_ARROWS==2", verification="planned"),
        ),
        L9_ROOM_51: (
            _e(DoorDir.DOWN, L9_ROOM_61, notes=f"{obs}; 6 Like-Likes; south pred of 0x41", verification="observed"),
            _e(
                DoorDir.UP,
                L9_ROOM_41,
                notes=f"{hyp}; selected Magical Key join; dest walk NO (statue diamond)",
                verification="planned",
            ),
        ),
        L9_ROOM_41: (
            _e(DoorDir.DOWN, L9_ROOM_51, notes=f"{obs}; south shutter", verification="observed"),
            _e(
                DoorDir.UP,
                L9_ROOM_31,
                GateKind.KILL_CLEAR,
                notes=f"{obs}; north after Like-Like clear; suffix join",
                verification="observed",
            ),
        ),
        L9_ROOM_31: (
            _e(DoorDir.DOWN, L9_ROOM_41, notes=obs, verification="observed"),
            _e(
                DoorDir.LEFT,
                L9_ROOM_30,
                GateKind.BOMB,
                bomb_stand=BOMB_WALL_31_WEST.stand,
                notes=f"{obs}; bomb-west",
                verification="observed",
            ),
        ),
        L9_ROOM_30: (
            _e(DoorDir.RIGHT, L9_ROOM_31, notes=obs, verification="observed"),
            _e(DoorDir.UP, L9_CELLAR_67, notes=f"{obs}; block-stairs → cellar 0x67", verification="observed"),
        ),
        L9_CELLAR_67: (
            _e(DoorDir.DOWN, L9_ROOM_30, notes=obs, verification="observed"),
            _e(DoorDir.RIGHT, L9_ROOM_04, notes=f"{obs}; cellar right mouth → 0x04", verification="observed"),
        ),
        L9_ROOM_04: (
            _e(
                DoorDir.LEFT,
                L9_ROOM_03,
                GateKind.BOMB,
                bomb_stand=BOMB_WALL_04_WEST.stand,
                notes=f"{obs}; bomb-west → 0x03",
                verification="observed",
            ),
        ),
        L9_ROOM_03: (
            _e(DoorDir.RIGHT, L9_ROOM_04, notes=obs, verification="observed"),
            _e(DoorDir.UP, L9_CELLAR_77, notes=f"{obs}; stairs → cellar 0x77", verification="observed"),
        ),
        L9_CELLAR_77: (
            _e(DoorDir.RIGHT, L9_ROOM_03, notes=f"{obs}; right dest", verification="observed"),
            _e(DoorDir.LEFT, L9_PATRA, notes=f"{obs}; left dest live final Patra", verification="observed"),
        ),
        L9_PATRA: (
            RoomExit(
                DoorDir.DOWN, None, GateKind.SEALED,
                notes="dead belief: 0x52 south is wall; 0x62 is not the predecessor",
                verification="observed",
            ),
            _e(DoorDir.UP, L9_GANON, GateKind.KILL_CLEAR, notes=f"{obs}; after Patra north bit", verification="observed"),
        ),
        L9_GANON: (
            _e(DoorDir.UP, L9_ZELDA, GateKind.KILL_CLEAR, notes=f"{obs}; Ganon + Power TF", verification="observed"),
        ),
        L9_ZELDA: (
            _e(DoorDir.DOWN, L9_GANON, notes=obs, verification="observed"),
        ),
    }


LEVEL_9_NATURAL_DOOR_GRAPH = DungeonDoorGraph.from_exits(
    _natural_exits(),
    level=9,
    name="level_9_natural_hypothesis",
)


def level_9_door_graph() -> DungeonDoorGraph:
    """Return a fresh copy of the L9 fixture suffix graph."""
    return clone_graph(LEVEL_9_DOOR_GRAPH)


def level_9_natural_door_graph() -> DungeonDoorGraph:
    """Return the hypothesized Magical Key graph; fixture-only edges stay labeled."""
    return clone_graph(LEVEL_9_NATURAL_DOOR_GRAPH)


def natural_route_requires_51_to_41() -> bool:
    """Selected Magical Key join uses 0x51 north; dest walk is unverified."""
    return True


def selected_natural_route_rooms() -> tuple[int, ...]:
    """Prefix through Silver Arrows, then join into the proven 0x41 suffix."""
    seen: list[int] = []
    for room in (*_SELECTED_PREFIX_ROOMS, *_SELECTED_JOIN_ROOMS):
        if room not in seen:
            seen.append(room)
    return tuple(seen)


def magic_key_caps() -> InventoryCaps:
    """Planning caps: Magical Key as unlimited keys; bombs for walls."""
    return _MAGIC
