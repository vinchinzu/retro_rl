"""Live Level 9 room ``0x62`` predicates and door/nav policy.

Room ``0x62`` was the hypothesized south neighbor of final Patra ``0x52``.
Live fceumm + first-quest ROM door bytes (2026-08-14) **disprove** that
cardinal adjacency:

- loader ``0x72`` + UP settles uncleared ``0x62`` with eight Keese ``0x1B``;
- ``CurOpenedDoors`` / ``OpenDoorwayMask`` stay 0;
- kill-clear does not raise any door bit;
- north bomb stands do not open a passage;
- ROM L7–9 N/S byte ``0x26``: north and south are wall code 1;
- ROM L7–9 E/W byte ``0x16``: west open, east key;
- ROM ``0x52`` N/S byte ``0xE6``: north shutter, south wall.

``0x52`` is therefore not entered by walking north from ``0x62``.  The
route predecessor is a stairs / underground-passage drop into ``0x52``
(StrategyWiki L9; L9 stair sources ``0x60,0x70,0x72,0x75,0x67,0x77,0x00,0x4F``).
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import ROOM_LEVEL9_62, ROOM_LEVEL9_72
from zelda_i.dungeon.ids import KEESE_OBJECT_TYPE, object_name
from zelda_i.level9.ganon import LEVEL9, ROOM_BEFORE_GANON
from zelda_i.level9.path import NORTH_DOOR_X, NORTH_DOOR_X_TOL
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

NORTH_DOOR = 0x08
EAST_DOOR = 0x01
WEST_DOOR = 0x02
SOUTH_DOOR = 0x04

# Live uncleared settle (loader 0x72 hold UP, 20 idle frames).
ROOM62_KEESE_COUNT = 8
ROOM62_ROOM_ITEM_ID = 0x0F

# First-quest L7–9 door codes (Data Crystal / cbmeeks): 0 open, 4 bomb,
# 5/6 key, 7 shutter, 1–3 wall.
ROOM62_ROM_NORTH = 1  # wall
ROOM62_ROM_SOUTH = 1  # wall
ROOM62_ROM_WEST = 0  # open
ROOM62_ROM_EAST = 5  # key
ROOM52_ROM_NORTH = 7  # shutter to Ganon
ROOM52_ROM_SOUTH = 1  # wall — not a 0x62 connection

# L9 stairway-list sources (ROM 0x19C10).  0x52 is not among them; it is
# the hypothesized drop destination of the final underground passage.
LEVEL9_STAIR_SOURCES: tuple[int, ...] = (
    0x60,
    0x70,
    0x72,
    0x75,
    0x67,
    0x77,
    0x00,
    0x4F,
)

_PLAY_SLOTS = range(1, 13)


@dataclass(frozen=True)
class Room62LoaderCandidate:
    """One game-loader seed that can scroll into uncleared room ``0x62``."""

    from_room: int
    direction: str
    link_x: int
    link_y: int
    label: str


# First try is the same -0x10 north arithmetic used to load 0x52 from 0x62.
LOADER_CANDIDATES: tuple[Room62LoaderCandidate, ...] = (
    Room62LoaderCandidate(
        from_room=ROOM_LEVEL9_72,
        direction="UP",
        link_x=0x78,
        link_y=0x58,
        label="south_0x72_hold_up",
    ),
    Room62LoaderCandidate(
        from_room=0x61,
        direction="RIGHT",
        link_x=0xD0,
        link_y=0xBD,
        label="west_0x61_hold_right",
    ),
    Room62LoaderCandidate(
        from_room=0x63,
        direction="LEFT",
        link_x=0x20,
        link_y=0xBD,
        label="east_0x63_hold_left",
    ),
    Room62LoaderCandidate(
        from_room=ROOM_BEFORE_GANON,
        direction="DOWN",
        link_x=0x78,
        link_y=0xDD,
        label="north_0x52_hold_down",
    ),
)


def in_room_62(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == PLAY_MODE
        and snap.level == LEVEL9
        and snap.screen == ROOM_LEVEL9_62
    )


def room62_objects(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot in _PLAY_SLOTS and (obj.type_id or obj.hp)
    )


def room62_keese(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(
        obj
        for obj in room62_objects(snap)
        if obj.type_id == KEESE_OBJECT_TYPE
    )


def uncleared_room62(snap: ZeldaSnapshot) -> bool:
    """True for the live loader settle: eight Keese, no opened door bits."""
    return (
        in_room_62(snap)
        and len(room62_keese(snap)) == ROOM62_KEESE_COUNT
        and snap.cur_opened_doors == 0
        and snap.open_doorway_mask == 0
    )


def room62_object_summary(snap: ZeldaSnapshot) -> list[dict[str, int | str]]:
    return [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "type_name": object_name(obj.type_id),
            "hp": obj.hp,
            "x": obj.x,
            "y": obj.y,
            "state": obj.state,
        }
        for obj in room62_objects(snap)
    ]


def door_bits(mask: int) -> dict[str, bool | int]:
    value = int(mask) & 0x0F
    return {
        "east": bool(value & EAST_DOOR),
        "west": bool(value & WEST_DOOR),
        "south": bool(value & SOUTH_DOOR),
        "north": bool(value & NORTH_DOOR),
        "raw": value,
    }


def room62_is_cardinal_predecessor_of_patra() -> bool:
    """ROM+live: 0x62 north is a wall; 0x52 south is a wall."""
    return False


def room62_to_patra_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of a north push ``0x62`` → ``0x52``.

    Kept as the nav micro that would apply IF a north door existed.  Live
    and ROM both say it does not; callers must treat a stuck north-wall
    finish as a retarget, not a success.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "patra_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM_BEFORE_GANON:
        return FrameAction(nes_idle_action(), "patra_arrived")
    if snap.screen != ROOM_LEVEL9_62:
        return FrameAction(
            nes_idle_action(),
            f"unexpected_room_0x{snap.screen:02x}",
        )
    if abs(int(snap.link_x) - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "patra_align_x")
    return FrameAction(nes_action("UP"), "patra_push_north")


__all__ = [
    "EAST_DOOR",
    "LEVEL9_STAIR_SOURCES",
    "LOADER_CANDIDATES",
    "NORTH_DOOR",
    "ROOM52_ROM_NORTH",
    "ROOM52_ROM_SOUTH",
    "ROOM62_KEESE_COUNT",
    "ROOM62_ROM_EAST",
    "ROOM62_ROM_NORTH",
    "ROOM62_ROM_SOUTH",
    "ROOM62_ROM_WEST",
    "ROOM62_ROOM_ITEM_ID",
    "ROOM_LEVEL9_62",
    "ROOM_LEVEL9_72",
    "Room62LoaderCandidate",
    "SOUTH_DOOR",
    "WEST_DOOR",
    "door_bits",
    "in_room_62",
    "room62_is_cardinal_predecessor_of_patra",
    "room62_keese",
    "room62_object_summary",
    "room62_objects",
    "room62_to_patra_step",
    "uncleared_room62",
]
