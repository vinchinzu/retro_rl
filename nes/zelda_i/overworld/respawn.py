"""Overworld wave respawn: six-slot RoomHistory, no duplicate appends.

ModifyObjCountByHistoryOW (Z_05.asm) clears a screen's kill flags only when
the screen is absent from RoomHistory ($621, 6 entries) and those flags
already read 7. RunCrossRoomTasksAndBeginUpdateMode (Z_07.asm) appends a
room only if it is not already in the history, so an out-and-back evicts
nothing. ``enter`` / ``respawn_visits`` are the absence half: necessary for
a full wave, not sufficient (flags 3 still subtracts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from zelda_i.ram import read_u8

__all__ = [
    "ADDR_ROOM_HISTORY",
    "ADDR_ROOM_HISTORY_INDEX",
    "KILL_FLAGS_MASK",
    "KILL_FLAGS_MAX",
    "ROOM_HISTORY_LEN",
    "RoomHistory",
    "cleared",
    "kill_flags",
    "read_room_history",
    "respawn_visits",
]

# ``Variables.inc``: CurRoomHistoryIndex := $620, RoomHistory := $621.
ADDR_ROOM_HISTORY_INDEX = 0x0620
ADDR_ROOM_HISTORY = 0x0621
ROOM_HISTORY_LEN = 6
# OW kill count is the low three bits of ``WorldFlags[$067F+screen]``. The
# ``0xC0`` mask in ``ram.WORLD_FLAG_KILLS`` is the *underworld* packing.
KILL_FLAGS_MASK = 0x07
KILL_FLAGS_MAX = 0x07


@dataclass
class RoomHistory:
    """``RoomHistory`` ($621) as the ROM keeps it: six slots, no duplicates.

    ``ClearRoomHistory`` fills the slots with ``0x00``, which is a real room
    id in the far north-west and nowhere near any route here.
    """

    slots: list[int] = field(default_factory=lambda: [0x00] * ROOM_HISTORY_LEN)
    index: int = 0

    def enter(self, room: int) -> bool:
        """Append ``room`` if it is not already in the ring.

        True when the room was absent (ModifyObjCountByHistoryOW's history
        miss). Necessary for a full respawn, not sufficient: flags must
        also read 7.
        """
        room = int(room) & 0xFF
        absent = room not in self.slots
        if absent:
            self.slots[self.index] = room
            self.index = (self.index + 1) % len(self.slots)
        return absent


def respawn_visits(route: Iterable[int]) -> tuple[bool, ...]:
    """Per screen entry along ``route``, True when its wave comes back.

    "Comes back" is the ROM's condition minus the flags half: absent from the
    history. A screen whose kill flags are short of 7 keeps its partial
    subtraction instead, so this is the *upper* bound on what a route can
    farm, and the reason to clear a screen rather than cross it.
    """
    history = RoomHistory()
    return tuple(history.enter(room) for room in route)


def read_room_history(ram: np.ndarray) -> tuple[tuple[int, ...], int]:
    """``(slots, index)`` live, for a probe that has to name the ROM's view."""
    slots = tuple(
        int(read_u8(ram, ADDR_ROOM_HISTORY + i)) for i in range(ROOM_HISTORY_LEN)
    )
    return slots, int(read_u8(ram, ADDR_ROOM_HISTORY_INDEX))


def kill_flags(world_flags_byte: int) -> int:
    """OW kill count stored for a screen: 0-6 partial, 7 = cleared."""
    return int(world_flags_byte) & KILL_FLAGS_MASK


def cleared(world_flags_byte: int) -> bool:
    return kill_flags(world_flags_byte) == KILL_FLAGS_MAX
