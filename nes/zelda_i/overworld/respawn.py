"""When an overworld wave comes back, and why no out-and-back can bring it.

One pass of the pre-L1 corridor cannot fund the 20R bomb pack: 32 bodies pay
~12R of random drops, and the forced 5-rupees want a 26-kill streak
(``docs/PRE_L1.md``). The only other supply is a *second* wave on screens
already fought, which every prior sitting left as "untested".

The ROM is exact about it, and it is not "walk two screens away"
(`aldonunez/zelda1-disassembly`):

* ``ModifyObjCountByHistoryOW`` (``Z_05.asm``) runs inside
  ``CreateRoomObjects`` on every room load. It clears a room's kill-count
  flags -- the full respawn -- only when the room is **absent from the
  six-entry ``RoomHistory``** *and* those flags already read the max, 7.
  A room that *is* in the history instead has its kill count **subtracted**
  from the spawn count, which is why a screen comes back thinner.
* ``SaveKillCountOW`` writes 7 exactly when ``RoomKillCount >= RoomObjCount``
  -- the screen was cleared of whatever it spawned -- and otherwise adds the
  partial count in, capped at 7. So partial clears converge on 7 over visits
  rather than blocking the respawn forever.
* ``RunCrossRoomTasksAndBeginUpdateMode`` (``Z_07.asm``) appends the room to
  the history **only when it is not already in it**, and leaves
  ``CurRoomHistoryIndex`` alone when it is.

That last rule is the one that decides route shape, and it is the reason the
``0x4A <-> 0x49`` restock in ``overworld.rupee_farm`` was never a rupee
supply: **every screen on the way back is already in the history, so an
out-and-back evicts nothing, ever.** Eviction needs *new* rooms. The pre-L1
corridor has seven distinct screens against six history slots, so the
smallest thing that works is a full lap -- walking back onto ``0x77`` evicts
``0x78``, and from there each screen Link enters evicts the next one in front
of him, so the whole corridor respawns on the way east and every lap after.

Measured live (``scratch/probe_respawn_lap.py``, ``lap1``): the history filled
``0x77 0x78 0x68 0x58 0x59 0x49`` and ``0x4A`` overwrote ``0x77`` at the wrap,
exactly as :class:`RoomHistory` models it; re-entering ``0x49`` four kills
later read flags ``4`` and spawned ``6 - 4 = 2`` bodies.

Pure arithmetic and one RAM read. No emulator.
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
        """Enter ``room``. True when the ROM would clear its kill flags.

        The absence test is the one ``ModifyObjCountByHistoryOW`` makes, and
        it happens *before* the append — ``CreateRoomObjects`` runs first in
        ``RunCrossRoomTasksAndBeginUpdateMode``.
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
