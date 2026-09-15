"""When an overworld wave comes back, transcribed from the ROM.

No emulator. Every rule here is ``ModifyObjCountByHistoryOW`` /
``SaveKillCountOW`` (``Z_05.asm``) and the room-history append in
``RunCrossRoomTasksAndBeginUpdateMode`` (``Z_07.asm``); a test that disagrees
with the ROM is the thing to fix, not the disassembly. Live confirmation of
the history and the partial-kill subtraction is ``scratch/probe_respawn_lap.py``
(``lap1``, 2026-09-15) and ``docs/PRE_L1.md``.
"""

from __future__ import annotations

import numpy as np

from zelda_i.overworld.gathering import (
    PRE_L1_BOMB_HOPS,
    PRE_L1_LAP_WEST_HOPS,
    pre_l1_walk_hops,
)
from zelda_i.overworld.graph import SCREEN_START, path_screens_from_hops
from zelda_i.overworld.respawn import (
    ADDR_ROOM_HISTORY,
    ADDR_ROOM_HISTORY_INDEX,
    ROOM_HISTORY_LEN,
    RoomHistory,
    cleared,
    kill_flags,
    read_room_history,
    respawn_visits,
)

CORRIDOR = (0x77, 0x78, 0x68, 0x58, 0x59, 0x49, 0x4A)


def _route(laps: int) -> tuple[int, ...]:
    return path_screens_from_hops(SCREEN_START, pre_l1_walk_hops(laps))


def test_a_room_already_in_history_is_not_re_appended() -> None:
    """``RunCrossRoomTasksAndBeginUpdateMode``: no duplicates, index frozen.

    This one rule is why an out-and-back can never farm: it means walking
    back over screens Link already visited evicts nothing at all.
    """
    history = RoomHistory()
    assert history.enter(0x68) is True
    assert history.index == 1
    assert history.enter(0x68) is False
    assert history.index == 1, "a repeat must not advance CurRoomHistoryIndex"


def test_the_ring_wraps_and_evicts_the_oldest_entry() -> None:
    history = RoomHistory()
    for room in CORRIDOR:
        history.enter(room)
    # Seven distinct rooms into six slots: 0x4A overwrote 0x77 at the wrap.
    assert history.slots == [0x4A, 0x78, 0x68, 0x58, 0x59, 0x49]
    assert history.index == 1


def test_an_out_and_back_respawns_nothing() -> None:
    """``rupee_farm``'s 0x4A<->0x49 restock, generalised to any depth.

    Turning round and walking home only re-enters rooms the history already
    holds, and a repeat neither evicts nor advances the index — so no depth of
    out-and-back ever brings a wave back.
    """
    back = tuple(reversed(CORRIDOR[:-1]))
    verdicts = respawn_visits(CORRIDOR + back)
    assert all(verdicts[: len(CORRIDOR)]), "the first pass is all fresh"
    returning = verdicts[len(CORRIDOR) :]
    # 0x77 is the exception, and only because 0x4A evicted it at the wrap.
    assert returning == (False,) * (len(back) - 1) + (True,)


def test_the_lap_respawns_every_screen_it_fought() -> None:
    """Seven distinct screens against six history slots is the whole trick.

    Walking back onto 0x77 is the only new room on the westbound leg, and it
    evicts 0x78; from there each screen Link enters evicts the next one in
    front of him, so the entire eastbound leg comes back whole.
    """
    route = _route(1)
    verdicts = dict(zip(range(len(route)), respawn_visits(route)))
    west = route.index(0x4A) + 1  # first index of the westbound leg
    east = len(route) - len(PRE_L1_BOMB_HOPS)  # first index of the second pass
    assert route[east - 1] == 0x77
    assert not any(verdicts[i] for i in range(west, east - 1)), (
        "the westbound leg is all history; nothing may respawn on it"
    )
    assert all(verdicts[i] for i in range(east, len(route))), (
        "every screen of the second eastbound pass respawns"
    )


def test_one_pass_has_no_second_wave() -> None:
    assert all(respawn_visits(_route(0))), "a first pass is all fresh screens"
    assert len(set(_route(0))) == len(CORRIDOR)


def test_the_lap_is_the_corridor_backwards_then_forwards() -> None:
    east = tuple(h.target for h in PRE_L1_BOMB_HOPS)
    assert tuple(h.target for h in PRE_L1_LAP_WEST_HOPS) == tuple(
        reversed(east[:-1])
    ) + (SCREEN_START,)
    assert _route(1)[-len(east) - 1 :] == CORRIDOR


def test_kill_flags_are_the_low_three_bits_not_the_underworld_pair() -> None:
    assert kill_flags(0x27) == 7
    assert cleared(0x27)
    assert kill_flags(0xC4) == 4
    assert not cleared(0xC4)


def test_read_room_history_reads_620_and_621() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_ROOM_HISTORY_INDEX] = 3
    for i, room in enumerate((0x4A, 0x78, 0x68, 0x58, 0x59, 0x49)):
        ram[ADDR_ROOM_HISTORY + i] = room
    slots, index = read_room_history(ram)
    assert slots == (0x4A, 0x78, 0x68, 0x58, 0x59, 0x49)
    assert index == 3
    assert len(slots) == ROOM_HISTORY_LEN
