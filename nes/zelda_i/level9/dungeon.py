"""Level 9 natural-spine endpoint specs and stop predicates.

This module intentionally contains no navigation or combat policy.  The
natural Level 9 topology is not decoded yet, so only RAM-observable chapter
boundaries are named here.  Existing backward-recon fixtures are not evidence
for any of the natural-prefix predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from zelda_i.level9.ganon import credits_rolling, final_ending_screen
from zelda_i.level9.patra import (
    NORTH_DOOR,
    PATRA_EYE_COUNT,
    final_patra_live,
    patra_eyes,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

LEVEL9 = 9
FULL_TRIFORCE = 0xFF
ROOM_LEVEL9_ENTRY = 0x76
ROOM_FINAL_PATRA = 0x52
SILVER_ARROWS = 2
MAGICAL_SWORD = 3


@dataclass(frozen=True)
class Level9EndpointSpec:
    """Public chapter boundary without implying a route to that boundary."""

    through: str
    stop: str
    evidence: str
    description: str
    predicate: Callable[[ZeldaSnapshot], bool] | None


def level9_entry_snapshot_stop(snap: ZeldaSnapshot) -> bool:
    """Snapshot-visible part of the natural L9 entry contract."""
    return (
        snap.level == LEVEL9
        and snap.screen == ROOM_LEVEL9_ENTRY
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.triforce == FULL_TRIFORCE
    )


def level9_entry_stop(snap: ZeldaSnapshot, *, magic_key: bool) -> bool:
    """Natural L9 entry contract after the Old Man full-Triforce gate."""
    return level9_entry_snapshot_stop(snap) and magic_key


def level9_silver_arrows_stop(
    snap: ZeldaSnapshot,
    *,
    room: int | None,
) -> bool:
    """Exact Silver Arrow endpoint after topology selects its live room."""
    return (
        room is not None
        and snap.level == LEVEL9
        and snap.screen == room
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.triforce == FULL_TRIFORCE
        and snap.bow > 0
        and snap.arrows == SILVER_ARROWS
    )


def level9_live_patra_stop(snap: ZeldaSnapshot) -> bool:
    """Exact natural-prefix join expected by the proven ending policies."""
    return (
        snap.level == LEVEL9
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.triforce == FULL_TRIFORCE
        and snap.bow > 0
        and snap.arrows == SILVER_ARROWS
        and snap.screen == ROOM_FINAL_PATRA
        and snap.sword >= MAGICAL_SWORD
        and final_patra_live(snap)
        and len(patra_eyes(snap)) == PATRA_EYE_COUNT
        and not (snap.cur_opened_doors & NORTH_DOOR)
    )


def level9_credits_stop(snap: ZeldaSnapshot) -> bool:
    """The accepted ending update loop: rolling credits or final page."""
    return credits_rolling(snap) or final_ending_screen(snap)


L9_ENTRY_ENDPOINT = Level9EndpointSpec(
    through="level9-entry",
    stop="level9_entry_0x76",
    evidence="hypothesis",
    description="natural bomb entrance and full-Triforce gate into room 0x76",
    predicate=level9_entry_snapshot_stop,
)
L9_SILVER_ARROWS_ENDPOINT = Level9EndpointSpec(
    through="level9-silver-arrows",
    stop="level9_silver_arrows",
    evidence="hypothesis",
    description="natural Silver Arrows acquisition in an undecoded live room",
    predicate=None,
)
L9_PATRA_ENDPOINT = Level9EndpointSpec(
    through="level9-patra",
    stop="level9_live_patra_0x52",
    evidence="hypothesis",
    description="natural join into live uncleared final Patra room 0x52",
    predicate=level9_live_patra_stop,
)
L9_CREDITS_ENDPOINT = Level9EndpointSpec(
    through="level9-credits",
    stop="level9_credits",
    evidence="fixture-live",
    description="write-free Patra, Ganon, Zelda, and ending input policies",
    predicate=level9_credits_stop,
)

L9_ENDPOINTS = (
    L9_ENTRY_ENDPOINT,
    L9_SILVER_ARROWS_ENDPOINT,
    L9_PATRA_ENDPOINT,
    L9_CREDITS_ENDPOINT,
)

__all__ = [
    "FULL_TRIFORCE",
    "L9_CREDITS_ENDPOINT",
    "L9_ENDPOINTS",
    "L9_ENTRY_ENDPOINT",
    "L9_PATRA_ENDPOINT",
    "L9_SILVER_ARROWS_ENDPOINT",
    "LEVEL9",
    "MAGICAL_SWORD",
    "ROOM_FINAL_PATRA",
    "ROOM_LEVEL9_ENTRY",
    "SILVER_ARROWS",
    "Level9EndpointSpec",
    "level9_credits_stop",
    "level9_entry_stop",
    "level9_entry_snapshot_stop",
    "level9_live_patra_stop",
    "level9_silver_arrows_stop",
]
