"""Clean ladder for ALTTP: C1 Sanctuary through C15 credits.

Status table only. It does not dispatch hops. Stop strings are the phase-map
labels in ``docs/PORT_PLAN.md``. ``acceptance`` is the predicate name a later
task will implement; this module does not call it. C1 is verified by the
no-assist power-on replay in ``recordings/clean_escape_20261002/replay.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = (
    "CLEAN_LADDER",
    "CleanRung",
    "LadderStatus",
    "next_open",
)


class LadderStatus(str, Enum):
    """``open`` until the rung task marks the row green."""

    OPEN = "open"
    GREEN = "green"


@dataclass(frozen=True)
class CleanRung:
    """One Clean stop. ``acceptance`` is a predicate name, not a callable."""

    id: str
    stop: str
    acceptance: str
    status: LadderStatus = LadderStatus.OPEN


# Phase-map order. Do not reorder: next_open walks this tuple.
_ROWS: tuple[tuple[str, str, str], ...] = (
    ("C1", "Sanctuary", "sanctuary_accepted"),
    ("C2", "Eastern", "eastern_prize_accepted"),
    ("C3", "Desert", "desert_prize_accepted"),
    ("C4", "Hera", "hera_prize_accepted"),
    ("C5", "Master Sword", "master_sword_accepted"),
    ("C6", "Agahnim 1", "agahnim_1_accepted"),
    ("C7", "PoD", "pod_prize_accepted"),
    ("C8", "Swamp", "swamp_prize_accepted"),
    ("C9", "Skull", "skull_prize_accepted"),
    ("C10", "Thieves", "thieves_prize_accepted"),
    ("C11", "Ice", "ice_prize_accepted"),
    ("C12", "Mire", "mire_prize_accepted"),
    ("C13", "Turtle Rock", "turtle_rock_prize_accepted"),
    ("C14", "GT / Agahnim 2", "gt_agahnim_2_accepted"),
    ("C15", "Ganon + credits", "ganon_credits_accepted"),
)

CLEAN_LADDER: tuple[CleanRung, ...] = tuple(
    CleanRung(rung_id, stop, acceptance,
              LadderStatus.GREEN if rung_id == "C1" else LadderStatus.OPEN)
    for rung_id, stop, acceptance in _ROWS
)


def next_open(ladder: tuple[CleanRung, ...] = CLEAN_LADDER) -> str | None:
    """Id of the first open rung, or None when every row is green.

    The shipped table has C1 green, so this returns ``C2``.
    """
    for row in ladder:
        if row.status is LadderStatus.OPEN:
            return row.id
    return None
