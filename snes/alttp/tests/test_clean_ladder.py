"""Clean ladder rows C1..C15. No ROM."""

from __future__ import annotations

import re

from alttp.spine import CLEAN_LADDER, LadderStatus, next_open
from alttp.spine.clean_ladder import CleanRung

_STOPS = (
    "Sanctuary",
    "Eastern",
    "Desert",
    "Hera",
    "Master Sword",
    "Agahnim 1",
    "PoD",
    "Swamp",
    "Skull",
    "Thieves",
    "Ice",
    "Mire",
    "Turtle Rock",
    "GT / Agahnim 2",
    "Ganon + credits",
)
_ACCEPTANCE = (
    "sanctuary_accepted",
    "eastern_prize_accepted",
    "desert_prize_accepted",
    "hera_prize_accepted",
    "master_sword_accepted",
    "agahnim_1_accepted",
    "pod_prize_accepted",
    "swamp_prize_accepted",
    "skull_prize_accepted",
    "thieves_prize_accepted",
    "ice_prize_accepted",
    "mire_prize_accepted",
    "turtle_rock_prize_accepted",
    "gt_agahnim_2_accepted",
    "ganon_credits_accepted",
)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def test_rows_match_phase_map() -> None:
    assert len(CLEAN_LADDER) == 15
    assert [row.id for row in CLEAN_LADDER] == [f"C{n}" for n in range(1, 16)]
    assert [row.stop for row in CLEAN_LADDER] == list(_STOPS)
    assert [row.acceptance for row in CLEAN_LADDER] == list(_ACCEPTANCE)
    assert all(_IDENT.match(row.acceptance) for row in CLEAN_LADDER)
    assert len({row.acceptance for row in CLEAN_LADDER}) == 15


def test_only_proved_sanctuary_rung_is_green() -> None:
    assert CLEAN_LADDER[0].status is LadderStatus.GREEN
    assert all(row.status is LadderStatus.OPEN for row in CLEAN_LADDER[1:])


def test_next_open_returns_c2() -> None:
    assert next_open() == "C2"
    assert CLEAN_LADDER[0].id == "C1"


def test_next_open_skips_a_green_prefix() -> None:
    green = CleanRung("C1", "Sanctuary", "sanctuary_accepted", LadderStatus.GREEN)
    assert next_open((green,) + CLEAN_LADDER[1:]) == "C2"
    assert next_open(()) is None
