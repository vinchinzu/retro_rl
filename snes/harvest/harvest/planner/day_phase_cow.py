"""Barn cow phase specs (primitives only; composed lists live in catalog)."""

from __future__ import annotations

from typing import List

from harvest.planner.day_phase_types import PhaseSpec
from harvest.tasks.cow_geometry import BARN_SHIFT_REASON, BLOCKED_REASON

# ── Barn cow phases ──

NAV_TO_BARN_PHASE = PhaseSpec(
    "NAV_TO_BARN",
    "multi_nav",
    {"route": "farm_to_barn", "timeout": 10000},
    required_maps=(0x00,),
    estimated_frames=800,
)

ENTER_BARN_PHASE = PhaseSpec(
    "ENTER_BARN",
    "directional_transition",
    {
        "direction": "up",
        "origin_tilemap": 0x00,
        "target_tilemap": 0x27,
        "timeout": 900,
        "stand_tile": (20, 22),
        "stand_tolerance": 0,
        "target_stand_tile": (8, 22),
        "target_stand_tolerance": 1,
        "settle_frames": 45,
        "door_align_px": 20 * 16 + 8,
        "overshoot_limit_px": 330,
        "require_empty_hands": True,
    },
    required_maps=(0x00,),
    estimated_frames=120,
)

COW_CHORES_PHASE = PhaseSpec(
    "COW_CHORES",
    "cow_chores",
    {"talk": True, "brush": True, "milk": True, "feed": True},
    required_maps=(0x27,),
    estimated_frames=8000,
    failure_modes=("slot_timeout", "nav_unreachable", "pixel_nav_stall", "sleeping_animal", "no_grass"),
)

# Cure only: no feed, talk, brush or milk, so a stall in one of them cannot hold the dose back.
COW_CURE_PHASE = PhaseSpec(
    "COW_CURE",
    "cow_chores",
    {"talk": False, "brush": False, "milk": False, "feed": False, "medicine": True, "timeout": 6000},
    required_maps=(0x27,),
    estimated_frames=1500,
    failure_modes=("slot_timeout", "nav_unreachable", "pixel_nav_stall", "sleeping_animal"),
)

# A cow that left by the door gets no chores until she is back. Optional: failing costs her one day of care.
COW_ROUNDUP_PHASE = PhaseSpec("COW_ROUNDUP", "cow_roundup", {"timeout": 3000}, required_maps=(0x00,), failure_policy="optional")

EXIT_BARN_PHASE = PhaseSpec(
    "EXIT_BARN",
    "directional_transition",
    {
        "direction": "down",
        "origin_tilemap": 0x27,
        "target_tilemap": 0x00,
        "timeout": 3600,  # two cows can plug a wing for 1,500 frames, and the clock is stopped indoors
        "stand_tile": (8, 22),
        "stand_tolerance": 1,
        "door_align_px": 8 * 16 + 8,
        "settle_frames": 5,
    },
    required_maps=(0x27,),
    estimated_frames=100,
)


# A reset is a ~450-frame exit and entry (the fade is ~170 frames each way) plus the walk to the door, with the clock stopped
# indoors, so it is cheap next to a cow left unfed.
MAX_BARN_RESETS = 16


def splice_barn_reset(schedule, index: int, chores: PhaseSpec, reason: str) -> bool:
    """On blocked walking or incomplete care, re-enter and rerun ``chores``.

    Entering the barn puts every cow back in its stall (ROM probe, Fall D5 and
    D10: a cow parked on the fodder dispenser for 24k frames stood at its stall
    after the round trip). The exit lands on the door stand (20,22) the entry
    starts from, so there is no ``NAV_TO_BARN`` in between (it was a 61-frame settle).
    """
    if not any(marker in str(reason or "") for marker in (BLOCKED_REASON, BARN_SHIFT_REASON, "incomplete cow care", "missing required cow feed")):
        return False
    schedule.splice_at(index, [EXIT_BARN_PHASE, ENTER_BARN_PHASE, chores])
    print(f"[DAY_PLAN] COW_CHORES unfinished ({reason}); re-entering the barn to reset the herd")
    return True


def splice_barn_reentry(schedule, index: int, chores: PhaseSpec, tilemap: int, done: set) -> bool:
    """After a recovery walked the farmer out of the barn, walk back in once.

    A ``COW_CHORES`` that stalls on talk or brush (a cow in the north aisle) is
    recovered by an exit to the farm; the retry then fails the barn map lock and
    ends the day with the herd fed but the crops unwatered (Y2 Summer D13, Fall
    D8). Entering also puts every cow back in its stall. ``done`` is the day's
    one-shot ledger.
    """
    if chores.phase != "COW_CHORES" or "BARN_REENTRY" in done or int(tilemap) != 0x00:
        return False
    done.add("BARN_REENTRY")
    schedule.splice_at(index, [NAV_TO_BARN_PHASE, ENTER_BARN_PHASE, chores])
    print("[DAY_PLAN] COW_CHORES recovered outside the barn; walking back in once")
    return True


BARN_CURRENT_COW_PHASES: List[PhaseSpec] = [
    COW_CHORES_PHASE,
    EXIT_BARN_PHASE,
]

__all__ = [
    "NAV_TO_BARN_PHASE",
    "ENTER_BARN_PHASE",
    "COW_CHORES_PHASE",
    "COW_CURE_PHASE",
    "COW_ROUNDUP_PHASE",
    "EXIT_BARN_PHASE",
    "BARN_CURRENT_COW_PHASES",
    "MAX_BARN_RESETS",
    "splice_barn_reset",
    "splice_barn_reentry",
]
