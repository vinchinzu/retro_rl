"""Spring D2 work sections — composable PhaseSpecs for the shop splice.

Product path is grape → shop → these sections → 5pm wait. Two carry
slots: plant is hoe+seeds, water is can, field work is lift work then hammer
then axe (never both).

Section order after BUY_SEEDS::

    ENSURE_CROP_SEEDS → CLEAR_PLOT (plot-ring lift)
    → CROP_ESTABLISH (8-ring hoe + plant)
    → ENSURE_WATERING_CAN → CROP_WATER (8 wet)
    leftover (after plant+water, not 06:08 plan-time hour>=17):
      spa? → CLEAR_BUSHES (10 pick+toss, lanes first) → CLEAR_FENCES
      (all posts to pond) → CLEAR_STONES (all to pond) → ENSURE_HAMMER → spa?
      → CLEAR_ROCKS (all large 2×2) → ENSURE_AXE → spa? → CLEAR_STUMPS (2)

Quota handoffs must not use pocket ``plot_ring`` SUCCESS. Spa inserts when
stamina cannot finish an 8-swing 2×2 (do not spa on D2 morning).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from harvest.core.stamina import Stamina
from harvest.core.tile_catalog import Tool
from harvest.planner.day_phase_catalog import (
    CROP_ESTABLISH_PHASE,
    ENSURE_CROP_SEEDS_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    NAV_CROP_PHASE,
)
from harvest.planner.day_phase_stamina import coerce_stamina, full_restore_spa_phase
from harvest.maps.map_config import WEST_PLANT_POCKET_BOUNDS
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec


# Crop establishment targets are separate from the evening debris quotas.
D2_TARGETS = {
    "plant": 8,
    "water": 8,
}

D2_LEFTOVER_PHASE_NAMES = (
    "HOT_SPRING_STAMINA",
    "CLEAR_BUSHES",
    "CLEAR_FENCES",
    "CLEAR_STONES",
    "ENSURE_HAMMER",
    "CLEAR_ROCKS",
    "ENSURE_AXE",
    "CLEAR_STUMPS",
)


def _optional_clear(
    phase: str,
    params: dict,
    *,
    required_tools: Sequence[str] = (),
    estimated_frames: int = 8000,
    failure_modes: Sequence[str] = (
        "timeout_budget",
        "tool_missing",
        "stamina_low",
        "debris_remaining",
    ),
) -> PhaseSpec:
    return PhaseSpec(
        phase,
        "clear_field",
        params,
        failure_policy="optional",
        required_maps=(0x00,),
        required_tools=tuple(required_tools),
        estimated_frames=estimated_frames,
        failure_modes=tuple(failure_modes),
    )


def pocket_clear_phase() -> PhaseSpec:
    """Lift weeds/stones on the 3x3+stands. Hands off via plot_ring."""
    return PhaseSpec(
        "CLEAR_PLOT",
        "clear_field",
        {
            "timeout": 7000,
            "fetch_tools": False,
            "prefer_lift_for_weeds": True,
            "prefer_lift_for_stones": True,
            "farm_bounds": WEST_PLANT_POCKET_BOUNDS,
            "priority": ["weed", "stone"],
            "handoff": "plot_ring",
        },
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=5000,
        failure_modes=("timeout_budget", "pocket_sealed"),
    )


def pocket_water_phase() -> PhaseSpec:
    """8-ring water from the untilled notch. Can pass, not the plant pair."""
    return PhaseSpec(
        "CROP_WATER",
        "crop",
        {
            "work_mode": "pocket",
            "refill_bounds": (3, 10, 62, 60),
            "min_wet": D2_TARGETS["water"],
        },
        failure_policy="optional",
        required_maps=(0x00,),
        required_tools=("watering_can",),
        estimated_frames=6000,
        failure_modes=("empty_can", "refill_fail", "dry_ring", "precheck_tool_success"),
    )


def ensure_hammer_phase() -> PhaseSpec:
    return PhaseSpec(
        "ENSURE_HAMMER",
        "ensure_tool",
        {"tool_id": int(Tool.HAMMER)},
        failure_policy="optional",
        required_tools=("hammer",),
        estimated_frames=8000,
        failure_modes=("shelf_miss", "carry_full"),
    )


def ensure_axe_phase() -> PhaseSpec:
    return PhaseSpec(
        "ENSURE_AXE",
        "ensure_tool",
        {"tool_id": int(Tool.AXE)},
        failure_policy="optional",
        required_tools=("axe",),
        estimated_frames=8000,
        failure_modes=("shelf_miss", "carry_full"),
    )


def bush_clear_phase() -> PhaseSpec:
    """Lift ten weeds before tool-driven debris."""
    return _optional_clear(
        "CLEAR_BUSHES",
        {
            "timeout": 0,
            "fetch_tools": False,
            "prefer_lift_for_weeds": True,
            "prefer_lift_for_stones": True,
            "priority": ["weed"],
            "quota": {"weeds": 10},
            "handoff": "quota",
        },
        estimated_frames=100000,
    )


def fence_dump_phase() -> PhaseSpec:
    """Lift every fence post and dump it in a pond. Not corridor-only."""
    return PhaseSpec(
        "CLEAR_FENCES",
        "fence_clear",
        {
            "timeout": 0,
            "max_fences": None,
            "corridor_only": False,
            "pond_dump": True,
            "max_steps_per_fence": 2800,
            "max_failures": 20,
            "debris_types": ["fence"],
        },
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=200000,
        failure_modes=("timeout_budget", "no_reachable_fence"),
    )


def stone_pond_phase() -> PhaseSpec:
    """Lift every remaining stone and dump it in a pond. Hammer is for 2×2.

    After_Stumps (axe selected, hoe backpack) still lifts; do not stow first.
    """
    return PhaseSpec(
        "CLEAR_STONES",
        "fence_clear",
        {
            "timeout": 0,
            "max_fences": None,
            "corridor_only": False,
            "pond_dump": True,
            "max_steps_per_fence": 2800,
            "max_failures": 60,
            "debris_types": ["stone"],
        },
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=400000,
        failure_modes=("timeout_budget", "no_reachable_fence"),
    )


def rock_clear_phase() -> PhaseSpec:
    """Hammer every remaining large 2×2 boulder. Quota 4 was the first slice."""
    return _optional_clear(
        "CLEAR_ROCKS",
        {
            "timeout": 0,
            "fetch_tools": False,
            "prefer_lift_for_weeds": True,
            "prefer_lift_for_stones": False,
            "priority": ["rock"],
            "quota": {"large_rocks": 10_000},
            "handoff": "quota",
        },
        required_tools=("hammer",),
        estimated_frames=400000,
    )


def stump_clear_phase() -> PhaseSpec:
    """Axe two distinct stumps. Axe replaces the hammer in carry."""
    return _optional_clear(
        "CLEAR_STUMPS",
        {
            "timeout": 120000,
            "fetch_tools": False,
            "priority": ["stump"],
            "quota": {"stumps": 2},
            "handoff": "quota",
        },
        required_tools=("axe",),
        estimated_frames=90000,
    )


_SPA_RETRY_PHASES = frozenset({"CLEAR_ROCKS", "CLEAR_STUMPS"})


def _maybe_spa(
    stamina: Stamina | int | None,
    *,
    include_spa: bool,
) -> List[PhaseSpec]:
    if not include_spa:
        return []
    stam = coerce_stamina(stamina)
    if stam is None or stam.can_finish_multi_hit():
        return []
    return [full_restore_spa_phase()]


def should_spa_retry(
    phase: str,
    reason: str | None,
    stamina: Stamina | int | None,
    *,
    include_spa: bool,
) -> bool:
    """Insert spa+retry when a smash phase stops on stamina, not aim."""
    if not include_spa or phase not in _SPA_RETRY_PHASES:
        return False
    if "stamina_low" not in (reason or ""):
        return False
    stam = coerce_stamina(stamina)
    return stam is not None and not stam.can_finish_multi_hit()


def needs_spa_before_next_smash(
    just_finished: str,
    stamina: Stamina | int | None,
    *,
    include_spa: bool,
    remaining_phases: Sequence[str],
) -> bool:
    """After rocks, soak if the axe section cannot finish an 8-swing 2×2."""
    if not include_spa or just_finished != "CLEAR_ROCKS":
        return False
    if "CLEAR_STUMPS" not in remaining_phases:
        return False
    stam = coerce_stamina(stamina)
    return stam is not None and not stam.can_finish_multi_hit()


def d2_leftover_phases(
    *,
    stamina: Stamina | int | None = None,
    policy: Optional[DayPlannerPolicy] = None,
) -> List[PhaseSpec]:
    """Lift leftover after plant+water, then hammer/axe. Spa between smash.

    Morning 06:08 ``build_day_phases`` must not attach this (hour<17). The
    shop splice / CROP_WATER splice owns insertion so leftover still runs
    on a 6am plan.
    """
    policy = policy or DayPlannerPolicy()
    if not policy.include_field_clear:
        return []
    include_spa = bool(getattr(policy, "include_spa", True))
    phases: List[PhaseSpec] = []
    phases.extend(_maybe_spa(stamina, include_spa=include_spa))
    phases.append(bush_clear_phase())
    phases.append(fence_dump_phase())
    phases.append(stone_pond_phase())
    phases.append(ensure_hammer_phase())
    phases.extend(_maybe_spa(stamina, include_spa=include_spa))
    phases.append(rock_clear_phase())
    phases.append(ensure_axe_phase())
    phases.extend(_maybe_spa(stamina, include_spa=include_spa))
    phases.append(stump_clear_phase())
    return phases


def d2_post_shop_work_phases(
    *,
    stamina: Stamina | int | None = None,
    policy: Optional[DayPlannerPolicy] = None,
    include_leftover: bool = True,
) -> List[PhaseSpec]:
    """Shed hoe+seeds → pocket clear → 8-plant → 8-water → leftover smash."""
    policy = policy or DayPlannerPolicy()
    phases: List[PhaseSpec] = [
        ENSURE_CROP_SEEDS_PHASE,
        pocket_clear_phase(),
        NAV_CROP_PHASE,
        CROP_ESTABLISH_PHASE,
        ENSURE_WATERING_CAN_PHASE,
        pocket_water_phase(),
    ]
    if include_leftover:
        phases.extend(d2_leftover_phases(stamina=stamina, policy=policy))
    return phases


def leftover_already_queued(remaining: Sequence[str]) -> bool:
    names = set(remaining)
    return any(name in names for name in D2_LEFTOVER_PHASE_NAMES if name != "HOT_SPRING_STAMINA")


__all__ = [
    "D2_LEFTOVER_PHASE_NAMES",
    "D2_TARGETS",
    "bush_clear_phase",
    "d2_leftover_phases",
    "d2_post_shop_work_phases",
    "ensure_axe_phase",
    "ensure_hammer_phase",
    "fence_dump_phase",
    "leftover_already_queued",
    "needs_spa_before_next_smash",
    "pocket_clear_phase",
    "pocket_water_phase",
    "rock_clear_phase",
    "should_spa_retry",
    "stone_pond_phase",
    "stump_clear_phase",
]
