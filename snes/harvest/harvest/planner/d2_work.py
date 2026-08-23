"""Spring D2 work sections — composable PhaseSpecs for the shop splice.

Product path is grape → shop → these sections → 5pm wait. Do not restore a
morning whole-farm wipe. Quotas are RAM-count contracts, not 800-target
CLEAR_FIELD. Two carry slots: plant is hoe+seeds, water is can, leftover
hammer then axe (never both).

Section order after BUY_SEEDS::

    ENSURE_CROP_SEEDS → CLEAR_PLOT (plot-ring lift)
    → CROP_ESTABLISH (8-ring hoe + plant)
    → ENSURE_WATERING_CAN → CROP_WATER (8 wet)
    leftover (after plant+water, not 06:08 plan-time hour>=17):
      spa? → CLEAR_BUSHES (10) → ENSURE_HAMMER → spa?
      → CLEAR_ROCKS (10 small + 4 large) → ENSURE_AXE → spa?
      → CLEAR_STUMPS (2)

``handoff=quota`` must not use pocket ``plot_ring`` SUCCESS. Spa inserts
when stamina cannot finish an 8-swing 2×2 (do not spa on D2 morning).
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


# RAM-count contracts for leftover smash/lift. Pocket CLEAR_PLOT still
# hands off on the 3x3+stands, not these numbers.
D2_QUOTAS = {
    "plant": 8,
    "water": 8,
    "bushes": 10,
    "small_rocks": 10,
    "large_boulders": 4,
    "stumps": 2,
}

D2_LEFTOVER_PHASE_NAMES = (
    "HOT_SPRING_STAMINA",
    "CLEAR_BUSHES",
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
        "quota_short",
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
            "min_wet": D2_QUOTAS["water"],
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
        estimated_frames=2500,
        failure_modes=("shelf_miss", "carry_full"),
    )


def ensure_axe_phase() -> PhaseSpec:
    return PhaseSpec(
        "ENSURE_AXE",
        "ensure_tool",
        {"tool_id": int(Tool.AXE)},
        failure_policy="optional",
        required_tools=("axe",),
        estimated_frames=2500,
        failure_modes=("shelf_miss", "carry_full"),
    )


def bush_quota_phase() -> PhaseSpec:
    """Lift 10 weeds. No hammer/axe. Not plot-ring handoff."""
    return _optional_clear(
        "CLEAR_BUSHES",
        {
            "timeout": 9000,
            "fetch_tools": False,
            "prefer_lift_for_weeds": True,
            "prefer_lift_for_stones": True,
            "priority": ["weed"],
            "handoff": "quota",
            "quota": {"weeds": D2_QUOTAS["bushes"]},
        },
        estimated_frames=7000,
    )


def rock_quota_phase() -> PhaseSpec:
    """Hammer 10 small rocks (0x06) + 4 large 2×2. Hammer already in carry."""
    return _optional_clear(
        "CLEAR_ROCKS",
        {
            "timeout": 18000,
            "fetch_tools": False,
            "prefer_lift_for_weeds": True,
            "prefer_lift_for_stones": False,
            "priority": ["rock"],
            "handoff": "quota",
            "quota": {
                "small_rocks": D2_QUOTAS["small_rocks"],
                "large_rocks": D2_QUOTAS["large_boulders"],
            },
        },
        required_tools=("hammer",),
        estimated_frames=15000,
    )


def stump_quota_phase() -> PhaseSpec:
    """Axe 2 stumps. Axe already in carry (hammer swapped out)."""
    return _optional_clear(
        "CLEAR_STUMPS",
        {
            "timeout": 12000,
            "fetch_tools": False,
            "priority": ["stump"],
            "handoff": "quota",
            "quota": {"stumps": D2_QUOTAS["stumps"]},
        },
        required_tools=("axe",),
        estimated_frames=8000,
    )


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


def d2_leftover_phases(
    *,
    stamina: Stamina | int | None = None,
    policy: Optional[DayPlannerPolicy] = None,
) -> List[PhaseSpec]:
    """Hammer/axe leftover after plant+water. Spa between smash sections.

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
    phases.append(bush_quota_phase())
    phases.append(ensure_hammer_phase())
    phases.extend(_maybe_spa(stamina, include_spa=include_spa))
    phases.append(rock_quota_phase())
    phases.append(ensure_axe_phase())
    phases.extend(_maybe_spa(stamina, include_spa=include_spa))
    phases.append(stump_quota_phase())
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
    "D2_QUOTAS",
    "bush_quota_phase",
    "d2_leftover_phases",
    "d2_post_shop_work_phases",
    "ensure_axe_phase",
    "ensure_hammer_phase",
    "leftover_already_queued",
    "pocket_clear_phase",
    "pocket_water_phase",
    "rock_quota_phase",
    "stump_quota_phase",
]
