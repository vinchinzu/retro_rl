"""Berry forage / ship phase specs and builders."""

from __future__ import annotations

from typing import List, Optional

from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec

GET_BERRIES_AND_SHIP_PHASE = PhaseSpec(
    "GET_BERRIES_AND_SHIP",
    "recorded",
    # Legacy blind recording — prefer SHIP_BERRY_* multi_nav on the farm.
    # Kept for named sequences / tests that still reference the phase name.
    {"task_name": "get_two_berries_and_ship_after_farm_exit"},
    failure_policy="optional",
)

# y=31 fence (x≈11–29) seals house north-pocket from south berry bush.
# Corridor-only lift opens ≥1 gap so berry_ship BFS can go south (not thrash
# into the wall at ~(336,486)).
OPEN_FENCE_GAP_PHASE = PhaseSpec(
    "OPEN_FENCE_GAP",
    "fence_clear",
    {
        # Berry route only needs the carry-south crossing; it never returns
        # through the gap. One wall post avoids a second north-side re-entry.
        "max_fences": 1,
        "corridor_only": True,
        "timeout": 8000,
    },
    failure_policy="optional",
    required_maps=(0x00,),
    estimated_frames=4000,
    failure_modes=("no_fence", "lift_fail", "timeout"),
)

# Farm forage bush ~(585,920) → shipping bin ~(1001,969). Wild berries are ON
# the farm south of the fence (not mountain / not west exit).
SHIP_BERRY_PHASE = PhaseSpec(
    "SHIP_BERRY",
    "berry_ship",
    {"route": "berry_ship", "timeout": 18000, "initial_settle_frames": 20},
    failure_policy="optional",
    required_maps=(0x00,),
    estimated_frames=10000,
    failure_modes=("bush_unreachable", "bin_path_fail", "fence_sealed", "no_berry_tile"),
)


def ship_berry_phases(*, count: int = 2, open_fence: bool = True) -> list[PhaseSpec]:
    """Fence-gap (if needed) then multi_nav berry_ship loops to the bin."""
    n = max(1, int(count))
    contract = SHIP_BERRY_PHASE.contract
    phases: list[PhaseSpec] = []
    if open_fence:
        phases.append(OPEN_FENCE_GAP_PHASE)
    for i in range(1, n + 1):
        params = dict(SHIP_BERRY_PHASE.params)
        if i > 1:
            params["route"] = "berry_ship_repeat"
        phases.append(
            PhaseSpec(
                f"SHIP_BERRY_{i}" if n > 1 else "SHIP_BERRY",
                SHIP_BERRY_PHASE.kind,
                params,
                failure_policy=SHIP_BERRY_PHASE.failure_policy,
                contract=contract,
            )
        )
    return phases


# Spring D2 house → path fork → first mountain grape/berry. Reactive, not tape.
MOUNTAIN_BERRY_PHASE = PhaseSpec(
    "MOUNTAIN_BERRY",
    "mountain_berry",
    {
        "timeout": 20000,
        "nav_timeout": 12000,
        "approach_only": False,
        "pick_attempts": 3,
        "ship": True,
    },
    failure_policy="optional",
    required_maps=(0x15, 0x00, 0x0C, 0x10),
    estimated_frames=3300,
    failure_modes=("nav_fail", "no_forage", "hands_full", "ship_unverified"),
)

MOUNTAIN_BERRY_PHASES: list[PhaseSpec] = [
    PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
    MOUNTAIN_BERRY_PHASE,
]

BERRY_CUTOFF_HOUR = 15  # latest hour to start a berry run
# Berry forage is independent of seed-shop; a failed bush run must not
# cascade-skip NAV_FARM_EXIT / BUY_SEEDS (wallet still funds potato).
OPTIONAL_BERRY_PHASES = frozenset({
    "BERRY_RUN_WINDOW",
    "LEAVE_FARM_WEST",
    "EXIT_FARM_WEST",
    "BERRY_RECORDING_WINDOW",
    "GET_BERRIES_AND_SHIP",
    "OPEN_FENCE_GAP",
    "SHIP_BERRY",
    "SHIP_BERRY_1",
    "SHIP_BERRY_2",
    "MOUNTAIN_BERRY",
})


def _seed_purchase_cost_g(season: int, day: int) -> int:
    """Gold cost of today's seasonal seed bag (0 when no plantable crop)."""
    from harvest.planner.crop_planner import CROP_SPECS, resolve_seed_type_for_date

    name = resolve_seed_type_for_date(season, day)
    if not name:
        return 0
    crop = CROP_SPECS.get(name)
    return int(crop.seed_cost_g) if crop is not None else 0


def _can_afford_seed_purchase(money: Optional[int], season: int, day: int) -> bool:
    """True when wallet is unknown (tests) or covers the seasonal seed bag."""
    if money is None:
        return True
    cost = _seed_purchase_cost_g(season, day)
    if cost <= 0:
        return False
    return int(money) >= cost


def _berry_run_phases(
    *,
    is_sunday: bool,
    hour: int,
    has_seeds: bool,
    policy: DayPlannerPolicy,
    season: int = 0,
    day: int = 1,
    money: Optional[int] = None,
) -> List[PhaseSpec]:
    """Build early money / seed-shop phases when the hour window allows.

    Priority within this list (Spring D2 empty-farm path):
    1. Mountain berry pick + ship (must hit bin before the 5pm window)
    2. Seed shop only when the wallet covers the bag (potato $200)
    """
    from harvest.core.game_clock import ClockTime
    from harvest.planner.crop_planner import (
        seed_purchase_recording_for_season,
        should_buy_seeds_for_date,
    )
    from harvest.planner.day_phase_catalog import NAV_FARM_EXIT_PHASE, buy_seeds_phase

    now = ClockTime(hour, 0)
    if now.hour >= policy.berry_cutoff_hour and now.hour > policy.buy_seed_hour:
        return []

    phases: List[PhaseSpec] = []
    if policy.include_berry_run and now.hour < policy.berry_cutoff_hour:
        phases.append(
            PhaseSpec(
                "BERRY_RUN_WINDOW",
                "deadline",
                {"latest_hour": policy.berry_exit_cutoff_hour, "latest_minute": 0},
                failure_policy="optional",
            )
        )
        if season == 0 and day == 2:
            phases.append(MOUNTAIN_BERRY_PHASE)
        else:
            phases.extend(ship_berry_phases(count=2))

    can_buy = (
        policy.include_shop_run
        and policy.include_planting
        and not is_sunday
        and not has_seeds
        and hour <= policy.buy_seed_hour
        and should_buy_seeds_for_date(season, day)
        and _can_afford_seed_purchase(money, season, day)
    )
    if can_buy:
        recording = (
            policy.seed_purchase_recording
            or seed_purchase_recording_for_season(season)
            or "buy_potato_seeds"
        )
        phases.extend(
            [
                PhaseSpec(
                    "BUY_SEEDS_WINDOW",
                    "deadline",
                    {"latest_hour": policy.buy_seed_hour + 1, "latest_minute": 0},
                    failure_policy="optional",
                ),
                NAV_FARM_EXIT_PHASE,
                buy_seeds_phase(recording_name=recording),
            ]
        )
    return phases


__all__ = [
    "GET_BERRIES_AND_SHIP_PHASE",
    "OPEN_FENCE_GAP_PHASE",
    "SHIP_BERRY_PHASE",
    "MOUNTAIN_BERRY_PHASE",
    "MOUNTAIN_BERRY_PHASES",
    "ship_berry_phases",
    "BERRY_CUTOFF_HOUR",
    "OPTIONAL_BERRY_PHASES",
    "_berry_run_phases",
]
