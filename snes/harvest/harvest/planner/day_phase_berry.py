"""Canonical mountain-grape forage / ship phase specs."""

from __future__ import annotations

from typing import List, Optional

from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec

# House/farm/path → first mountain grape → bin. ``count`` is clipped at
# runtime from the live pose. D2 ships one (bench ~10:10 then shop); D3+
# tries two. A failed second pick still ends SUCCESS once one grape shipped.
_MOUNTAIN_BERRY_PARAMS = {
    "timeout": 20_000,
    "nav_timeout": 12_000,
    "approach_only": False,
    "pick_attempts": 3,
    "ship": True,
    "count": 1,
}


def mountain_berry_count_for_day(day: int) -> int:
    """D2 keeps the one-grape shop window. D3+ asks for two."""
    return 2 if int(day) >= 3 else 1


def shop_latest_hour_for_day(day: int, policy: DayPlannerPolicy) -> int:
    """Latest hour the seed shop may still start.

    D2 uses buy_seed_hour+1 (13). D3+ two-grape days keep 16:00 so a 13:12
    bin toss can still buy potato before the 17:00 shipper.
    """
    latest = int(policy.buy_seed_hour) + 1
    if mountain_berry_count_for_day(day) >= 2 and policy.include_berry_run:
        return max(latest, 16)
    return latest


def mountain_berry_phase(*, count: int = 1) -> PhaseSpec:
    n = max(1, int(count))
    params = dict(_MOUNTAIN_BERRY_PARAMS)
    params["count"] = n
    params["timeout"] = 20_000 if n == 1 else 40_000
    return PhaseSpec(
        "MOUNTAIN_BERRY",
        "mountain_berry",
        params,
        failure_policy="optional",
        required_maps=(0x15, 0x00, 0x0C, 0x10),
        estimated_frames=3300 if n == 1 else 6200,
        failure_modes=("nav_fail", "no_forage", "hands_full", "ship_unverified"),
    )


MOUNTAIN_BERRY_PHASE = mountain_berry_phase(count=1)

MOUNTAIN_BERRY_PHASES: list[PhaseSpec] = [
    PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
    MOUNTAIN_BERRY_PHASE,
]

BERRY_CUTOFF_HOUR = 15  # latest hour to start a berry run
# Berry forage is independent of seed-shop; a failed grape run must not
# cascade-skip NAV_FARM_EXIT / BUY_SEEDS (wallet still funds potato).
OPTIONAL_BERRY_PHASES = frozenset({
    "BERRY_RUN_WINDOW",
    "MOUNTAIN_BERRY",
    # Stale names stay in the skip group so a failed berry window cannot
    # cascade into shop/water when an old sequence still lists them.
    "LEAVE_FARM_WEST",
    "EXIT_FARM_WEST",
    "BERRY_RECORDING_WINDOW",
    "GET_BERRIES_AND_SHIP",
    "OPEN_FENCE_GAP",
    "SHIP_BERRY",
    "SHIP_BERRY_1",
    "SHIP_BERRY_2",
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
    """Mountain grape then seed shop when the hour window allows.

    D2: one grape (lands ~10:10) then shop. D3+: two grapes; shop window
    stays open until 16:00 so a slower second loop can still buy.
    """
    from harvest.core.game_clock import ClockTime
    from harvest.planner.crop_planner import (
        seed_purchase_recording_for_season,
        should_buy_seeds_for_date,
    )
    from harvest.planner.day_phase_catalog import NAV_FARM_EXIT_PHASE, buy_seeds_phase

    now = ClockTime(hour, 0)
    berry_count = mountain_berry_count_for_day(day)
    shop_latest = shop_latest_hour_for_day(day, policy)
    if now.hour >= policy.berry_cutoff_hour and now.hour >= shop_latest:
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
        phases.append(mountain_berry_phase(count=berry_count))

    can_buy = (
        policy.include_shop_run
        and policy.include_planting
        and not is_sunday
        and not has_seeds
        and hour < shop_latest
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
                    {"latest_hour": shop_latest, "latest_minute": 0},
                    failure_policy="optional",
                ),
                NAV_FARM_EXIT_PHASE,
                buy_seeds_phase(recording_name=recording),
            ]
        )
    return phases


__all__ = [
    "MOUNTAIN_BERRY_PHASE",
    "MOUNTAIN_BERRY_PHASES",
    "mountain_berry_count_for_day",
    "shop_latest_hour_for_day",
    "mountain_berry_phase",
    "BERRY_CUTOFF_HOUR",
    "OPTIONAL_BERRY_PHASES",
    "_berry_run_phases",
]
