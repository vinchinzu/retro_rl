"""Canonical mountain forage (grape in Spring, green fruit in Summer) phase specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec
from harvest.planner.forage_calendar import SEASON_SPRING, ForageKind, forage_for_name, forage_for_season

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


# A further mountain loop costs ~4 in-game hours, so it may only *start*
# this early; after that it cannot land back at the bin before the 17:00
# ShippingScene. Applies to the second+ loop only — the first one is gated
# by ``policy.berry_cutoff_hour``. 12 is a start-time gate, not an abort.
GRAPE_BAIL_HOUR_NO_SHOP = 12
GRAPE_BAIL_HOUR_SHOP_DAY = 9


@dataclass(frozen=True)
class GrapeDaySpec:
    count: int
    bail_hour: int


def grape_day_spec(
    day: int, *, has_harvest: bool = False, kind: Optional[ForageKind] = None
) -> GrapeDaySpec:
    """Trips and the second-loop bail hour. A pickup tile gives again the same day, so a free
    day takes two loops (``kind.max_per_day``); D2 of Spring keeps one for the shop window."""
    cap = 2 if kind is None else kind.max_per_day
    if int(day) < 3 and (kind is None or kind.season == SEASON_SPRING):
        return GrapeDaySpec(1, 10)
    if has_harvest:
        return GrapeDaySpec(1, GRAPE_BAIL_HOUR_SHOP_DAY)  # 9
    return GrapeDaySpec(min(2, cap), GRAPE_BAIL_HOUR_NO_SHOP)  # 12


def mountain_berry_count_for_day(day: int, *, has_harvest: bool = False) -> int:
    return grape_day_spec(day, has_harvest=has_harvest).count


def grape_bail_hour_for_day(day: int, *, has_harvest: bool = False) -> int:
    return grape_day_spec(day, has_harvest=has_harvest).bail_hour


def shop_latest_hour_for_day(
    day: int,
    policy: DayPlannerPolicy,
    *,
    has_harvest: bool = False,
    kind: Optional[ForageKind] = None,
) -> int:
    """Latest hour the seed shop may still start.

    D2 uses buy_seed_hour+1 (13). D3+ two-grape days keep 16:00 so a 13:12
    bin toss can still buy potato before the 17:00 shipper.
    """
    latest = int(policy.buy_seed_hour) + 1
    if (
        grape_day_spec(day, has_harvest=has_harvest, kind=kind).count >= 2
        and policy.include_berry_run
    ):
        return max(latest, 16)
    return latest


def mountain_berry_phase(
    *,
    count: int = 1,
    shop_bail_hour: Optional[int] = None,
    forage: str = "grape",
) -> PhaseSpec:
    n = max(1, int(count))
    kind = forage_for_name(forage)
    params = dict(_MOUNTAIN_BERRY_PARAMS)
    params["count"] = n
    params["timeout"] = 20_000 if n == 1 else 40_000
    if forage != "grape":
        params["forage"] = forage
    if shop_bail_hour is not None:
        params["shop_bail_hour"] = int(shop_bail_hour)
    return PhaseSpec(
        "MOUNTAIN_BERRY",
        "mountain_berry",
        params,
        failure_policy="optional",
        required_maps=(0x15, 0x00, 0x0C, 0x10),
        estimated_frames=kind.loop_frames if n == 1 else kind.loop_frames * 2 - 400,
        failure_modes=("nav_fail", "no_forage", "hands_full", "ship_unverified"),
    )


MOUNTAIN_BERRY_PHASE = mountain_berry_phase(count=1)

MOUNTAIN_BERRY_PHASES: list[PhaseSpec] = [
    PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
    MOUNTAIN_BERRY_PHASE,
]

BERRY_CUTOFF_HOUR = 15  # latest hour to start a berry run
# Year 3 plays for the score: its forage gold is long since capped, but a crop day is not. y3e Summer D1 ran two
# loops to 14:00, left for the shop at 15:13 and reached a closed door at 17:05 (the walk is ~1h50, the door shuts
# at 17:00); Saturday and Sunday are shut and Monday went to grass seed, so corn went in on D7 and tomatoes never.
SHOP_FIRST_FROM_YEAR = 2
SHOP_FIRST_LATEST_HOUR = 14
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
    has_plant_capacity: bool = True,
    has_harvest: bool = False,
    is_festival: bool = False,
    year: int = 0,
) -> List[PhaseSpec]:
    """Mountain grape then seed shop when the hour window allows.

    D2: one grape (lands ~10:10) then shop. D3+ restock: two grapes then
    shop (ROM 13:12 / 16:08). Harvest mornings ask for one grape as an
    option after crop work, not a forced second loop.
    """
    from harvest.core.game_clock import ClockTime
    from harvest.planner.crop_planner import (
        seed_purchase_recording_for_season,
        should_buy_seeds_for_date,
    )
    from harvest.planner.day_phase_catalog import NAV_FARM_EXIT_PHASE, buy_seeds_phase

    now = ClockTime(hour, 0)
    kind = forage_for_season(season)
    spec = grape_day_spec(day, has_harvest=has_harvest, kind=kind)
    shop_latest = shop_latest_hour_for_day(day, policy, has_harvest=has_harvest, kind=kind)
    shop_first = int(year) >= SHOP_FIRST_FROM_YEAR and int(season) in (0, 1)
    if shop_first:
        shop_latest = min(shop_latest, SHOP_FIRST_LATEST_HOUR)
    if now.hour >= policy.berry_cutoff_hour and now.hour >= shop_latest:
        return []

    can_buy = (
        policy.include_shop_run
        and policy.include_planting
        and has_plant_capacity
        and not is_sunday
        and not is_festival
        and not has_seeds
        and hour < shop_latest
        and should_buy_seeds_for_date(season, day)
        and _can_afford_seed_purchase(money, season, day)
    )

    phases: List[PhaseSpec] = []
    forages = policy.include_berry_run and now.hour < policy.berry_cutoff_hour and not (shop_first and can_buy)
    if forages:
        phases.append(
            PhaseSpec(
                "BERRY_RUN_WINDOW",
                "deadline",
                {"latest_hour": policy.berry_exit_cutoff_hour, "latest_minute": 0},
                failure_policy="optional",
            )
        )
        phases.append(
            mountain_berry_phase(
                count=spec.count,
                shop_bail_hour=spec.bail_hour,
                forage=kind.name if kind is not None else "grape",
            )
        )

    if can_buy:
        target_crop = (
            policy.summer_crop
            if season == 1
            else (policy.fall_crop if season == 2 else policy.spring_crop)
        )
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
                buy_seeds_phase(recording_name=recording, seed_type=target_crop),
            ]
        )
    return phases


__all__ = [
    "MOUNTAIN_BERRY_PHASE",
    "MOUNTAIN_BERRY_PHASES",
    "GrapeDaySpec",
    "grape_day_spec",
    "mountain_berry_count_for_day",
    "shop_latest_hour_for_day",
    "grape_bail_hour_for_day",
    "GRAPE_BAIL_HOUR_NO_SHOP",
    "GRAPE_BAIL_HOUR_SHOP_DAY",
    "mountain_berry_phase",
    "BERRY_CUTOFF_HOUR",
    "OPTIONAL_BERRY_PHASES",
    "_berry_run_phases",
]
