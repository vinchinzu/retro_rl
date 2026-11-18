"""Dynamic day-plan builders and phase catalog re-exports."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import List, Optional
import numpy as np
from harvest.core.tile_catalog import Tool
from harvest.planner.day_phase_types import (
    DayPlannerPolicy,
    PhaseKind,
    PhaseSpec,
    day_planner_policy_for_season,
)
from harvest.planner.world_probe import WorldProbe
from harvest.planner.day_plan_status import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    BARN_TILEMAP,
    COOP_TILEMAP,
    is_farm_tilemap,
    is_house_tilemap,
    SHOP_CLOSED_WEEKDAYS,
    SUNDAY_WEEKDAY,
)
from harvest.core.ram_catalog import read_ram_u8, read_ram_value
from harvest.planner.livestock_econ import (
    animal_shop_errand,
    cow_tool_trip_affordable,
    hatch_allowed,
    kept_adults,
    tool_shop_open,
    weather_blocks_animal_delivery,
)
from harvest.core.stamina import Stamina
from harvest.planner.cow_cure import cure_day_phases
from harvest.planner.day_phase_stamina import evening_clear_phases
from harvest.planner.day_phase_berry import _berry_run_phases
from harvest.planner.forage_calendar import (
    forage_first_today,
    forage_for_season,
    loop_hours,
    trips_today,
)
from harvest.planner.day_phase_calendar import (
    ATTEND_FESTIVAL_PHASE,
    MEET_HORSE_PHASE,
    horse_event_due,
    SEASON_WINTER,
    REPAIR_CROPS_PHASE,
    SUNDAY_CHURCH_PHASE,
    SUNDAY_MOUNTAIN_PHASE,
    attend_festival_phase,
    default_crop_for_season,
    festival_name_for_date,
    is_festival_day,
    is_storm_day,
    is_storm_weather,
    next_crop_in_rotation,
    post_storm_recovery_phases,
    seasonal_crops_for_season,
)
from harvest.planner.day_phase_chicken import (
    _chicken_oversupplied,
    _chicken_phases,
    _chicken_sale_phases,
    _coop_current_phases,
)
from harvest.planner.day_phase_catalog import (
    EXIT_HOUSE_PHASE,
    EXIT_TO_FARM_PHASE,
    LEAVE_HOUSE_TO_FARM_PHASE,
    NAV_FARM_EXIT_PHASE,
    LEAVE_FARM_WEST_PHASE,
    EXIT_FARM_WEST_PHASE,
    BUY_SEEDS_PHASE,
    buy_seeds_phase,
    pasture_day_phases,
    animal_purchase_phases,
    cow_day_phases,
    tool_purchase_phases,
    MOUNTAIN_BERRY_PHASE,
    NAV_CROP_PHASE,
    HARVEST_ROUTE_PHASE,
    CLEAR_FIELD_PHASE,
    CLEAR_BUSHES_PHASE,
    CLEAR_PHASES,
    DYNAMIC_OUTDOOR_PLAN_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    ENSURE_CROP_SEEDS_PHASE,
    ENSURE_ANIMAL_TOOLS_PHASE,
    ENSURE_MILKER_PHASE,
    CROP_ESTABLISH_PHASE,
    CROP_WATER_PHASE,
    HOT_SPRING_STAMINA_PHASE,
    RETURN_HOME_PHASE,
    GO_TO_SLEEP_PHASE,
    TOWN_EXPLORE_PHASE,
    READY_TO_GO_HOME_PHASE,
    GET_HAMMER_MACRO_PHASE,
    GET_AXE_MACRO_PHASE,
    GET_SICKLE_MACRO_PHASE,
    LEAVE_HOUSE_MACRO_PHASE,
    BOOT_TO_DAY2_PHASES,
    GO_HOME_TRIGGER_PHASES,
    EVE_TALK_LOOP_PHASE,
    EVE_TALK_LOOP_PHASES,
    NAV_TO_COOP_PHASE,
    NAV_BARN_TO_COOP_PHASE,
    ENTER_COOP_PHASE,
    COOP_CHORES_PHASE,
    COOP_GIFT_PHASE,
    EXIT_COOP_PHASE,
    NAV_TO_COOP_FOR_CHICKEN_SALE_PHASE,
    ENTER_COOP_FOR_CHICKEN_SALE_PHASE,
    PICKUP_CHICKEN_FOR_SALE_PHASE,
    EXIT_COOP_FOR_CHICKEN_SALE_PHASE,
    DROP_CHICKEN_FOR_SALE_PHASE,
    NAV_TO_ANIMAL_SHOP_FOR_CHICKEN_SALE_PHASE,
    REQUEST_CHICKEN_SALE_PHASE,
    EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_PHASE,
    RETURN_FARM_AFTER_CHICKEN_SALE_PHASE,
    SELL_CHICKEN_PHASE,
    CHICKEN_SALE_BATCH_DROP_POINTS,
    chicken_sale_stage_phases,
    chicken_sale_cycle_phases,
    chicken_sale_batch_phases,
    CHICKEN_PHASES,
    CHICKEN_AFTER_BARN_PHASES,
    COOP_CURRENT_PHASES,
    NAV_TO_BARN_PHASE,
    ENTER_BARN_PHASE,
    COW_CHORES_PHASE,
    EXIT_BARN_PHASE,
    COW_PHASES,
    BARN_CURRENT_COW_PHASES,
    BUY_COW_PHASES,
    BUY_COW_FIRST_PHASES,
    DAY1_PHASES,
    SPRING4_PHASES,
    BERRIES_WATER_PHASES,
    SUNDAY_PHASES,
    RESUME_WATER_PHASES,
    HARVEST_PHASES,
    SELL_CHICKEN_TEST_PHASES,
    SELL_THREE_CHICKENS_TEST_PHASES,
    SELL_THREE_CHICKENS_BATCH_TEST_PHASES,
    PHASE_SEQUENCES,
    PHASE_SEQUENCE,
    BERRY_CUTOFF_HOUR,
    OPTIONAL_MONEY_PHASES,
    crop_establish_phases,
    crop_water_phases,
    pocket_clear_phase,
    pocket_plant_phases,
    _crop_work_phases,
)

def _uses_d2_exhaustive_clear(
    season: int | None, day: int | None, year: int = 0
) -> bool:
    """Year-1 S0D2 folds grape/shop/clear/plant/water into one reactive tactic.

    Year 2+ Spring D2 is an ordinary day: the intro tactic assumes the
    starting wilderness and a hand-to-mouth wallet.
    """
    return int(year or 0) == 0 and int(season or 0) == 0 and int(day or 0) == 2

def _clear_after_shop(
    policy: DayPlannerPolicy, season: int | None, day: int | None, year: int = 0
) -> bool:
    """Post-shop clear: the D2 tactic is required even when daily CLEAR_FIELD is off."""
    return bool(policy.include_field_clear or _uses_d2_exhaustive_clear(season, day, year))

def _daytime_clear_phase(
    season: int | None, day: int | None, year: int = 0
) -> PhaseSpec:
    """D2 uses the exhaustive tactic; other days keep quota CLEAR_FIELD."""
    if _uses_d2_exhaustive_clear(season, day, year):
        from harvest.planner.d2_work import d2_farm_clear_phase

        return d2_farm_clear_phase()
    return CLEAR_FIELD_PHASE

def _planting_today(
    *,
    has_seeds: bool,
    buying_seeds: bool,
    season: int | None,
    day: int | None,
    late_day: bool,
    policy: DayPlannerPolicy,
    has_plant_capacity: bool = True,
    year: int = 0,
) -> bool:
    """Establish when a plot can receive owned seeds or today's purchase; D2 plants in its tactic."""
    if late_day or not policy.include_planting or not has_plant_capacity:
        return False
    if has_seeds:
        return True
    return buying_seeds and not _uses_d2_exhaustive_clear(season, day, year)

def _calendar_gates(
    *,
    weekday: int,
    hour: int,
    season: int,
    day: int,
    year: int,
    is_festival: Optional[bool],
    policy: DayPlannerPolicy,
) -> tuple[DayPlannerPolicy, bool, bool, bool, bool, bool]:
    """Season policy plus sunday/festival shop closure.

    Returns ``(policy, is_sunday, is_fest, shops_closed, late_day, festival_today)``.
    ``policy`` is the off-season copy (no fall/winter field work, berries only
    in year-1 spring, no winter shop). ``is_fest`` also requires
    ``include_festivals``; ``festival_today`` does not.
    """
    policy = day_planner_policy_for_season(season, policy, year=year)
    is_sunday = weekday == SUNDAY_WEEKDAY
    festival_today = (
        is_festival_day(season, day, year) if is_festival is None else bool(is_festival)
    )
    is_fest = bool(festival_today and policy.include_festivals)
    shops_closed = is_sunday or is_fest or weekday in SHOP_CLOSED_WEEKDAYS
    late_day = hour >= policy.late_water_hour
    return policy, is_sunday, is_fest, shops_closed, late_day, festival_today

def build_day_phases(
    state_name: Optional[str] = None,
    *,
    weekday: Optional[int] = None,
    hour: Optional[int] = None,
    season: Optional[int] = None,
    day: Optional[int] = None,
    year: Optional[int] = None,
    has_chickens: Optional[bool] = None,
    adult_chickens: Optional[int] = None,
    has_cows: Optional[bool] = None,
    has_harvest: Optional[bool] = None,
    has_waterable: Optional[bool] = None,
    has_seeds: Optional[bool] = None,
    has_debris: Optional[bool] = None,
    should_buy_cow: Optional[bool] = None,
    is_rainy: Optional[bool] = None,
    money: Optional[int] = None,
    stamina: Optional[Stamina | int] = None,
    has_plant_capacity: Optional[bool] = None,
    is_storm: Optional[bool] = None,
    is_post_storm: Optional[bool] = None,
    is_festival: Optional[bool] = None,
    festival_name: Optional[str] = None,
    weather_code: Optional[int] = None,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
) -> List[PhaseSpec]:
    """Assemble a day's phase list dynamically from state inspection.

    Explicit keyword overrides let callers (and tests) control each flag
    without needing a real save state.
    """
    # ── Inspect state for defaults ──
    if state_name is not None:
        probe = WorldProbe.from_inputs(state_name=state_name)
        if weekday is None:
            weekday = probe.weekday()
        if hour is None:
            hour = probe.day_time()[1]
        if season is None or day is None:
            probe_season, probe_day = probe.calendar_date()
            if season is None:
                season = probe_season
            if day is None:
                day = probe_day
        if year is None:
            year = probe.year()
        if has_chickens is None:
            has_chickens = probe.needs_chicken_chores()
        if adult_chickens is None:
            adult_chickens = probe.chicken_counts()[0]
        if has_cows is None:
            has_cows = probe.needs_cow_chores()
        if should_buy_cow is None:
            should_buy_cow = probe.should_buy_cow()
        if has_harvest is None:
            has_harvest = probe.has_harvestable_crops()
        if has_waterable is None:
            has_waterable = probe.has_waterable_crops()
        if has_seeds is None:
            has_seeds = probe.has_seasonal_plantable_seeds()
        if has_debris is None:
            has_debris = probe.has_farm_debris()
        if is_rainy is None:
            is_rainy = probe.is_rainy()
        if money is None:
            money = probe.money()
        if stamina is None:
            stamina = probe.stamina()
        if has_plant_capacity is None and probe.source_ram is not None:
            has_plant_capacity = probe.pocket_has_plant_capacity()
        if is_storm is None:
            is_storm = probe.is_storm()
        if is_festival is None:
            is_festival = probe.is_festival()

    # Fill remaining defaults
    if has_plant_capacity is None:
        has_plant_capacity = True
    if weekday is None:
        weekday = 1
    if hour is None:
        hour = 6
    if season is None:
        season = 0
    if day is None:
        day = 1
    if year is None:
        year = 0
    if has_chickens is None:
        has_chickens = False
    if adult_chickens is None:
        adult_chickens = 0
    if has_cows is None:
        has_cows = False
    if should_buy_cow is None:
        should_buy_cow = False
    if has_harvest is None:
        has_harvest = False
    if has_waterable is None:
        has_waterable = False
    if has_seeds is None:
        has_seeds = False
    if has_debris is None:
        has_debris = False
    if is_rainy is None:
        is_rainy = False
    if is_storm is None:
        if weather_code is not None:
            is_storm = is_storm_day(season, weather_code=weather_code)
        else:
            is_storm = False
    if is_post_storm is None:
        is_post_storm = False
    policy, _is_sunday, is_fest, shops_closed, late_day, festival_today = _calendar_gates(
        weekday=weekday,
        hour=hour,
        season=season,
        day=day,
        year=year,
        is_festival=is_festival,
        policy=policy,
    )
    if festival_today and festival_name is None:
        festival_name = festival_name_for_date(season, day, year)
    if is_storm:
        # Severe storm (hurricane/blizzard): farmer cannot leave the house.
        if policy.include_end_day:
            return [GO_TO_SLEEP_PHASE]
        return []

    buy_cow_first = (
        policy.include_cows
        and policy.include_animal_purchases
        and policy.include_shop_run
        and should_buy_cow
        and not late_day
        and not shops_closed
    )
    phases: List[PhaseSpec] = []
    if buy_cow_first:
        phases.extend(BUY_COW_FIRST_PHASES)
    else:
        phases.append(EXIT_TO_FARM_PHASE)

    # Livestock sits between morning clear and harvest; the field list is outdoor.
    animal_phases: List[PhaseSpec] = []
    if policy.include_cows and has_cows and not late_day and not buy_cow_first:
        animal_phases.extend(COW_PHASES)
        exited_barn = True
    else:
        exited_barn = buy_cow_first
    if policy.include_chickens and has_chickens and not late_day:
        animal_phases.extend(
            _chicken_phases(
                exited_barn=exited_barn,
                oversupplied=_chicken_oversupplied(adult_chickens, policy),
                policy=policy,
            )
        )
    animal_phases.extend(
        _chicken_sale_phases(
            adult_chickens=adult_chickens,
            hour=hour,
            is_sunday=shops_closed,
            policy=policy,
            is_festival=is_fest,
        )
    )
    phases.extend(
        build_outdoor_day_phases(
            weekday=weekday,
            hour=hour,
            has_harvest=has_harvest,
            has_waterable=has_waterable,
            has_seeds=has_seeds,
            has_debris=has_debris,
            is_rainy=is_rainy,
            season=season,
            day=day,
            year=year,
            money=money,
            stamina=stamina,
            has_plant_capacity=has_plant_capacity,
            is_post_storm=is_post_storm,
            is_festival=festival_today,
            has_chickens=has_chickens,
            has_cows=has_cows,
            between_clear_and_harvest=animal_phases,
            policy=policy,
        )
    )
    return phases

def build_outdoor_day_phases(
    *,
    weekday: int,
    hour: int,
    has_harvest: bool,
    has_waterable: bool,
    has_seeds: bool,
    has_debris: bool = False,
    is_rainy: bool = False,
    season: int = 0,
    day: int = 1,
    year: int = 0,
    money: Optional[int] = None,
    stamina: Optional[Stamina | int] = None,
    has_plant_capacity: bool = True,
    is_storm: bool = False,
    is_post_storm: bool = False,
    is_festival: Optional[bool] = None,
    has_chickens: bool = False,
    has_cows: bool = False,
    between_clear_and_harvest: Optional[List[PhaseSpec]] = None,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
) -> List[PhaseSpec]:
    """Assemble the outdoor portion of the day's work from current farm state.

    ``has_chickens`` / ``has_cows`` hold berries behind chores (full-day plan).
    ``between_clear_and_harvest`` is spliced after the morning clear; livestock
    from ``build_day_phases`` goes there. Both default to an outdoor-only list.
    """
    if is_storm:
        return []
    if _uses_d2_exhaustive_clear(season, day, year):
        # Power-on D2 is all debris, so no ring looks plantable. The tactic
        # clears those pockets itself; skipping the shop left spring grape-only.
        has_plant_capacity = True
    policy, is_sunday, is_fest, shops_closed, late_day, _festival_today = _calendar_gates(
        weekday=weekday,
        hour=hour,
        season=season,
        day=day,
        year=year,
        is_festival=is_festival,
        policy=policy,
    )
    berry_phases = _berry_run_phases(
        is_sunday=shops_closed,
        hour=hour,
        has_seeds=has_seeds,
        policy=policy,
        season=season,
        day=day,
        money=money,
        has_plant_capacity=has_plant_capacity,
        has_harvest=has_harvest,
        is_festival=is_fest,
        year=year,
    )
    phases: List[PhaseSpec] = []

    # Post-storm recovery: morning debris clearing and fence repair
    if is_post_storm and not late_day:
        phases.append(_daytime_clear_phase(season, day, year))
        phases.append(REPAIR_CROPS_PHASE)

    seed_buy_phases = [
        phase
        for phase in berry_phases
        if phase.phase in {"BUY_SEEDS_WINDOW", "NAV_FARM_EXIT", "BUY_SEEDS"}
    ]
    other_berry_phases = [
        phase for phase in berry_phases if phase not in seed_buy_phases
    ]
    plant_intent = _planting_today(
        has_seeds=has_seeds,
        buying_seeds=bool(seed_buy_phases),
        season=season,
        day=day,
        late_day=late_day,
        policy=policy,
        has_plant_capacity=has_plant_capacity,
        year=year,
    )

    # Early money (berries) before optional field wipe when nothing needs
    # keep-alive water. Full day clear burns the shipping window on bush thrash.
    defer_field_clear = bool(
        policy.include_field_clear
        and has_debris
        and not late_day
        and has_waterable
        and not is_rainy
        and policy.include_watering
    )
    # An empty field still gives the morning to forage after the animals.
    # The buy gate already proves affordability. A D3+ grape round trip can
    # consume the whole shop window; its deposit is paid later, not at pickup.
    shop_before_forage = bool(seed_buy_phases and season == 0 and year == 0 and day is not None and day >= 3)
    from harvest.planner import run_strategy

    if (
        seed_buy_phases and not shop_before_forage and run_strategy.field_cover()
        # Opt-in after the shop-first A/B lost total points (Y1_BENCH.md).
        and os.environ.get("HARVEST_SHOP_FIRST", "0").strip().lower() in {"1", "true", "yes"}
    ):
        # Cover plus forage once pushed the crop-seed trip to the shut 17:11 door.
        shop_before_forage = bool(season in (0, 1) and not late_day)
    berry_before_clear = bool(
        not shop_before_forage
        and not late_day
        and other_berry_phases
        and not has_waterable
        and not has_harvest
        and not has_chickens
        and not has_cows
    )
    # First-Spring restocks already have bag money and keep the shop window.
    restock_berries_first = bool(
        not shop_before_forage
        and not berry_before_clear
        and not late_day
        and seed_buy_phases
        and other_berry_phases
        and not has_harvest
    )
    # Forage is free money while the morning has nothing to ship: watering, sowing and clearing
    # carry on after 18:00 (the clock freezes), but a ripe crop pays only if it reaches the bin
    # before 17:00. So with no harvest the trip goes ahead of the crop work (after the shop on a
    # restock day) instead of being starved by it, which left two grape trips in all of Spring.
    forage_after_shop = bool(
        not (berry_before_clear or restock_berries_first)
        and other_berry_phases
        and forage_first_today(
            kind=forage_for_season(season),
            hour=hour,
            has_harvest=has_harvest,
            late_day=late_day,
            is_post_storm=is_post_storm,
            is_festival=bool(is_fest),
        )
    )
    early_berries = berry_before_clear or restock_berries_first or forage_after_shop
    if berry_before_clear:
        phases.extend(other_berry_phases)
        phases.extend(seed_buy_phases)
        if seed_buy_phases and has_debris and _clear_after_shop(policy, season, day, year):
            phases.append(_daytime_clear_phase(season, day, year))
    elif seed_buy_phases:
        if restock_berries_first:
            phases.extend(other_berry_phases)
        phases.extend(seed_buy_phases)
        if forage_after_shop:
            phases.extend(other_berry_phases)
    elif forage_after_shop:
        phases.extend(other_berry_phases)

    day_clear = bool(
        policy.include_field_clear
        and has_debris
        and not late_day
        and not defer_field_clear
        and not berry_before_clear
        and not is_post_storm
    )
    if day_clear:
        phases.append(_daytime_clear_phase(season, day, year))

    if between_clear_and_harvest:
        phases.extend(between_clear_and_harvest)

    if policy.include_harvest and has_harvest and not late_day:
        phases.append(NAV_CROP_PHASE)
        phases.append(HARVEST_ROUTE_PHASE)

    phases.extend(
        _crop_work_phases(
            has_harvest=has_harvest,
            has_waterable=has_waterable,
            has_seeds=plant_intent,
            is_rainy=is_rainy,
            late_day=late_day,
            policy=policy,
        )
    )

    if defer_field_clear and not berry_before_clear:
        phases.append(_daytime_clear_phase(season, day, year))

    # 4. Festival attendance: on festival days, farm work happens early, then attend festival
    if is_fest and not late_day and policy.attend_festivals:
        festival = attend_festival_phase(season, day, year)
        phases.append(festival)
        if policy.include_end_day and not festival.params.get("ends_day"):
            phases.append(RETURN_HOME_PHASE)
            phases.append(GO_TO_SLEEP_PHASE)
        return phases

    # 4b. Sunday church visit and mountain foraging
    if is_sunday and not late_day:
        if policy.include_sunday_church:
            phases.append(SUNDAY_CHURCH_PHASE)
        if policy.include_sunday_mountain:
            phases.append(SUNDAY_MOUNTAIN_PHASE)

    # Berries after crop work when keep-alive / harvest claimed the morning
    # (skip when a restock day already ran them first).
    if not late_day and not early_berries:
        phases.extend(other_berry_phases)

    if late_day:
        phases.extend(
            evening_clear_phases(
                has_debris=has_debris,
                late_day=True,
                policy=policy,
                stamina=stamina,
                season=season,
                day=day,
            )
        )
        if policy.include_end_day:
            phases.append(RETURN_HOME_PHASE)
            phases.append(GO_TO_SLEEP_PHASE)

    return phases

def build_outdoor_day_phases_from_ram(
    ram: np.ndarray,
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
    state_name: Optional[str] = None,
    is_post_storm: bool = False,
) -> List[PhaseSpec]:
    """Inspect live farm RAM and build only the outdoor work that remains."""
    probe = WorldProbe.from_inputs(ram=ram, state_name=state_name)
    _calendar_day, hour, _minute = probe.day_time()
    season, day = probe.calendar_date()
    phases = build_outdoor_day_phases(
        weekday=1 if probe.weekday() is None else probe.weekday(),
        hour=hour,
        has_harvest=probe.has_harvestable_crops(),
        has_waterable=probe.has_waterable_crops(),
        has_seeds=probe.has_seasonal_plantable_seeds(),
        has_debris=probe.has_farm_debris(),
        is_rainy=probe.is_rainy(),
        season=season,
        day=day,
        year=probe.year(),
        money=probe.money(),
        stamina=probe.stamina(),
        has_plant_capacity=probe.pocket_has_plant_capacity(),
        is_storm=probe.is_storm(),
        is_post_storm=is_post_storm,
        is_festival=probe.is_festival(),
        policy=policy,
    )
    pasture = []
    if not probe.is_storm() and policy.include_pasture:
        pasture = pasture_day_phases(ram, afternoon=hour >= 12)
        from harvest.planner.run_strategy import active

        # A ripe crop has a 17:00 deadline; pasture work continues after shipping closes.
        at = next((i + 1 for i, p in enumerate(phases) if p.phase == "HARVEST_ROUTE"), 0) if active().caps_shipping else 0
        phases[at:at] = pasture
    kind = forage_for_season(season)
    if (
        season == 2
        and policy.include_mushroom_gather
        and kind is not None
        and not probe.is_storm()
        and not probe.is_festival()
    ):
        trips = trips_today(kind, hour, probe.day_time()[2])
        if trips > 0:
            # Fall grows nothing, so the trips go first (after the grass errand): a loop is ~2,400 frames
            # (~3 game hours) and each poison mushroom ships for 200G. y1new stopped at 13:00 after two
            # loops on most days; a loop started before 14:00 still lands before the bin closes.
            phases[len(pasture):len(pasture)] = [PhaseSpec(
                "MUSHROOM_GATHER", "mountain_berry",
                {"ship": True, "forage": kind.name, "count": trips, "shop_bail_hour": 14,
                 "loop_hours": loop_hours(kind), "timeout": 12_000 * trips},
                failure_policy="optional",
            )]
    return phases

def build_day_phases_from_ram(
    ram: np.ndarray,
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
    state_name: Optional[str] = None,
    is_post_storm: bool = False,
) -> List[PhaseSpec]:
    """Assemble a day's phase list directly from live RAM."""
    probe = WorldProbe.from_inputs(ram=ram, state_name=state_name)
    season, _calendar_day = probe.calendar_date()
    policy = day_planner_policy_for_season(season, policy, year=probe.year())
    policy = replace(
        policy,
        hold_flock_growth=not hatch_allowed(ram, reserve_first_cow=policy.include_cows),
        max_adult_chickens=max(policy.max_adult_chickens, kept_adults(ram)),
    )
    if probe.is_storm():
        return [GO_TO_SLEEP_PHASE] if policy.include_end_day else []
    _day, hour, _minute = probe.day_time()
    tilemap = probe.tilemap() or 0
    on_farm = is_farm_tilemap(tilemap)
    in_barn = tilemap == BARN_TILEMAP
    in_coop = tilemap == COOP_TILEMAP
    late_day = hour >= policy.late_water_hour
    if is_house_tilemap(tilemap) and late_day and policy.include_end_day:
        from harvest.tasks.proposal import apply_eve_phases

        return apply_eve_phases([GO_TO_SLEEP_PHASE], ram)
    from harvest.planner import power_berry_calendar
    if season == SEASON_WINTER and power_berry_calendar.due_source(ram) is None and not (
        (policy.include_cows and probe.needs_cow_chores())
        or (policy.include_chickens and probe.needs_chicken_chores())
    ):
        # Nothing grows, sells or plants in Winter: an empty plan goes straight
        # to the bed (see MultiDayPlannerTask._build_sleep_task).
        return []
    adult_chickens = probe.chicken_counts()[0]
    oversupplied_chickens = _chicken_oversupplied(adult_chickens, policy)
    probe_weekday = probe.weekday()
    is_sunday = probe_weekday == SUNDAY_WEEKDAY
    is_fest = probe.is_festival() and policy.include_festivals
    shops_closed = is_sunday or is_fest or probe_weekday in SHOP_CLOSED_WEEKDAYS
    cows_need_chores = policy.include_cows and not late_day and probe.needs_cow_chores()
    chickens_need_chores = (
        policy.include_chickens and not late_day and probe.needs_chicken_chores()
    )
    animal_tools_ready = _animal_tools_ready(ram)
    species = animal_shop_errand(ram, cows=policy.include_cows, chickens=policy.include_chickens)
    buy_animal = bool(
        species
        and policy.include_animal_purchases
        and policy.include_shop_run
        and not late_day
        and not shops_closed
    )
    # The brush and milker come from the general store once a cow exists. Calves
    # take 21 nights to give milk, so the trip waits for a morning with the gold.
    # A scheduled power berry trip (statue, fish, cave, hawker, festivals): one source a day. It yields to an animal
    # purchase (5,000G, rare) but not to the brush / milker / sprinkler trip, which simply waits a day: on the old
    # cash-run Fall pins that trip was planned every weekday and would have starved the hen trade all season.
    from harvest.planner import power_berry_calendar, social_calendar

    berry_source = None if (buy_animal or late_day) else power_berry_calendar.due_source(
        ram, shops_closed=shops_closed
    )
    from harvest.tasks.sprinkler import sprinkler_purchase_due
    buy_sprinkler = sprinkler_purchase_due(ram)
    buy_tools = bool(
        not buy_animal
        and berry_source is None
        and policy.include_animal_purchases
        and policy.include_shop_run
        and hour < policy.buy_seed_hour
        and not is_fest
        and (buy_sprinkler or (int(read_ram_value(ram, "num_cows")) >= 1 and cow_tool_trip_affordable(ram)))
        and tool_shop_open(ram)
    )
    # Raise a cow already in the barn before the shop; any other animal buy goes first.
    care_first = cows_need_chores and species != "cow"
    buy_first = buy_animal and not care_first
    phases: List[PhaseSpec] = []
    barn_done = False
    coop_done = False
    animal_trip = False
    if in_barn and cows_need_chores and animal_tools_ready and not buy_first:
        phases.extend(BARN_CURRENT_COW_PHASES)
        barn_done = True
        exited_barn = True
    elif in_coop and chickens_need_chores:
        phases.extend(_coop_current_phases(oversupplied=oversupplied_chickens, policy=policy))
        coop_done = True
        exited_barn = False
    elif in_coop:
        phases.append(EXIT_COOP_PHASE)
        exited_barn = False
    elif buy_first:
        phases.append(EXIT_TO_FARM_PHASE)
        phases.extend(animal_purchase_phases(species))
        animal_trip = True
        exited_barn = False
    elif not on_farm:
        phases.append(EXIT_TO_FARM_PHASE)
        exited_barn = in_barn
    else:
        exited_barn = False
    if in_coop and buy_first:
        phases.extend(animal_purchase_phases(species))
        animal_trip = True
        exited_barn = False
    if buy_tools and not (barn_done or coop_done or animal_trip):
        phases.extend(tool_purchase_phases(("brush", "milker", "sprinkler") if buy_sprinkler else None))
    if cows_need_chores and not barn_done:
        phases.extend(cure_day_phases(ram, other_town_trip=bool(buy_animal or buy_tools)))
        phases.extend(cow_day_phases(ram))
        exited_barn = True
        barn_done = True
    if care_first and buy_animal and not animal_trip:
        phases.extend(animal_purchase_phases(species))
        animal_trip = True
    if chickens_need_chores and not coop_done:
        phases.extend(
            _chicken_phases(
                exited_barn=exited_barn,
                oversupplied=oversupplied_chickens,
                policy=policy,
            )
        )
    # After the chores: a sale that fails in the coop or in town then costs
    # the day's outdoor work, not the feed (Fall D7 of flock_cap went to bed
    # at 07:00 with the hens unfed). One town trip a day: a second one fails
    # or lands past the shops' hours. In rain or snow the counter refuses
    # every request ($81DE45), and a hen already carried out would stay out.
    if not (buy_animal or buy_tools or weather_blocks_animal_delivery(ram) or berry_source == "chicken_trade"):
        phases.extend(
            _chicken_sale_phases(
                adult_chickens=adult_chickens,
                hour=hour,
                is_sunday=shops_closed,
                policy=policy,
                is_festival=is_fest,
            )
        )
    church = None if (late_day or animal_trip) else social_calendar.social_phase(ram)
    from harvest.planner.run_strategy import quota_harvest_due
    if (church is not None and berry_source is None and policy.include_harvest
            and quota_harvest_due(ram) and not late_day and not social_calendar.court_before_harvest(ram, church)):
        phases.extend((NAV_CROP_PHASE, HARVEST_ROUTE_PHASE))
    from harvest.planner.tasks.festival_escape import harvest_happiness_due
    happiness_trip = harvest_happiness_due(ram)
    if berry_source is not None or church is not None or happiness_trip:
        # The trip ends anywhere (mountain, town, house); the rest of the day is planned from live RAM afterwards.
        phases.extend(power_berry_calendar.power_berry_phases(berry_source) if berry_source else [])
        if happiness_trip:
            phases.append(PhaseSpec("HARVEST_HAPPINESS", PhaseKind.HARVEST_HAPPINESS,
                                    {"ends_day": True}, failure_policy="optional"))
        # A berry trip can end on the mountain (fish camp); the courting walks start from the farm.
        phases.extend(([EXIT_TO_FARM_PHASE] if berry_source else []) + [church] if church else [])
        if church is not None:
            phases.append(EXIT_TO_FARM_PHASE)
        phases.append(DYNAMIC_OUTDOOR_PLAN_PHASE)
    elif on_farm:
        phases.extend(
            build_outdoor_day_phases_from_ram(
                ram, policy=policy, state_name=state_name, is_post_storm=is_post_storm
            )
        )
    else:
        phases.append(DYNAMIC_OUTDOOR_PLAN_PHASE)
    if not late_day and not is_fest and horse_event_due(ram):
        # Its actors seal the road west for the rest of that path visit, so
        # the day's town trip starts from a fresh path load.
        at = next((i + 1 for i, p in enumerate(phases) if p.phase == EXIT_TO_FARM_PHASE.phase),
                  0 if on_farm else None)
        if at is not None:
            phases.insert(at, MEET_HORSE_PHASE)
    from harvest.tasks.eve_gift import apply_wife_gift
    from harvest.tasks.proposal import apply_eve_phases

    from harvest.core.animal_probe import chicken_slot_snapshots
    if policy.include_chickens and not in_coop and not late_day and any(
        row["stage"] == "adult" and int(row["map_raw"]) < 4
        for row in chicken_slot_snapshots(ram)
    ):
        phases[:0] = [EXIT_TO_FARM_PHASE,
            PhaseSpec("RECOVER_FARM_HEN", "hen_delivery", failure_policy="optional"),
            COOP_CHORES_PHASE, EXIT_COOP_PHASE]
    return apply_wife_gift(apply_eve_phases(phases, ram), ram)

def _animal_tools_ready(ram: np.ndarray) -> bool:
    selected = read_ram_u8(ram, ADDR_TOOL_SELECTED)
    backpack = read_ram_u8(ram, ADDR_TOOL_BACKPACK)
    carried = {selected, backpack}
    return int(Tool.MILKER) in carried and int(Tool.BRUSH) in carried

def auto_day_plan_name_for_weekday(weekday: Optional[int]) -> str:
    """Pick the default day-plan sequence for a known weekday."""
    return "sunday" if weekday == SUNDAY_WEEKDAY else "day1"

def auto_day_plan_name_for_ram(ram: np.ndarray, fallback_state_name: Optional[str] = None) -> str:
    """Pick a day plan from live RAM, with optional save-state fallback heuristics.

    NOTE: This returns a sequence *name* for backward compat with the
    ``--day-plan`` CLI flag.  The preferred path is ``auto_day_phases``
    which returns the phase list directly from ``build_day_phases``.
    """
    probe = WorldProbe.from_inputs(ram=ram, state_name=fallback_state_name)
    _day, hour, minute = probe.day_time()
    if hour < DayPlannerPolicy().late_water_hour and probe.should_buy_cow():
        return "buy_cow"
    if probe.has_harvestable_crops():
        return "harvest"
    if not probe.is_rainy() and probe.has_waterable_crops():
        return "resume_water"
    if fallback_state_name:
        return auto_day_plan_name_for_state(fallback_state_name)
    if hour > 6 or minute > 0:
        return "resume_water"
    return auto_day_plan_name_for_weekday(probe.weekday())

def auto_day_plan_name_for_state(state_name: Optional[str]) -> str:
    """Pick the default day-plan sequence for a save state.

    NOTE: Kept for backward compat.  Prefer ``auto_day_phases`` instead.
    """
    probe = WorldProbe.from_inputs(state_name=state_name)
    _day, hour, minute = probe.day_time()
    if (
        probe.should_buy_cow()
        and hour < DayPlannerPolicy().late_water_hour
    ):
        return "buy_cow"
    if probe.has_harvestable_crops():
        return "harvest"
    if not probe.is_rainy() and probe.has_waterable_crops():
        return "resume_water"
    if hour > 6 or minute > 0:
        return "resume_water"
    if probe.has_any_crop_seeds():
        return "berries_water"
    return auto_day_plan_name_for_weekday(probe.weekday())

def auto_day_phases(
    state_name: Optional[str] = None,
    ram: Optional[np.ndarray] = None,
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
    is_post_storm: bool = False,
    is_storm: Optional[bool] = None,
    is_festival: Optional[bool] = None,
) -> List[PhaseSpec]:
    """Build the day's phase list dynamically from state/RAM inspection.

    This is the primary entry point — returns the phase list directly
    instead of a sequence name.
    """
    if ram is not None:
        return build_day_phases_from_ram(
            ram, policy=policy, state_name=state_name, is_post_storm=is_post_storm
        )

    return build_day_phases(
        state_name,
        policy=policy,
        is_post_storm=is_post_storm,
        is_storm=is_storm,
        is_festival=is_festival,
    )

__all__ = [
    "PhaseSpec", "DayPlannerPolicy", "day_planner_policy_for_season",
    "EXIT_HOUSE_PHASE", "EXIT_TO_FARM_PHASE", "LEAVE_HOUSE_TO_FARM_PHASE",
    "NAV_FARM_EXIT_PHASE", "LEAVE_FARM_WEST_PHASE", "EXIT_FARM_WEST_PHASE",
    "BUY_SEEDS_PHASE", "buy_seeds_phase", "MOUNTAIN_BERRY_PHASE",
    "NAV_CROP_PHASE", "HARVEST_ROUTE_PHASE", "CLEAR_FIELD_PHASE", "CLEAR_PHASES",
    "DYNAMIC_OUTDOOR_PLAN_PHASE", "ENSURE_WATERING_CAN_PHASE", "ENSURE_CROP_SEEDS_PHASE",
    "ENSURE_ANIMAL_TOOLS_PHASE", "ENSURE_MILKER_PHASE", "CROP_ESTABLISH_PHASE",
    "CROP_WATER_PHASE", "HOT_SPRING_STAMINA_PHASE", "RETURN_HOME_PHASE", "GO_TO_SLEEP_PHASE",
    "EVE_TALK_LOOP_PHASE", "EVE_TALK_LOOP_PHASES", "NAV_TO_COOP_PHASE",
    "NAV_BARN_TO_COOP_PHASE", "ENTER_COOP_PHASE", "COOP_CHORES_PHASE", "COOP_GIFT_PHASE",
    "EXIT_COOP_PHASE", "NAV_TO_COOP_FOR_CHICKEN_SALE_PHASE", "ENTER_COOP_FOR_CHICKEN_SALE_PHASE",
    "PICKUP_CHICKEN_FOR_SALE_PHASE", "EXIT_COOP_FOR_CHICKEN_SALE_PHASE",
    "DROP_CHICKEN_FOR_SALE_PHASE", "NAV_TO_ANIMAL_SHOP_FOR_CHICKEN_SALE_PHASE",
    "REQUEST_CHICKEN_SALE_PHASE", "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_PHASE",
    "RETURN_FARM_AFTER_CHICKEN_SALE_PHASE", "SELL_CHICKEN_PHASE",
    "CHICKEN_SALE_BATCH_DROP_POINTS", "chicken_sale_stage_phases", "chicken_sale_cycle_phases",
    "chicken_sale_batch_phases", "CHICKEN_PHASES", "CHICKEN_AFTER_BARN_PHASES",
    "COOP_CURRENT_PHASES", "NAV_TO_BARN_PHASE", "ENTER_BARN_PHASE", "COW_CHORES_PHASE",
    "EXIT_BARN_PHASE", "COW_PHASES", "BARN_CURRENT_COW_PHASES", "BUY_COW_PHASES",
    "BUY_COW_FIRST_PHASES", "DAY1_PHASES", "BOOT_TO_DAY2_PHASES", "TOWN_EXPLORE_PHASE",
    "READY_TO_GO_HOME_PHASE", "GET_HAMMER_MACRO_PHASE", "GET_AXE_MACRO_PHASE",
    "GET_SICKLE_MACRO_PHASE", "LEAVE_HOUSE_MACRO_PHASE", "GO_HOME_TRIGGER_PHASES",
    "SPRING4_PHASES", "BERRIES_WATER_PHASES", "SUNDAY_PHASES", "RESUME_WATER_PHASES",
    "HARVEST_PHASES", "SELL_CHICKEN_TEST_PHASES", "SELL_THREE_CHICKENS_TEST_PHASES",
    "SELL_THREE_CHICKENS_BATCH_TEST_PHASES", "PHASE_SEQUENCES", "PHASE_SEQUENCE",
    "BERRY_CUTOFF_HOUR", "OPTIONAL_MONEY_PHASES", "auto_day_plan_name_for_weekday",
    "auto_day_plan_name_for_ram", "auto_day_plan_name_for_state", "auto_day_phases",
    "crop_establish_phases", "crop_water_phases", "pocket_clear_phase", "pocket_plant_phases",
    "build_day_phases", "build_outdoor_day_phases", "build_outdoor_day_phases_from_ram",
    "build_day_phases_from_ram", "ATTEND_FESTIVAL_PHASE", "REPAIR_CROPS_PHASE",
    "SUNDAY_CHURCH_PHASE", "SUNDAY_MOUNTAIN_PHASE", "attend_festival_phase",
    "default_crop_for_season", "is_festival_day", "festival_name_for_date",
    "is_storm_day", "is_storm_weather", "post_storm_recovery_phases",
    "seasonal_crops_for_season",
]
