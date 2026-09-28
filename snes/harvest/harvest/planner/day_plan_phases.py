"""Dynamic day-plan builders and phase catalog re-exports."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from harvest.core.tile_catalog import Tool
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec, day_planner_policy_for_season
from harvest.planner.world_probe import WorldProbe
from harvest.planner.day_plan_status import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    BARN_TILEMAP,
    COOP_TILEMAP,
    is_farm_tilemap,
    is_house_tilemap,
    SUNDAY_WEEKDAY,
)
from harvest.core.ram_catalog import read_ram_u8
from harvest.core.stamina import Stamina
from harvest.planner.day_phase_stamina import evening_clear_phases
from harvest.planner.day_phase_berry import _berry_run_phases
from harvest.planner.day_phase_calendar import (
    ATTEND_FESTIVAL_PHASE,
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
    MOUNTAIN_BERRY_PHASE,
    NAV_CROP_PHASE,
    HARVEST_ROUTE_PHASE,
    CLEAR_FIELD_PHASE,
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


def _uses_d2_exhaustive_clear(season: int | None, day: int | None) -> bool:
    """S0D2 folds grape/shop/clear/plant/water into one reactive tactic."""
    return int(season or 0) == 0 and int(day or 0) == 2


def _daytime_clear_phase(season: int | None, day: int | None) -> PhaseSpec:
    """D2 uses the exhaustive tactic; other days keep quota CLEAR_FIELD."""
    if _uses_d2_exhaustive_clear(season, day):
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
) -> bool:
    """True when today's plan should hoe + establish a plot.

    A same-day seed purchase counts as plant intent: the bag lands on the
    shed shelf, ``ENSURE_CROP_SEEDS`` picks it up, then ``CROP_ESTABLISH``
    tills and sows it (a fresh plot beside any existing rows). D2's own
    reactive tactic already plants, so it is excluded here.

    ``has_plant_capacity`` is False when no pocket ring can still receive a
    full bag (both rings full / boxed in) — planting is pointless then.
    """
    if late_day or not policy.include_planting or not has_plant_capacity:
        return False
    if has_seeds:
        return True
    return buying_seeds and not _uses_d2_exhaustive_clear(season, day)


def build_day_phases(
    state_name: Optional[str] = None,
    *,
    weekday: Optional[int] = None,
    hour: Optional[int] = None,
    season: Optional[int] = None,
    day: Optional[int] = None,
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
    if is_festival is None:
        is_festival = is_festival_day(season, day)
    if is_festival and festival_name is None:
        festival_name = festival_name_for_date(season, day)

    policy = day_planner_policy_for_season(season, policy)

    if is_storm:
        # Severe storm (hurricane/blizzard): farmer cannot leave the house.
        if policy.include_end_day:
            return [GO_TO_SLEEP_PHASE]
        return []

    is_sunday = weekday == SUNDAY_WEEKDAY
    is_fest = bool(is_festival and policy.include_festivals)
    shops_closed = is_sunday or is_fest
    late_day = hour >= policy.late_water_hour
    oversupplied_chickens = _chicken_oversupplied(adult_chickens, policy)
    berry_phases = _berry_run_phases(
        is_sunday=shops_closed,
        hour=hour,
        has_seeds=has_seeds,
        policy=policy,
        season=season,
        day=day,
        money=money,
        has_plant_capacity=has_plant_capacity,
        has_harvest=bool(has_harvest),
        is_festival=is_fest,
    )

    buy_cow_first = (
        policy.include_cows
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

    # Post-storm recovery: morning debris clearing and fence repair
    if is_post_storm and not late_day:
        phases.append(_daytime_clear_phase(season, day))
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
    )

    # Field wipe is valuable, but day CLEAR thrash starves berry ship on empty
    # Spring mornings. Bushes/weeds only clear in the evening after shipping.
    # When dry crops already exist, still defer any day clear until after water.
    defer_field_clear = bool(
        policy.include_field_clear
        and has_debris
        and not late_day
        and has_waterable
        and not is_rainy
        and policy.include_watering
    )
    # Empty early-spring day: berries (+ optional seed buy) before chores.
    berry_before_clear = bool(
        not late_day
        and other_berry_phases
        and not has_waterable
        and not has_harvest
        and not has_chickens
        and not has_cows
    )
    # Restock day (D3+): grapes ship before the shop hop even with keep-alive
    # crops — both are morning deadlines and grape income precedes the spend.
    # Harvest mornings are the exception: do not force a 2-grape run in front
    # of ripe tiles (2 grapes stay an option after crop work).
    restock_berries_first = bool(
        not berry_before_clear
        and not late_day
        and seed_buy_phases
        and other_berry_phases
        and not has_harvest
    )
    early_berries = berry_before_clear or restock_berries_first
    if berry_before_clear:
        # Berries ship first, then potato seeds only if wallet can pay.
        phases.extend(other_berry_phases)
        phases.extend(seed_buy_phases)
        # Shop hop used to starve morning CLEAR. After a real buy we are back
        # on the farm with time left — clear before go-home.
        if seed_buy_phases and has_debris and policy.include_field_clear:
            phases.append(_daytime_clear_phase(season, day))
    elif seed_buy_phases:
        # Keep-alive farm: still buy seeds early so plant/water is not starved.
        if restock_berries_first:
            phases.extend(other_berry_phases)
        phases.extend(seed_buy_phases)

    # Daytime clear only when crops need pathing and we are not on the empty
    # berry-first path. Weed/bush lift thrash is evening-only on D2 empty days.
    day_clear = bool(
        policy.include_field_clear
        and has_debris
        and not late_day
        and not defer_field_clear
        and not berry_before_clear
        and not is_post_storm
    )
    if day_clear:
        phases.append(_daytime_clear_phase(season, day))

    if policy.include_cows and has_cows and not late_day and not buy_cow_first:
        phases.extend(COW_PHASES)
        exited_barn = True
    else:
        exited_barn = buy_cow_first

    phases.extend(
        _chicken_sale_phases(
            adult_chickens=adult_chickens,
            hour=hour,
            is_sunday=shops_closed,
            policy=policy,
            is_festival=is_fest,
        )
    )

    if policy.include_chickens and has_chickens and not late_day:
        phases.extend(
            _chicken_phases(
                exited_barn=exited_barn,
                oversupplied=oversupplied_chickens,
                policy=policy,
            )
        )

    # 2. Harvest ripe crops
    if policy.include_harvest and has_harvest and not late_day:
        phases.append(NAV_CROP_PHASE)
        phases.append(HARVEST_ROUTE_PHASE)

    # 3. Crop work before optional money routes so watering is not pushed past 5pm.
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

    # 3b. Deferred field clear after keep-alive water (rr-3v9) — day path only
    # when crops claimed the morning (not empty berry days).
    if defer_field_clear and not berry_before_clear:
        phases.append(_daytime_clear_phase(season, day))

    # 4. Festival attendance: on festival days, farm work happens early, then attend festival
    if is_fest and not late_day:
        phases.append(attend_festival_phase(season, day))
        if policy.include_end_day:
            phases.append(RETURN_HOME_PHASE)
            phases.append(GO_TO_SLEEP_PHASE)
        return phases

    # 4b. Sunday church visit and mountain foraging
    if is_sunday and not late_day:
        if policy.include_sunday_church:
            phases.append(SUNDAY_CHURCH_PHASE)
        if policy.include_sunday_mountain:
            phases.append(SUNDAY_MOUNTAIN_PHASE)

    # 4c. Early money route after animals/crops (or skipped if already first).
    if not late_day and not early_berries:
        phases.extend(other_berry_phases)
        # Seeds already placed early on keep-alive path when applicable.

    # Evening: bush/debris clear after the 5pm shipping window, then sleep.
    if late_day:
        phases.extend(
            evening_clear_phases(
                has_debris=has_debris,
                late_day=True,
                policy=policy,
                stamina=stamina,
            )
        )
        if policy.include_end_day:
            phases.append(RETURN_HOME_PHASE)
            phases.append(GO_TO_SLEEP_PHASE)

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
    money: Optional[int] = None,
    stamina: Optional[Stamina | int] = None,
    has_plant_capacity: bool = True,
    is_storm: bool = False,
    is_post_storm: bool = False,
    is_festival: Optional[bool] = None,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
) -> List[PhaseSpec]:
    """Assemble the outdoor portion of the day's work from current farm state."""
    if is_storm:
        return []
    policy = day_planner_policy_for_season(season, policy)
    is_sunday = weekday == SUNDAY_WEEKDAY
    if is_festival is None:
        is_festival = is_festival_day(season, day)
    is_fest = bool(is_festival and policy.include_festivals)
    shops_closed = is_sunday or is_fest
    late_day = hour >= policy.late_water_hour
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
    )
    phases: List[PhaseSpec] = []

    # Post-storm recovery: morning debris clearing and fence repair
    if is_post_storm and not late_day:
        phases.append(_daytime_clear_phase(season, day))
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
    berry_before_clear = bool(
        not late_day
        and other_berry_phases
        and not has_waterable
        and not has_harvest
    )
    # Restock day (D3+): ship the grapes before the shop hop even when
    # keep-alive crops still need water — grape < 5pm and shop < noon are
    # both morning deadlines, and the grape wallet credit precedes the spend.
    # Harvest mornings do not force a 2-grape run ahead of ripe tiles.
    restock_berries_first = bool(
        not berry_before_clear
        and not late_day
        and seed_buy_phases
        and other_berry_phases
        and not has_harvest
    )
    early_berries = berry_before_clear or restock_berries_first
    if berry_before_clear:
        phases.extend(other_berry_phases)
        phases.extend(seed_buy_phases)
        if seed_buy_phases and has_debris and policy.include_field_clear:
            phases.append(_daytime_clear_phase(season, day))
    elif seed_buy_phases:
        if restock_berries_first:
            phases.extend(other_berry_phases)
        phases.extend(seed_buy_phases)

    day_clear = bool(
        policy.include_field_clear
        and has_debris
        and not late_day
        and not defer_field_clear
        and not berry_before_clear
        and not is_post_storm
    )
    if day_clear:
        phases.append(_daytime_clear_phase(season, day))

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
        phases.append(_daytime_clear_phase(season, day))

    # 4. Festival attendance: on festival days, farm work happens early, then attend festival
    if is_fest and not late_day:
        phases.append(attend_festival_phase(season, day))
        if policy.include_end_day:
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
    return build_outdoor_day_phases(
        weekday=probe.weekday() or 1,
        hour=hour,
        has_harvest=probe.has_harvestable_crops(),
        has_waterable=probe.has_waterable_crops(),
        has_seeds=probe.has_seasonal_plantable_seeds(),
        has_debris=probe.has_farm_debris(),
        is_rainy=probe.is_rainy(),
        season=season,
        day=day,
        money=probe.money(),
        stamina=probe.stamina(),
        has_plant_capacity=probe.pocket_has_plant_capacity(),
        is_storm=probe.is_storm(),
        is_post_storm=is_post_storm,
        is_festival=probe.is_festival(),
        policy=policy,
    )


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
    policy = day_planner_policy_for_season(season, policy)
    if probe.is_storm():
        return [GO_TO_SLEEP_PHASE] if policy.include_end_day else []
    _day, hour, _minute = probe.day_time()
    tilemap = probe.tilemap() or 0
    on_farm = is_farm_tilemap(tilemap)
    in_barn = tilemap == BARN_TILEMAP
    in_coop = tilemap == COOP_TILEMAP
    late_day = hour >= policy.late_water_hour
    if is_house_tilemap(tilemap) and late_day and policy.include_end_day:
        return [GO_TO_SLEEP_PHASE]
    adult_chickens = probe.chicken_counts()[0]
    oversupplied_chickens = _chicken_oversupplied(adult_chickens, policy)
    is_sunday = (probe.weekday() or 1) == SUNDAY_WEEKDAY
    is_fest = probe.is_festival() and policy.include_festivals
    shops_closed = is_sunday or is_fest
    cows_need_chores = policy.include_cows and not late_day and probe.needs_cow_chores()
    chickens_need_chores = (
        policy.include_chickens and not late_day and probe.needs_chicken_chores()
    )
    animal_tools_ready = _animal_tools_ready(ram)
    buy_cow_first = (
        policy.include_cows
        and policy.include_shop_run
        and not late_day
        and not shops_closed
        and probe.should_buy_cow()
    )
    phases: List[PhaseSpec] = []
    started_in_barn_cows = False
    started_in_coop_chickens = False
    handled_buy_cow = False
    if in_barn and cows_need_chores and animal_tools_ready and not buy_cow_first:
        phases.extend(BARN_CURRENT_COW_PHASES)
        started_in_barn_cows = True
        exited_barn = True
    elif in_coop and chickens_need_chores:
        phases.extend(_coop_current_phases(oversupplied=oversupplied_chickens, policy=policy))
        started_in_coop_chickens = True
        exited_barn = False
    elif in_coop:
        phases.append(EXIT_COOP_PHASE)
        exited_barn = False
    elif buy_cow_first:
        phases.extend(BUY_COW_FIRST_PHASES)
        handled_buy_cow = True
        exited_barn = True
    elif not on_farm:
        phases.append(EXIT_TO_FARM_PHASE)
        exited_barn = in_barn
    else:
        exited_barn = False
    if in_coop and buy_cow_first:
        phases.extend(BUY_COW_FIRST_PHASES)
        handled_buy_cow = True
        exited_barn = True
    if started_in_barn_cows:
        pass
    elif cows_need_chores and not handled_buy_cow:
        phases.extend(COW_PHASES)
        exited_barn = True
    else:
        exited_barn = exited_barn or handled_buy_cow
    phases.extend(
        _chicken_sale_phases(
            adult_chickens=adult_chickens,
            hour=hour,
            is_sunday=shops_closed,
            policy=policy,
            is_festival=is_fest,
        )
    )
    if chickens_need_chores and not started_in_coop_chickens:
        phases.extend(
            _chicken_phases(
                exited_barn=exited_barn,
                oversupplied=oversupplied_chickens,
                policy=policy,
            )
        )
    if on_farm:
        phases.extend(
            build_outdoor_day_phases_from_ram(
                ram, policy=policy, state_name=state_name, is_post_storm=is_post_storm
            )
        )
    else:
        phases.append(DYNAMIC_OUTDOOR_PLAN_PHASE)
    return phases


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
    "PhaseSpec",
    "DayPlannerPolicy",
    "day_planner_policy_for_season",
    "EXIT_HOUSE_PHASE",
    "EXIT_TO_FARM_PHASE",
    "LEAVE_HOUSE_TO_FARM_PHASE",
    "NAV_FARM_EXIT_PHASE",
    "LEAVE_FARM_WEST_PHASE",
    "EXIT_FARM_WEST_PHASE",
    "BUY_SEEDS_PHASE",
    "buy_seeds_phase",
    "MOUNTAIN_BERRY_PHASE",
    "NAV_CROP_PHASE",
    "HARVEST_ROUTE_PHASE",
    "CLEAR_FIELD_PHASE",
    "CLEAR_PHASES",
    "DYNAMIC_OUTDOOR_PLAN_PHASE",
    "ENSURE_WATERING_CAN_PHASE",
    "ENSURE_CROP_SEEDS_PHASE",
    "ENSURE_ANIMAL_TOOLS_PHASE",
    "ENSURE_MILKER_PHASE",
    "CROP_ESTABLISH_PHASE",
    "CROP_WATER_PHASE",
    "HOT_SPRING_STAMINA_PHASE",
    "RETURN_HOME_PHASE",
    "GO_TO_SLEEP_PHASE",
    "EVE_TALK_LOOP_PHASE",
    "EVE_TALK_LOOP_PHASES",
    "NAV_TO_COOP_PHASE",
    "NAV_BARN_TO_COOP_PHASE",
    "ENTER_COOP_PHASE",
    "COOP_CHORES_PHASE",
    "COOP_GIFT_PHASE",
    "EXIT_COOP_PHASE",
    "NAV_TO_COOP_FOR_CHICKEN_SALE_PHASE",
    "ENTER_COOP_FOR_CHICKEN_SALE_PHASE",
    "PICKUP_CHICKEN_FOR_SALE_PHASE",
    "EXIT_COOP_FOR_CHICKEN_SALE_PHASE",
    "DROP_CHICKEN_FOR_SALE_PHASE",
    "NAV_TO_ANIMAL_SHOP_FOR_CHICKEN_SALE_PHASE",
    "REQUEST_CHICKEN_SALE_PHASE",
    "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_PHASE",
    "RETURN_FARM_AFTER_CHICKEN_SALE_PHASE",
    "SELL_CHICKEN_PHASE",
    "CHICKEN_SALE_BATCH_DROP_POINTS",
    "chicken_sale_stage_phases",
    "chicken_sale_cycle_phases",
    "chicken_sale_batch_phases",
    "CHICKEN_PHASES",
    "CHICKEN_AFTER_BARN_PHASES",
    "COOP_CURRENT_PHASES",
    "NAV_TO_BARN_PHASE",
    "ENTER_BARN_PHASE",
    "COW_CHORES_PHASE",
    "EXIT_BARN_PHASE",
    "COW_PHASES",
    "BARN_CURRENT_COW_PHASES",
    "BUY_COW_PHASES",
    "BUY_COW_FIRST_PHASES",
    "DAY1_PHASES",
    "BOOT_TO_DAY2_PHASES",
    "TOWN_EXPLORE_PHASE",
    "READY_TO_GO_HOME_PHASE",
    "GET_HAMMER_MACRO_PHASE",
    "GET_AXE_MACRO_PHASE",
    "GET_SICKLE_MACRO_PHASE",
    "LEAVE_HOUSE_MACRO_PHASE",
    "GO_HOME_TRIGGER_PHASES",
    "SPRING4_PHASES",
    "BERRIES_WATER_PHASES",
    "SUNDAY_PHASES",
    "RESUME_WATER_PHASES",
    "HARVEST_PHASES",
    "SELL_CHICKEN_TEST_PHASES",
    "SELL_THREE_CHICKENS_TEST_PHASES",
    "SELL_THREE_CHICKENS_BATCH_TEST_PHASES",
    "PHASE_SEQUENCES",
    "PHASE_SEQUENCE",
    "BERRY_CUTOFF_HOUR",
    "OPTIONAL_MONEY_PHASES",
    "auto_day_plan_name_for_weekday",
    "auto_day_plan_name_for_ram",
    "auto_day_plan_name_for_state",
    "auto_day_phases",
    "crop_establish_phases",
    "crop_water_phases",
    "pocket_clear_phase",
    "pocket_plant_phases",
    "build_day_phases",
    "build_outdoor_day_phases",
    "build_outdoor_day_phases_from_ram",
    "build_day_phases_from_ram",
    "ATTEND_FESTIVAL_PHASE",
    "REPAIR_CROPS_PHASE",
    "SUNDAY_CHURCH_PHASE",
    "SUNDAY_MOUNTAIN_PHASE",
    "attend_festival_phase",
    "default_crop_for_season",
    "is_festival_day",
    "festival_name_for_date",
    "is_storm_day",
    "is_storm_weather",
    "post_storm_recovery_phases",
    "seasonal_crops_for_season",
]
