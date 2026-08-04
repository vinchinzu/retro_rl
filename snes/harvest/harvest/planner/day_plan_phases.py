"""Dynamic day-plan builders and phase catalog re-exports."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from harvest.tasks.farm_clearer import Tool
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec, day_planner_policy_for_season
from harvest.planner.world_probe import WorldProbe
from harvest.planner.day_plan_status import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    BARN_TILEMAP,
    COOP_TILEMAP,
    FARM_TILEMAP,
    HOUSE_TILEMAP,
    is_farm_tilemap,
    is_house_tilemap,
    SUNDAY_WEEKDAY,
)
from harvest.core.ram_catalog import read_ram_u8
from harvest.planner.day_phase_catalog import (
    EXIT_HOUSE_PHASE,
    EXIT_TO_FARM_PHASE,
    LEAVE_HOUSE_TO_FARM_PHASE,
    NAV_FARM_EXIT_PHASE,
    LEAVE_FARM_WEST_PHASE,
    EXIT_FARM_WEST_PHASE,
    BUY_SEEDS_PHASE,
    buy_seeds_phase,
    GET_BERRIES_AND_SHIP_PHASE,
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
)


def _chicken_oversupplied(adult_chickens: int, policy: DayPlannerPolicy) -> bool:
    return policy.include_chickens and adult_chickens > policy.max_adult_chickens


def _coop_chores_phase(*, oversupplied: bool, policy: DayPlannerPolicy) -> PhaseSpec:
    if not oversupplied:
        return COOP_CHORES_PHASE
    return PhaseSpec(
        "COOP_CHORES",
        "coop_chores",
        {"egg_mode": "ship", "max_feed_adults": policy.max_adult_chickens},
    )


def _chicken_phases(
    *,
    exited_barn: bool,
    oversupplied: bool,
    policy: DayPlannerPolicy,
) -> List[PhaseSpec]:
    phases = list(CHICKEN_AFTER_BARN_PHASES if exited_barn else CHICKEN_PHASES)
    chores = _coop_chores_phase(oversupplied=oversupplied, policy=policy)
    return [chores if phase.phase == "COOP_CHORES" else phase for phase in phases]


def _coop_current_phases(*, oversupplied: bool, policy: DayPlannerPolicy) -> List[PhaseSpec]:
    chores = _coop_chores_phase(oversupplied=oversupplied, policy=policy)
    return [chores, EXIT_COOP_PHASE]


def _chicken_sale_phases(
    *,
    adult_chickens: int,
    hour: int,
    is_sunday: bool,
    policy: DayPlannerPolicy,
) -> List[PhaseSpec]:
    if (
        not policy.include_chicken_sales
        or not policy.include_shop_run
        or is_sunday
        or not _chicken_oversupplied(adult_chickens, policy)
        or hour >= policy.chicken_sale_cutoff_hour
    ):
        return []
    return [
        PhaseSpec(
            "SELL_CHICKEN_WINDOW",
            "deadline",
            {"latest_hour": policy.chicken_sale_cutoff_hour, "latest_minute": 0},
            failure_policy="optional",
        ),
        *chicken_sale_cycle_phases(failure_policy="optional"),
    ]


def _berry_run_phases(
    *,
    is_sunday: bool,
    hour: int,
    has_seeds: bool,
    policy: DayPlannerPolicy,
    season: int = 0,
    day: int = 1,
) -> List[PhaseSpec]:
    """Build early money phases when a berry/shop route is available."""
    from harvest.planner.crop_planner import (
        seed_purchase_recording_for_season,
        should_buy_seeds_for_date,
    )

    if hour >= policy.berry_cutoff_hour:
        return []

    can_buy = (
        policy.include_shop_run
        and policy.include_planting
        and not is_sunday
        and not has_seeds
        and hour <= policy.buy_seed_hour
        and should_buy_seeds_for_date(season, day)
    )
    if can_buy:
        recording = (
            policy.seed_purchase_recording
            or seed_purchase_recording_for_season(season)
            or "buy_potato_seeds"
        )
        return [
            PhaseSpec(
                "BUY_SEEDS_WINDOW",
                "deadline",
                {"latest_hour": policy.buy_seed_hour + 1, "latest_minute": 0},
                failure_policy="optional",
            ),
            NAV_FARM_EXIT_PHASE,
            buy_seeds_phase(recording_name=recording),
        ]

    if not policy.include_berry_run:
        return []

    if is_sunday or has_seeds:
        return [
            PhaseSpec(
                "BERRY_RUN_WINDOW",
                "deadline",
                {"latest_hour": policy.berry_exit_cutoff_hour, "latest_minute": 0},
                failure_policy="optional",
            ),
            EXIT_FARM_WEST_PHASE,
            PhaseSpec(
                "BERRY_RECORDING_WINDOW",
                "deadline",
                {"latest_hour": policy.berry_cutoff_hour, "latest_minute": 0},
                failure_policy="optional",
            ),
            GET_BERRIES_AND_SHIP_PHASE,
        ]

    return []


def crop_establish_phases() -> List[PhaseSpec]:
    """Hoe + plant pass: ensure seeds/hoe, walk to field, establish plots.

    Only two carry slots, so this pass keeps seeds+hoe and does not fetch the can.
    """
    return [
        ENSURE_CROP_SEEDS_PHASE,
        NAV_CROP_PHASE,
        CROP_ESTABLISH_PHASE,
    ]


def crop_water_phases(*, include_nav: bool = True) -> List[PhaseSpec]:
    """Water pass: ensure can, optional field nav, water established crops."""
    phases: List[PhaseSpec] = [ENSURE_WATERING_CAN_PHASE]
    if include_nav:
        phases.append(NAV_CROP_PHASE)
    phases.append(CROP_WATER_PHASE)
    return phases


def _crop_work_phases(
    *,
    has_harvest: bool,
    has_waterable: bool,
    has_seeds: bool,
    is_rainy: bool,
    late_day: bool,
    policy: DayPlannerPolicy,
) -> List[PhaseSpec]:
    plant_seeds = bool(has_seeds and policy.include_planting and not late_day)
    needs_manual_water = bool(not is_rainy and (has_waterable or plant_seeds))
    needs_crop_phase = plant_seeds or needs_manual_water
    if late_day:
        needs_crop_phase = bool(has_waterable and not is_rainy)
        plant_seeds = False
        needs_manual_water = bool(has_waterable and not is_rainy)
    if not policy.include_watering or not needs_crop_phase:
        return []

    phases: List[PhaseSpec] = []
    # Only two carry slots. Plant pass uses hoe+seeds; water pass re-fetches the
    # can afterward (seed bag frees a slot once the bag is spent).
    if plant_seeds:
        phases.extend(crop_establish_phases())
    if needs_manual_water:
        # When a harvest route will already walk the field, skip a second nav.
        prefer_harvest_nav = bool(
            policy.include_harvest and has_harvest and not late_day and not plant_seeds
        )
        phases.extend(crop_water_phases(include_nav=not prefer_harvest_nav))
    return phases


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

    # Fill remaining defaults
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

    policy = day_planner_policy_for_season(season, policy)

    is_sunday = weekday == SUNDAY_WEEKDAY
    late_day = hour >= policy.late_water_hour
    oversupplied_chickens = _chicken_oversupplied(adult_chickens, policy)
    berry_phases = _berry_run_phases(
        is_sunday=is_sunday,
        hour=hour,
        has_seeds=has_seeds,
        policy=policy,
        season=season,
        day=day,
    )

    buy_cow_first = (
        policy.include_cows
        and policy.include_shop_run
        and should_buy_cow
        and not late_day
    )
    phases: List[PhaseSpec] = []
    if buy_cow_first:
        phases.extend(BUY_COW_FIRST_PHASES)
    else:
        phases.append(EXIT_TO_FARM_PHASE)

    # Time-critical seed shop before field clear — clear can burn past 7am.
    seed_buy_phases = [
        phase
        for phase in berry_phases
        if phase.phase in {"BUY_SEEDS_WINDOW", "NAV_FARM_EXIT", "BUY_SEEDS"}
    ]
    other_berry_phases = [
        phase for phase in berry_phases if phase not in seed_buy_phases
    ]
    if seed_buy_phases:
        phases.extend(seed_buy_phases)

    # Early-game field wipe before animals/crops when debris remains.
    if policy.include_field_clear and has_debris and not late_day:
        phases.append(CLEAR_FIELD_PHASE)

    if policy.include_cows and has_cows and not late_day and not buy_cow_first:
        phases.extend(COW_PHASES)
        exited_barn = True
    else:
        exited_barn = buy_cow_first

    phases.extend(
        _chicken_sale_phases(
            adult_chickens=adult_chickens,
            hour=hour,
            is_sunday=is_sunday,
            policy=policy,
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
            has_seeds=has_seeds,
            is_rainy=is_rainy,
            late_day=late_day,
            policy=policy,
        )
    )

    # 4. Early money route, only after required animal/crop work.
    if not late_day:
        phases.extend(other_berry_phases)

    if policy.include_end_day and late_day:
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
    policy: DayPlannerPolicy = DayPlannerPolicy(),
) -> List[PhaseSpec]:
    """Assemble the outdoor portion of the day's work from current farm state."""
    policy = day_planner_policy_for_season(season, policy)
    is_sunday = weekday == SUNDAY_WEEKDAY
    late_day = hour >= policy.late_water_hour
    berry_phases = _berry_run_phases(
        is_sunday=is_sunday,
        hour=hour,
        has_seeds=has_seeds,
        policy=policy,
        season=season,
        day=day,
    )
    phases: List[PhaseSpec] = []

    seed_buy_phases = [
        phase
        for phase in berry_phases
        if phase.phase in {"BUY_SEEDS_WINDOW", "NAV_FARM_EXIT", "BUY_SEEDS"}
    ]
    other_berry_phases = [
        phase for phase in berry_phases if phase not in seed_buy_phases
    ]

    # Buy seeds before clearing so the 7am shop window is not burned.
    if seed_buy_phases:
        phases.extend(seed_buy_phases)

    if policy.include_field_clear and has_debris and not late_day:
        phases.append(CLEAR_FIELD_PHASE)

    if policy.include_harvest and has_harvest and not late_day:
        phases.append(NAV_CROP_PHASE)
        phases.append(HARVEST_ROUTE_PHASE)

    phases.extend(
        _crop_work_phases(
            has_harvest=has_harvest,
            has_waterable=has_waterable,
            has_seeds=has_seeds,
            is_rainy=is_rainy,
            late_day=late_day,
            policy=policy,
        )
    )

    if not late_day:
        phases.extend(other_berry_phases)

    if policy.include_end_day and late_day:
        phases.append(RETURN_HOME_PHASE)
        phases.append(GO_TO_SLEEP_PHASE)

    return phases


def build_outdoor_day_phases_from_ram(
    ram: np.ndarray,
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
    state_name: Optional[str] = None,
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
        policy=policy,
    )


def build_day_phases_from_ram(
    ram: np.ndarray,
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
    state_name: Optional[str] = None,
) -> List[PhaseSpec]:
    """Assemble a day's phase list directly from live RAM."""
    probe = WorldProbe.from_inputs(ram=ram, state_name=state_name)
    season, _calendar_day = probe.calendar_date()
    policy = day_planner_policy_for_season(season, policy)
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
    cows_need_chores = policy.include_cows and not late_day and probe.needs_cow_chores()
    chickens_need_chores = (
        policy.include_chickens and not late_day and probe.needs_chicken_chores()
    )
    animal_tools_ready = _animal_tools_ready(ram)
    buy_cow_first = (
        policy.include_cows
        and policy.include_shop_run
        and not late_day
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
            is_sunday=is_sunday,
            policy=policy,
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
        phases.extend(build_outdoor_day_phases_from_ram(ram, policy=policy, state_name=state_name))
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
) -> List[PhaseSpec]:
    """Build the day's phase list dynamically from state/RAM inspection.

    This is the primary entry point — returns the phase list directly
    instead of a sequence name.
    """
    if ram is not None:
        return build_day_phases_from_ram(ram, policy=policy, state_name=state_name)

    return build_day_phases(state_name, policy=policy)


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
    "GET_BERRIES_AND_SHIP_PHASE",
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
    "build_day_phases",
    "build_outdoor_day_phases",
    "build_outdoor_day_phases_from_ram",
    "build_day_phases_from_ram",
]
