"""Shared phase data types for day planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PhaseKind(StrEnum):
    """Typed phase kinds for day-plan task construction."""

    EXIT = "exit"
    FARM_BUILDING_EXIT = "farm_building_exit"
    FARM_EXIT = "farm_exit"
    NAV = "nav"
    RECORDED = "recorded"
    RECORDED_SLICE = "recorded_slice"
    RECORDED_TRANSITION = "recorded_transition"
    CROSS_MAP = "cross_map"
    DIRECTIONAL_TRANSITION = "directional_transition"
    MULTI_NAV = "multi_nav"
    ENSURE_TOOL = "ensure_tool"
    ENSURE_ANIMAL_TOOLS = "ensure_animal_tools"
    ENSURE_SEED = "ensure_seed"
    DEADLINE = "deadline"
    WAIT_UNTIL_TIME = "wait_until_time"
    HARVEST = "harvest"
    CLEAR_FIELD = "clear_field"
    COOP_CHORES = "coop_chores"
    COW_CHORES = "cow_chores"
    COW_PURCHASE = "cow_purchase"
    CROP = "crop"
    RETURN_HOME = "return_home"
    SLEEP = "sleep"
    READY_TO_GO_HOME = "ready_to_go_home"
    EVE_TALK_LOOP = "eve_talk_loop"
    PICKUP_CHICKEN = "pickup_chicken"
    DROP_CHICKEN = "drop_chicken"
    CHICKEN_SALE_FOLLOWUP = "chicken_sale_followup"
    CHICKEN_SALE_REQUEST = "chicken_sale_request"
    CHICKEN_SALE_EVENT = "chicken_sale_event"
    DYNAMIC_OUTDOOR_PLAN = "dynamic_outdoor_plan"


SKIP_MAP_LOCK_KINDS = frozenset(
    {
        PhaseKind.RECORDED,
        PhaseKind.RECORDED_SLICE,
        PhaseKind.RECORDED_TRANSITION,
        PhaseKind.CROSS_MAP,
        PhaseKind.MULTI_NAV,
        PhaseKind.ENSURE_TOOL,
        PhaseKind.ENSURE_ANIMAL_TOOLS,
        PhaseKind.ENSURE_SEED,
        PhaseKind.CLEAR_FIELD,
        PhaseKind.COW_PURCHASE,
        PhaseKind.EVE_TALK_LOOP,
    }
)


def coerce_phase_kind(kind: PhaseKind | str) -> PhaseKind | str:
    """Return a :class:`PhaseKind` when known; preserve unknown strings for tests."""
    if isinstance(kind, PhaseKind):
        return kind
    try:
        return PhaseKind(kind)
    except ValueError:
        return kind


@dataclass
class PhaseSpec:
    phase: str
    kind: PhaseKind | str
    params: dict[str, Any] = field(default_factory=dict)
    failure_policy: str = "required"

    def __init__(
        self,
        phase: str,
        kind: PhaseKind | str,
        params: dict[str, Any] | None = None,
        failure_policy: str = "required",
    ) -> None:
        self.phase = phase
        self.kind = coerce_phase_kind(kind)
        self.params = dict(params or {})
        self.failure_policy = failure_policy


@dataclass(frozen=True)
class DayPlannerPolicy:
    berry_cutoff_hour: int = 15
    berry_exit_cutoff_hour: int = 14
    buy_seed_hour: int = 6
    late_water_hour: int = 17
    include_chickens: bool = True
    include_cows: bool = True
    include_harvest: bool = True
    include_field_clear: bool = True
    include_watering: bool = True
    include_planting: bool = True
    include_berry_run: bool = False
    include_shop_run: bool = True
    include_end_day: bool = True
    include_chicken_sales: bool = True
    max_adult_chickens: int = 2
    chicken_sale_cutoff_hour: int = 10
    # None = derive from calendar season via crop_planner.
    seed_purchase_recording: str | None = None


def day_planner_policy_for_season(
    season: int | str,
    base: DayPlannerPolicy | None = None,
) -> DayPlannerPolicy:
    """Adjust a day-plan policy for spring/summer planting vs fall/winter.

    HM SNES has no fall/winter field crops, so planting and seed shopping stop
    after summer while harvest/water remain gated by live crop tiles.
    """
    from harvest.planner.crop_planner import (
        is_crop_planting_season,
        normalize_season,
        seed_purchase_recording_for_season,
    )

    policy = base or DayPlannerPolicy()
    season_id = normalize_season(season)
    planting = is_crop_planting_season(season_id)
    recording = policy.seed_purchase_recording
    if recording is None and planting:
        recording = seed_purchase_recording_for_season(season_id)
    return DayPlannerPolicy(
        berry_cutoff_hour=policy.berry_cutoff_hour,
        berry_exit_cutoff_hour=policy.berry_exit_cutoff_hour,
        buy_seed_hour=policy.buy_seed_hour,
        late_water_hour=policy.late_water_hour,
        include_chickens=policy.include_chickens,
        include_cows=policy.include_cows,
        include_harvest=policy.include_harvest,
        include_field_clear=policy.include_field_clear,
        include_watering=policy.include_watering,
        include_planting=planting and policy.include_planting,
        include_berry_run=policy.include_berry_run,
        include_shop_run=policy.include_shop_run,
        include_end_day=policy.include_end_day,
        include_chicken_sales=policy.include_chicken_sales,
        max_adult_chickens=policy.max_adult_chickens,
        chicken_sale_cutoff_hour=policy.chicken_sale_cutoff_hour,
        seed_purchase_recording=recording if planting else None,
    )


__all__ = [
    "DayPlannerPolicy",
    "PhaseKind",
    "PhaseSpec",
    "SKIP_MAP_LOCK_KINDS",
    "coerce_phase_kind",
    "day_planner_policy_for_season",
]
