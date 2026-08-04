"""Structured day-plan decisions and advisor hook points."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

from harvest.planner.day_plan_phases import (
    BUY_COW_PHASES,
    BUY_SEEDS_PHASE,
    COOP_CHORES_PHASE,
    COW_CHORES_PHASE,
    CROP_WATER_PHASE,
    DayPlannerPolicy,
    ENSURE_CROP_SEEDS_PHASE,
    GET_BERRIES_AND_SHIP_PHASE,
    HARVEST_ROUTE_PHASE,
    OPTIONAL_MONEY_PHASES,
    PhaseSpec,
    auto_day_phases,
)
from harvest.planner.world_probe import WorldProbe
from harvest.planner.day_plan_status import (
    BARN_TILEMAP,
    COOP_TILEMAP,
    SUNDAY_WEEKDAY,
    is_farm_tilemap,
    is_house_tilemap,
)


@dataclass(frozen=True)
class PlanningFacts:
    """Small, JSON-friendly fact set used to build a day plan."""

    source: str
    weekday: int
    hour: int
    minute: int = 0
    tilemap: Optional[int] = None
    on_farm: bool = False
    in_house: bool = False
    in_barn: bool = False
    in_coop: bool = False
    late_day: bool = False
    is_sunday: bool = False
    is_rainy: bool = False
    needs_chickens: bool = False
    needs_cows: bool = False
    should_buy_cow: bool = False
    has_harvest: bool = False
    has_waterable: bool = False
    has_seeds: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "weekday": self.weekday,
            "hour": self.hour,
            "minute": self.minute,
            "tilemap": self.tilemap,
            "on_farm": self.on_farm,
            "in_house": self.in_house,
            "in_barn": self.in_barn,
            "in_coop": self.in_coop,
            "late_day": self.late_day,
            "is_sunday": self.is_sunday,
            "is_rainy": self.is_rainy,
            "needs_chickens": self.needs_chickens,
            "needs_cows": self.needs_cows,
            "should_buy_cow": self.should_buy_cow,
            "has_harvest": self.has_harvest,
            "has_waterable": self.has_waterable,
            "has_seeds": self.has_seeds,
        }


@dataclass(frozen=True)
class DeferredPlan:
    """A task intention that should be reconsidered by a later day plan."""

    phase: str
    kind: str
    reason: str
    retry: str = "tomorrow"
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_phase(cls, phase: PhaseSpec, reason: str, *, retry: str = "tomorrow") -> "DeferredPlan":
        return cls(
            phase=phase.phase,
            kind=str(phase.kind),
            reason=reason,
            retry=retry,
            params=dict(phase.params),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "kind": self.kind,
            "reason": self.reason,
            "retry": self.retry,
            "params": _jsonable(self.params),
        }


@dataclass(frozen=True)
class DayPlanDecision:
    """Planner output that can be logged, tested, or passed to an advisor."""

    phases: tuple[PhaseSpec, ...]
    facts: PlanningFacts
    deferred: tuple[DeferredPlan, ...] = ()
    notes: tuple[str, ...] = ()
    source: str = "rules"

    @property
    def phase_names(self) -> tuple[str, ...]:
        return tuple(phase.phase for phase in self.phases)

    def with_notes(self, notes: Sequence[str], *, source: Optional[str] = None) -> "DayPlanDecision":
        clean_notes = tuple(note for note in notes if note)
        if not clean_notes:
            return self
        return replace(self, notes=self.notes + clean_notes, source=source or self.source)

    def with_deferred(
        self,
        deferred: Sequence[DeferredPlan],
        *,
        source: Optional[str] = None,
    ) -> "DayPlanDecision":
        if not deferred:
            return self
        combined = _dedupe_deferred(self.deferred + tuple(deferred))
        return replace(self, deferred=tuple(combined), source=source or self.source)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "facts": self.facts.to_jsonable(),
            "phases": [phase_spec_to_dict(phase) for phase in self.phases],
            "phase_names": list(self.phase_names),
            "deferred": [item.to_jsonable() for item in self.deferred],
            "notes": list(self.notes),
        }


class DayPlanAdvisor(Protocol):
    """Optional advisor that can annotate or adjust a deterministic plan."""

    def advise_day_plan(self, decision: DayPlanDecision) -> DayPlanDecision:
        ...


def build_day_plan_decision(
    state_name: Optional[str] = None,
    ram: Optional[np.ndarray] = None,
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
) -> DayPlanDecision:
    """Build rule-based phases plus explicit tomorrow-facing deferrals."""

    phases = tuple(auto_day_phases(state_name=state_name, ram=ram, policy=policy))
    facts = planning_facts(state_name=state_name, ram=ram, policy=policy)
    deferred = tuple(collect_deferred_plans(facts, phases, policy=policy))
    notes = _planning_notes(facts, phases)
    return DayPlanDecision(phases=phases, facts=facts, deferred=deferred, notes=notes)


def auto_day_plan_decision(
    state_name: Optional[str] = None,
    ram: Optional[np.ndarray] = None,
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
    advisor: Optional[DayPlanAdvisor] = None,
) -> DayPlanDecision:
    """Build a day-plan decision and optionally pass it through an advisor."""

    decision = build_day_plan_decision(state_name=state_name, ram=ram, policy=policy)
    if advisor is None:
        return decision
    advised = advisor.advise_day_plan(decision)
    return _validated_advised_decision(decision, advised)


def planning_facts(
    state_name: Optional[str] = None,
    ram: Optional[np.ndarray] = None,
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
) -> PlanningFacts:
    probe = WorldProbe.from_inputs(ram=ram, state_name=state_name)
    source = "ram" if ram is not None else "state"
    return _planning_facts_from_probe(probe, source=source, policy=policy)


def collect_deferred_plans(
    facts: PlanningFacts,
    phases: Sequence[PhaseSpec],
    *,
    policy: DayPlannerPolicy = DayPlannerPolicy(),
) -> list[DeferredPlan]:
    """Return omitted work that should be reconsidered by a future plan."""

    planned = {phase.phase for phase in phases}
    deferred: list[DeferredPlan] = []
    if facts.should_buy_cow and "BUY_COW_VENDOR" not in planned:
        reason = _omission_reason(facts, policy, "cow_purchase")
        if reason:
            deferred.append(DeferredPlan.from_phase(BUY_COW_PHASES[2], reason))
    if facts.needs_cows and COW_CHORES_PHASE.phase not in planned:
        reason = _omission_reason(facts, policy, "cows")
        if reason:
            deferred.append(DeferredPlan.from_phase(COW_CHORES_PHASE, reason))
    if facts.needs_chickens and COOP_CHORES_PHASE.phase not in planned:
        reason = _omission_reason(facts, policy, "chickens")
        if reason:
            deferred.append(DeferredPlan.from_phase(COOP_CHORES_PHASE, reason))
    if facts.has_harvest and HARVEST_ROUTE_PHASE.phase not in planned:
        reason = _omission_reason(facts, policy, "harvest")
        if reason:
            deferred.append(DeferredPlan.from_phase(HARVEST_ROUTE_PHASE, reason))
    if facts.has_waterable and CROP_WATER_PHASE.phase not in planned:
        reason = _omission_reason(facts, policy, "water")
        if reason:
            deferred.append(DeferredPlan.from_phase(CROP_WATER_PHASE, reason))
    if facts.has_seeds and ENSURE_CROP_SEEDS_PHASE.phase not in planned and CROP_WATER_PHASE.phase not in planned:
        reason = _omission_reason(facts, policy, "seeds")
        if reason:
            deferred.append(DeferredPlan.from_phase(CROP_WATER_PHASE, reason))
    if (
        policy.include_berry_run
        and facts.hour >= policy.berry_cutoff_hour
        and GET_BERRIES_AND_SHIP_PHASE.phase not in planned
    ):
        deferred.append(DeferredPlan.from_phase(GET_BERRIES_AND_SHIP_PHASE, "berry_cutoff"))
    if (
        policy.include_shop_run
        and policy.include_planting
        and not facts.has_seeds
        and BUY_SEEDS_PHASE.phase not in planned
        and not facts.should_buy_cow
    ):
        if facts.is_sunday:
            deferred.append(DeferredPlan.from_phase(BUY_SEEDS_PHASE, "shop_closed_sunday"))
        elif facts.hour > policy.buy_seed_hour:
            deferred.append(DeferredPlan.from_phase(BUY_SEEDS_PHASE, "seed_shop_cutoff"))
    return _dedupe_deferred(deferred)


def phase_spec_to_dict(phase: PhaseSpec) -> dict[str, Any]:
    payload = {
        "phase": phase.phase,
        "kind": phase.kind,
        "params": _jsonable(phase.params),
        "failure_policy": phase.failure_policy,
    }
    contract = getattr(phase, "contract", None)
    if contract is not None and hasattr(contract, "to_jsonable"):
        payload["contract"] = contract.to_jsonable()
    return payload


def deferred_from_phase_name(phase_name: str, reason: str, *, retry: str = "tomorrow") -> DeferredPlan:
    """Build a deferral from a known phase name for planner advisors."""

    phase = _PHASE_BY_NAME.get(phase_name)
    if phase is None:
        return DeferredPlan(phase=phase_name, kind="unknown", reason=reason, retry=retry)
    return DeferredPlan.from_phase(phase, reason, retry=retry)


def _planning_facts_from_probe(
    probe: WorldProbe,
    *,
    source: str,
    policy: DayPlannerPolicy,
) -> PlanningFacts:
    from harvest.planner.day_phase_types import day_planner_policy_for_season

    _day, hour, minute = probe.day_time()
    season, _calendar_day = probe.calendar_date()
    policy = day_planner_policy_for_season(season, policy)
    weekday = probe.weekday() or 1
    tilemap = probe.tilemap()
    on_farm = tilemap is not None and is_farm_tilemap(tilemap)
    return PlanningFacts(
        source=source,
        weekday=weekday,
        hour=hour,
        minute=minute,
        tilemap=tilemap if source == "ram" else None,
        on_farm=on_farm if source == "ram" else False,
        in_house=tilemap is not None and is_house_tilemap(tilemap) if source == "ram" else False,
        in_barn=tilemap == BARN_TILEMAP if source == "ram" else False,
        in_coop=tilemap == COOP_TILEMAP if source == "ram" else False,
        late_day=hour >= policy.late_water_hour,
        is_sunday=weekday == SUNDAY_WEEKDAY,
        is_rainy=probe.is_rainy(),
        needs_chickens=probe.needs_chicken_chores(),
        needs_cows=probe.needs_cow_chores(),
        should_buy_cow=probe.should_buy_cow(),
        has_harvest=probe.has_harvestable_crops(),
        has_waterable=probe.has_waterable_crops(),
        has_seeds=probe.has_seasonal_plantable_seeds(),
    )


def _omission_reason(facts: PlanningFacts, policy: DayPlannerPolicy, category: str) -> str:
    if category in {"cows", "cow_purchase"} and not policy.include_cows:
        return "disabled_by_policy"
    if category == "chickens" and not policy.include_chickens:
        return "disabled_by_policy"
    if category == "harvest" and not policy.include_harvest:
        return "disabled_by_policy"
    if category in {"water", "seeds"} and not policy.include_watering:
        return "disabled_by_policy"
    if category == "seeds" and not policy.include_planting:
        return "disabled_by_policy"
    if category == "cow_purchase" and not policy.include_shop_run:
        return "shop_run_disabled"
    if category == "water" and facts.is_rainy:
        return "rainy_day"
    if category == "seeds" and facts.is_rainy:
        return "rainy_day_seed_planting"
    if facts.late_day:
        return "late_day"
    return ""


def _planning_notes(facts: PlanningFacts, phases: Sequence[PhaseSpec]) -> tuple[str, ...]:
    notes: list[str] = []
    if facts.late_day and any(phase.phase in {"RETURN_HOME", "GO_TO_SLEEP"} for phase in phases):
        notes.append("late-day plan includes end-day route")
    if facts.is_rainy:
        notes.append("rainy day suppresses watering work")
    if any(phase.phase in OPTIONAL_MONEY_PHASES for phase in phases):
        notes.append("optional money route is present")
    return tuple(notes)


def _dedupe_deferred(items: Sequence[DeferredPlan]) -> list[DeferredPlan]:
    seen: set[tuple[str, str]] = set()
    deduped: list[DeferredPlan] = []
    for item in items:
        key = (item.phase, item.reason)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _validated_advised_decision(base: DayPlanDecision, advised: DayPlanDecision) -> DayPlanDecision:
    if not isinstance(advised, DayPlanDecision):
        raise TypeError("day-plan advisor must return DayPlanDecision")
    for phase in advised.phases:
        if not isinstance(phase, PhaseSpec):
            raise TypeError("day-plan advisor returned a non-PhaseSpec phase")
    if advised.facts != base.facts:
        raise ValueError("day-plan advisor must not rewrite planning facts")
    return advised


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_PHASE_BY_NAME = {
    phase.phase: phase
    for phase in (
        BUY_COW_PHASES[2],
        BUY_SEEDS_PHASE,
        COOP_CHORES_PHASE,
        COW_CHORES_PHASE,
        CROP_WATER_PHASE,
        ENSURE_CROP_SEEDS_PHASE,
        GET_BERRIES_AND_SHIP_PHASE,
        HARVEST_ROUTE_PHASE,
    )
}


__all__ = [
    "DayPlanAdvisor",
    "DayPlanDecision",
    "DeferredPlan",
    "PlanningFacts",
    "auto_day_plan_decision",
    "build_day_plan_decision",
    "collect_deferred_plans",
    "deferred_from_phase_name",
    "phase_spec_to_dict",
    "planning_facts",
]
