"""Bounded planning over progression, consumables, and observed skill risk."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, log
from typing import Any, Iterable, Mapping, Protocol

from retro_harness.adventure.graph import (
    GraphEdge,
)
from retro_harness.adventure.planner import (
    PlanRequest,
    PlanResult,
    PlanSearchDimension,
    ResourceBlocker,
    SearchTransition,
    search,
)
from retro_harness.identity import canonical_json, sha256_bytes


class ResourceKind(str, Enum):
    CONSUMABLE = "consumable"
    RENEWABLE = "renewable"
    SAFETY = "safety"


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    minimum: float = 0.0
    maximum: float = float("inf")
    kind: ResourceKind = ResourceKind.CONSUMABLE

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("resource name must be non-empty")
        object.__setattr__(self, "name", self.name.strip())
        if not isinstance(self.kind, ResourceKind):
            raise TypeError("resource kind must be ResourceKind")
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"resource {name} must be numeric")
        if not isfinite(float(self.minimum)):
            raise ValueError("resource minimum must be finite")
        if float(self.maximum) != float("inf") and not isfinite(float(self.maximum)):
            raise ValueError("resource maximum must be finite or +inf")
        if self.maximum < self.minimum:
            raise ValueError("resource maximum is below minimum")
        object.__setattr__(self, "minimum", float(self.minimum))
        object.__setattr__(self, "maximum", float(self.maximum))

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "minimum": self.minimum,
            "maximum": None if self.maximum == float("inf") else self.maximum,
            "kind": self.kind.value,
        }


def _amounts(values: Mapping[str, float], name: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} resource names must be non-empty")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} resource amounts must be numeric")
        number = float(value)
        if not isfinite(number) or number < 0:
            raise ValueError(f"{name} resource amounts must be finite and non-negative")
        normalized[key.strip()] = number
    return dict(sorted(normalized.items()))


@dataclass(frozen=True, slots=True)
class EdgeResourceProfile:
    edge_id: str
    consumes: Mapping[str, float] = field(default_factory=dict)
    produces: Mapping[str, float] = field(default_factory=dict)
    minimum_required: Mapping[str, float] = field(default_factory=dict)
    expected_frames: float = 1.0
    prior_success_probability: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.edge_id, str) or not self.edge_id.strip():
            raise ValueError("edge_id must be non-empty")
        object.__setattr__(self, "edge_id", self.edge_id.strip())
        for name in ("consumes", "produces", "minimum_required"):
            object.__setattr__(self, name, _amounts(getattr(self, name), name))
        if (
            isinstance(self.expected_frames, bool)
            or not isinstance(self.expected_frames, (int, float))
            or not isfinite(float(self.expected_frames))
            or self.expected_frames < 0
        ):
            raise ValueError("expected_frames must be finite and non-negative")
        probability = float(self.prior_success_probability)
        if not 0 < probability <= 1:
            raise ValueError("prior_success_probability must be in (0, 1]")
        object.__setattr__(self, "expected_frames", float(self.expected_frames))
        object.__setattr__(self, "prior_success_probability", probability)

    def to_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "consumes": dict(self.consumes),
            "produces": dict(self.produces),
            "minimum_required": dict(self.minimum_required),
            "expected_frames": self.expected_frames,
            "prior_success_probability": self.prior_success_probability,
        }


@dataclass(frozen=True, slots=True)
class SkillReliabilityStats:
    edge_id: str
    attempts: int
    successes: int
    mean_frames: float

    def __post_init__(self) -> None:
        if not isinstance(self.edge_id, str) or not self.edge_id.strip():
            raise ValueError("edge_id must be non-empty")
        if self.attempts < 1 or not 0 <= self.successes <= self.attempts:
            raise ValueError("invalid reliability attempt counts")
        if not isfinite(float(self.mean_frames)) or self.mean_frames < 0:
            raise ValueError("mean_frames must be finite and non-negative")

    @property
    def success_probability(self) -> float:
        # Beta(1, 1) posterior mean prevents zero/one certainty from tiny samples.
        return (self.successes + 1.0) / (self.attempts + 2.0)

    def to_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "attempts": self.attempts,
            "successes": self.successes,
            "mean_frames": self.mean_frames,
            "success_probability": self.success_probability,
        }


class OutcomeStatusLike(Protocol):
    value: str


class SkillOutcomeLike(Protocol):
    edge_id: str
    status: OutcomeStatusLike
    frames: int


def reliability_from_outcomes(
    outcomes: Iterable[SkillOutcomeLike],
) -> tuple[SkillReliabilityStats, ...]:
    """Aggregate outcomes through the adventure-layer event protocol."""
    grouped: dict[str, list[SkillOutcomeLike]] = defaultdict(list)
    for outcome in outcomes:
        edge_id = outcome.edge_id
        status = outcome.status.value
        frames = outcome.frames
        if not isinstance(edge_id, str) or not isinstance(status, str):
            raise TypeError("outcomes must expose edge_id, status.value, and frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise TypeError("outcome frames must be a non-negative integer")
        grouped[edge_id].append(outcome)
    return tuple(
        SkillReliabilityStats(
            edge_id=edge_id,
            attempts=len(values),
            successes=sum(
                value.status.value == "success" for value in values
            ),
            mean_frames=sum(value.frames for value in values) / len(values),
        )
        for edge_id, values in sorted(grouped.items())
    )


@dataclass(frozen=True, slots=True)
class RiskCostModel:
    failure_weight: float = 1.0
    frame_weight: float = 0.0
    consumption_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("failure_weight", "frame_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        object.__setattr__(
            self,
            "consumption_weights",
            _amounts(self.consumption_weights, "consumption_weights"),
        )

    def edge_cost(
        self,
        edge: GraphEdge,
        profile: EdgeResourceProfile,
        stats: SkillReliabilityStats | None,
    ) -> float:
        probability = (
            stats.success_probability if stats else profile.prior_success_probability
        )
        frames = stats.mean_frames if stats else profile.expected_frames
        consumption = sum(
            amount * self.consumption_weights.get(name, 0.0)
            for name, amount in profile.consumes.items()
        )
        return (
            edge.cost
            + self.failure_weight * -log(probability)
            + self.frame_weight * frames
            + consumption
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "failure_weight": self.failure_weight,
            "frame_weight": self.frame_weight,
            "consumption_weights": dict(self.consumption_weights),
        }


@dataclass(frozen=True, slots=True)
class ResourcePlanRequest:
    plan_request: PlanRequest
    resources: tuple[ResourceSpec, ...]
    initial_resources: Mapping[str, float]
    profiles: tuple[EdgeResourceProfile, ...] = ()
    reliability: tuple[SkillReliabilityStats, ...] = ()
    cost_model: RiskCostModel = field(default_factory=RiskCostModel)

    def __post_init__(self) -> None:
        if not isinstance(self.plan_request, PlanRequest):
            raise TypeError("plan_request must be PlanRequest")
        specs = tuple(sorted(self.resources, key=lambda value: value.name))
        if not specs or not all(isinstance(value, ResourceSpec) for value in specs):
            raise ValueError("at least one ResourceSpec is required")
        names = [value.name for value in specs]
        if len(names) != len(set(names)):
            raise ValueError("resource names must be unique")
        initial = {key: float(value) for key, value in self.initial_resources.items()}
        if set(initial) != set(names):
            raise ValueError("initial_resources must exactly match resource specs")
        for spec in specs:
            if not spec.minimum <= initial[spec.name] <= spec.maximum:
                raise ValueError(f"initial {spec.name} is outside its bounds")
        edge_ids = {edge.edge_id for edge in self.plan_request.edges}
        profiles = tuple(sorted(self.profiles, key=lambda value: value.edge_id))
        if len({value.edge_id for value in profiles}) != len(profiles):
            raise ValueError("edge resource profiles must be unique")
        reliability = tuple(sorted(self.reliability, key=lambda value: value.edge_id))
        if len({value.edge_id for value in reliability}) != len(reliability):
            raise ValueError("reliability stats must be unique per edge")
        for value in (*profiles, *reliability):
            if value.edge_id not in edge_ids:
                raise ValueError(f"unknown profiled edge: {value.edge_id}")
        known = set(names)
        for profile in profiles:
            used = set(profile.consumes) | set(profile.produces) | set(profile.minimum_required)
            if not used <= known:
                raise ValueError(f"profile {profile.edge_id} references unknown resources")
        object.__setattr__(self, "resources", specs)
        object.__setattr__(self, "initial_resources", dict(sorted(initial.items())))
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "reliability", reliability)
        if not isinstance(self.cost_model, RiskCostModel):
            raise TypeError("cost_model must be RiskCostModel")

    def to_record(self) -> dict[str, Any]:
        return {
            "plan_request_digest": self.plan_request.identity_digest,
            "resources": [value.to_record() for value in self.resources],
            "initial_resources": dict(self.initial_resources),
            "profiles": [value.to_record() for value in self.profiles],
            "reliability": [value.to_record() for value in self.reliability],
            "cost_model": self.cost_model.to_record(),
        }

    @property
    def identity_digest(self) -> str:
        return sha256_bytes(canonical_json(self.to_record()).encode("utf-8"))


ResourceTuple = tuple[float, ...]


class _ResourceDimension(PlanSearchDimension):
    """Resource/risk adapter for the shared progression search."""

    risk_adjusted = True

    def __init__(self, request: ResourcePlanRequest) -> None:
        self.request = request
        self.request_digest = request.identity_digest
        self.specs = request.resources
        self.names = tuple(spec.name for spec in self.specs)
        self.initial_state: ResourceTuple = tuple(
            request.initial_resources[name] for name in self.names
        )
        self.profiles = {value.edge_id: value for value in request.profiles}
        self.reliability = {value.edge_id: value for value in request.reliability}

    def transition(self, edge: GraphEdge, state: Any) -> SearchTransition:
        values = tuple(float(value) for value in state)
        available = dict(zip(self.names, values, strict=True))
        profile = self.profiles.get(edge.edge_id, EdgeResourceProfile(edge.edge_id))
        blockers = _resource_blockers(edge, profile, available, self.specs)
        if blockers:
            return SearchTransition(None, None, blockers)
        next_available = _apply_profile(profile, available, self.specs)
        next_state = tuple(next_available[name] for name in self.names)
        cost = self.request.cost_model.edge_cost(
            edge,
            profile,
            self.reliability.get(edge.edge_id),
        )
        return SearchTransition(next_state, cost)

    def dominates(self, first: Any, second: Any) -> bool:
        first_values = tuple(float(value) for value in first)
        second_values = tuple(float(value) for value in second)
        return all(
            available >= required
            for available, required in zip(
                first_values,
                second_values,
                strict=True,
            )
        )

    def state_record(self, state: Any) -> Mapping[str, float]:
        values = tuple(float(value) for value in state)
        return dict(zip(self.names, values, strict=True))


def resource_plan(request: ResourcePlanRequest) -> PlanResult:
    """Adapt a resource/risk request to the shared bounded search core."""
    if not isinstance(request, ResourcePlanRequest):
        raise TypeError("request must be ResourcePlanRequest")
    return search(request.plan_request, _ResourceDimension(request))


def _resource_blockers(
    edge: GraphEdge,
    profile: EdgeResourceProfile,
    available: Mapping[str, float],
    specs: tuple[ResourceSpec, ...],
) -> tuple[ResourceBlocker, ...]:
    blockers: list[ResourceBlocker] = []
    spec_by_name = {value.name: value for value in specs}
    for name in sorted(set(profile.minimum_required) | set(profile.consumes)):
        required = max(
            profile.minimum_required.get(name, spec_by_name[name].minimum),
            profile.consumes.get(name, 0.0) + spec_by_name[name].minimum,
        )
        if available[name] < required:
            blockers.append(
                ResourceBlocker(
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    name,
                    required,
                    available[name],
                    "minimum or consumption bound",
                )
            )
    return tuple(blockers)


def _apply_profile(
    profile: EdgeResourceProfile,
    available: Mapping[str, float],
    specs: tuple[ResourceSpec, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for spec in specs:
        value = (
            available[spec.name]
            - profile.consumes.get(spec.name, 0.0)
            + profile.produces.get(spec.name, 0.0)
        )
        result[spec.name] = min(value, spec.maximum)
    return result


__all__ = [
    "EdgeResourceProfile",
    "ResourceKind",
    "ResourcePlanRequest",
    "ResourceSpec",
    "RiskCostModel",
    "SkillReliabilityStats",
    "reliability_from_outcomes",
    "resource_plan",
]
