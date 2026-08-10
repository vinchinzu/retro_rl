"""Shared observe → plan → invoke skill → validate → replan runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol

from retro_harness.adventure.bindings import (
    BindingCatalog,
    ExecutionReadiness,
    SkillBinding,
)
from retro_harness.adventure.planner import PlanResult, PlanStatus
from retro_harness.benchmark_claims import PolicyIdentity
from retro_harness.identity import (
    canonical_json as _canonical_json,
    digest_record as _digest,
    require_nonempty as _nonempty,
)


def _node_record(value: Hashable) -> dict[str, str]:
    return {"type": type(value).__qualname__, "repr": repr(value)}


def _action_value(value: Any, path: str = "action") -> Any:
    """Convert an action to stable JSON without falling back to ``repr``."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return _action_value(value.value, path)
    if isinstance(value, Mapping):
        record: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            record[key] = _action_value(item, f"{path}.{key}")
        return dict(sorted(record.items()))
    if isinstance(value, (list, tuple)):
        return [_action_value(item, f"{path}[]") for item in value]
    to_record = getattr(value, "to_action_record", None)
    if callable(to_record):
        return _action_value(to_record(), path)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _action_value(tolist(), path)
    raise TypeError(
        f"{path} has unsupported type {type(value).__qualname__}; "
        "implement to_action_record()"
    )


def canonical_action_record(value: Any) -> dict[str, Any]:
    """Return the replayable, type-tagged representation stored in traces."""
    record = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": _action_value(value),
    }
    _canonical_json(record)
    return record


@dataclass(frozen=True, slots=True)
class SolverObservation:
    """Canonical planner/runtime observation independent of game RAM layout."""

    frame: int
    node_id: Hashable
    schema_digest: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    resources: Mapping[str, float] = field(default_factory=dict)
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise ValueError("observation frame must be a non-negative integer")
        object.__setattr__(
            self,
            "schema_digest",
            _nonempty(self.schema_digest, "schema_digest"),
        )
        normalized_caps = frozenset(
            _nonempty(value, "capability") for value in self.capabilities
        )
        normalized_resources: dict[str, float] = {}
        for name, value in self.resources.items():
            key = _nonempty(name, "resource name")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("resource values must be numeric")
            normalized_resources[key] = float(value)
        object.__setattr__(self, "capabilities", normalized_caps)
        object.__setattr__(self, "resources", dict(sorted(normalized_resources.items())))
        object.__setattr__(self, "values", dict(self.values))
        # Validate payload now so traces cannot fail after execution.
        _canonical_json(self.to_record(include_digest=False))

    @property
    def identity_digest(self) -> str:
        return _digest("solver-observation-v1", self.to_record(include_digest=False))

    def to_record(self, *, include_digest: bool = True) -> dict[str, Any]:
        record = {
            "frame": self.frame,
            "node": _node_record(self.node_id),
            "schema_digest": self.schema_digest,
            "capabilities": sorted(self.capabilities),
            "resources": dict(self.resources),
            "values": dict(self.values),
        }
        if include_digest:
            record["identity_digest"] = self.identity_digest
        return record


@dataclass(frozen=True, slots=True)
class ObservationRequirement:
    """Typed skill applicability contract."""

    schema_digest: str
    allowed_nodes: tuple[Hashable, ...] = ()
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    required_values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_digest",
            _nonempty(self.schema_digest, "schema_digest"),
        )
        object.__setattr__(self, "allowed_nodes", tuple(self.allowed_nodes))
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(
                _nonempty(value, "required capability")
                for value in self.required_capabilities
            ),
        )
        object.__setattr__(self, "required_values", dict(self.required_values))
        _canonical_json(self.to_record())

    @property
    def identity_digest(self) -> str:
        return _digest("observation-requirement-v1", self.to_record())

    def mismatches(self, observation: SolverObservation) -> tuple[str, ...]:
        errors: list[str] = []
        if observation.schema_digest != self.schema_digest:
            errors.append("observation schema digest")
        if self.allowed_nodes and observation.node_id not in self.allowed_nodes:
            errors.append("observation node")
        missing = self.required_capabilities - observation.capabilities
        if missing:
            errors.append(f"capabilities:{','.join(sorted(missing))}")
        for key, expected in sorted(self.required_values.items()):
            if observation.values.get(key) != expected:
                errors.append(f"value:{key}")
        return tuple(errors)

    def matches(self, observation: SolverObservation) -> bool:
        return not self.mismatches(observation)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_digest": self.schema_digest,
            "allowed_nodes": [_node_record(value) for value in self.allowed_nodes],
            "required_capabilities": sorted(self.required_capabilities),
            "required_values": dict(self.required_values),
        }


@dataclass(frozen=True, slots=True)
class ProgressionDelta:
    """Expected postcondition for a successful skill."""

    target_node: Hashable | None = None
    acquired_capabilities: frozenset[str] = field(default_factory=frozenset)
    resource_deltas: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acquired_capabilities",
            frozenset(
                _nonempty(value, "acquired capability")
                for value in self.acquired_capabilities
            ),
        )
        normalized: dict[str, float] = {}
        for name, value in self.resource_deltas.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("expected resource deltas must be numeric")
            normalized[_nonempty(name, "resource name")] = float(value)
        object.__setattr__(self, "resource_deltas", dict(sorted(normalized.items())))

    @property
    def identity_digest(self) -> str:
        return _digest("progression-delta-v1", self.to_record())

    def mismatches(
        self,
        before: SolverObservation,
        after: SolverObservation,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        if self.target_node is not None and after.node_id != self.target_node:
            errors.append("target node")
        missing = self.acquired_capabilities - (
            after.capabilities - before.capabilities
        )
        if missing:
            errors.append(f"capability delta:{','.join(sorted(missing))}")
        for name, expected in self.resource_deltas.items():
            actual = after.resources.get(name, 0.0) - before.resources.get(name, 0.0)
            if actual != expected:
                errors.append(f"resource delta:{name}")
        return tuple(errors)

    def to_record(self) -> dict[str, Any]:
        return {
            "target_node": (
                _node_record(self.target_node)
                if self.target_node is not None
                else None
            ),
            "acquired_capabilities": sorted(self.acquired_capabilities),
            "resource_deltas": dict(self.resource_deltas),
        }


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """Lifecycle and progression contract for one reusable skill."""

    skill_id: str
    dispatch_key: str
    observation_requirement: ObservationRequirement
    expected_delta: ProgressionDelta
    timeout_frames: int
    max_retries: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _nonempty(self.skill_id, "skill_id"))
        object.__setattr__(
            self,
            "dispatch_key",
            _nonempty(self.dispatch_key, "dispatch_key"),
        )
        if not isinstance(self.observation_requirement, ObservationRequirement):
            raise TypeError("observation_requirement must be ObservationRequirement")
        if not isinstance(self.expected_delta, ProgressionDelta):
            raise TypeError("expected_delta must be ProgressionDelta")
        for field_name, minimum in (("timeout_frames", 1), ("max_retries", 0)):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be >= {minimum}")

    @property
    def identity_digest(self) -> str:
        return _digest(
            "skill-spec-v1",
            {
                "skill_id": self.skill_id,
                "dispatch_key": self.dispatch_key,
                "observation_requirement_digest": (
                    self.observation_requirement.identity_digest
                ),
                "expected_delta_digest": self.expected_delta.identity_digest,
                "timeout_frames": self.timeout_frames,
                "max_retries": self.max_retries,
            },
        )


class SkillSignal(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"


@dataclass(frozen=True, slots=True)
class SkillStep:
    signal: SkillSignal
    action: Any = None
    reason: str | None = None
    recovery_hint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.signal, SkillSignal):
            raise TypeError("signal must be a SkillSignal")


class SkillPolicy(Protocol):
    def reset(
        self,
        observation: SolverObservation,
        config: Mapping[str, Any],
    ) -> None: ...

    def step(self, observation: SolverObservation) -> SkillStep: ...


@dataclass(frozen=True, slots=True)
class SkillInstance:
    """Policy/config/identity bound to a validated SkillSpec and edge binding."""

    spec: SkillSpec
    binding: SkillBinding
    policy: SkillPolicy
    policy_identity: PolicyIdentity
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.binding.skill_id != self.spec.skill_id:
            raise ValueError("SkillBinding skill_id does not match SkillSpec")
        if self.binding.dispatch_key != self.spec.dispatch_key:
            raise ValueError("SkillBinding dispatch_key does not match SkillSpec")
        if (
            self.binding.entry_requirement_digest
            != self.spec.observation_requirement.identity_digest
        ):
            raise ValueError("SkillBinding entry requirement does not match SkillSpec")
        if (
            self.binding.progression_delta_digest
            != self.spec.expected_delta.identity_digest
        ):
            raise ValueError("SkillBinding progression delta does not match SkillSpec")
        if not isinstance(self.policy_identity, PolicyIdentity):
            raise TypeError("policy_identity must be a PolicyIdentity")
        object.__setattr__(self, "config", dict(self.config))
        _canonical_json(dict(self.config))


class SkillOutcomeStatus(str, Enum):
    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class SkillOutcome:
    edge_id: str
    skill_id: str
    status: SkillOutcomeStatus
    frames: int
    start_observation_digest: str
    end_observation_digest: str
    observed_capability_delta: frozenset[str]
    observed_resource_delta: Mapping[str, float]
    reason: str | None = None
    recovery_hint: str | None = None
    replan: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "skill_id": self.skill_id,
            "status": self.status.value,
            "frames": self.frames,
            "start_observation_digest": self.start_observation_digest,
            "end_observation_digest": self.end_observation_digest,
            "observed_capability_delta": sorted(self.observed_capability_delta),
            "observed_resource_delta": dict(self.observed_resource_delta),
            "reason": self.reason,
            "recovery_hint": self.recovery_hint,
            "replan": self.replan,
        }


class SolverLifecycle(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    EXECUTING = "executing"
    VALIDATING = "validating"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SolverTraceEvent:
    sequence: int
    lifecycle: SolverLifecycle
    observation_digest: str
    edge_id: str | None = None
    skill_id: str | None = None
    policy_identity_digest: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "lifecycle": self.lifecycle.value,
            "observation_digest": self.observation_digest,
            "edge_id": self.edge_id,
            "skill_id": self.skill_id,
            "policy_identity_digest": self.policy_identity_digest,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class SolverActionEvent:
    """One exact action application with its observation boundary."""

    sequence: int
    edge_id: str
    skill_id: str
    policy_identity_digest: str
    frame_start: int
    frame_end: int
    applied_frames: int
    observation_before: SolverObservation
    observation_after: SolverObservation
    action: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("action sequence must be non-negative")
        if self.frame_end < self.frame_start:
            raise ValueError("action frame range is reversed")
        if self.applied_frames < 1:
            raise ValueError("applied_frames must be positive")
        object.__setattr__(self, "action", dict(self.action))
        _canonical_json(self.to_record())

    def to_record(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "edge_id": self.edge_id,
            "skill_id": self.skill_id,
            "policy_identity_digest": self.policy_identity_digest,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "applied_frames": self.applied_frames,
            "observation_before": self.observation_before.to_record(),
            "observation_after": self.observation_after.to_record(),
            "action": dict(self.action),
        }


class SolverResultStatus(str, Enum):
    COMPLETED = "completed"
    PLAN_FAILED = "plan_failed"
    TERMINAL_FAILURE = "terminal_failure"
    REPLAN_EXHAUSTED = "replan_exhausted"


@dataclass(frozen=True, slots=True)
class SolverSessionResult:
    status: SolverResultStatus
    final_observation: SolverObservation
    outcomes: tuple[SkillOutcome, ...]
    trace: tuple[SolverTraceEvent, ...]
    replans: int
    completed_edges: tuple[str, ...]
    actions: tuple[SolverActionEvent, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "final_observation": self.final_observation.to_record(),
            "outcomes": [outcome.to_record() for outcome in self.outcomes],
            "trace": [event.to_record() for event in self.trace],
            "replans": self.replans,
            "completed_edges": list(self.completed_edges),
            "actions": [event.to_record() for event in self.actions],
        }


PlanFunction = Callable[[SolverObservation, frozenset[str]], PlanResult]


class SolverSession:
    """Execute bounded plans and recover by excluding retryable failed edges."""

    def __init__(
        self,
        *,
        observe: Callable[[], SolverObservation],
        apply_action: Callable[[Any], int | None],
        plan_fn: PlanFunction,
        bindings: BindingCatalog,
        skills: Iterable[SkillInstance],
        minimum_readiness: ExecutionReadiness = ExecutionReadiness.NATURAL_ENTRY,
        max_replans: int = 3,
    ) -> None:
        if not isinstance(bindings, BindingCatalog):
            raise TypeError("bindings must be a BindingCatalog")
        if not isinstance(minimum_readiness, ExecutionReadiness):
            raise TypeError("minimum_readiness must be an ExecutionReadiness")
        if isinstance(max_replans, bool) or not isinstance(max_replans, int) or max_replans < 0:
            raise ValueError("max_replans must be a non-negative integer")
        by_dispatch: dict[str, SkillInstance] = {}
        for instance in skills:
            key = instance.spec.dispatch_key
            if key in by_dispatch:
                raise ValueError(f"duplicate skill dispatch key: {key!r}")
            by_dispatch[key] = instance
        self._observe = observe
        self._apply_action = apply_action
        self._plan_fn = plan_fn
        self.bindings = bindings
        self.skills = by_dispatch
        self.minimum_readiness = minimum_readiness
        self.max_replans = max_replans
        self.lifecycle = SolverLifecycle.IDLE
        self._trace: list[SolverTraceEvent] = []
        self._actions: list[SolverActionEvent] = []

    def run(self) -> SolverSessionResult:
        self._trace.clear()
        self._actions.clear()
        observation = self._observe()
        outcomes: list[SkillOutcome] = []
        completed: list[str] = []
        excluded: set[str] = set()
        replans = 0

        while True:
            self._transition(
                SolverLifecycle.REPLANNING if replans else SolverLifecycle.PLANNING,
                observation,
                detail={"excluded_edges": sorted(excluded)},
            )
            plan_result = self._plan_fn(observation, frozenset(excluded))
            self._transition(
                SolverLifecycle.REPLANNING if replans else SolverLifecycle.PLANNING,
                observation,
                detail={"plan_result": plan_result.to_record()},
            )
            if plan_result.status is not PlanStatus.FOUND:
                return self._finish(
                    SolverResultStatus.PLAN_FAILED,
                    observation,
                    outcomes,
                    completed,
                    replans,
                    detail={"plan_status": plan_result.status.value},
                )
            if not plan_result.path:
                return self._finish(
                    SolverResultStatus.COMPLETED,
                    observation,
                    outcomes,
                    completed,
                    replans,
                )

            should_replan = False
            for edge in plan_result.path:
                binding = self.bindings.binding_for(edge.edge_id)
                if binding is None or binding.readiness < self.minimum_readiness:
                    return self._finish(
                        SolverResultStatus.TERMINAL_FAILURE,
                        observation,
                        outcomes,
                        completed,
                        replans,
                        detail={"reason": "edge binding is not execution-ready", "edge_id": edge.edge_id},
                    )
                instance = self.skills.get(binding.dispatch_key)
                if instance is None or instance.binding.identity_digest != binding.identity_digest:
                    return self._finish(
                        SolverResultStatus.TERMINAL_FAILURE,
                        observation,
                        outcomes,
                        completed,
                        replans,
                        detail={"reason": "skill dispatch binding mismatch", "edge_id": edge.edge_id},
                    )
                self._transition(
                    SolverLifecycle.DISPATCHING,
                    observation,
                    edge_id=edge.edge_id,
                    instance=instance,
                )
                retry_index = 0
                while True:
                    outcome, observation = self._invoke(
                        edge.edge_id,
                        instance,
                        observation,
                    )
                    outcomes.append(outcome)
                    if (
                        outcome.status is SkillOutcomeStatus.RETRYABLE_FAILURE
                        and retry_index < instance.spec.max_retries
                    ):
                        retry_index += 1
                        continue
                    break
                if outcome.status is SkillOutcomeStatus.SUCCESS:
                    completed.append(edge.edge_id)
                    continue
                if outcome.status is SkillOutcomeStatus.RETRYABLE_FAILURE:
                    excluded.add(edge.edge_id)
                    replans += 1
                    if replans > self.max_replans:
                        return self._finish(
                            SolverResultStatus.REPLAN_EXHAUSTED,
                            observation,
                            outcomes,
                            completed,
                            replans,
                        )
                    should_replan = True
                    break
                return self._finish(
                    SolverResultStatus.TERMINAL_FAILURE,
                    observation,
                    outcomes,
                    completed,
                    replans,
                )
            if should_replan:
                continue
            # Observe and plan again: edges may acquire items or reveal a new node.
            observation = self._observe()

    def _invoke(
        self,
        edge_id: str,
        instance: SkillInstance,
        start: SolverObservation,
    ) -> tuple[SkillOutcome, SolverObservation]:
        mismatches = instance.spec.observation_requirement.mismatches(start)
        if mismatches:
            outcome = self._outcome(
                edge_id,
                instance,
                SkillOutcomeStatus.RETRYABLE_FAILURE,
                0,
                start,
                start,
                reason="precondition failed: " + ", ".join(mismatches),
                recovery_hint="reobserve_and_replan",
                replan=True,
            )
            self._trace_outcome(start, instance, outcome)
            return outcome, start

        instance.policy.reset(start, instance.config)
        current = start
        frames = 0
        self._transition(
            SolverLifecycle.EXECUTING,
            current,
            edge_id=edge_id,
            instance=instance,
        )
        while frames < instance.spec.timeout_frames:
            step = instance.policy.step(current)
            if not isinstance(step, SkillStep):
                raise TypeError("skill policy step must return SkillStep")
            if step.action is not None:
                before_action = current
                action_record = canonical_action_record(step.action)
                applied_frames = self._apply_action(step.action)
                if applied_frames is None:
                    applied_frames = 1
                if (
                    isinstance(applied_frames, bool)
                    or not isinstance(applied_frames, int)
                    or applied_frames < 1
                ):
                    raise ValueError(
                        "apply_action must return a positive frame count or None"
                    )
                frames += applied_frames
                current = self._observe()
                self._actions.append(
                    SolverActionEvent(
                        sequence=len(self._actions),
                        edge_id=edge_id,
                        skill_id=instance.spec.skill_id,
                        policy_identity_digest=(
                            instance.policy_identity.identity_digest
                        ),
                        frame_start=before_action.frame,
                        frame_end=current.frame,
                        applied_frames=applied_frames,
                        observation_before=before_action,
                        observation_after=current,
                        action=action_record,
                    )
                )
                self._transition(
                    SolverLifecycle.EXECUTING,
                    current,
                    edge_id=edge_id,
                    instance=instance,
                    detail={"action": action_record, "skill_frame": frames},
                )
            if step.signal is SkillSignal.RUNNING:
                continue
            if step.signal is SkillSignal.SUCCESS:
                self._transition(
                    SolverLifecycle.VALIDATING,
                    current,
                    edge_id=edge_id,
                    instance=instance,
                )
                post_errors = instance.spec.expected_delta.mismatches(start, current)
                status = (
                    SkillOutcomeStatus.SUCCESS
                    if not post_errors
                    else SkillOutcomeStatus.RETRYABLE_FAILURE
                )
                outcome = self._outcome(
                    edge_id,
                    instance,
                    status,
                    frames,
                    start,
                    current,
                    reason=(
                        None
                        if not post_errors
                        else "postcondition failed: " + ", ".join(post_errors)
                    ),
                    recovery_hint=(
                        step.recovery_hint or "reobserve_and_replan"
                        if post_errors
                        else step.recovery_hint
                    ),
                    replan=bool(post_errors),
                )
                self._trace_outcome(current, instance, outcome)
                return outcome, current
            status = (
                SkillOutcomeStatus.RETRYABLE_FAILURE
                if step.signal is SkillSignal.RETRYABLE_FAILURE
                else SkillOutcomeStatus.TERMINAL_FAILURE
            )
            outcome = self._outcome(
                edge_id,
                instance,
                status,
                frames,
                start,
                current,
                reason=step.reason,
                recovery_hint=step.recovery_hint,
                replan=status is SkillOutcomeStatus.RETRYABLE_FAILURE,
            )
            self._trace_outcome(current, instance, outcome)
            return outcome, current

        outcome = self._outcome(
            edge_id,
            instance,
            SkillOutcomeStatus.TIMEOUT,
            frames,
            start,
            current,
            reason="skill timeout",
            recovery_hint="terminal_timeout",
        )
        self._trace_outcome(current, instance, outcome)
        return outcome, current

    def _outcome(
        self,
        edge_id: str,
        instance: SkillInstance,
        status: SkillOutcomeStatus,
        frames: int,
        start: SolverObservation,
        end: SolverObservation,
        *,
        reason: str | None = None,
        recovery_hint: str | None = None,
        replan: bool = False,
    ) -> SkillOutcome:
        resource_names = set(start.resources) | set(end.resources)
        return SkillOutcome(
            edge_id=edge_id,
            skill_id=instance.spec.skill_id,
            status=status,
            frames=frames,
            start_observation_digest=start.identity_digest,
            end_observation_digest=end.identity_digest,
            observed_capability_delta=end.capabilities - start.capabilities,
            observed_resource_delta={
                name: end.resources.get(name, 0.0) - start.resources.get(name, 0.0)
                for name in sorted(resource_names)
            },
            reason=reason,
            recovery_hint=recovery_hint,
            replan=replan,
        )

    def _transition(
        self,
        lifecycle: SolverLifecycle,
        observation: SolverObservation,
        *,
        edge_id: str | None = None,
        instance: SkillInstance | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        self._trace.append(
            SolverTraceEvent(
                sequence=len(self._trace),
                lifecycle=lifecycle,
                observation_digest=observation.identity_digest,
                edge_id=edge_id,
                skill_id=instance.spec.skill_id if instance else None,
                policy_identity_digest=(
                    instance.policy_identity.identity_digest if instance else None
                ),
                detail=dict(detail or {}),
            )
        )

    def _trace_outcome(
        self,
        observation: SolverObservation,
        instance: SkillInstance,
        outcome: SkillOutcome,
    ) -> None:
        self._transition(
            SolverLifecycle.VALIDATING,
            observation,
            edge_id=outcome.edge_id,
            instance=instance,
            detail={"outcome": outcome.to_record()},
        )

    def _finish(
        self,
        status: SolverResultStatus,
        observation: SolverObservation,
        outcomes: list[SkillOutcome],
        completed: list[str],
        replans: int,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> SolverSessionResult:
        lifecycle = (
            SolverLifecycle.COMPLETED
            if status is SolverResultStatus.COMPLETED
            else SolverLifecycle.FAILED
        )
        self._transition(lifecycle, observation, detail=detail)
        return SolverSessionResult(
            status=status,
            final_observation=observation,
            outcomes=tuple(outcomes),
            trace=tuple(self._trace),
            replans=replans,
            completed_edges=tuple(completed),
            actions=tuple(self._actions),
        )
