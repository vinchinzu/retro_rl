"""SolverSession execution kernel: plan, dispatch, validate, replan.

Immutable observation/spec/outcome/trace types live in
:mod:`retro_harness.solver_domain`. The public compatibility surface remains
:mod:`retro_harness.solver`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from retro_harness.adventure.bindings import (
    BindingCatalog,
    ExecutionReadiness,
)
from retro_harness.adventure.planner import PlanResult, PlanStatus
from retro_harness.solver_domain import (
    SkillInstance,
    SkillOutcome,
    SkillOutcomeStatus,
    SkillSignal,
    SkillStep,
    SolverActionEvent,
    SolverLifecycle,
    SolverObservation,
    SolverResultStatus,
    SolverSessionResult,
    SolverTraceEvent,
    canonical_action_record,
)

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
