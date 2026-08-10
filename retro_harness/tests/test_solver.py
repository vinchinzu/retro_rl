"""Lifecycle, dispatch, validation, and recovery tests for SolverSession."""

from __future__ import annotations

from retro_harness.adventure import (
    BindingCatalog,
    ExecutionReadiness,
    GraphEdge,
    PlanRequest,
    SkillBinding,
    plan,
)
from retro_harness.benchmark import PolicyIdentity
from retro_harness.solver import (
    ObservationRequirement,
    ProgressionDelta,
    SkillInstance,
    SkillOutcomeStatus,
    SkillSignal,
    SkillSpec,
    SkillStep,
    SolverLifecycle,
    SolverObservation,
    SolverResultStatus,
    SolverSession,
)


class FixturePolicy:
    def __init__(self, steps):
        self.steps = list(steps)
        self.index = 0
        self.reset_calls = 0

    def reset(self, observation, config):
        self.index = 0
        self.reset_calls += 1

    def step(self, observation):
        value = self.steps[min(self.index, len(self.steps) - 1)]
        self.index += 1
        return value


def _spec(edge_id, source, target, *, timeout=3):
    requirement = ObservationRequirement(
        schema_digest="fixture-obs-v1",
        allowed_nodes=(source,),
    )
    delta = ProgressionDelta(target_node=target)
    spec = SkillSpec(
        skill_id=f"skill:{edge_id}",
        dispatch_key=f"dispatch:{edge_id}",
        observation_requirement=requirement,
        expected_delta=delta,
        timeout_frames=timeout,
    )
    binding = SkillBinding(
        edge_id=edge_id,
        skill_id=spec.skill_id,
        dispatch_key=spec.dispatch_key,
        entry_requirement_digest=requirement.identity_digest,
        progression_delta_digest=delta.identity_digest,
        readiness=ExecutionReadiness.NATURAL_ENTRY,
        evidence_digest=f"evidence:{edge_id}",
    )
    return spec, binding


def _instance(edge_id, source, target, policy, *, timeout=3):
    spec, binding = _spec(edge_id, source, target, timeout=timeout)
    return SkillInstance(
        spec=spec,
        binding=binding,
        policy=policy,
        policy_identity=PolicyIdentity(f"policy:{edge_id}"),
    )


def _runtime(edges, instances, world, *, max_replans=3):
    by_action = {
        "bad": "start",
        "good": "goal",
        "middle": "middle",
        "finish": "goal",
        "noop": None,
    }

    def observe():
        return SolverObservation(
            frame=world["frame"],
            node_id=world["node"],
            schema_digest="fixture-obs-v1",
            capabilities=frozenset(world.get("capabilities", ())),
            resources=world.get("resources", {}),
        )

    def apply_action(action):
        world["frame"] += 1
        target = by_action[action]
        if target is not None:
            world["node"] = target
        return 1

    def plan_fn(observation, excluded):
        allowed = tuple(edge for edge in edges if edge.edge_id not in excluded)
        return plan(PlanRequest(allowed, observation.node_id, "goal"))

    return SolverSession(
        observe=observe,
        apply_action=apply_action,
        plan_fn=plan_fn,
        bindings=BindingCatalog(instance.binding for instance in instances),
        skills=instances,
        max_replans=max_replans,
    )


def test_solver_session_executes_multi_edge_lifecycle_and_trace_identity():
    world = {"frame": 0, "node": "start"}
    first = _instance(
        "first",
        "start",
        "middle",
        FixturePolicy([SkillStep(SkillSignal.SUCCESS, action="middle")]),
    )
    second = _instance(
        "second",
        "middle",
        "goal",
        FixturePolicy([SkillStep(SkillSignal.SUCCESS, action="finish")]),
    )
    result = _runtime(
        (
            GraphEdge("start", "middle", edge_id="first"),
            GraphEdge("middle", "goal", edge_id="second"),
        ),
        (first, second),
        world,
    ).run()

    assert result.status is SolverResultStatus.COMPLETED
    assert result.completed_edges == ("first", "second")
    assert [outcome.status for outcome in result.outcomes] == [
        SkillOutcomeStatus.SUCCESS,
        SkillOutcomeStatus.SUCCESS,
    ]
    assert result.final_observation.node_id == "goal"
    assert [event.action["value"] for event in result.actions] == [
        "middle",
        "finish",
    ]
    assert result.actions[0].observation_before.node_id == "start"
    assert result.actions[0].observation_after.node_id == "middle"
    lifecycle = [event.lifecycle for event in result.trace]
    assert SolverLifecycle.PLANNING in lifecycle
    assert SolverLifecycle.DISPATCHING in lifecycle
    assert SolverLifecycle.EXECUTING in lifecycle
    assert SolverLifecycle.VALIDATING in lifecycle
    assert lifecycle[-1] is SolverLifecycle.COMPLETED
    policy_events = [event for event in result.trace if event.policy_identity_digest]
    assert all(event.policy_identity_digest for event in policy_events)


def test_injected_retryable_failure_replans_to_successful_parallel_edge():
    world = {"frame": 0, "node": "start"}
    failed = _instance(
        "bad-edge",
        "start",
        "goal",
        FixturePolicy(
            [
                SkillStep(
                    SkillSignal.RETRYABLE_FAILURE,
                    reason="blocked transition",
                    recovery_hint="try_parallel_edge",
                )
            ]
        ),
    )
    recovered = _instance(
        "good-edge",
        "start",
        "goal",
        FixturePolicy([SkillStep(SkillSignal.SUCCESS, action="good")]),
    )
    session = _runtime(
        (
            GraphEdge("start", "goal", edge_id="bad-edge", cost=1),
            GraphEdge("start", "goal", edge_id="good-edge", cost=2),
        ),
        (failed, recovered),
        world,
    )
    result = session.run()

    assert result.status is SolverResultStatus.COMPLETED
    assert result.replans == 1
    assert [outcome.status for outcome in result.outcomes] == [
        SkillOutcomeStatus.RETRYABLE_FAILURE,
        SkillOutcomeStatus.SUCCESS,
    ]
    assert result.completed_edges == ("good-edge",)
    assert any(event.lifecycle is SolverLifecycle.REPLANNING for event in result.trace)
    assert result.outcomes[0].recovery_hint == "try_parallel_edge"


def test_timeout_is_classified_and_stops_session():
    world = {"frame": 0, "node": "start"}
    instance = _instance(
        "timeout",
        "start",
        "goal",
        FixturePolicy([SkillStep(SkillSignal.RUNNING, action="noop")]),
        timeout=2,
    )
    result = _runtime(
        (GraphEdge("start", "goal", edge_id="timeout"),),
        (instance,),
        world,
    ).run()

    assert result.status is SolverResultStatus.TERMINAL_FAILURE
    assert result.outcomes[0].status is SkillOutcomeStatus.TIMEOUT
    assert result.outcomes[0].frames == 2


def test_success_signal_with_bad_postcondition_becomes_retryable_failure():
    world = {"frame": 0, "node": "start"}
    instance = _instance(
        "wrong-target",
        "start",
        "goal",
        FixturePolicy([SkillStep(SkillSignal.SUCCESS, action="bad")]),
    )
    result = _runtime(
        (GraphEdge("start", "goal", edge_id="wrong-target"),),
        (instance,),
        world,
    ).run()

    assert result.status is SolverResultStatus.PLAN_FAILED
    assert result.replans == 1
    assert result.outcomes[0].status is SkillOutcomeStatus.RETRYABLE_FAILURE
    assert "postcondition failed" in result.outcomes[0].reason


def test_action_adapter_can_report_multi_frame_skill_duration():
    world = {"frame": 0, "node": "start"}
    instance = _instance(
        "macro",
        "start",
        "goal",
        FixturePolicy([SkillStep(SkillSignal.SUCCESS, action="finish")]),
        timeout=10,
    )

    def observe():
        return SolverObservation(
            frame=world["frame"],
            node_id=world["node"],
            schema_digest="fixture-obs-v1",
        )

    def apply_action(action):
        assert action == "finish"
        world.update(frame=7, node="goal")
        return 7

    edge = GraphEdge("start", "goal", edge_id="macro")
    result = SolverSession(
        observe=observe,
        apply_action=apply_action,
        plan_fn=lambda observation, excluded: plan(
            PlanRequest((edge,), observation.node_id, "goal")
        ),
        bindings=BindingCatalog((instance.binding,)),
        skills=(instance,),
    ).run()

    assert result.status is SolverResultStatus.COMPLETED
    assert result.outcomes[0].frames == 7
