"""Canonical trajectory export and counterexample replay tests."""

from __future__ import annotations

import json

import pytest

from retro_harness.solver import (
    SkillOutcome,
    SkillOutcomeStatus,
    SolverActionEvent,
    SolverObservation,
    SolverResultStatus,
    SolverSessionResult,
    canonical_action_record,
)
from retro_harness.trajectory import (
    CounterexampleLibrary,
    TRAJECTORY_SCHEMA_DIGEST,
    Trajectory,
    TrajectoryError,
    counterexamples_from_solver_result,
    trajectory_from_solver_result,
)


def _failed_result() -> SolverSessionResult:
    before = SolverObservation(0, "start", "observation-v1", resources={"keys": 1})
    after = SolverObservation(1, "start", "observation-v1", resources={"keys": 0})
    action = SolverActionEvent(
        sequence=0,
        edge_id="locked-door",
        skill_id="open-door",
        policy_identity_digest="policy-v1",
        frame_start=0,
        frame_end=1,
        applied_frames=1,
        observation_before=before,
        observation_after=after,
        action=canonical_action_record({"buttons": [0, 1]}),
    )
    outcome = SkillOutcome(
        edge_id="locked-door",
        skill_id="open-door",
        status=SkillOutcomeStatus.RETRYABLE_FAILURE,
        frames=1,
        start_observation_digest=before.identity_digest,
        end_observation_digest=after.identity_digest,
        observed_capability_delta=frozenset(),
        observed_resource_delta={"keys": -1},
        reason="door did not open",
        recovery_hint="find another key",
        replan=True,
    )
    return SolverSessionResult(
        status=SolverResultStatus.PLAN_FAILED,
        final_observation=after,
        outcomes=(outcome,),
        trace=(),
        replans=1,
        completed_edges=(),
        actions=(action,),
    )


def _export(result: SolverSessionResult):
    return trajectory_from_solver_result(
        result,
        action_schema_digest="actions-v1",
        reward_schema_digest="rewards-v1",
        contract_bundle_digest="bundle-v1",
        policy_identity_digest="policy-v1",
        provenance={"run_id": "fixture-1"},
        reward_fn=lambda event: {
            "key_cost": event.observation_after.resources["keys"]
            - event.observation_before.resources["keys"]
        },
    )


def test_solver_export_has_contract_identity_actions_rewards_and_provenance(tmp_path):
    trajectory = _export(_failed_result())

    assert trajectory.to_record()["schema_digest"] == TRAJECTORY_SCHEMA_DIGEST
    assert trajectory.steps[0].action["value"] == {"buttons": [0, 1]}
    assert trajectory.steps[0].reward_components == {"key_cost": -1.0}
    assert trajectory.provenance == {"run_id": "fixture-1", "source": "SolverSession"}
    assert trajectory.terminal_reason == "door did not open"

    path = trajectory.write(tmp_path / "trajectory.json")
    assert Trajectory.load(path) == trajectory

    record = json.loads(path.read_text(encoding="utf-8"))
    record["steps"][0]["action"]["value"]["buttons"] = [1, 0]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(TrajectoryError, match="identity digest mismatch"):
        Trajectory.load(path)


def test_failed_outcome_is_retained_and_imported_by_cluster(tmp_path):
    failures = counterexamples_from_solver_result(
        _failed_result(),
        action_schema_digest="actions-v1",
        reward_schema_digest="rewards-v1",
        contract_bundle_digest="bundle-v1",
        policy_identity_digest="policy-v1",
        provenance={"run_id": "fixture-1"},
    )
    assert len(failures) == 1
    assert failures[0].succeeded is False

    library = CounterexampleLibrary(tmp_path / "counterexamples")
    library.add(failures[0])
    cluster = library.cluster_key(failures[0])

    assert library.trajectories(cluster=cluster) == failures
    assert library.offline_actions(cluster=cluster) == (
        {"type": "builtins.dict", "value": {"buttons": [0, 1]}},
    )


def test_counterexample_library_rejects_successes(tmp_path):
    failed = _export(_failed_result())
    successful = Trajectory(
        observation_schema_digest=failed.observation_schema_digest,
        action_schema_digest=failed.action_schema_digest,
        reward_schema_digest=failed.reward_schema_digest,
        contract_bundle_digest=failed.contract_bundle_digest,
        policy_identity_digest=failed.policy_identity_digest,
        steps=failed.steps,
        succeeded=True,
        terminal_reason="completed",
    )

    with pytest.raises(TrajectoryError, match="failed trajectories only"):
        CounterexampleLibrary(tmp_path).add(successful)
