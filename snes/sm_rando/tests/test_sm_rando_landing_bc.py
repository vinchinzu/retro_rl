"""Pure checks for the held-out Landing behavior-cloning experiment."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.benchmark import validate_claim
from retro_harness.entry_states import EntryStateRecord
from retro_harness.trajectory import Trajectory
from sm_rando.entry_corpus import landing_corpus_contracts
from sm_rando.landing_bc import (
    EXPERT_HANDOFF_LANDING_FRAME,
    LANDING_BC_REPORT,
    LandingBCEvalResult,
    LandingBCTrajectoryCapture,
    build_landing_bc_contracts,
    build_landing_bc_trajectory,
    export_landing_bc_trajectories,
    fit_landing_bc_model,
    package_landing_bc_report,
    partition_metrics,
    write_landing_bc_report,
)
from sm_rando.paths import REPO_ROOT


def _record(index: int, landing_frame: int, y: int, y_sub: int) -> EntryStateRecord:
    return EntryStateRecord(
        state_digest=f"state-{index}",
        ram_snapshot_digest=f"ram-{index}",
        state_path=f"states/{index}.state",
        source_skill_id="predecessor",
        source_segment_id="landing",
        source_trajectory_digest="trajectory",
        frame=index,
        observation_schema_digest="observation",
        contract_bundle_digest="bundle",
        metadata={
            "observation_metadata_version": 2,
            "room_id": 0x91F8,
            "game_state": 8,
            "door_transition": 0,
            "samus_x": 1152,
            "samus_x_sub": 0,
            "samus_y": y,
            "samus_y_sub": y_sub,
            "velocity_x": 0,
            "velocity_y": 0,
            "health": 99,
            "missiles": 0,
            "pose": 0,
            "timing": {"landing_frame": landing_frame},
        },
    )


def test_bc_contract_reuses_corpus_observation_but_declares_macro_actions() -> None:
    corpus = landing_corpus_contracts()
    bc = build_landing_bc_contracts()

    assert bc.observation.identity_digest == corpus.observation.identity_digest
    assert bc.action.action_count == 2
    assert [entry.action_id for entry in bc.action.entries] == ["wait", "dispatch"]
    assert bc.environment.action_space_size == 2


def test_linear_behavior_clone_fits_expert_wait_demonstrations() -> None:
    records = tuple(
        _record(index, landing_frame, 200 + 4 * landing_frame, 0)
        for index, landing_frame in enumerate((1, 10, 40, 80, 120, 180))
    )

    model, metrics = fit_landing_bc_model(records)

    prediction = model.predict_wait(
        [0, 0, 0, 0, 0, 200 + 4 * 60, 0, 0, 0, 0, 0, 0]
    )
    assert prediction == EXPERT_HANDOFF_LANDING_FRAME - 60
    assert metrics["max_abs_error_frames"] == 0


def test_retained_bc_report_is_held_out_audited_and_replayable() -> None:
    report = json.loads(LANDING_BC_REPORT.read_text(encoding="utf-8"))

    assert report["training"]["eval_states_used_for_fit"] == 0
    assert report["metrics"]["train"]["successes"] == 58
    assert report["metrics"]["eval"]["successes"] == 6
    assert report["beats_structured_baseline"] is True
    assert "not_deployed" in report["decision"]
    for attempt in report["attempts"]:
        assert validate_claim(attempt)
    trajectories = tuple(
        Trajectory.load(REPO_ROOT / path) for path in report["eval_trajectories"]
    )
    assert len(trajectories) == 6
    assert all(value.succeeded for value in trajectories)


def test_partition_metrics_and_report_packaging_are_pure() -> None:
    """Report ownership: metrics + packaging need no ROM."""
    attempts = (
        {
            "partition": "train",
            "success": True,
            "predicted_wait": 10,
            "wait_error": 0,
        },
        {
            "partition": "train",
            "success": False,
            "predicted_wait": 12,
            "wait_error": 2,
        },
        {
            "partition": "eval",
            "success": True,
            "predicted_wait": 8,
            "wait_error": -1,
        },
        {
            "partition": "eval",
            "success": True,
            "predicted_wait": 9,
            "wait_error": 0,
        },
    )
    train = partition_metrics(attempts, "train")
    eval_m = partition_metrics(attempts, "eval")
    assert train["attempts"] == 2
    assert train["successes"] == 1
    assert train["success_rate"] == 0.5
    assert eval_m["successes"] == 2
    assert eval_m["success_rate"] == 1.0

    # Synthetic eval result under REPO_ROOT-relative paths for packaging.
    corpus_path = REPO_ROOT / "snes" / "sm_rando" / "custom_integrations" / "x"
    checkpoint_path = REPO_ROOT / "snes" / "sm_rando" / "models" / "landing_wait_bc_v1.json"
    eval_result = LandingBCEvalResult(
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
        corpus_digest="corpus-d",
        contract_bundle_digest="contracts-d",
        observation_schema_digest="obs-d",
        action_schema_digest="act-d",
        reward_schema_digest="rew-d",
        policy_identity_digest="policy-d",
        policy_artifact_digest="artifact-d",
        train_state_count=2,
        train_metrics={"mae_frames": 0.0, "examples": 2.0},
        attempts=tuple(dict(a) for a in attempts),
        eval_captures=(),
    )
    report = package_landing_bc_report(
        eval_result,
        eval_trajectories=["snes/sm_rando/recordings/t0.json"],
        baseline_report_path=Path("/nonexistent/baseline.json"),
        generated_at="2026-08-09T00:00:00+00:00",
    )
    assert report["training"]["eval_states_used_for_fit"] == 0
    assert report["training"]["states"] == 2
    assert report["metrics"]["eval"]["success_rate"] == 1.0
    assert report["metrics"]["generalization_gap"] == -0.5
    assert report["structured_baseline_eval_rate"] is None
    assert report["beats_structured_baseline"] is False
    assert "do_not_deploy" in report["decision"]
    assert report["intervention_class"] == "Clean"
    assert report["runtime_observation_class"] == "Bronze"
    assert report["eval_trajectories"] == ["snes/sm_rando/recordings/t0.json"]


def test_trajectory_export_is_report_owned_and_disk_local(
    tmp_path: Path,
) -> None:
    """Trajectory export ownership does not require ROM re-entry."""
    attempt = {
        "state_digest": "eval-state-1",
        "partition": "eval",
        "success": True,
        "predicted_wait": 42,
        "expert_wait": 42,
        "wait_error": 0,
        "frames": 100,
        "final_room_id": 0x96BA,
        "failure": None,
        "claim_valid": True,
    }
    capture = LandingBCTrajectoryCapture(
        state_digest="eval-state-1",
        before_values=(0.0, 0.0, 0.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        after_values=(0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        after_state_digest="after-digest",
        predicted_wait=42,
        frames=100,
        success=True,
        failure=None,
        attempt=attempt,
    )
    checkpoint_path = REPO_ROOT / "snes" / "sm_rando" / "models" / "landing_wait_bc_v1.json"
    eval_result = LandingBCEvalResult(
        corpus_path=REPO_ROOT / "snes" / "sm_rando" / "custom_integrations" / "x",
        checkpoint_path=checkpoint_path,
        corpus_digest="corpus-d",
        contract_bundle_digest="contracts-d",
        observation_schema_digest="obs-d",
        action_schema_digest="act-d",
        reward_schema_digest="rew-d",
        policy_identity_digest="policy-d",
        policy_artifact_digest="artifact-d",
        train_state_count=1,
        train_metrics={},
        attempts=(attempt,),
        eval_captures=(capture,),
    )
    trajectory = build_landing_bc_trajectory(capture, eval_result=eval_result)
    assert trajectory.succeeded is True
    assert trajectory.steps[0].action["value"]["wait_frames"] == 42

    traj_dir = tmp_path / "traj"
    cx_dir = tmp_path / "cx"
    paths = export_landing_bc_trajectories(
        eval_result, trajectory_dir=traj_dir, counterexample_dir=cx_dir
    )
    assert len(paths) == 1
    loaded = Trajectory.load(traj_dir / "eval-state-1.json")
    assert loaded.succeeded is True
    assert loaded.provenance["partition"] == "eval"


def test_write_landing_bc_report_is_report_owned(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    written = write_landing_bc_report(
        {"schema_version": 1, "decision": "candidate_only_not_deployed"},
        output_path=path,
    )
    assert written == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_fit_requires_metadata_v2_observation_fields() -> None:
    """Train ownership still routes features through metadata-v2 mapping."""
    bad = EntryStateRecord(
        state_digest="state-bad",
        ram_snapshot_digest="ram-bad",
        state_path="states/bad.state",
        source_skill_id="predecessor",
        source_segment_id="landing",
        source_trajectory_digest="trajectory",
        frame=0,
        observation_schema_digest="observation",
        contract_bundle_digest="bundle",
        metadata={
            # intentionally omit observation_metadata_version
            "room_id": 0x91F8,
            "samus_y": 240,
            "samus_y_sub": 0,
            "timing": {"landing_frame": 10},
        },
    )
    try:
        fit_landing_bc_model((bad,))
    except ValueError as exc:
        assert "metadata" in str(exc).lower() or "v2" in str(exc).lower()
        return
    raise AssertionError("fit should reject missing metadata-v2 version")
