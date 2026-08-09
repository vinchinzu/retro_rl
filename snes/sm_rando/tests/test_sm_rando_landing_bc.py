"""Pure checks for the held-out Landing behavior-cloning experiment."""

from __future__ import annotations

import json

from retro_harness.benchmark import validate_claim
from retro_harness.entry_states import EntryStateRecord
from retro_harness.trajectory import Trajectory
from sm_rando.entry_corpus import landing_corpus_contracts
from sm_rando.landing_bc import (
    EXPERT_HANDOFF_LANDING_FRAME,
    LANDING_BC_REPORT,
    build_landing_bc_contracts,
    fit_landing_bc_model,
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
