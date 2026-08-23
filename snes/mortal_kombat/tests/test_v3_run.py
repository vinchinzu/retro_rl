"""V3 run spec and artifact naming (no ROM, no PPO.learn)."""

from __future__ import annotations

import pytest

from mortal_kombat.v3_run import V3Run, v3_artifact_name


def test_fresh_run_allows_missing_candidate() -> None:
    run = V3Run(state="Match7_LiuKang", stage="Match7", steps=100)
    assert run.load is None
    assert run.candidate is None
    assert run.output_stage == "Match7"


def test_load_without_candidate_raises() -> None:
    with pytest.raises(ValueError, match="output-prefix|incumbent"):
        V3Run(state="Match7_LiuKang", stage="Match7", steps=100, load="foo.zip")


def test_load_with_same_candidate_as_stage_raises() -> None:
    with pytest.raises(ValueError, match="output-prefix|incumbent"):
        V3Run(
            state="Match7_LiuKang",
            stage="Match7",
            steps=100,
            load="foo.zip",
            candidate="Match7",
        )


def test_load_with_distinct_candidate_is_valid() -> None:
    run = V3Run(
        state="Match7_LiuKang",
        stage="Match7",
        steps=100,
        load="foo.zip",
        candidate="Match7_cont",
    )
    assert run.output_stage == "Match7_cont"


def test_artifact_name_finished_budget_is_final() -> None:
    assert (
        v3_artifact_name("Match7", wall_stopped=False, timesteps=100)
        == "mk1_v3_Match7_ppo_final.zip"
    )


def test_artifact_name_wall_stop_is_steps_zip() -> None:
    assert (
        v3_artifact_name("Match7", wall_stopped=True, timesteps=1_200_000)
        == "mk1_v3_Match7_ppo_1200000_steps.zip"
    )
