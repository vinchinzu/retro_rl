"""Unit tests for hybrid SMB pipeline helpers (no emulator required)."""

from pathlib import Path

from super_mario_bros.hybrid_pipeline import (
    CandidateMetric,
    build_sub_agent_plan,
    candidate_rank_key,
    discover_candidate_files,
    infer_algorithm,
    mine_splice_seeds,
    weakness_score,
)


def _row(*, completed: bool, completion_rate: float, frames: int, fitness: float, progress: float) -> CandidateMetric:
    return CandidateMetric(
        path="/tmp/x.json",
        algorithm="replay",
        source="existing",
        is_raw=False,
        completed=completed,
        completion_rate=completion_rate,
        died_rate=0.0,
        frames_mean=float(frames),
        total_frames=frames,
        fitness_mean=fitness,
        max_progress=progress,
        num_actions=frames,
    )


def test_candidate_rank_prefers_completed_then_faster():
    slow_complete = _row(completed=True, completion_rate=1.0, frames=3000, fitness=97000.0, progress=3200.0)
    fast_complete = _row(completed=True, completion_rate=1.0, frames=2200, fitness=97800.0, progress=3200.0)
    incomplete = _row(completed=False, completion_rate=0.0, frames=1800, fitness=2000.0, progress=1200.0)

    assert candidate_rank_key(fast_complete) < candidate_rank_key(slow_complete)
    assert candidate_rank_key(fast_complete) < candidate_rank_key(incomplete)


def test_weakness_score_prioritizes_incomplete_segments():
    complete = _row(completed=True, completion_rate=1.0, frames=3500, fitness=96500.0, progress=3300.0)
    incomplete = _row(completed=False, completion_rate=0.0, frames=2000, fitness=5000.0, progress=900.0)

    assert weakness_score(incomplete) > weakness_score(complete)


def test_build_sub_agent_plan_uses_raw_ga_neuro_ppo_flags():
    baseline = _row(completed=False, completion_rate=0.0, frames=5000, fitness=1200.0, progress=800.0)

    plan = build_sub_agent_plan(
        baseline,
        has_raw=True,
        use_ga=True,
        use_neuro=True,
        use_ppo=True,
    )
    assert plan == ["replay", "ga_raw", "hillclimb_raw", "neuro", "ppo"]


def test_mine_splice_seeds_generates_children():
    a = [1] * 20 + [2] * 20
    b = [3] * 15 + [4] * 25
    c = [5] * 30 + [6] * 10

    mined = mine_splice_seeds([a, b, c], max_generated=5)
    assert len(mined) > 0
    assert len(mined) <= 5
    assert all(len(x) >= 4 for x in mined)


def test_discover_candidates_filters_companion_and_trace(tmp_path: Path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir(parents=True)
    (run_dir / "candidates" / "123" / "mined").mkdir(parents=True)

    (run_dir / "recording_000.json").write_text('{"actions": [0, 1], "num_frames": 2}')
    (run_dir / "recording_000_raw.json").write_text('{"raw_buttons": [[0]*12], "num_frames": 1}')
    (run_dir / "hillclimb_best_final.json").write_text('{"completed": true, "total_frames": 100}')
    (run_dir / "recording_000_trace.json").write_text('{"trace": []}')
    (run_dir / "candidates" / "123" / "mined" / "seed_indices_00.json").write_text('{"actions": [0]}')

    found = discover_candidate_files(run_dir, max_candidates=20)
    names = {p.name for p in found}

    assert "recording_000.json" in names
    assert "hillclimb_best_final.json" in names
    assert "recording_000_raw.json" not in names
    assert "recording_000_trace.json" not in names
    assert "seed_indices_00.json" not in names


def test_infer_algorithm_detects_variants():
    assert infer_algorithm(Path("/tmp/ga_raw_best.json")) == "ga_raw"
    assert infer_algorithm(Path("/tmp/hillclimb_best_final.json")) == "hillclimb"
    assert infer_algorithm(Path("/tmp/neuro/neuro_best_buttons.json")) == "neat"
    assert infer_algorithm(Path("/tmp/smb_1_1.json")) == "unknown"
