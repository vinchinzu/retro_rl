"""Tests for retro_harness.benchmark module."""

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from retro_harness.benchmark import (
    BenchmarkCase,
    BenchmarkTier,
    IdlePolicy,
    RandomPolicy,
    SeedAttemptResult,
    SeedRobustnessConfig,
    SeedRobustnessReport,
    run_benchmark,
    run_seed_robustness,
    write_seed_robustness_report,
    zero_action_for_env,
)
from retro_harness.recordings import iter_jsonl


class FakeDiscreteActionSpace:
    n = 4

    def sample(self):
        return 3


class FakeArrayActionSpace:
    shape = (12,)
    dtype = np.int8

    def sample(self):
        return np.ones(self.shape, dtype=self.dtype)


class FakeEnv:
    def __init__(
        self,
        *,
        success_after=2,
        truncated_after=None,
        array_actions=False,
        info_extra=None,
    ):
        self.success_after = success_after
        self.truncated_after = truncated_after
        self.info_extra = dict(info_extra or {})
        self.action_space = FakeArrayActionSpace() if array_actions else FakeDiscreteActionSpace()
        self.closed = False
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        return np.zeros((2, 2), dtype=np.uint8), {
            "count": 0,
            "flag": np.int64(1),
            **self.info_extra,
        }

    def step(self, action):
        self.step_count += 1
        info = {
            "count": self.step_count,
            "array": np.array([1, 2, 3], dtype=np.int64),
            **self.info_extra,
        }
        terminated = self.success_after is not None and self.step_count >= self.success_after
        truncated = self.truncated_after is not None and self.step_count >= self.truncated_after
        reward = 1.5
        return np.zeros((2, 2), dtype=np.uint8), reward, terminated, truncated, info

    def close(self):
        self.closed = True


class CountingPolicy:
    name = "counting"

    def __init__(self):
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self, env, case):
        self.reset_calls += 1

    def act(self, obs, info, env, case):
        self.act_calls += 1
        return zero_action_for_env(env)


def _success(info, terminated, truncated):
    return info.get("count", 0) >= 2


def _seed_config(**kwargs):
    values = {
        "generator": "fixture-generator",
        "generator_version": "1.0",
        "logic": "standard",
        "goal": "reach house chest",
        "seeds": ("alpha", "beta"),
        "budget": 3,
        "success_threshold": 1,
        "runtime_observation_class": "Bronze",
        "intervention_class": "Clean",
    }
    values.update(kwargs)
    return SeedRobustnessConfig(**values)


def test_zero_action_for_discrete_env():
    env = FakeEnv()
    assert zero_action_for_env(env) == 0


def test_zero_action_for_array_env():
    env = FakeEnv(array_actions=True)
    action = zero_action_for_env(env)
    assert action.shape == (12,)
    assert np.all(action == 0)


def test_idle_policy_uses_zero_action():
    env = FakeEnv(array_actions=True)
    action = IdlePolicy().act(None, {}, env, None)
    assert np.all(action == 0)


def test_random_policy_delegates_to_action_space():
    env = FakeEnv()
    assert RandomPolicy().act(None, {}, env, None) == 3


def test_run_benchmark_success_records_attempts():
    case = BenchmarkCase(
        benchmark_id="fake_success",
        display_name="Fake Success",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 2",
        max_steps=5,
        build_env=lambda: FakeEnv(success_after=2),
        is_success=_success,
    )
    policy = CountingPolicy()
    result = run_benchmark(case, policy, attempts=2)

    assert len(result.attempts) == 2
    assert result.successes == 2
    assert result.success_rate == 1.0
    assert policy.reset_calls == 2
    assert policy.act_calls == 4


def test_run_benchmark_timeout_sets_failure_reason():
    case = BenchmarkCase(
        benchmark_id="fake_timeout",
        display_name="Fake Timeout",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Never succeeds",
        max_steps=3,
        build_env=lambda: FakeEnv(success_after=None),
        is_success=lambda info, terminated, truncated: False,
    )
    result = run_benchmark(case, IdlePolicy())

    attempt = result.attempts[0]
    assert attempt.success is False
    assert attempt.failure_reason == "max_steps"
    assert attempt.steps == 3


def test_run_benchmark_writes_jsonl_log():
    case = BenchmarkCase(
        benchmark_id="fake_log",
        display_name="Fake Log",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 2",
        max_steps=5,
        build_env=lambda: FakeEnv(success_after=2),
        is_success=_success,
    )
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "benchmarks.jsonl"
        result = run_benchmark(case, IdlePolicy(), log_path=log_path)

        entries = iter_jsonl(log_path)
        assert len(entries) == 2
        assert entries[0]["event"] == "benchmark_attempt"
        assert entries[1]["event"] == "benchmark_summary"
        assert entries[1]["success_rate"] == 1.0
        assert result.log_path == log_path


def test_attempt_log_is_json_safe():
    case = BenchmarkCase(
        benchmark_id="fake_json",
        display_name="Fake Json",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective="Reach count 2",
        max_steps=5,
        build_env=lambda: FakeEnv(success_after=2),
        is_success=_success,
    )
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "benchmarks.jsonl"
        run_benchmark(case, IdlePolicy(), log_path=log_path)
        entries = iter_jsonl(log_path)
        assert entries[0]["final_info"]["array"] == [1, 2, 3]


def test_run_seed_robustness_writes_deterministic_st_fixture_report():
    config = SeedRobustnessConfig(
        generator="fixture-generator",
        generator_version="1.0",
        logic="standard",
        goal="reach house chest",
        seeds=("alpha", "beta", "gamma"),
        budget=3,
        success_threshold=2,
        runtime_observation_class="Bronze",
        intervention_class="Clean",
    )
    outcomes = {
        "alpha": (2, {"terminal_milestone": "house_chest", "assists": {"missile": 1}}),
        "beta": (
            None,
            {
                "terminal_milestone": "red_door",
                "failure_mode": "stalled",
                "assists": {"missile": 2},
            },
        ),
        "gamma": (2, {"terminal_milestone": "house_chest", "assists": {}}),
    }
    seen_seeds = []

    def build_case(seed):
        seen_seeds.append(seed)
        success_after, info_extra = outcomes[seed]
        return BenchmarkCase(
            benchmark_id=f"fixture_{seed}",
            display_name="Seed fixture",
            game="FakeGame",
            start_state=f"power_on_{seed}",
            tier=BenchmarkTier.BRONZE,
            objective=config.goal,
            max_steps=config.budget,
            build_env=lambda: FakeEnv(
                success_after=success_after,
                info_extra=info_extra,
            ),
            is_success=lambda info, terminated, truncated: success_after is not None
            and _success(info, terminated, truncated),
        )

    with tempfile.TemporaryDirectory() as td:
        report_path = Path(td) / "seed_report.json"
        report = run_seed_robustness(
            config,
            build_case,
            IdlePolicy(),
            report_path=report_path,
        )

        first_bytes = report_path.read_bytes()
        write_seed_robustness_report(report_path, report)
        assert report_path.read_bytes() == first_bytes
        record = json.loads(first_bytes)

    assert seen_seeds == ["alpha", "beta", "gamma"]
    assert report.successes == 2
    assert report.threshold_met is True
    assert record["config"]["seed_count"] == 3
    assert record["config"]["success_threshold"] == 2
    assert record["summary"] == {
        "required_successes": 2,
        "seeds_successful": 2,
        "seeds_total": 3,
        "success_rate": 2 / 3,
        "threshold_met": True,
    }
    assert record["seed_results"][0]["frames"] == 2
    assert record["seed_results"][0]["terminal_milestone"] == "house_chest"
    assert record["seed_results"][0]["assists"] == {"missile": 1}
    assert record["seed_results"][1]["outcome"] == "failure"
    assert record["seed_results"][1]["failure_mode"] == "stalled"
    assert record["seed_results"][1]["terminal_milestone"] == "red_door"


@pytest.mark.parametrize("case_steps", [2, 4])
def test_run_seed_robustness_requires_case_budget_to_match(case_steps):
    config = _seed_config(seeds=("alpha",), success_threshold=1)
    case = BenchmarkCase(
        benchmark_id="budget_mismatch",
        display_name="Budget mismatch",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective=config.goal,
        max_steps=case_steps,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=_success,
    )

    with pytest.raises(ValueError, match="must use exactly"):
        run_seed_robustness(config, lambda seed: case, IdlePolicy())


def test_seed_robustness_report_rejects_over_budget_frames():
    config = _seed_config()
    results = tuple(
        SeedAttemptResult(
            seed=seed,
            success=False,
            frames=config.budget + 1 if seed == config.seeds[0] else config.budget,
        )
        for seed in config.seeds
    )

    with pytest.raises(ValueError, match="exceed the published frame budget"):
        SeedRobustnessReport(config, "idle", results)


def test_run_seed_robustness_rejects_over_budget_extracted_frames():
    config = _seed_config(seeds=("alpha",), success_threshold=1)
    case = BenchmarkCase(
        benchmark_id="extracted_frames",
        display_name="Extracted frames",
        game="FakeGame",
        start_state="Start",
        tier=BenchmarkTier.BRONZE,
        objective=config.goal,
        max_steps=config.budget,
        build_env=lambda: FakeEnv(success_after=1),
        is_success=_success,
    )

    def extract(seed, attempt):
        return SeedAttemptResult(seed=seed, success=attempt.success, frames=config.budget + 1)

    with pytest.raises(ValueError, match="exceed the published frame budget"):
        run_seed_robustness(config, lambda seed: case, IdlePolicy(), result_extractor=extract)


def test_write_seed_robustness_report_rejects_nonfinite_metadata(tmp_path):
    config = _seed_config(metadata={"score": float("nan")})
    results = tuple(
        SeedAttemptResult(seed=seed, success=False, frames=0) for seed in config.seeds
    )
    report = SeedRobustnessReport(config, "idle", results)
    report_path = tmp_path / "nested" / "seed_report.json"

    with pytest.raises(ValueError, match="finite JSON numbers"):
        write_seed_robustness_report(report_path, report)

    assert not report_path.exists()
    assert not report_path.parent.exists()


@pytest.mark.parametrize(
    "metadata",
    [{"bad": object()}, {1: "non-string key"}],
)
def test_write_seed_robustness_report_rejects_non_json_metadata(tmp_path, metadata):
    config = _seed_config(metadata=metadata)
    results = tuple(
        SeedAttemptResult(seed=seed, success=False, frames=0) for seed in config.seeds
    )
    report = SeedRobustnessReport(config, "idle", results)

    with pytest.raises(TypeError, match="JSON"):
        write_seed_robustness_report(tmp_path / "seed_report.json", report)
