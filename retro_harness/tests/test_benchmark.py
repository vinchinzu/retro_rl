"""Tests for retro_harness.benchmark module."""

from pathlib import Path
import tempfile

import numpy as np

from retro_harness.benchmark import (
    BenchmarkCase,
    BenchmarkTier,
    IdlePolicy,
    RandomPolicy,
    run_benchmark,
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
    def __init__(self, *, success_after=2, truncated_after=None, array_actions=False):
        self.success_after = success_after
        self.truncated_after = truncated_after
        self.action_space = FakeArrayActionSpace() if array_actions else FakeDiscreteActionSpace()
        self.closed = False
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        return np.zeros((2, 2), dtype=np.uint8), {"count": 0, "flag": np.int64(1)}

    def step(self, action):
        self.step_count += 1
        info = {"count": self.step_count, "array": np.array([1, 2, 3], dtype=np.int64)}
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
