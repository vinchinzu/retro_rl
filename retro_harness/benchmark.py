"""Generic benchmark definitions and runners for emulator-backed tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np

from retro_harness.recordings import append_jsonl


class BenchmarkTier(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


@dataclass(frozen=True)
class BenchmarkCase:
    """Static definition of a reproducible benchmark."""

    benchmark_id: str
    display_name: str
    game: str
    start_state: str
    tier: BenchmarkTier
    objective: str
    max_steps: int
    build_env: Callable[[], Any]
    is_success: Callable[[dict[str, Any], bool, bool], bool]
    stop_on_success: bool = True
    tags: tuple[str, ...] = ()
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkAttemptResult:
    """Outcome for a single benchmark attempt."""

    attempt_index: int
    success: bool
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    wall_time_seconds: float
    failure_reason: str | None
    final_info: dict[str, Any] = field(default_factory=dict)

    def to_record(self, case: BenchmarkCase, policy_name: str) -> dict[str, Any]:
        return {
            "event": "benchmark_attempt",
            "benchmark_id": case.benchmark_id,
            "display_name": case.display_name,
            "game": case.game,
            "start_state": case.start_state,
            "tier": case.tier.value,
            "policy": policy_name,
            "attempt_index": self.attempt_index,
            "success": self.success,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "wall_time_seconds": self.wall_time_seconds,
            "failure_reason": self.failure_reason,
            "final_info": _to_jsonable(self.final_info),
        }


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Aggregate result for a benchmark run with one or more attempts."""

    case: BenchmarkCase
    policy_name: str
    attempts: tuple[BenchmarkAttemptResult, ...]
    started_at_unix: float
    finished_at_unix: float
    log_path: Path | None = None

    @property
    def successes(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.success)

    @property
    def success_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return self.successes / len(self.attempts)

    @property
    def total_wall_time_seconds(self) -> float:
        return self.finished_at_unix - self.started_at_unix

    def to_record(self) -> dict[str, Any]:
        return {
            "event": "benchmark_summary",
            "benchmark_id": self.case.benchmark_id,
            "display_name": self.case.display_name,
            "game": self.case.game,
            "start_state": self.case.start_state,
            "tier": self.case.tier.value,
            "policy": self.policy_name,
            "attempts": len(self.attempts),
            "successes": self.successes,
            "success_rate": self.success_rate,
            "objective": self.case.objective,
            "tags": list(self.case.tags),
            "notes": self.case.notes,
            "metadata": _to_jsonable(self.case.metadata),
            "started_at_unix": self.started_at_unix,
            "finished_at_unix": self.finished_at_unix,
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "attempt_summaries": [
                {
                    "attempt_index": attempt.attempt_index,
                    "success": attempt.success,
                    "steps": attempt.steps,
                    "total_reward": attempt.total_reward,
                    "failure_reason": attempt.failure_reason,
                }
                for attempt in self.attempts
            ],
        }


class BenchmarkPolicy(Protocol):
    """Optional protocol for policies passed to run_benchmark."""

    name: str

    def reset(self, env: Any, case: BenchmarkCase) -> None:
        ...

    def act(self, obs: Any, info: dict[str, Any], env: Any, case: BenchmarkCase) -> Any:
        ...


class IdlePolicy:
    """Return a no-op action for both discrete and button-array envs."""

    name = "idle"

    def reset(self, env: Any, case: BenchmarkCase) -> None:
        return None

    def act(self, obs: Any, info: dict[str, Any], env: Any, case: BenchmarkCase) -> Any:
        return zero_action_for_env(env)


class RandomPolicy:
    """Sample directly from the environment action space."""

    name = "random"

    def reset(self, env: Any, case: BenchmarkCase) -> None:
        return None

    def act(self, obs: Any, info: dict[str, Any], env: Any, case: BenchmarkCase) -> Any:
        return env.action_space.sample()


def zero_action_for_env(env: Any) -> Any:
    """Build an idle action that matches the environment action space."""
    action_space = env.action_space
    if hasattr(action_space, "n"):
        return 0

    shape = getattr(action_space, "shape", None)
    if shape in (None, ()):
        return 0

    dtype = getattr(action_space, "dtype", np.int8)
    return np.zeros(shape, dtype=dtype)


def run_benchmark(
    case: BenchmarkCase,
    policy: BenchmarkPolicy | Callable[..., Any],
    *,
    attempts: int = 1,
    log_path: str | Path | None = None,
) -> BenchmarkRunResult:
    """Run a benchmark case for one or more attempts."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    started_at = time.time()
    attempt_results: list[BenchmarkAttemptResult] = []
    policy_name = getattr(policy, "name", getattr(policy, "__name__", policy.__class__.__name__))
    log_path_obj = Path(log_path) if log_path is not None else None

    for attempt_index in range(1, attempts + 1):
        env = case.build_env()
        try:
            obs, info = env.reset()
            _reset_policy(policy, env, case)

            total_reward = 0.0
            terminated = False
            truncated = False
            success = case.is_success(info, terminated, truncated)
            step_count = 0

            attempt_start = time.time()
            while step_count < case.max_steps and not success and not (terminated or truncated):
                action = _policy_action(policy, obs, info, env, case)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                step_count += 1
                success = case.is_success(info, terminated, truncated)
                if success and case.stop_on_success:
                    break

            wall_time = time.time() - attempt_start
            failure_reason = None if success else _failure_reason(step_count, case.max_steps, terminated, truncated)
            attempt = BenchmarkAttemptResult(
                attempt_index=attempt_index,
                success=success,
                steps=step_count,
                total_reward=total_reward,
                terminated=terminated,
                truncated=truncated,
                wall_time_seconds=wall_time,
                failure_reason=failure_reason,
                final_info=_to_jsonable(info),
            )
            attempt_results.append(attempt)
            if log_path_obj is not None:
                append_jsonl(log_path_obj, attempt.to_record(case, policy_name))
        finally:
            env.close()

    finished_at = time.time()
    result = BenchmarkRunResult(
        case=case,
        policy_name=policy_name,
        attempts=tuple(attempt_results),
        started_at_unix=started_at,
        finished_at_unix=finished_at,
        log_path=log_path_obj,
    )
    if log_path_obj is not None:
        append_jsonl(log_path_obj, result.to_record())
    return result


def _reset_policy(policy: BenchmarkPolicy | Callable[..., Any], env: Any, case: BenchmarkCase) -> None:
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn(env, case)


def _policy_action(
    policy: BenchmarkPolicy | Callable[..., Any],
    obs: Any,
    info: dict[str, Any],
    env: Any,
    case: BenchmarkCase,
) -> Any:
    act_fn = getattr(policy, "act", None)
    if callable(act_fn):
        return act_fn(obs, info, env, case)
    return policy(obs, info, env, case)


def _failure_reason(steps: int, max_steps: int, terminated: bool, truncated: bool) -> str:
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    if steps >= max_steps:
        return "max_steps"
    return "incomplete"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
