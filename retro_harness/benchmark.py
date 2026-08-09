"""Generic benchmark definitions and runners for emulator-backed tasks."""

from __future__ import annotations

import json
import math
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


SeedValue = str | int


@dataclass(frozen=True)
class SeedRobustnessConfig:
    """Published contract for a deterministic multi-seed benchmark report."""

    generator: str
    generator_version: str
    logic: str
    goal: str
    seeds: tuple[SeedValue, ...]
    budget: int
    success_threshold: int
    runtime_observation_class: str
    intervention_class: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.seeds, (set, frozenset)):
            raise TypeError("seeds must be an ordered sequence")
        seeds = tuple(self.seeds)
        if not seeds:
            raise ValueError("seeds must contain at least one published seed")
        for seed in seeds:
            _validate_seed_value(seed)
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int)
            or self.budget < 1
        ):
            raise ValueError("budget must be a positive frame count")
        if not 1 <= self.success_threshold <= len(seeds):
            raise ValueError("success_threshold must be between 1 and the seed count")
        for field_name in (
            "generator",
            "generator_version",
            "logic",
            "goal",
            "runtime_observation_class",
            "intervention_class",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def seed_count(self) -> int:
        """The published T in the S/T contract."""
        return len(self.seeds)

    def to_record(self) -> dict[str, Any]:
        """Return the stable, JSON-safe contract portion of a report."""
        return {
            "generator": self.generator,
            "generator_version": self.generator_version,
            "logic": self.logic,
            "goal": self.goal,
            "seeds": [_to_jsonable(seed) for seed in self.seeds],
            "seed_count": self.seed_count,
            "budget": self.budget,
            "budget_unit": "frames",
            "success_threshold": self.success_threshold,
            "runtime_observation_class": self.runtime_observation_class,
            "intervention_class": self.intervention_class,
            "metadata": _canonicalize_metadata(self.metadata),
        }


@dataclass(frozen=True)
class SeedAttemptResult:
    """Stable per-seed outcome used by :class:`SeedRobustnessReport`."""

    seed: SeedValue
    success: bool
    frames: int
    terminal_milestone: str | int | None = None
    failure_mode: str | None = None
    assists: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_seed_value(self.seed)
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames < 0:
            raise ValueError("frames must be a non-negative integer")
        if self.terminal_milestone is not None and not isinstance(
            self.terminal_milestone, (str, int)
        ):
            raise TypeError("terminal_milestone must be a string, integer, or None")
        if self.failure_mode is not None and not isinstance(self.failure_mode, str):
            raise TypeError("failure_mode must be a string or None")
        object.__setattr__(self, "assists", _normalize_assists(self.assists))

    @classmethod
    def from_benchmark_attempt(
        cls,
        seed: SeedValue,
        attempt: BenchmarkAttemptResult,
    ) -> "SeedAttemptResult":
        """Adapt an existing benchmark attempt into the seed report schema.

        Seed-aware environments can expose ``terminal_milestone``,
        ``failure_mode``, and an ``assists`` count mapping in their final
        ``info`` dictionary. The benchmark failure reason is retained as the
        fallback failure mode when an environment does not provide one.
        """
        info = attempt.final_info if isinstance(attempt.final_info, dict) else {}
        terminal_milestone = info.get("terminal_milestone")
        failure_mode = info.get("failure_mode")
        if failure_mode is None and not attempt.success:
            failure_mode = attempt.failure_reason
        return cls(
            seed=seed,
            success=attempt.success,
            frames=attempt.steps,
            terminal_milestone=terminal_milestone,
            failure_mode=failure_mode,
            assists=info.get("assists", {}),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "seed": _to_jsonable(self.seed),
            "outcome": "success" if self.success else "failure",
            "success": self.success,
            "frames": self.frames,
            "terminal_milestone": _to_jsonable(self.terminal_milestone),
            "failure_mode": self.failure_mode,
            "assists": _to_jsonable(self.assists),
        }


SEED_ROBUSTNESS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SeedRobustnessReport:
    """Deterministic aggregate artifact for one published S/T evaluation."""

    config: SeedRobustnessConfig
    policy_name: str
    seed_results: tuple[SeedAttemptResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError("policy_name must be a non-empty string")
        results = tuple(self.seed_results)
        if len(results) != self.config.seed_count:
            raise ValueError("seed_results must contain exactly one result per published seed")
        for expected_seed, result in zip(self.config.seeds, results, strict=True):
            if not isinstance(result, SeedAttemptResult):
                raise TypeError("seed_results must contain SeedAttemptResult values")
            if result.seed != expected_seed:
                raise ValueError("seed_results must follow the published seed order")
            _validate_seed_result_budget(self.config, result)
        object.__setattr__(self, "seed_results", results)

    @property
    def successes(self) -> int:
        return sum(1 for result in self.seed_results if result.success)

    @property
    def success_rate(self) -> float:
        return self.successes / self.config.seed_count

    @property
    def threshold_met(self) -> bool:
        return self.successes >= self.config.success_threshold

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe report without timestamps or wall-time noise."""
        return {
            "event": "seed_robustness_report",
            "schema_version": SEED_ROBUSTNESS_SCHEMA_VERSION,
            "policy": self.policy_name,
            "config": self.config.to_record(),
            "summary": {
                "seeds_total": self.config.seed_count,
                "seeds_successful": self.successes,
                "success_rate": self.success_rate,
                "required_successes": self.config.success_threshold,
                "threshold_met": self.threshold_met,
            },
            "seed_results": [result.to_record() for result in self.seed_results],
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


def run_seed_robustness(
    config: SeedRobustnessConfig,
    build_case: Callable[[SeedValue], BenchmarkCase],
    policy: BenchmarkPolicy | Callable[..., Any],
    *,
    report_path: str | Path | None = None,
    result_extractor: Callable[
        [SeedValue, BenchmarkAttemptResult], SeedAttemptResult
    ]
    | None = None,
) -> SeedRobustnessReport:
    """Run an existing benchmark policy once for each published seed.

    ``build_case`` owns seed/ROM setup and must return a case whose
    ``max_steps`` equals the report's frame budget. Seeds are consumed exactly
    in the order published in ``config``; this runner does not sample, shuffle,
    or silently replace them.
    """
    seed_results: list[SeedAttemptResult] = []
    for seed in config.seeds:
        case = build_case(seed)
        if not isinstance(case, BenchmarkCase):
            raise TypeError("build_case must return a BenchmarkCase")
        if case.max_steps != config.budget:
            raise ValueError(
                f"benchmark case for seed {seed!r} must use exactly the published "
                f"frame budget ({config.budget})"
            )
        run_result = run_benchmark(case, policy)
        attempt = run_result.attempts[0]
        if result_extractor is None:
            seed_result = SeedAttemptResult.from_benchmark_attempt(seed, attempt)
        else:
            seed_result = result_extractor(seed, attempt)
        if not isinstance(seed_result, SeedAttemptResult):
            raise TypeError("result_extractor must return a SeedAttemptResult")
        _validate_seed_result_budget(config, seed_result)
        seed_results.append(seed_result)

    report = SeedRobustnessReport(
        config=config,
        policy_name=_policy_name(policy),
        seed_results=tuple(seed_results),
    )
    if report_path is not None:
        write_seed_robustness_report(report_path, report)
    return report


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
    policy_name = _policy_name(policy)
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


def write_seed_robustness_report(
    path: str | Path,
    report: SeedRobustnessReport,
) -> Path:
    """Write a canonical JSON artifact and return its path."""
    if not isinstance(report, SeedRobustnessReport):
        raise TypeError("report must be a SeedRobustnessReport")
    record = report.to_record()
    serialized = json.dumps(
        record,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(serialized, encoding="utf-8")
    return report_path


def _policy_name(policy: BenchmarkPolicy | Callable[..., Any]) -> str:
    return getattr(policy, "name", getattr(policy, "__name__", policy.__class__.__name__))


def _validate_seed_value(seed: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (str, int)):
        raise TypeError("seed values must be strings or integers")


def _normalize_assists(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("assists must be a mapping of assist name to count")
    normalized: dict[str, int] = {}
    for name, count in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("assist names must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("assist counts must be non-negative integers")
        normalized[name] = count
    return dict(sorted(normalized.items()))


def _validate_seed_result_budget(
    config: SeedRobustnessConfig,
    result: SeedAttemptResult,
) -> None:
    if result.frames > config.budget:
        raise ValueError(
            f"frames for seed {result.seed!r} exceed the published frame budget"
        )


def _canonicalize_metadata(value: Any, *, path: str = "metadata") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{path} JSON object keys must be strings")
    return {
        key: _canonicalize_metadata_value(item, path=f"{path}.{key}")
        for key, item in sorted(value.items())
    }


def _canonicalize_metadata_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize_metadata_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"{path} JSON object keys must be strings")
        return {
            key: _canonicalize_metadata_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items())
        }
    raise TypeError(
        f"{path} contains unsupported JSON value type {type(value).__name__}"
    )


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
