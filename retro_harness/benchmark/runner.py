"""Benchmark case execution and seed-robustness orchestration.

Seed DTO ownership lives in :mod:`retro_harness.benchmark.seed_robustness`.
Claim identities live in :mod:`retro_harness.benchmark.claims`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from retro_harness.audit import (
    AttemptAudit,
    InterventionClass,
    RuntimeObservationClass,
)
from retro_harness.benchmark.claims import (
    ClaimValidationError,
    EvaluationContract,
    PolicyIdentity,
    StartIdentity,
    policy_identity_for,
    validate_claim,
)
from retro_harness.benchmark.seed_robustness import (
    SEED_ROBUSTNESS_SCHEMA_VERSION,
    SeedAttemptResult,
    SeedRobustnessConfig,
    SeedRobustnessReport,
    SeedValue,
    _audit_with_contract_identity,
    _to_jsonable,
    _validate_seed_result_budget,
    write_seed_robustness_report,
)
from retro_harness.recordings import append_jsonl


class BenchmarkTier(str, Enum):
    """Deprecated compatibility adapter for runtime observation classes."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"

    def to_runtime_observation_class(self) -> RuntimeObservationClass:
        """Translate the legacy tier spelling to the typed class."""
        return RuntimeObservationClass.from_value(self.value)

    def to_observation_class(self) -> RuntimeObservationClass:
        """Alias for :meth:`to_runtime_observation_class`."""
        return self.to_runtime_observation_class()


@dataclass(frozen=True)
class BenchmarkCase:
    """Static definition of a reproducible benchmark."""

    benchmark_id: str
    display_name: str
    game: str
    start_state: str
    tier: BenchmarkTier | RuntimeObservationClass | str | None
    objective: str
    max_steps: int
    build_env: Callable[[], Any]
    is_success: Callable[[dict[str, Any], bool, bool], bool]
    stop_on_success: bool = True
    tags: tuple[str, ...] = ()
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    contract: EvaluationContract | None = None

    @property
    def evaluation_contract(self) -> EvaluationContract | None:
        return self.contract

    def __post_init__(self) -> None:
        if self.tier is None:
            if self.contract is None:
                raise ValueError("tier or contract is required")
            tier = BenchmarkTier(self.contract.runtime_observation_class.value.casefold())
        elif isinstance(self.tier, BenchmarkTier):
            tier = self.tier
        else:
            observation = RuntimeObservationClass.from_value(self.tier)
            tier = BenchmarkTier(observation.value.casefold())
        if self.contract is not None:
            if not isinstance(self.contract, EvaluationContract):
                raise TypeError("contract must be an EvaluationContract or None")
            if self.contract.runtime_observation_class.value.casefold() != tier.value:
                raise ValueError("case tier does not match its evaluation contract")
            contract = self.contract
        else:
            contract = EvaluationContract(
                runtime_observation_class=tier.to_runtime_observation_class(),
                intervention_class=InterventionClass.CLEAN,
                start_identity=StartIdentity(self.start_state),
                policy_identity=PolicyIdentity("unbound-policy"),
                benchmark_id=self.benchmark_id,
                objective=self.objective,
            )
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", dict(self.metadata))


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
    attempt_audit: AttemptAudit | None = None
    contract: EvaluationContract | None = None

    def to_record(self, case: BenchmarkCase, policy_name: str) -> dict[str, Any]:
        contract = self.contract or _contract_for_case(case, policy_name)
        _validate_contract_for_case(case, contract)
        audit = self.attempt_audit or AttemptAudit.from_info(self.final_info)
        audit = _audit_with_contract_identity(audit, contract)
        validate_claim(contract, audit)
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
            "runtime_observation_class": contract.runtime_observation_class.value,
            "intervention_class": contract.intervention_class.value,
            "start_identity_digest": contract.start_identity.identity_digest,
            "policy_identity_digest": contract.policy_identity.identity_digest,
            "attempt_audit": audit.to_record(),
            "contract": contract.to_record(),
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
    contract: EvaluationContract | None = None

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
        contract = self.contract or _contract_for_case(self.case, self.policy_name)
        _validate_contract_for_case(self.case, contract)
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
            "runtime_observation_class": contract.runtime_observation_class.value,
            "intervention_class": contract.intervention_class.value,
            "start_identity_digest": contract.start_identity.identity_digest,
            "policy_identity_digest": contract.policy_identity.identity_digest,
            "contract": contract.to_record(),
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
    actual_policy_identity = policy_identity_for(policy)
    for seed in config.seeds:
        case = build_case(seed)
        if not isinstance(case, BenchmarkCase):
            raise TypeError("build_case must return a BenchmarkCase")
        if case.max_steps != config.budget:
            raise ValueError(
                f"benchmark case for seed {seed!r} must use exactly the published "
                f"frame budget ({config.budget})"
            )
        contract = _contract_for_seed_case(config, case, actual_policy_identity)
        run_result = run_benchmark(case, policy, contract=contract)
        attempt = run_result.attempts[0]
        if result_extractor is None:
            seed_result = SeedAttemptResult.from_benchmark_attempt(
                seed,
                attempt,
                contract=contract,
            )
        else:
            seed_result = result_extractor(seed, attempt)
        if not isinstance(seed_result, SeedAttemptResult):
            raise TypeError("result_extractor must return a SeedAttemptResult")
        seed_result = seed_result.bind_contract(contract)
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
    contract: EvaluationContract | None = None,
) -> BenchmarkRunResult:
    """Run a benchmark case for one or more attempts."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    started_at = time.time()
    attempt_results: list[BenchmarkAttemptResult] = []
    policy_name = _policy_name(policy)
    actual_policy_identity = policy_identity_for(policy)
    if contract is not None and not isinstance(contract, EvaluationContract):
        raise TypeError("contract must be an EvaluationContract or None")
    if contract is not None and not _policy_identity_is_verifiable(actual_policy_identity):
        raise ClaimValidationError(
            "explicit evaluation contract cannot verify an opaque policy identity: "
            "the policy has no inspectable source or bytecode and is unverifiable"
        )
    resolved_contract = (
        contract
        if contract is not None
        else _contract_for_case(case, actual_policy_identity)
    )
    if not isinstance(resolved_contract, EvaluationContract):
        raise TypeError("contract must be an EvaluationContract or None")
    _validate_contract_for_case(case, resolved_contract)
    if resolved_contract.policy_identity.identity_digest != actual_policy_identity.identity_digest:
        raise ClaimValidationError(
            "evaluation contract policy identity does not match the supplied policy"
        )
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
            failure_reason = None if success else _failure_reason(
                step_count, case.max_steps, terminated, truncated
            )
            audit = _audit_with_contract_identity(
                AttemptAudit.from_info(info),
                resolved_contract,
            )
            validate_claim(resolved_contract, audit)
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
                attempt_audit=audit,
                contract=resolved_contract,
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
        contract=resolved_contract,
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


def _validate_contract_for_case(
    case: BenchmarkCase,
    contract: EvaluationContract,
) -> None:
    """Ensure a contract is bound to the case it is used to evaluate."""
    if not isinstance(contract, EvaluationContract):
        raise TypeError("contract must be an EvaluationContract")

    expected_observation = case.tier.to_runtime_observation_class()
    expected_start = StartIdentity(case.start_state)
    case_contract = case.contract
    if case_contract is not None:
        if case_contract.benchmark_id != case.benchmark_id:
            raise ValueError("BenchmarkCase contract benchmark ID does not match the case")
        if case_contract.objective != case.objective:
            raise ValueError("BenchmarkCase contract objective does not match the case")
        if case_contract.runtime_observation_class is not expected_observation:
            raise ValueError("BenchmarkCase contract observation class does not match its tier")
        if case_contract.start_identity.start_state != case.start_state:
            raise ValueError("BenchmarkCase contract start identity does not match the case")
        expected_start = case_contract.start_identity

        if case_contract.intervention_class is not contract.intervention_class:
            raise ValueError(
                "evaluation contract intervention class does not match the BenchmarkCase contract"
            )
        for field_name in (
            "assist_contract_path",
            "assist_contract_digest",
            "assist_mode",
        ):
            if getattr(case_contract, field_name) != getattr(contract, field_name):
                raise ValueError(
                    f"evaluation contract {field_name} does not match the "
                    "BenchmarkCase contract"
                )
        if (
            not _is_unbound_policy_identity(case_contract.policy_identity)
            and not _same_policy_identity(
                case_contract.policy_identity,
                contract.policy_identity,
            )
        ):
            raise ValueError(
                "evaluation contract policy identity does not match the "
                "BenchmarkCase contract"
            )

    if contract.benchmark_id != case.benchmark_id:
        raise ValueError("evaluation contract benchmark ID does not match the BenchmarkCase")
    if contract.objective != case.objective:
        raise ValueError("evaluation contract objective does not match the BenchmarkCase")
    if contract.runtime_observation_class is not expected_observation:
        raise ValueError("evaluation contract observation class does not match the BenchmarkCase tier")
    if contract.start_identity.start_state != case.start_state:
        raise ValueError("evaluation contract start identity does not match the BenchmarkCase")
    if contract.start_identity.identity_digest != expected_start.identity_digest:
        raise ValueError("evaluation contract start identity does not match the BenchmarkCase")


def _contract_for_case(
    case: BenchmarkCase,
    policy: PolicyIdentity | str,
) -> EvaluationContract:
    policy_identity = policy if isinstance(policy, PolicyIdentity) else PolicyIdentity(policy)
    if case.contract is not None:
        contract = case.contract
        _validate_contract_for_case(case, contract)
        if _is_unbound_policy_identity(contract.policy_identity):
            contract = contract.with_policy(policy_identity)
        _validate_contract_for_case(case, contract)
        return contract
    contract = EvaluationContract(
        runtime_observation_class=case.tier.to_runtime_observation_class(),
        intervention_class=InterventionClass.CLEAN,
        start_identity=StartIdentity(case.start_state),
        policy_identity=policy_identity,
        benchmark_id=case.benchmark_id,
        objective=case.objective,
    )
    _validate_contract_for_case(case, contract)
    return contract


def _contract_for_seed_result(result: SeedAttemptResult) -> EvaluationContract:
    if result.contract is not None:
        return result.contract
    return EvaluationContract(
        runtime_observation_class=result.runtime_observation_class,
        intervention_class=result.intervention_class,
        start_identity=StartIdentity(
            f"seed-record:{result.seed}",
            digest=result.start_identity_digest,
        ),
        policy_identity=PolicyIdentity(
            "seed-record-policy",
            digest=result.policy_identity_digest,
        ),
        assist_contract_path=result.assist_contract_path,
        assist_contract_digest=result.assist_contract_digest,
        assist_mode=result.assist_mode,
    )


def _contract_for_seed_case(
    config: SeedRobustnessConfig,
    case: BenchmarkCase,
    policy: PolicyIdentity | str,
) -> EvaluationContract:
    policy_identity = policy if isinstance(policy, PolicyIdentity) else PolicyIdentity(policy)
    assist_mode = config.assist_mode
    if case.contract is not None:
        contract = case.contract
        if not _is_unbound_policy_identity(contract.policy_identity):
            if contract.runtime_observation_class is not config.runtime_observation_class:
                raise ValueError(
                    "seed case contract runtime observation class does not match config"
                )
            if contract.intervention_class is not config.intervention_class:
                raise ValueError("seed case contract intervention class does not match config")
            if contract.assist_contract_path != config.assist_contract_path:
                raise ValueError("seed case assist contract path does not match config")
            if contract.assist_contract_digest != config.assist_contract_digest:
                raise ValueError("seed case assist contract digest does not match config")
            if contract.assist_mode != assist_mode:
                raise ValueError("seed case assist mode does not match config")
            if (
                config._policy_identity_explicit
                and not _same_policy_identity(
                    contract.policy_identity,
                    config.policy_identity,
                )
            ):
                raise ValueError("seed case policy identity does not match config")
            if (
                config._start_identity_explicit
                and contract.start_identity != config.start_identity
            ):
                raise ValueError("seed case start identity does not match config")
            return contract
        start_identity = contract.start_identity
        if assist_mode is not None and contract.assist_mode != assist_mode:
            raise ValueError("seed case assist mode does not match config")
        if assist_mode is None and contract.assist_mode is not None:
            assist_mode = contract.assist_mode
    else:
        start_identity = (
            config.start_identity
            if config._start_identity_explicit
            else StartIdentity(case.start_state)
        )
    if config._policy_identity_explicit:
        policy_identity = config.policy_identity
    contract = EvaluationContract(
        runtime_observation_class=config.runtime_observation_class,
        intervention_class=config.intervention_class,
        start_identity=start_identity,
        policy_identity=policy_identity,
        benchmark_id=case.benchmark_id,
        objective=case.objective,
        assist_contract_path=config.assist_contract_path,
        assist_contract_digest=config.assist_contract_digest,
        assist_mode=assist_mode,
    )
    if (
        config._policy_identity_explicit
        and not _same_policy_identity(contract.policy_identity, config.policy_identity)
    ):
        raise ValueError("seed case policy identity does not match config")
    if (
        config._start_identity_explicit
        and contract.start_identity != config.start_identity
    ):
        raise ValueError("seed case start identity does not match config")
    return contract


def _same_policy_identity(
    first: PolicyIdentity,
    second: PolicyIdentity,
) -> bool:
    """Compare the verifiable identity, not its display label or metadata."""
    return first.identity_digest == second.identity_digest


def _policy_identity_is_verifiable(identity: PolicyIdentity) -> bool:
    """Return whether an identity carries source or bytecode evidence."""
    if identity.metadata.get("policy_kind") == "learned":
        return (
            identity.metadata.get("policy_artifact_digest")
            == identity.identity_digest
        )
    fingerprint_kind = identity.metadata.get("fingerprint_kind")
    if fingerprint_kind == "source":
        return isinstance(identity.metadata.get("source_sha256"), str)
    if fingerprint_kind == "bytecode":
        return isinstance(identity.metadata.get("bytecode_sha256"), str)
    return False


def _is_unbound_policy_identity(identity: PolicyIdentity) -> bool:
    """Return whether an identity is the compatibility placeholder."""
    return (
        identity.name == "unbound-policy"
        and identity.identity_digest == PolicyIdentity("unbound-policy").identity_digest
    )


def _policy_name(policy: BenchmarkPolicy | Callable[..., Any]) -> str:
    value = getattr(policy, "name", getattr(policy, "__name__", policy.__class__.__name__))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return str(value)


__all__ = [
    "BenchmarkAttemptResult",
    "BenchmarkCase",
    "BenchmarkPolicy",
    "BenchmarkRunResult",
    "BenchmarkTier",
    "IdlePolicy",
    "RandomPolicy",
    "SEED_ROBUSTNESS_SCHEMA_VERSION",
    "SeedAttemptResult",
    "SeedRobustnessConfig",
    "SeedRobustnessReport",
    "SeedValue",
    "run_benchmark",
    "run_seed_robustness",
    "write_seed_robustness_report",
    "zero_action_for_env",
]
