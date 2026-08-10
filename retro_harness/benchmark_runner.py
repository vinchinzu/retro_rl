"""Benchmark runners and deterministic report serialization.

Claim identities and validation live in :mod:`retro_harness.benchmark_claims`.
The legacy :mod:`retro_harness.benchmark` module re-exports this public
surface for compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np

from retro_harness.audit import (
    AttemptAudit,
    InterventionClass,
    RuntimeObservationClass,
    normalize_assists as _normalize_assists,
)
from retro_harness.benchmark_claims import (
    ClaimValidationError,
    EvaluationContract,
    PolicyIdentity,
    StartIdentity,
    _canonicalize_metadata,
    _normalize_assist_mode,
    _validate_identity_digest,
    policy_identity_for,
    validate_claim,
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


SeedValue = str | int


@dataclass(frozen=True)
class SeedRobustnessConfig:
    """Published contract for a deterministic multi-seed benchmark report.

    An explicitly supplied start or policy identity is a shared constraint for
    every seed. When omitted, the identity is intentionally per-seed: each
    result carries and validates its own typed contract.
    """

    generator: str
    generator_version: str
    logic: str
    goal: str
    seeds: tuple[SeedValue, ...]
    budget: int
    success_threshold: int
    runtime_observation_class: RuntimeObservationClass | str
    intervention_class: InterventionClass | str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_identity: StartIdentity | None = None
    policy_identity: PolicyIdentity | None = None
    assist_contract_path: str | None = None
    assist_contract_digest: str | None = None
    assist_mode: str | None = None
    contract: EvaluationContract | None = None
    _start_identity_explicit: bool = field(init=False, repr=False, compare=False, default=False)
    _policy_identity_explicit: bool = field(init=False, repr=False, compare=False, default=False)

    @property
    def evaluation_contract(self) -> EvaluationContract | None:
        return self.contract

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
        for field_name in ("generator", "generator_version", "logic", "goal"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        observation = RuntimeObservationClass.from_value(self.runtime_observation_class)
        intervention = InterventionClass.from_value(self.intervention_class)
        if self.start_identity is not None and not isinstance(self.start_identity, StartIdentity):
            raise TypeError("start_identity must be a StartIdentity or None")
        if self.policy_identity is not None and not isinstance(self.policy_identity, PolicyIdentity):
            raise TypeError("policy_identity must be a PolicyIdentity or None")
        start_identity_explicit = self.start_identity is not None
        policy_identity_explicit = self.policy_identity is not None
        contract = self.contract
        if contract is not None:
            if not isinstance(contract, EvaluationContract):
                raise TypeError("contract must be an EvaluationContract or None")
            if contract.runtime_observation_class is not observation:
                raise ValueError("contract runtime observation class does not match config")
            if contract.intervention_class is not intervention:
                raise ValueError("contract intervention class does not match config")
            if (
                self.start_identity is not None
                and self.start_identity != contract.start_identity
            ):
                raise ValueError("config start identity does not match its evaluation contract")
            if (
                self.policy_identity is not None
                and self.policy_identity != contract.policy_identity
            ):
                raise ValueError("config policy identity does not match its evaluation contract")
            start_identity_explicit = True
            policy_identity_explicit = True
        assist_path = self.assist_contract_path
        assist_digest = self.assist_contract_digest
        assist_mode = _normalize_assist_mode(self.assist_mode)
        if contract is not None:
            if assist_path is None:
                assist_path = contract.assist_contract_path
            elif assist_path != contract.assist_contract_path:
                raise ValueError("config assist contract path does not match its evaluation contract")
            if assist_digest is None:
                assist_digest = contract.assist_contract_digest
            elif assist_digest != contract.assist_contract_digest:
                raise ValueError(
                    "config assist contract digest does not match its evaluation contract"
                )
            if assist_mode is None:
                assist_mode = contract.assist_mode
            elif assist_mode != contract.assist_mode:
                raise ValueError("config assist mode does not match its evaluation contract")
        if intervention.is_clean and assist_mode is not None:
            raise ValueError("Clean configs cannot carry assist_mode")
        _validate_assist_contract_fields(intervention, assist_path, assist_digest)
        start_identity = self.start_identity
        policy_identity = self.policy_identity
        if contract is not None:
            if start_identity is None:
                start_identity = contract.start_identity
            if policy_identity is None:
                policy_identity = contract.policy_identity
        if start_identity is None:
            start_identity = StartIdentity(
                f"seed-set:{self.generator}:{self.generator_version}:{','.join(map(str, seeds))}"
            )
        if policy_identity is None:
            policy_identity = PolicyIdentity("unbound-policy")

        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "runtime_observation_class", observation)
        object.__setattr__(self, "intervention_class", intervention)
        object.__setattr__(self, "assist_contract_path", assist_path)
        object.__setattr__(self, "assist_contract_digest", assist_digest)
        object.__setattr__(self, "assist_mode", assist_mode)
        object.__setattr__(self, "start_identity", start_identity)
        object.__setattr__(self, "policy_identity", policy_identity)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "_start_identity_explicit", start_identity_explicit)
        object.__setattr__(self, "_policy_identity_explicit", policy_identity_explicit)

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
            "runtime_observation_class": self.runtime_observation_class.value,
            "intervention_class": self.intervention_class.value,
            "start_identity_digest": self.start_identity.identity_digest,
            "start_identity_scope": (
                "shared" if self._start_identity_explicit else "per-seed"
            ),
            "start_identity": self.start_identity.to_record(),
            "policy_identity_digest": self.policy_identity.identity_digest,
            "policy_identity_scope": (
                "shared" if self._policy_identity_explicit else "per-seed"
            ),
            "policy_identity": self.policy_identity.to_record(),
            "assist_contract_path": self.assist_contract_path,
            "assist_contract_digest": self.assist_contract_digest,
            "assist_mode": self.assist_mode,
            "metadata": _canonicalize_metadata(self.metadata),
            **(
                {"contract": self.contract.to_record()}
                if self.contract is not None
                else {}
            ),
        }

    def to_contract(self) -> EvaluationContract:
        """Build the typed contract represented by this seed configuration."""
        if self.contract is not None:
            return self.contract
        return EvaluationContract(
            runtime_observation_class=self.runtime_observation_class,
            intervention_class=self.intervention_class,
            start_identity=self.start_identity,
            policy_identity=self.policy_identity,
            benchmark_id=f"seed-robust:{self.generator}",
            objective=self.goal,
            assist_contract_path=self.assist_contract_path,
            assist_contract_digest=self.assist_contract_digest,
            assist_mode=self.assist_mode,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class SeedAttemptResult:
    """Stable per-seed outcome used by :class:`SeedRobustnessReport`."""

    seed: SeedValue
    success: bool
    frames: int
    terminal_milestone: str | int | None = None
    failure_mode: str | None = None
    assists: dict[str, int] = field(default_factory=dict)
    runtime_observation_class: RuntimeObservationClass | str = RuntimeObservationClass.BRONZE
    intervention_class: InterventionClass | str = InterventionClass.CLEAN
    start_identity_digest: str | None = None
    policy_identity_digest: str | None = None
    assist_contract_path: str | None = None
    assist_contract_digest: str | None = None
    assist_mode: str | None = None
    ram_writes: int | bool = 0
    mid_run_loads: int | bool = 0
    attempt_audit: AttemptAudit | None = None
    contract: EvaluationContract | None = None
    _start_identity_explicit: bool = field(init=False, repr=False, compare=False, default=False)
    _policy_identity_explicit: bool = field(init=False, repr=False, compare=False, default=False)

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
        observation = RuntimeObservationClass.from_value(self.runtime_observation_class)
        intervention = InterventionClass.from_value(self.intervention_class)
        start_identity_explicit = self.start_identity_digest is not None
        policy_identity_explicit = self.policy_identity_digest is not None
        if self.contract is not None:
            if not isinstance(self.contract, EvaluationContract):
                raise TypeError("contract must be an EvaluationContract or None")
            if self.contract.runtime_observation_class is not observation:
                raise ValueError("seed result runtime observation class does not match contract")
            if self.contract.intervention_class is not intervention:
                raise ValueError("seed result intervention class does not match contract")
            if (
                self.start_identity_digest is not None
                and self.start_identity_digest != self.contract.start_identity_digest
            ):
                raise ValueError("seed result start identity does not match contract")
            if (
                self.policy_identity_digest is not None
                and self.policy_identity_digest != self.contract.policy_identity_digest
            ):
                raise ValueError("seed result policy identity does not match contract")
            if self.start_identity_digest is None:
                object.__setattr__(
                    self,
                    "start_identity_digest",
                    self.contract.start_identity.identity_digest,
                )
            if self.policy_identity_digest is None:
                object.__setattr__(
                    self,
                    "policy_identity_digest",
                    self.contract.policy_identity.identity_digest,
                )
            if self.assist_contract_path is None:
                object.__setattr__(
                    self,
                    "assist_contract_path",
                    self.contract.assist_contract_path,
                )
            if self.assist_contract_digest is None:
                object.__setattr__(
                    self,
                    "assist_contract_digest",
                    self.contract.assist_contract_digest,
                )
            if self.assist_mode is None:
                object.__setattr__(self, "assist_mode", self.contract.assist_mode)
            if self.assist_contract_path != self.contract.assist_contract_path:
                raise ValueError("seed result assist contract path does not match contract")
            if self.assist_contract_digest != self.contract.assist_contract_digest:
                raise ValueError("seed result assist contract digest does not match contract")
            if self.assist_mode != self.contract.assist_mode:
                raise ValueError("seed result assist mode does not match contract")
        object.__setattr__(self, "assists", _normalize_assists(self.assists))
        object.__setattr__(self, "ram_writes", _normalize_event_count(self.ram_writes, "ram_writes"))
        object.__setattr__(
            self,
            "mid_run_loads",
            _normalize_event_count(self.mid_run_loads, "mid_run_loads"),
        )
        audit = self.attempt_audit
        if audit is not None and not isinstance(audit, AttemptAudit):
            raise TypeError("attempt_audit must be an AttemptAudit or None")
        if audit is not None:
            if not self.assists:
                object.__setattr__(self, "assists", dict(audit.assists))
            if not self.ram_writes:
                object.__setattr__(self, "ram_writes", audit.ram_writes)
            if not self.mid_run_loads:
                object.__setattr__(self, "mid_run_loads", audit.mid_run_loads)
            if self.start_identity_digest is None:
                object.__setattr__(self, "start_identity_digest", audit.start_identity_digest)
            if self.policy_identity_digest is None:
                object.__setattr__(self, "policy_identity_digest", audit.policy_identity_digest)
            if audit.start_identity_digest is not None:
                start_identity_explicit = True
            if audit.policy_identity_digest is not None:
                policy_identity_explicit = True
            if self.contract is not None:
                if (
                    audit.start_identity_digest is not None
                    and audit.start_identity_digest != self.contract.start_identity_digest
                ):
                    raise ValueError("seed attempt audit start identity does not match contract")
                if (
                    audit.policy_identity_digest is not None
                    and audit.policy_identity_digest != self.contract.policy_identity_digest
                ):
                    raise ValueError("seed attempt audit policy identity does not match contract")
        if self.start_identity_digest is None:
            object.__setattr__(
                self,
                "start_identity_digest",
                StartIdentity(f"seed:{self.seed}").identity_digest,
            )
        else:
            object.__setattr__(
                self,
                "start_identity_digest",
                _validate_identity_digest(self.start_identity_digest, "start_identity_digest"),
            )
        if self.policy_identity_digest is None:
            object.__setattr__(
                self,
                "policy_identity_digest",
                PolicyIdentity("unbound-policy").identity_digest,
            )
        else:
            object.__setattr__(
                self,
                "policy_identity_digest",
                _validate_identity_digest(self.policy_identity_digest, "policy_identity_digest"),
            )
        if self.assist_contract_path is not None and (
            not isinstance(self.assist_contract_path, str)
            or not self.assist_contract_path.strip()
        ):
            raise ValueError("assist_contract_path must be a non-empty string or None")
        if self.assist_contract_digest is not None:
            object.__setattr__(
                self,
                "assist_contract_digest",
                _validate_identity_digest(
                    self.assist_contract_digest,
                    "assist_contract_digest",
                ),
            )
        assist_mode = _normalize_assist_mode(self.assist_mode)
        _validate_assist_contract_fields(
            intervention,
            self.assist_contract_path,
            self.assist_contract_digest,
        )
        if intervention.is_clean and assist_mode is not None:
            raise ValueError("Clean seed results cannot carry assist_mode")
        if intervention.is_clean and (
            self.ram_writes or self.mid_run_loads or sum(self.assists.values())
        ):
            raise ClaimValidationError("Clean seed attempt contains an intervention")
        object.__setattr__(self, "runtime_observation_class", observation)
        object.__setattr__(self, "intervention_class", intervention)
        object.__setattr__(self, "assist_mode", assist_mode)
        object.__setattr__(self, "_start_identity_explicit", start_identity_explicit)
        object.__setattr__(self, "_policy_identity_explicit", policy_identity_explicit)

    @classmethod
    def from_benchmark_attempt(
        cls,
        seed: SeedValue,
        attempt: BenchmarkAttemptResult,
        contract: EvaluationContract | None = None,
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
        resolved_contract = contract or attempt.contract
        audit = attempt.attempt_audit or AttemptAudit.from_info(info)
        if resolved_contract is None:
            resolved_contract = EvaluationContract(
                runtime_observation_class=RuntimeObservationClass.BRONZE,
                intervention_class=InterventionClass.CLEAN,
                start_identity=StartIdentity(f"seed:{seed}"),
                policy_identity=PolicyIdentity("unbound-policy"),
            )
        audit = _audit_with_contract_identity(audit, resolved_contract)
        validate_claim(resolved_contract, audit)
        return cls(
            seed=seed,
            success=attempt.success,
            frames=attempt.steps,
            terminal_milestone=terminal_milestone,
            failure_mode=failure_mode,
            assists=info.get("assists", {}),
            runtime_observation_class=resolved_contract.runtime_observation_class,
            intervention_class=resolved_contract.intervention_class,
            start_identity_digest=resolved_contract.start_identity.identity_digest,
            policy_identity_digest=resolved_contract.policy_identity.identity_digest,
            assist_contract_path=resolved_contract.assist_contract_path,
            assist_contract_digest=resolved_contract.assist_contract_digest,
            assist_mode=resolved_contract.assist_mode,
            ram_writes=audit.ram_writes,
            mid_run_loads=audit.mid_run_loads,
            attempt_audit=audit,
            contract=resolved_contract,
        )

    def bind_contract(self, contract: EvaluationContract) -> "SeedAttemptResult":
        """Attach a published contract to an extracted per-seed result."""
        if not isinstance(contract, EvaluationContract):
            raise TypeError("contract must be an EvaluationContract")
        if self.runtime_observation_class is not contract.runtime_observation_class:
            raise ValueError("seed result runtime observation class does not match contract")
        if self.intervention_class is not contract.intervention_class:
            raise ValueError("seed result intervention class does not match contract")
        if self.contract is not None and self.contract != contract:
            raise ValueError("seed result already carries a different evaluation contract")
        if (
            self._start_identity_explicit
            and self.start_identity_digest != contract.start_identity_digest
        ):
            raise ValueError("seed result start identity does not match contract")
        if (
            self._policy_identity_explicit
            and self.policy_identity_digest != contract.policy_identity_digest
        ):
            raise ValueError("seed result policy identity does not match contract")
        audit = _audit_with_contract_identity(self.attempt_audit or AttemptAudit(
            ram_writes=self.ram_writes,
            mid_run_loads=self.mid_run_loads,
            assists=self.assists,
        ), contract)
        result = replace(
            self,
            start_identity_digest=contract.start_identity.identity_digest,
            policy_identity_digest=contract.policy_identity.identity_digest,
            assist_contract_path=contract.assist_contract_path,
            assist_contract_digest=contract.assist_contract_digest,
            assist_mode=contract.assist_mode,
            attempt_audit=audit,
            contract=contract,
        )
        validate_claim(contract, audit)
        return result

    def to_record(self) -> dict[str, Any]:
        contract = self.contract or EvaluationContract(
            runtime_observation_class=self.runtime_observation_class,
            intervention_class=self.intervention_class,
            start_identity=StartIdentity(
                f"seed-record:{self.seed}",
                digest=self.start_identity_digest,
            ),
            policy_identity=PolicyIdentity(
                "seed-record-policy",
                digest=self.policy_identity_digest,
            ),
            assist_contract_path=self.assist_contract_path,
            assist_contract_digest=self.assist_contract_digest,
            assist_mode=self.assist_mode,
        )
        audit = _audit_with_contract_identity(
            self.attempt_audit
            or AttemptAudit(
                ram_writes=self.ram_writes,
                mid_run_loads=self.mid_run_loads,
                assists=self.assists,
            ),
            contract,
        )
        validate_claim(contract, audit)
        return {
            "seed": _to_jsonable(self.seed),
            "outcome": "success" if self.success else "failure",
            "success": self.success,
            "frames": self.frames,
            "terminal_milestone": _to_jsonable(self.terminal_milestone),
            "failure_mode": self.failure_mode,
            "assists": _to_jsonable(self.assists),
            "runtime_observation_class": contract.runtime_observation_class.value,
            "intervention_class": contract.intervention_class.value,
            "start_identity_digest": contract.start_identity.identity_digest,
            "policy_identity_digest": contract.policy_identity.identity_digest,
            "assist_contract_path": contract.assist_contract_path,
            "assist_contract_digest": contract.assist_contract_digest,
            "assist_mode": contract.assist_mode,
            "ram_writes": audit.ram_writes,
            "mid_run_loads": audit.mid_run_loads,
            "attempt_audit": audit.to_record(),
            "contract": contract.to_record(),
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
            if result.runtime_observation_class is not self.config.runtime_observation_class:
                raise ValueError("seed result runtime observation class does not match config")
            if result.intervention_class is not self.config.intervention_class:
                raise ValueError("seed result intervention class does not match config")
            if result.assist_contract_path != self.config.assist_contract_path:
                raise ValueError("seed result assist contract path does not match config")
            if result.assist_contract_digest != self.config.assist_contract_digest:
                raise ValueError("seed result assist contract digest does not match config")
            if result.assist_mode != self.config.assist_mode:
                raise ValueError("seed result assist mode does not match config")
            _validate_seed_result_budget(self.config, result)
            contract = _contract_for_seed_result(result)
            if (
                self.config._start_identity_explicit
                and result.start_identity_digest != self.config.start_identity.identity_digest
            ):
                raise ValueError("seed result start identity does not match config")
            if (
                self.config._policy_identity_explicit
                and result.policy_identity_digest != self.config.policy_identity.identity_digest
            ):
                raise ValueError("seed result policy identity does not match config")
            audit = _audit_with_contract_identity(
                result.attempt_audit
                or AttemptAudit(
                    ram_writes=result.ram_writes,
                    mid_run_loads=result.mid_run_loads,
                    assists=result.assists,
                ),
                contract,
            )
            validate_claim(contract, audit)
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
            failure_reason = None if success else _failure_reason(step_count, case.max_steps, terminated, truncated)
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


def write_seed_robustness_report(
    path: str | Path,
    report: SeedRobustnessReport,
) -> Path:
    """Write a canonical JSON artifact via atomic replace and return its path."""
    if not isinstance(report, SeedRobustnessReport):
        raise TypeError("report must be a SeedRobustnessReport")
    from retro_harness.seed_campaign import atomic_write_json

    return atomic_write_json(path, report.to_record())


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
        if case_contract.assist_mode != contract.assist_mode:
            raise ValueError(
                "evaluation contract assist_mode does not match the BenchmarkCase contract"
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


def _audit_with_contract_identity(
    audit: AttemptAudit,
    contract: EvaluationContract,
) -> AttemptAudit:
    if not isinstance(audit, AttemptAudit):
        raise TypeError("audit must be an AttemptAudit")
    return replace(
        audit,
        start_identity_digest=(
            audit.start_identity_digest or contract.start_identity.identity_digest
        ),
        policy_identity_digest=(
            audit.policy_identity_digest or contract.policy_identity.identity_digest
        ),
        runtime_observation_class=(
            audit.runtime_observation_class or contract.runtime_observation_class
        ),
        intervention_class=audit.intervention_class or contract.intervention_class,
    )


def _validate_assist_contract_fields(
    intervention: InterventionClass,
    path: str | None,
    digest: str | None,
) -> None:
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("assist_contract_path must be a non-empty string or None")
    if digest is not None:
        _validate_identity_digest(digest, "assist_contract_digest")
    if intervention.is_clean and (path is not None or digest is not None):
        raise ValueError("Clean contracts cannot carry an assist contract")
    if not intervention.is_clean and (path is None or digest is None):
        raise ValueError(
            "assisted contracts require both assist_contract_path and "
            "assist_contract_digest"
        )


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


def _validate_seed_value(seed: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (str, int)):
        raise TypeError("seed values must be strings or integers")


def _normalize_event_count(value: Any, field_name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value >= 0:
        return value
    raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_seed_result_budget(
    config: SeedRobustnessConfig,
    result: SeedAttemptResult,
) -> None:
    if result.frames > config.budget:
        raise ValueError(
            f"frames for seed {result.seed!r} exceed the published frame budget"
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

# Campaign runner re-exports (canonical home: retro_harness.seed_campaign).
def __getattr__(name: str) -> Any:
    if name in {
        "SEED_CAMPAIGN_SCHEMA_VERSION",
        "SeedCampaignContractMismatch",
        "SeedCampaignError",
        "SeedCampaignInfraError",
        "SeedCampaignLedger",
        "SeedCampaignResult",
        "SeedCampaignRunner",
        "SeedExecutionRow",
        "SeedExecutionStatus",
        "atomic_write_json",
        "atomic_write_text",
        "config_contract_digest",
        "run_seed_campaign",
    }:
        import retro_harness.seed_campaign as _seed_campaign

        return getattr(_seed_campaign, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
