"""Generic benchmark definitions and runners for emulator-backed tasks."""

from __future__ import annotations

import json
import hashlib
import inspect
import marshal
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np

from retro_harness.recordings import append_jsonl


class RuntimeObservationClass(str, Enum):
    """What the policy may observe while an attempt is active."""

    GOLD = "Gold"
    SILVER = "Silver"
    BRONZE = "Bronze"

    @classmethod
    def _missing_(cls, value: object) -> "RuntimeObservationClass | None":
        if isinstance(value, str):
            normalized = value.strip().casefold()
            for member in cls:
                if member.value.casefold() == normalized:
                    return member
        return None

    @classmethod
    def from_value(
        cls,
        value: "RuntimeObservationClass | str | BenchmarkTier",
    ) -> "RuntimeObservationClass":
        if isinstance(value, cls):
            return value
        if isinstance(value, BenchmarkTier):
            return cls(value.value.title())
        if not isinstance(value, str):
            raise TypeError(
                "runtime observation class must be a RuntimeObservationClass or string"
            )
        normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
        values = {member.value.casefold(): member for member in cls}
        if normalized in values:
            return values[normalized]
        raise ValueError(
            f"invalid runtime observation class {value!r}; "
            "expected Gold, Silver, or Bronze"
        )


class InterventionClass(str, Enum):
    """What an attempt may mutate, independent of runtime observations."""

    CLEAN = "Clean"
    ASSISTED = "Assisted"
    SURVIVAL_ASSISTED = "Survival-assisted"
    RESOURCE_ASSISTED = "Resource-assisted"
    PROTECTION_ASSISTED = "Protection-assisted"
    PROGRESSION_ASSISTED = "Progression-assisted"

    @classmethod
    def _missing_(cls, value: object) -> "InterventionClass | None":
        if isinstance(value, str):
            normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
            for member in cls:
                if member.value.casefold() == normalized:
                    return member
        return None

    @classmethod
    def from_value(cls, value: "InterventionClass | str") -> "InterventionClass":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("intervention class must be an InterventionClass or string")
        normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
        values = {member.value.casefold(): member for member in cls}
        aliases = {
            "clean": cls.CLEAN,
            "assisted": cls.ASSISTED,
            "survival": cls.SURVIVAL_ASSISTED,
            "resource": cls.RESOURCE_ASSISTED,
            "protection": cls.PROTECTION_ASSISTED,
            "progression": cls.PROGRESSION_ASSISTED,
        }
        if normalized in values:
            return values[normalized]
        if normalized in aliases:
            return aliases[normalized]

        # Existing manifests sometimes spell a composed intervention as
        # ``resource_assisted+protection_assisted``. Preserve that input as
        # the typed assisted class without accepting arbitrary class strings.
        parts = [part for part in normalized.split("+") if part]
        if len(parts) > 1:
            known_assists = {
                "survival-assisted",
                "resource-assisted",
                "protection-assisted",
                "progression-assisted",
            }
            if all(part in known_assists for part in parts):
                return cls.ASSISTED
        raise ValueError(
            f"invalid intervention class {value!r}; "
            "expected Clean or a known assisted class"
        )

    @property
    def is_clean(self) -> bool:
        return self is type(self).CLEAN


def _identity_digest(kind: str, values: Mapping[str, str | None]) -> str:
    payload = json.dumps(
        {"kind": kind, **dict(sorted(values.items()))},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_identity_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty identity digest")
    return value.strip()


@dataclass(frozen=True)
class StartIdentity:
    """Stable identity for the ROM and published start condition."""

    start_state: str
    digest: str | None = None
    rom_sha256: str | None = None
    state_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rom_digest: str | None = None
    state_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.start_state, str) or not self.start_state.strip():
            raise ValueError("start_state must be a non-empty string")
        rom_sha256 = self.rom_sha256 or self.rom_digest
        state_sha256 = self.state_sha256 or self.state_digest
        for field_name, value in (
            ("rom_sha256", rom_sha256),
            ("state_sha256", state_sha256),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None")
        digest = self.digest
        if digest is None:
            digest = _identity_digest(
                "start",
                {
                    "start_state": self.start_state.strip(),
                    "rom_sha256": rom_sha256,
                    "state_sha256": state_sha256,
                },
            )
        object.__setattr__(self, "start_state", self.start_state.strip())
        object.__setattr__(self, "rom_sha256", rom_sha256)
        object.__setattr__(self, "state_sha256", state_sha256)
        object.__setattr__(self, "rom_digest", rom_sha256)
        object.__setattr__(self, "state_digest", state_sha256)
        object.__setattr__(self, "digest", _validate_identity_digest(digest, "digest"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def identity_digest(self) -> str:
        return self.digest  # type: ignore[return-value]

    @property
    def start_identity_digest(self) -> str:
        return self.identity_digest

    @classmethod
    def from_state(cls, start_state: str, **kwargs: Any) -> "StartIdentity":
        return cls(start_state=start_state, **kwargs)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "start_state": self.start_state,
            "digest": self.identity_digest,
        }
        if self.rom_sha256 is not None:
            record["rom_sha256"] = self.rom_sha256
        if self.state_sha256 is not None:
            record["state_sha256"] = self.state_sha256
        if self.metadata:
            record["metadata"] = _canonicalize_metadata(self.metadata)
        return record


@dataclass(frozen=True)
class PolicyIdentity:
    """Stable identity for the evaluated policy implementation."""

    name: str
    digest: str | None = None
    version: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    policy_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("policy name must be a non-empty string")
        for field_name in ("version", "source"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None")
        if self.digest is not None and self.policy_digest is not None and self.digest != self.policy_digest:
            raise ValueError("digest and policy_digest must match")
        digest = self.digest or self.policy_digest
        if digest is None:
            digest = _identity_digest(
                "policy",
                {
                    "name": self.name.strip(),
                    "version": self.version,
                    "source": self.source,
                },
            )
        object.__setattr__(self, "name", self.name.strip())
        normalized_digest = _validate_identity_digest(digest, "digest")
        object.__setattr__(self, "digest", normalized_digest)
        object.__setattr__(self, "policy_digest", normalized_digest)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def identity_digest(self) -> str:
        return self.digest  # type: ignore[return-value]

    @property
    def policy_identity_digest(self) -> str:
        return self.identity_digest

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> "PolicyIdentity":
        return cls(name=name, **kwargs)

    @classmethod
    def from_policy(cls, policy: Any) -> "PolicyIdentity":
        """Derive an identity from the policy implementation being executed."""
        return policy_identity_for(policy)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "digest": self.identity_digest,
        }
        if self.version is not None:
            record["version"] = self.version
        if self.source is not None:
            record["source"] = self.source
        if self.metadata:
            record["metadata"] = _canonicalize_metadata(self.metadata)
        return record


@dataclass(frozen=True)
class EvaluationContract:
    """Typed, auditable contract attached to a benchmark claim."""

    runtime_observation_class: RuntimeObservationClass | str
    intervention_class: InterventionClass | str
    start_identity: StartIdentity | None = None
    policy_identity: PolicyIdentity | None = None
    benchmark_id: str = ""
    objective: str = ""
    assist_contract_path: str | None = None
    assist_contract_digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start: StartIdentity | None = None
    policy: PolicyIdentity | None = None
    assist_mode: str | None = None

    def __post_init__(self) -> None:
        observation = RuntimeObservationClass.from_value(self.runtime_observation_class)
        intervention = InterventionClass.from_value(self.intervention_class)
        start_identity = self.start_identity or self.start
        policy_identity = self.policy_identity or self.policy
        if self.start_identity is not None and self.start is not None and self.start_identity != self.start:
            raise ValueError("start_identity and start must match")
        if self.policy_identity is not None and self.policy is not None and self.policy_identity != self.policy:
            raise ValueError("policy_identity and policy must match")
        if not isinstance(start_identity, StartIdentity):
            raise TypeError("start_identity must be a StartIdentity")
        if not isinstance(policy_identity, PolicyIdentity):
            raise TypeError("policy_identity must be a PolicyIdentity")
        for field_name in ("benchmark_id", "objective"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
        path = self.assist_contract_path
        digest = self.assist_contract_digest
        assist_mode = _normalize_assist_mode(self.assist_mode)
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ValueError("assist_contract_path must be a non-empty string or None")
        if digest is not None:
            digest = _validate_identity_digest(digest, "assist_contract_digest")
        metadata = dict(self.metadata)
        metadata_assist_mode = metadata.get("assist_mode")
        if metadata_assist_mode is not None:
            metadata_assist_mode = _normalize_assist_mode(metadata_assist_mode)
        if assist_mode is not None and metadata_assist_mode is not None:
            if assist_mode != metadata_assist_mode:
                raise ValueError("assist_mode and metadata assist_mode must match")
        if assist_mode is None:
            assist_mode = metadata_assist_mode
        if intervention.is_clean and (path is not None or digest is not None):
            raise ValueError("Clean contracts cannot carry an assist contract")
        if intervention.is_clean and assist_mode is not None:
            raise ValueError("Clean contracts cannot carry assist_mode")
        if not intervention.is_clean and (path is None or digest is None):
            raise ValueError(
                "assisted contracts require both assist_contract_path and "
                "assist_contract_digest"
            )
        object.__setattr__(self, "runtime_observation_class", observation)
        object.__setattr__(self, "intervention_class", intervention)
        object.__setattr__(self, "start_identity", start_identity)
        object.__setattr__(self, "policy_identity", policy_identity)
        object.__setattr__(self, "start", start_identity)
        object.__setattr__(self, "policy", policy_identity)
        object.__setattr__(self, "assist_contract_path", path.strip() if path else None)
        object.__setattr__(self, "assist_contract_digest", digest)
        object.__setattr__(self, "assist_mode", assist_mode)
        object.__setattr__(self, "metadata", metadata)

    @property
    def observation_class(self) -> RuntimeObservationClass:
        """Short alias for callers that do not need the runtime qualifier."""
        return self.runtime_observation_class  # type: ignore[return-value]

    @property
    def start_identity_digest(self) -> str:
        return self.start_identity.identity_digest

    @property
    def policy_identity_digest(self) -> str:
        return self.policy_identity.identity_digest

    @property
    def intervention(self) -> InterventionClass:
        return self.intervention_class  # type: ignore[return-value]

    def with_policy(self, policy_identity: PolicyIdentity) -> "EvaluationContract":
        return replace(self, policy_identity=policy_identity, policy=policy_identity)

    def to_record(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "objective": self.objective,
            "runtime_observation_class": self.runtime_observation_class.value,
            "intervention_class": self.intervention_class.value,
            "start_identity_digest": self.start_identity.identity_digest,
            "policy_identity_digest": self.policy_identity.identity_digest,
            "start_identity": self.start_identity.to_record(),
            "policy_identity": self.policy_identity.to_record(),
            "assist_contract_path": self.assist_contract_path,
            "assist_contract_digest": self.assist_contract_digest,
            "assist_mode": self.assist_mode,
            "metadata": _canonicalize_metadata(self.metadata),
        }


@dataclass(frozen=True)
class AttemptAudit:
    """Observed interventions and identity evidence for one attempt."""

    ram_writes: int | bool = 0
    mid_run_loads: int | bool = 0
    assists: Mapping[str, int] | int | bool | None = field(default_factory=dict)
    start_identity_digest: str | None = None
    policy_identity_digest: str | None = None
    runtime_observation_class: RuntimeObservationClass | str | None = None
    intervention_class: InterventionClass | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ram_writes",
            _normalize_event_count(self.ram_writes, "ram_writes"),
        )
        object.__setattr__(
            self,
            "mid_run_loads",
            _normalize_event_count(self.mid_run_loads, "mid_run_loads"),
        )
        object.__setattr__(self, "assists", _normalize_assists(self.assists))
        for field_name in ("start_identity_digest", "policy_identity_digest"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _validate_identity_digest(value, field_name),
                )
        if self.runtime_observation_class is not None:
            object.__setattr__(
                self,
                "runtime_observation_class",
                RuntimeObservationClass.from_value(self.runtime_observation_class),
            )
        if self.intervention_class is not None:
            object.__setattr__(
                self,
                "intervention_class",
                InterventionClass.from_value(self.intervention_class),
            )

    @property
    def assist_count(self) -> int:
        return sum(self.assists.values())

    @property
    def has_interventions(self) -> bool:
        return bool(self.ram_writes or self.mid_run_loads or self.assist_count)

    @classmethod
    def from_info(cls, info: Mapping[str, Any] | None) -> "AttemptAudit":
        values = info if isinstance(info, Mapping) else {}
        return cls(
            ram_writes=values.get("ram_writes", values.get("ram_write_count", 0)),
            mid_run_loads=values.get(
                "mid_run_loads",
                values.get("mid_run_load_count", values.get("save_state_loads", 0)),
            ),
            assists=values.get("assists", {}),
            start_identity_digest=values.get("start_identity_digest"),
            policy_identity_digest=values.get("policy_identity_digest"),
            runtime_observation_class=values.get("runtime_observation_class"),
            intervention_class=values.get("intervention_class"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "ram_writes": self.ram_writes,
            "mid_run_loads": self.mid_run_loads,
            "assists": _to_jsonable(self.assists),
            "runtime_observation_class": (
                self.runtime_observation_class.value
                if isinstance(self.runtime_observation_class, Enum)
                else self.runtime_observation_class
            ),
            "intervention_class": (
                self.intervention_class.value
                if isinstance(self.intervention_class, Enum)
                else self.intervention_class
            ),
            "start_identity_digest": self.start_identity_digest,
            "policy_identity_digest": self.policy_identity_digest,
        }


class ClaimValidationError(ValueError):
    """Raised when an attempt cannot support its published contract."""


def validate_claim(
    contract: EvaluationContract | Mapping[str, Any],
    audit: AttemptAudit | Mapping[str, Any] | None = None,
) -> bool:
    """Fail closed when an attempt violates its typed evaluation contract.

    The function returns ``True`` for a valid claim and raises
    :class:`ClaimValidationError` for a claim that must not be published. A
    complete result mapping may be passed as the sole argument for validating
    serialized benchmark records.
    """
    record: Mapping[str, Any] | None = None
    if audit is None:
        if not isinstance(contract, Mapping):
            raise TypeError("validate_claim requires an EvaluationContract and AttemptAudit")
        record = contract
        if "seed_results" in record:
            seed_results = record["seed_results"]
            if not isinstance(seed_results, list):
                raise TypeError("seed_results must be a list of claim records")
            for seed_record in seed_results:
                if not isinstance(seed_record, Mapping):
                    raise TypeError("seed_results must contain mapping records")
                validate_claim(seed_record)
            config = record.get("config")
            if isinstance(config, Mapping):
                _validate_seed_report_record(config, seed_results)
            return True
        contract_value = record.get("contract", record)
        audit_value = record.get("attempt_audit", record)
        contract = _contract_from_record(contract_value)
        audit = _audit_from_record(audit_value)
    elif isinstance(contract, AttemptAudit) and isinstance(audit, EvaluationContract):
        contract, audit = audit, contract

    if not isinstance(contract, EvaluationContract):
        raise TypeError("contract must be an EvaluationContract")
    if not isinstance(audit, AttemptAudit):
        if isinstance(audit, Mapping):
            audit = _audit_from_record(audit)
        else:
            raise TypeError("audit must be an AttemptAudit")

    if record is not None:
        _validate_serialized_claim_fields(record, contract, audit)

    errors: list[str] = []
    if audit.start_identity_digest != contract.start_identity.identity_digest:
        errors.append("start identity digest does not match the contract")
    if audit.policy_identity_digest != contract.policy_identity.identity_digest:
        errors.append("policy identity digest does not match the contract")
    if audit.runtime_observation_class is not None:
        if audit.runtime_observation_class is not contract.runtime_observation_class:
            errors.append("runtime observation class does not match the contract")
    if audit.intervention_class is not None:
        if audit.intervention_class is not contract.intervention_class:
            errors.append("intervention class does not match the contract")
    if contract.intervention_class.is_clean:
        if audit.ram_writes:
            errors.append("Clean claim contains RAM writes")
        if audit.mid_run_loads:
            errors.append("Clean claim contains mid-run loads")
        if audit.assist_count:
            errors.append("Clean claim contains assists")
    if errors:
        raise ClaimValidationError("; ".join(errors))
    return True


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


def _normalize_assist_mode(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("assist_mode must be a non-empty string or None")
    return value.strip()


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


def _contract_from_record(value: Any) -> EvaluationContract:
    if isinstance(value, EvaluationContract):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("record contract must be an EvaluationContract or mapping")
    start_digest = value.get("start_identity_digest")
    policy_digest = value.get("policy_identity_digest")
    if start_digest is None:
        start_value = value.get("start_identity", {})
        start_digest = start_value.get("digest") if isinstance(start_value, Mapping) else None
    if policy_digest is None:
        policy_value = value.get("policy_identity", {})
        policy_digest = (
            policy_value.get("digest") if isinstance(policy_value, Mapping) else None
        )
    start_state = "serialized-start"
    start_value = value.get("start_identity")
    if isinstance(start_value, Mapping):
        start_state = str(start_value.get("start_state", start_state))
    policy_name = "serialized-policy"
    policy_value = value.get("policy_identity")
    policy_identity_fields: Mapping[str, Any] = {}
    if isinstance(policy_value, Mapping):
        policy_name = str(policy_value.get("name", policy_name))
        policy_identity_fields = policy_value
    policy_metadata = policy_identity_fields.get("metadata", {})
    if not isinstance(policy_metadata, dict):
        policy_metadata = {}
    return EvaluationContract(
        runtime_observation_class=value.get("runtime_observation_class"),
        intervention_class=value.get("intervention_class"),
        start_identity=StartIdentity(start_state, digest=start_digest),
        policy_identity=PolicyIdentity(
            policy_name,
            digest=policy_digest,
            version=policy_identity_fields.get("version"),
            source=policy_identity_fields.get("source"),
            metadata=policy_metadata,
        ),
        benchmark_id=str(value.get("benchmark_id", "")),
        objective=str(value.get("objective", "")),
        assist_contract_path=value.get("assist_contract_path"),
        assist_contract_digest=value.get("assist_contract_digest"),
        assist_mode=value.get("assist_mode"),
        metadata=value.get("metadata", {}),
    )


def _record_identity_digest(
    value: Mapping[str, Any],
    field_name: str,
    *,
    label: str,
) -> str | None:
    direct = value.get(field_name)
    identity_name = field_name.removesuffix("_digest")
    nested = value.get(identity_name)
    nested_digest = nested.get("digest") if isinstance(nested, Mapping) else None
    if direct is not None and nested_digest is not None and direct != nested_digest:
        raise ClaimValidationError(f"{label} {field_name} contradicts its identity record")
    digest = direct if direct is not None else nested_digest
    if digest is None:
        return None
    return _validate_identity_digest(digest, f"{label} {field_name}")


def _validate_serialized_claim_fields(
    record: Mapping[str, Any],
    contract: EvaluationContract,
    audit: AttemptAudit,
) -> None:
    errors: list[str] = []
    start_digest = _record_identity_digest(
        record,
        "start_identity_digest",
        label="serialized claim",
    )
    if start_digest is not None and start_digest != contract.start_identity_digest:
        errors.append("serialized start identity digest does not match the contract")
    policy_digest = _record_identity_digest(
        record,
        "policy_identity_digest",
        label="serialized claim",
    )
    if policy_digest is not None and policy_digest != contract.policy_identity_digest:
        errors.append("serialized policy identity digest does not match the contract")

    for field_name, expected, normalizer in (
        (
            "runtime_observation_class",
            contract.runtime_observation_class,
            RuntimeObservationClass.from_value,
        ),
        ("intervention_class", contract.intervention_class, InterventionClass.from_value),
    ):
        value = record.get(field_name)
        if value is not None:
            try:
                normalized = normalizer(value)
            except (TypeError, ValueError) as exc:
                raise ClaimValidationError(
                    f"serialized claim has an invalid {field_name}"
                ) from exc
            if normalized is not expected:
                errors.append(f"serialized {field_name} does not match the contract")

    for field_name, expected in (
        ("assist_contract_path", contract.assist_contract_path),
        ("assist_contract_digest", contract.assist_contract_digest),
        ("assist_mode", contract.assist_mode),
    ):
        value = record.get(field_name)
        if value is not None and value != expected:
            errors.append(f"serialized {field_name} does not match the contract")

    nested_contract = record.get("contract")
    if isinstance(nested_contract, Mapping):
        nested_start = _record_identity_digest(
            nested_contract,
            "start_identity_digest",
            label="serialized contract",
        )
        nested_policy = _record_identity_digest(
            nested_contract,
            "policy_identity_digest",
            label="serialized contract",
        )
        if nested_start is not None and nested_start != contract.start_identity_digest:
            errors.append("serialized contract start identity does not match the contract")
        if nested_policy is not None and nested_policy != contract.policy_identity_digest:
            errors.append("serialized contract policy identity does not match the contract")

    for field_name, audit_value, contract_value in (
        (
            "start_identity_digest",
            audit.start_identity_digest,
            contract.start_identity_digest,
        ),
        (
            "policy_identity_digest",
            audit.policy_identity_digest,
            contract.policy_identity_digest,
        ),
    ):
        direct_value = record.get(field_name)
        if direct_value is not None and audit_value is not None and direct_value != audit_value:
            errors.append(f"serialized {field_name} does not match the attempt audit")
        if audit_value is not None and audit_value != contract_value:
            errors.append(f"serialized audit {field_name} does not match the contract")

    if errors:
        raise ClaimValidationError("; ".join(errors))


def _validate_seed_report_record(
    config: Mapping[str, Any],
    seed_results: list[Any],
) -> None:
    config_contract = config.get("contract")
    if config_contract is not None and not isinstance(config_contract, Mapping):
        raise TypeError("seed report config contract must be a mapping")

    config_start = _record_identity_digest(
        config,
        "start_identity_digest",
        label="seed report config",
    )
    config_policy = _record_identity_digest(
        config,
        "policy_identity_digest",
        label="seed report config",
    )
    if isinstance(config_contract, Mapping):
        for field_name, normalizer in (
            (
                "runtime_observation_class",
                RuntimeObservationClass.from_value,
            ),
            ("intervention_class", InterventionClass.from_value),
        ):
            config_value = config.get(field_name)
            contract_value = config_contract.get(field_name)
            if config_value is None or contract_value is None:
                continue
            try:
                config_class = normalizer(config_value)
                contract_class = normalizer(contract_value)
            except (TypeError, ValueError) as exc:
                raise ClaimValidationError(
                    f"seed report config contract has an invalid {field_name}"
                ) from exc
            if config_class is not contract_class:
                raise ClaimValidationError(
                    f"seed report config {field_name} contradicts its evaluation contract"
                )
        for field_name in (
            "assist_contract_path",
            "assist_contract_digest",
        ):
            config_value = config.get(field_name)
            contract_value = config_contract.get(field_name)
            if (
                config_value is not None
                and contract_value is not None
                and config_value != contract_value
            ):
                raise ClaimValidationError(
                    f"seed report config {field_name} contradicts its evaluation contract"
                )
        if config.get("assist_mode") != config_contract.get("assist_mode"):
            raise ClaimValidationError(
                "seed report config assist_mode contradicts its evaluation contract"
            )
        contract_start = _record_identity_digest(
            config_contract,
            "start_identity_digest",
            label="seed report config contract",
        )
        contract_policy = _record_identity_digest(
            config_contract,
            "policy_identity_digest",
            label="seed report config contract",
        )
        if config_start is not None and contract_start is not None and config_start != contract_start:
            raise ClaimValidationError(
                "seed report config start identity contradicts its evaluation contract"
            )
        if (
            config_policy is not None
            and contract_policy is not None
            and config_policy != contract_policy
        ):
            raise ClaimValidationError(
                "seed report config policy identity contradicts its evaluation contract"
            )
        config_start = contract_start if contract_start is not None else config_start
        config_policy = contract_policy if contract_policy is not None else config_policy

    for field_name, normalizer in (
        (
            "runtime_observation_class",
            RuntimeObservationClass.from_value,
        ),
        ("intervention_class", InterventionClass.from_value),
    ):
        config_value = config.get(field_name)
        if config_value is None:
            continue
        try:
            config_value = normalizer(config_value)
        except (TypeError, ValueError) as exc:
            raise ClaimValidationError(
                f"seed report config has an invalid {field_name}"
            ) from exc
        for seed_record in seed_results:
            if not isinstance(seed_record, Mapping):
                raise TypeError("seed_results must contain mapping records")
            seed_value = seed_record.get(field_name)
            if seed_value is None:
                continue
            try:
                seed_value = normalizer(seed_value)
            except (TypeError, ValueError) as exc:
                raise ClaimValidationError(
                    f"seed result has an invalid {field_name}"
                ) from exc
            if seed_value is not config_value:
                raise ClaimValidationError(
                    f"seed result {field_name} does not match the report config"
                )

    for field_name in (
        "assist_contract_path",
        "assist_contract_digest",
        "assist_mode",
    ):
        config_value = config.get(field_name)
        for seed_record in seed_results:
            if not isinstance(seed_record, Mapping):
                raise TypeError("seed_results must contain mapping records")
            if (
                config_value is not None
                and seed_record.get(field_name) != config_value
            ):
                raise ClaimValidationError(
                    f"seed result {field_name} does not match the report config"
                )

    scopes = {
        "start_identity_digest": config.get("start_identity_scope", "shared"),
        "policy_identity_digest": config.get("policy_identity_scope", "shared"),
    }
    for field_name, scope in scopes.items():
        if scope not in {"shared", "per-seed"}:
            raise ClaimValidationError(f"invalid seed report {field_name} scope")
        if scope == "shared":
            if field_name == "start_identity_digest":
                expected = config_start
            else:
                expected = config_policy
            if expected is None:
                continue
            for seed_record in seed_results:
                seed_digest = _record_identity_digest(
                    seed_record,
                    field_name,
                    label="seed result",
                )
                if seed_digest != expected:
                    raise ClaimValidationError(
                        f"seed result {field_name} contradicts the shared report identity"
                    )
        else:
            if config_contract is not None:
                raise ClaimValidationError(
                    f"seed report config contract cannot use {field_name} per-seed scope"
                )
            if field_name == "start_identity_digest" and config_start is not None:
                generator = config.get("generator")
                generator_version = config.get("generator_version")
                seeds = config.get("seeds")
                if (
                    isinstance(generator, str)
                    and isinstance(generator_version, str)
                    and isinstance(seeds, list)
                ):
                    expected_start = StartIdentity(
                        f"seed-set:{generator}:{generator_version}:{','.join(map(str, seeds))}"
                    )
                    if config_start != expected_start.identity_digest:
                        raise ClaimValidationError(
                            "per-seed report config start identity is not its seed-set identity"
                        )
            if field_name == "policy_identity_digest" and config_policy is not None:
                expected_policy = PolicyIdentity("unbound-policy")
                if config_policy != expected_policy.identity_digest:
                    raise ClaimValidationError(
                        "per-seed report config policy identity must be unbound"
                    )


def _audit_from_record(value: Any) -> AttemptAudit:
    if isinstance(value, AttemptAudit):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("record audit must be an AttemptAudit or mapping")
    nested = value.get("attempt_audit")
    if isinstance(nested, Mapping):
        value = nested
    return AttemptAudit(
        ram_writes=value.get("ram_writes", 0),
        mid_run_loads=value.get("mid_run_loads", 0),
        assists=value.get("assists", {}),
        start_identity_digest=value.get("start_identity_digest"),
        policy_identity_digest=value.get("policy_identity_digest"),
        runtime_observation_class=value.get("runtime_observation_class"),
        intervention_class=value.get("intervention_class"),
    )


def _same_policy_identity(
    first: PolicyIdentity,
    second: PolicyIdentity,
) -> bool:
    """Compare the verifiable identity, not its display label or metadata."""
    return first.identity_digest == second.identity_digest


def _policy_identity_is_verifiable(identity: PolicyIdentity) -> bool:
    """Return whether an identity carries source or bytecode evidence."""
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


def _source_for_identity(target: Any) -> str | None:
    try:
        return inspect.getsource(target)
    except (OSError, TypeError, IOError):
        return None


def _code_for_identity(target: Any) -> Any:
    code = getattr(target, "__code__", None)
    if code is not None:
        return code
    function = getattr(target, "__func__", None)
    return getattr(function, "__code__", None)


def _bytecode_digest_for_identity(targets: list[tuple[str, Any]]) -> str | None:
    records: list[str] = []
    for label, target in sorted(targets):
        code = _code_for_identity(target)
        if code is None:
            continue
        try:
            payload = marshal.dumps(code)
        except (TypeError, ValueError):
            continue
        digest = hashlib.sha256(payload).hexdigest()
        records.append(f"{label}:{digest}")
    if not records:
        return None
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def _policy_implementation_descriptor(
    policy: BenchmarkPolicy | Callable[..., Any],
) -> dict[str, str | None]:
    """Return stable implementation fields used by ``policy_identity_for``.

    Source is preferred because it captures the complete class or callable.
    If source is unavailable, the callable members' marshalled bytecode is
    used.  Objects with neither expose a deterministic module-qualified-name
    fallback; this intentionally avoids object reprs and memory addresses.
    """
    if inspect.isfunction(policy) or inspect.ismethod(policy) or inspect.isbuiltin(policy):
        target = policy
        implementation_kind = "callable"
        member_targets = [("callable", policy)]
    elif inspect.isclass(policy):
        target = policy
        implementation_kind = "class"
        member_targets = []
        for name in ("act", "__call__"):
            member = getattr(policy, name, None)
            if callable(member):
                member_targets.append((name, member))
    else:
        target = type(policy)
        implementation_kind = "class"
        member_targets = []
        for name in ("act", "__call__"):
            member = getattr(policy, name, None)
            if callable(member):
                member_targets.append((name, member))

    module = getattr(target, "__module__", None) or getattr(
        type(policy), "__module__", "builtins"
    )
    qualname = getattr(target, "__qualname__", None) or getattr(
        target, "__name__", type(policy).__qualname__
    )

    source = _source_for_identity(target)
    if source is None:
        source_parts = []
        for name, member in member_targets:
            member_source = _source_for_identity(member)
            if member_source is not None:
                source_parts.append(f"{name}\n{member_source}")
        if source_parts:
            source = "\n".join(source_parts)

    source_digest = (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        if source is not None
        else None
    )
    bytecode_digest = None
    if source_digest is None:
        bytecode_digest = _bytecode_digest_for_identity(member_targets)

    if source_digest is not None:
        fingerprint_kind = "source"
    elif bytecode_digest is not None:
        fingerprint_kind = "bytecode"
    else:
        fingerprint_kind = "module-qualified-name"

    return {
        "implementation_kind": implementation_kind,
        "module": str(module),
        "qualname": str(qualname),
        "fingerprint_kind": fingerprint_kind,
        "source_sha256": source_digest,
        "bytecode_sha256": bytecode_digest,
    }


def policy_identity_for(
    policy: BenchmarkPolicy | Callable[..., Any],
) -> PolicyIdentity:
    """Derive a verifiable identity for the policy implementation.

    The mutable ``policy.name`` remains a display label only.  The identity
    digest covers the implementation's module and qualified name plus source,
    bytecode, or the documented stable module-qualified-name fallback.
    """
    descriptor = _policy_implementation_descriptor(policy)
    return PolicyIdentity(
        name=_policy_name(policy),
        digest=_identity_digest("policy-implementation-v1", descriptor),
        version="implementation-v1",
        source=f"{descriptor['module']}:{descriptor['qualname']}",
        metadata={
            "implementation_kind": descriptor["implementation_kind"],
            "implementation_module": descriptor["module"],
            "implementation_qualname": descriptor["qualname"],
            "fingerprint_kind": descriptor["fingerprint_kind"],
            "source_sha256": descriptor["source_sha256"],
            "bytecode_sha256": descriptor["bytecode_sha256"],
        },
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


def _normalize_assists(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"assist": 1} if value else {}
    if isinstance(value, int):
        if value < 0:
            raise ValueError("assist counts must be non-negative integers")
        return {"assist": value} if value else {}
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
