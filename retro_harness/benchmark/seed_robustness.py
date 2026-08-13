"""Seed-robustness DTOs: campaign scalars + owned EvaluationContract.

Identity fields (observation/intervention/assist/digests) live on
:class:`~retro_harness.benchmark.claims.EvaluationContract`. Config and
per-seed result types keep campaign/outcome scalars and expose former
mirrored fields as properties for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace, InitVar
from pathlib import Path
from typing import Any

from retro_harness.audit import (
    AttemptAudit,
    InterventionClass,
    RuntimeObservationClass,
    normalize_assists as _normalize_assists,
)
from retro_harness.benchmark.claims import (
    ClaimValidationError,
    EvaluationContract,
    PolicyIdentity,
    StartIdentity,
    _canonicalize_metadata,
    _normalize_assist_mode,
    _validate_identity_digest,
    validate_claim,
)

SeedValue = str | int

SEED_ROBUSTNESS_SCHEMA_VERSION = 1


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


def _to_jsonable(value: Any) -> Any:
    from enum import Enum
    import numpy as np

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


def _build_identity_contract(
    *,
    runtime_observation_class: RuntimeObservationClass | str | None,
    intervention_class: InterventionClass | str | None,
    start_identity: StartIdentity | None,
    policy_identity: PolicyIdentity | None,
    assist_contract_path: str | None,
    assist_contract_digest: str | None,
    assist_mode: str | None,
    metadata: dict[str, Any],
    generator: str,
    generator_version: str,
    goal: str,
    seeds: tuple[SeedValue, ...],
    contract: EvaluationContract | None,
) -> tuple[EvaluationContract, bool, bool, bool]:
    """Resolve the owned contract and which identities were caller-explicit."""
    start_explicit = start_identity is not None
    policy_explicit = policy_identity is not None
    contract_explicit = contract is not None

    if contract is not None:
        if not isinstance(contract, EvaluationContract):
            raise TypeError("contract must be an EvaluationContract or None")
        observation = (
            RuntimeObservationClass.from_value(runtime_observation_class)
            if runtime_observation_class is not None
            else contract.runtime_observation_class
        )
        intervention = (
            InterventionClass.from_value(intervention_class)
            if intervention_class is not None
            else contract.intervention_class
        )
        if contract.runtime_observation_class is not observation:
            raise ValueError("contract runtime observation class does not match config")
        if contract.intervention_class is not intervention:
            raise ValueError("contract intervention class does not match config")
        if start_identity is not None and start_identity != contract.start_identity:
            raise ValueError("config start identity does not match its evaluation contract")
        if policy_identity is not None and policy_identity != contract.policy_identity:
            raise ValueError("config policy identity does not match its evaluation contract")
        assist_path = assist_contract_path
        assist_digest = assist_contract_digest
        mode = _normalize_assist_mode(assist_mode)
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
        if mode is None:
            mode = contract.assist_mode
        elif mode != contract.assist_mode:
            raise ValueError("config assist mode does not match its evaluation contract")
        if (
            assist_path != contract.assist_contract_path
            or assist_digest != contract.assist_contract_digest
            or mode != contract.assist_mode
        ):
            # Already checked; keep for clarity if contract fields differ only by None fill.
            pass
        start_explicit = True
        policy_explicit = True
        return contract, contract_explicit, start_explicit, policy_explicit

    if runtime_observation_class is None or intervention_class is None:
        raise ValueError(
            "runtime_observation_class and intervention_class are required "
            "when contract is omitted"
        )
    observation = RuntimeObservationClass.from_value(runtime_observation_class)
    intervention = InterventionClass.from_value(intervention_class)
    mode = _normalize_assist_mode(assist_mode)
    if intervention.is_clean and mode is not None:
        raise ValueError("Clean configs cannot carry assist_mode")
    resolved_start = start_identity or StartIdentity(
        f"seed-set:{generator}:{generator_version}:{','.join(map(str, seeds))}"
    )
    resolved_policy = policy_identity or PolicyIdentity("unbound-policy")
    identity = EvaluationContract(
        runtime_observation_class=observation,
        intervention_class=intervention,
        start_identity=resolved_start,
        policy_identity=resolved_policy,
        benchmark_id=f"seed-robust:{generator}",
        objective=goal,
        assist_contract_path=assist_contract_path,
        assist_contract_digest=assist_contract_digest,
        assist_mode=mode,
        metadata=metadata,
    )
    return identity, False, start_explicit, policy_explicit


@dataclass(frozen=True)
class SeedRobustnessConfig:
    """Published contract for a deterministic multi-seed benchmark report.

    Campaign scalars live on this DTO. Observation/intervention/assist and
    identity digests are owned by :attr:`_identity_contract` (exposed via
    properties and optional explicit :attr:`contract`).
    """

    generator: str
    generator_version: str
    logic: str
    goal: str
    seeds: tuple[SeedValue, ...]
    budget: int
    success_threshold: int
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_observation_class: InitVar[RuntimeObservationClass | str | None] = None
    intervention_class: InitVar[InterventionClass | str | None] = None
    start_identity: InitVar[StartIdentity | None] = None
    policy_identity: InitVar[PolicyIdentity | None] = None
    assist_contract_path: InitVar[str | None] = None
    assist_contract_digest: InitVar[str | None] = None
    assist_mode: InitVar[str | None] = None
    contract: InitVar[EvaluationContract | None] = None
    _identity_contract: EvaluationContract = field(init=False, repr=False, compare=True)
    _contract_explicit: bool = field(init=False, repr=False, compare=False, default=False)
    _start_identity_explicit: bool = field(init=False, repr=False, compare=False, default=False)
    _policy_identity_explicit: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(
        self,
        runtime_observation_class: RuntimeObservationClass | str | None,
        intervention_class: InterventionClass | str | None,
        start_identity: StartIdentity | None,
        policy_identity: PolicyIdentity | None,
        assist_contract_path: str | None,
        assist_contract_digest: str | None,
        assist_mode: str | None,
        contract: EvaluationContract | None,
    ) -> None:
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
        if start_identity is not None and not isinstance(start_identity, StartIdentity):
            raise TypeError("start_identity must be a StartIdentity or None")
        if policy_identity is not None and not isinstance(policy_identity, PolicyIdentity):
            raise TypeError("policy_identity must be a PolicyIdentity or None")

        metadata = dict(self.metadata)
        identity, contract_explicit, start_explicit, policy_explicit = _build_identity_contract(
            runtime_observation_class=runtime_observation_class,
            intervention_class=intervention_class,
            start_identity=start_identity,
            policy_identity=policy_identity,
            assist_contract_path=assist_contract_path,
            assist_contract_digest=assist_contract_digest,
            assist_mode=assist_mode,
            metadata=metadata,
            generator=self.generator,
            generator_version=self.generator_version,
            goal=self.goal,
            seeds=seeds,
            contract=contract,
        )
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_identity_contract", identity)
        object.__setattr__(self, "_contract_explicit", contract_explicit)
        object.__setattr__(self, "_start_identity_explicit", start_explicit)
        object.__setattr__(self, "_policy_identity_explicit", policy_explicit)

    def __getattribute__(self, name: str) -> Any:
        # InitVar defaults remain as class attributes; intercept before they shadow.
        if name in {
            "contract",
            "evaluation_contract",
            "runtime_observation_class",
            "intervention_class",
            "start_identity",
            "policy_identity",
            "assist_contract_path",
            "assist_contract_digest",
            "assist_mode",
        }:
            identity = object.__getattribute__(self, "_identity_contract")
            explicit = object.__getattribute__(self, "_contract_explicit")
            if name in {"contract", "evaluation_contract"}:
                return identity if explicit else None
            return getattr(identity, name)
        return object.__getattribute__(self, name)

    @property
    def seed_count(self) -> int:
        """The published T in the S/T contract."""
        return len(self.seeds)

    def to_record(self) -> dict[str, Any]:
        """Return the stable, JSON-safe contract portion of a report."""
        identity = self._identity_contract
        record = {
            "generator": self.generator,
            "generator_version": self.generator_version,
            "logic": self.logic,
            "goal": self.goal,
            "seeds": [_to_jsonable(seed) for seed in self.seeds],
            "seed_count": self.seed_count,
            "budget": self.budget,
            "budget_unit": "frames",
            "success_threshold": self.success_threshold,
            "runtime_observation_class": identity.runtime_observation_class.value,
            "intervention_class": identity.intervention_class.value,
            "start_identity_digest": identity.start_identity.identity_digest,
            "start_identity_scope": (
                "shared" if self._start_identity_explicit else "per-seed"
            ),
            "start_identity": identity.start_identity.to_record(),
            "policy_identity_digest": identity.policy_identity.identity_digest,
            "policy_identity_scope": (
                "shared" if self._policy_identity_explicit else "per-seed"
            ),
            "policy_identity": identity.policy_identity.to_record(),
            "assist_contract_path": identity.assist_contract_path,
            "assist_contract_digest": identity.assist_contract_digest,
            "assist_mode": identity.assist_mode,
            "metadata": _canonicalize_metadata(self.metadata),
        }
        if self._contract_explicit:
            record["contract"] = identity.to_record()
        return record


    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SeedRobustnessConfig":
        """Rehydrate from a serialized config record.

        Prefer the embedded ``contract`` when present so InitVar mirrors are
        derived from the owned EvaluationContract rather than reconstructed
        from flat duplicated fields.
        """
        from collections.abc import Mapping as MappingABC

        from retro_harness.benchmark.claims import _contract_from_record

        if not isinstance(record, MappingABC):
            raise TypeError("config record must be a mapping")
        seeds = record.get("seeds")
        if not isinstance(seeds, (list, tuple)) or not seeds:
            raise ValueError("config.seeds must be a non-empty sequence")
        scalars = dict(
            generator=str(record["generator"]),
            generator_version=str(record["generator_version"]),
            logic=str(record["logic"]),
            goal=str(record["goal"]),
            seeds=tuple(seeds),
            budget=int(record["budget"]),
            success_threshold=int(record["success_threshold"]),
            metadata=dict(record.get("metadata") or {}),
        )
        contract_record = record.get("contract")
        if isinstance(contract_record, MappingABC):
            return cls(**scalars, contract=_contract_from_record(contract_record))

        start_identity = None
        start_record = record.get("start_identity")
        if isinstance(start_record, MappingABC) and record.get("start_identity_scope") == "shared":
            start_identity = StartIdentity(
                str(start_record.get("start_state", "ledger-start")),
                digest=start_record.get("digest") or record.get("start_identity_digest"),
                rom_sha256=start_record.get("rom_sha256"),
                state_sha256=start_record.get("state_sha256"),
                metadata=start_record.get("metadata") or {},
            )
        policy_identity = None
        policy_record = record.get("policy_identity")
        if isinstance(policy_record, MappingABC) and record.get("policy_identity_scope") == "shared":
            policy_identity = PolicyIdentity(
                str(policy_record.get("name", "ledger-policy")),
                digest=policy_record.get("digest") or record.get("policy_identity_digest"),
                version=policy_record.get("version"),
                source=policy_record.get("source"),
                metadata=policy_record.get("metadata") or {},
            )
        return cls(
            **scalars,
            runtime_observation_class=RuntimeObservationClass.from_value(
                record["runtime_observation_class"]
            ),
            intervention_class=InterventionClass.from_value(record["intervention_class"]),
            start_identity=start_identity,
            policy_identity=policy_identity,
            assist_contract_path=record.get("assist_contract_path"),
            assist_contract_digest=record.get("assist_contract_digest"),
            assist_mode=record.get("assist_mode"),
        )

    def to_contract(self) -> EvaluationContract:
        """Build the typed contract represented by this seed configuration."""
        return self._identity_contract


@dataclass(frozen=True)
class SeedAttemptResult:
    """Stable per-seed outcome used by :class:`SeedRobustnessReport`.

    Outcome scalars live here; identity/assist/observation fields are owned by
    :attr:`contract` and exposed as compatibility properties.
    """

    seed: SeedValue
    success: bool
    frames: int
    terminal_milestone: str | int | None = None
    failure_mode: str | None = None
    assists: dict[str, int] = field(default_factory=dict)
    ram_writes: int | bool = 0
    mid_run_loads: int | bool = 0
    attempt_audit: AttemptAudit | None = None
    runtime_observation_class: InitVar[
        RuntimeObservationClass | str
    ] = RuntimeObservationClass.BRONZE
    intervention_class: InitVar[InterventionClass | str] = InterventionClass.CLEAN
    start_identity_digest: InitVar[str | None] = None
    policy_identity_digest: InitVar[str | None] = None
    assist_contract_path: InitVar[str | None] = None
    assist_contract_digest: InitVar[str | None] = None
    assist_mode: InitVar[str | None] = None
    contract: EvaluationContract | None = None
    _start_identity_explicit: bool = field(init=False, repr=False, compare=False, default=False)
    _policy_identity_explicit: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(
        self,
        runtime_observation_class: RuntimeObservationClass | str,
        intervention_class: InterventionClass | str,
        start_identity_digest: str | None,
        policy_identity_digest: str | None,
        assist_contract_path: str | None,
        assist_contract_digest: str | None,
        assist_mode: str | None,
    ) -> None:
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

        start_identity_explicit = start_identity_digest is not None
        policy_identity_explicit = policy_identity_digest is not None
        observation = RuntimeObservationClass.from_value(runtime_observation_class)
        intervention = InterventionClass.from_value(intervention_class)
        assists = _normalize_assists(self.assists)
        ram_writes = _normalize_event_count(self.ram_writes, "ram_writes")
        mid_run_loads = _normalize_event_count(self.mid_run_loads, "mid_run_loads")
        audit = self.attempt_audit
        if audit is not None and not isinstance(audit, AttemptAudit):
            raise TypeError("attempt_audit must be an AttemptAudit or None")
        if audit is not None:
            if not assists:
                assists = dict(audit.assists)
            if not ram_writes:
                ram_writes = audit.ram_writes
            if not mid_run_loads:
                mid_run_loads = audit.mid_run_loads
            if start_identity_digest is None:
                start_identity_digest = audit.start_identity_digest
            if policy_identity_digest is None:
                policy_identity_digest = audit.policy_identity_digest
            if audit.start_identity_digest is not None:
                start_identity_explicit = True
            if audit.policy_identity_digest is not None:
                policy_identity_explicit = True

        contract = self.contract
        if contract is not None:
            if not isinstance(contract, EvaluationContract):
                raise TypeError("contract must be an EvaluationContract or None")
            if contract.runtime_observation_class is not observation:
                raise ValueError("seed result runtime observation class does not match contract")
            if contract.intervention_class is not intervention:
                raise ValueError("seed result intervention class does not match contract")
            if (
                start_identity_digest is not None
                and start_identity_digest != contract.start_identity_digest
            ):
                raise ValueError("seed result start identity does not match contract")
            if (
                policy_identity_digest is not None
                and policy_identity_digest != contract.policy_identity_digest
            ):
                raise ValueError("seed result policy identity does not match contract")
            if assist_contract_path is None:
                assist_contract_path = contract.assist_contract_path
            elif assist_contract_path != contract.assist_contract_path:
                raise ValueError("seed result assist contract path does not match contract")
            if assist_contract_digest is None:
                assist_contract_digest = contract.assist_contract_digest
            elif assist_contract_digest != contract.assist_contract_digest:
                raise ValueError("seed result assist contract digest does not match contract")
            mode = _normalize_assist_mode(assist_mode)
            if mode is None:
                mode = contract.assist_mode
            elif mode != contract.assist_mode:
                raise ValueError("seed result assist mode does not match contract")
            if audit is not None:
                if (
                    audit.start_identity_digest is not None
                    and audit.start_identity_digest != contract.start_identity_digest
                ):
                    raise ValueError("seed attempt audit start identity does not match contract")
                if (
                    audit.policy_identity_digest is not None
                    and audit.policy_identity_digest != contract.policy_identity_digest
                ):
                    raise ValueError("seed attempt audit policy identity does not match contract")
            start_identity_digest = contract.start_identity_digest
            policy_identity_digest = contract.policy_identity_digest
            assist_contract_path = contract.assist_contract_path
            assist_contract_digest = contract.assist_contract_digest
            assist_mode = mode
        else:
            mode = _normalize_assist_mode(assist_mode)
            if start_identity_digest is None:
                start_identity_digest = StartIdentity(f"seed:{self.seed}").identity_digest
            else:
                start_identity_digest = _validate_identity_digest(
                    start_identity_digest, "start_identity_digest"
                )
            if policy_identity_digest is None:
                policy_identity_digest = PolicyIdentity("unbound-policy").identity_digest
            else:
                policy_identity_digest = _validate_identity_digest(
                    policy_identity_digest, "policy_identity_digest"
                )
            if assist_contract_path is not None and (
                not isinstance(assist_contract_path, str) or not assist_contract_path.strip()
            ):
                raise ValueError("assist_contract_path must be a non-empty string or None")
            if assist_contract_digest is not None:
                assist_contract_digest = _validate_identity_digest(
                    assist_contract_digest, "assist_contract_digest"
                )
            if intervention.is_clean and mode is not None:
                raise ValueError("Clean seed results cannot carry assist_mode")
            # EvaluationContract owns assist path/digest pairing validation.
            contract = EvaluationContract(
                runtime_observation_class=observation,
                intervention_class=intervention,
                start_identity=StartIdentity(
                    f"seed-record:{self.seed}",
                    digest=start_identity_digest,
                ),
                policy_identity=PolicyIdentity(
                    "seed-record-policy",
                    digest=policy_identity_digest,
                ),
                assist_contract_path=assist_contract_path,
                assist_contract_digest=assist_contract_digest,
                assist_mode=mode,
            )
            assist_mode = contract.assist_mode

        if intervention.is_clean and (ram_writes or mid_run_loads or sum(assists.values())):
            raise ClaimValidationError("Clean seed attempt contains an intervention")

        object.__setattr__(self, "assists", assists)
        object.__setattr__(self, "ram_writes", ram_writes)
        object.__setattr__(self, "mid_run_loads", mid_run_loads)
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "_start_identity_explicit", start_identity_explicit)
        object.__setattr__(self, "_policy_identity_explicit", policy_identity_explicit)

    def __getattribute__(self, name: str) -> Any:
        if name in {
            "runtime_observation_class",
            "intervention_class",
            "start_identity_digest",
            "policy_identity_digest",
            "assist_contract_path",
            "assist_contract_digest",
            "assist_mode",
        }:
            contract = object.__getattribute__(self, "contract")
            if contract is None:
                raise AttributeError(
                    f"{type(self).__name__!r} object has no attribute {name!r}"
                )
            if name == "start_identity_digest":
                return contract.start_identity_digest
            if name == "policy_identity_digest":
                return contract.policy_identity_digest
            return getattr(contract, name)
        return object.__getattribute__(self, name)


    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SeedAttemptResult":
        """Rehydrate from a ledger/campaign/seed-report row.

        When ``contract`` is embedded, construct from that owned contract and
        derive InitVar identity fields from it (avoid flat-mirror reconstruction).
        """
        from collections.abc import Mapping as MappingABC

        from retro_harness.audit import AuditCapabilities
        from retro_harness.benchmark.claims import _contract_from_record

        if not isinstance(record, MappingABC):
            raise TypeError("seed result record must be a mapping")

        audit = None
        audit_record = record.get("attempt_audit")
        if isinstance(audit_record, MappingABC):
            caps = audit_record.get("audit_capabilities")
            capabilities = (
                AuditCapabilities.from_value(caps) if caps is not None else None
            )
            audit = AttemptAudit(
                ram_writes=audit_record.get("ram_writes"),
                mid_run_loads=audit_record.get("mid_run_loads"),
                assists=audit_record.get("assists"),
                start_identity_digest=audit_record.get("start_identity_digest"),
                policy_identity_digest=audit_record.get("policy_identity_digest"),
                runtime_observation_class=audit_record.get("runtime_observation_class"),
                intervention_class=audit_record.get("intervention_class"),
                capabilities=capabilities,
            )

        outcome = dict(
            seed=record["seed"],
            success=bool(record.get("success")),
            frames=int(record.get("frames", 0)),
            terminal_milestone=record.get("terminal_milestone"),
            failure_mode=record.get("failure_mode"),
            assists=record.get("assists") or {},
            ram_writes=(
                record.get("ram_writes") if record.get("ram_writes") is not None else 0
            ),
            mid_run_loads=(
                record.get("mid_run_loads") if record.get("mid_run_loads") is not None else 0
            ),
            attempt_audit=audit,
        )

        contract_record = record.get("contract")
        if isinstance(contract_record, MappingABC):
            contract = _contract_from_record(contract_record)
            return cls(
                **outcome,
                runtime_observation_class=contract.runtime_observation_class,
                intervention_class=contract.intervention_class,
                start_identity_digest=contract.start_identity_digest,
                policy_identity_digest=contract.policy_identity_digest,
                assist_contract_path=contract.assist_contract_path,
                assist_contract_digest=contract.assist_contract_digest,
                assist_mode=contract.assist_mode,
                contract=contract,
            )

        return cls(
            **outcome,
            runtime_observation_class=RuntimeObservationClass.from_value(
                record.get("runtime_observation_class", "Bronze")
            ),
            intervention_class=InterventionClass.from_value(
                record.get("intervention_class", "Clean")
            ),
            start_identity_digest=record.get("start_identity_digest"),
            policy_identity_digest=record.get("policy_identity_digest"),
            assist_contract_path=record.get("assist_contract_path"),
            assist_contract_digest=record.get("assist_contract_digest"),
            assist_mode=record.get("assist_mode"),
        )

    @classmethod
    def from_benchmark_attempt(
        cls,
        seed: SeedValue,
        attempt: Any,
        contract: EvaluationContract | None = None,
    ) -> "SeedAttemptResult":
        """Adapt an existing benchmark attempt into the seed report schema."""
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
            ram_writes=audit.ram_writes,
            mid_run_loads=audit.mid_run_loads,
            attempt_audit=audit,
            runtime_observation_class=resolved_contract.runtime_observation_class,
            intervention_class=resolved_contract.intervention_class,
            start_identity_digest=resolved_contract.start_identity.identity_digest,
            policy_identity_digest=resolved_contract.policy_identity.identity_digest,
            assist_contract_path=resolved_contract.assist_contract_path,
            assist_contract_digest=resolved_contract.assist_contract_digest,
            assist_mode=resolved_contract.assist_mode,
            contract=resolved_contract,
        )

    def bind_contract(self, contract: EvaluationContract) -> "SeedAttemptResult":
        """Attach a published contract to an extracted per-seed result."""
        if not isinstance(contract, EvaluationContract):
            raise TypeError("contract must be an EvaluationContract")
        existing = self.contract
        if existing is None:
            raise TypeError("seed result contract missing after initialization")
        if existing.runtime_observation_class is not contract.runtime_observation_class:
            raise ValueError("seed result runtime observation class does not match contract")
        if existing.intervention_class is not contract.intervention_class:
            raise ValueError("seed result intervention class does not match contract")
        if existing != contract:
            if (
                self._start_identity_explicit
                and existing.start_identity_digest != contract.start_identity_digest
            ):
                raise ValueError("seed result start identity does not match contract")
            if (
                self._policy_identity_explicit
                and existing.policy_identity_digest != contract.policy_identity_digest
            ):
                raise ValueError("seed result policy identity does not match contract")
            # Synthesized placeholders (no published benchmark_id) may be upgraded.
            if existing.benchmark_id and existing != contract:
                raise ValueError("seed result already carries a different evaluation contract")
        audit = _audit_with_contract_identity(
            self.attempt_audit
            or AttemptAudit(
                ram_writes=self.ram_writes,
                mid_run_loads=self.mid_run_loads,
                assists=self.assists,
            ),
            contract,
        )
        # Avoid dataclasses.replace: InitVar names are re-read via __getattribute__
        # and would re-inject digests from the previous synthesized contract.
        result = SeedAttemptResult(
            seed=self.seed,
            success=self.success,
            frames=self.frames,
            terminal_milestone=self.terminal_milestone,
            failure_mode=self.failure_mode,
            assists=self.assists,
            ram_writes=self.ram_writes,
            mid_run_loads=self.mid_run_loads,
            attempt_audit=audit,
            runtime_observation_class=contract.runtime_observation_class,
            intervention_class=contract.intervention_class,
            start_identity_digest=contract.start_identity_digest,
            policy_identity_digest=contract.policy_identity_digest,
            assist_contract_path=contract.assist_contract_path,
            assist_contract_digest=contract.assist_contract_digest,
            assist_mode=contract.assist_mode,
            contract=contract,
        )
        validate_claim(contract, audit)
        return result

    def to_record(self) -> dict[str, Any]:
        contract = self.contract
        assert contract is not None
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
        config_contract = self.config.to_contract()
        for expected_seed, result in zip(self.config.seeds, results, strict=True):
            if not isinstance(result, SeedAttemptResult):
                raise TypeError("seed_results must contain SeedAttemptResult values")
            if result.seed != expected_seed:
                raise ValueError("seed_results must follow the published seed order")
            assert result.contract is not None
            if result.runtime_observation_class is not config_contract.runtime_observation_class:
                raise ValueError("seed result runtime observation class does not match config")
            if result.intervention_class is not config_contract.intervention_class:
                raise ValueError("seed result intervention class does not match config")
            if result.assist_contract_path != config_contract.assist_contract_path:
                raise ValueError("seed result assist contract path does not match config")
            if result.assist_contract_digest != config_contract.assist_contract_digest:
                raise ValueError("seed result assist contract digest does not match config")
            if result.assist_mode != config_contract.assist_mode:
                raise ValueError("seed result assist mode does not match config")
            _validate_seed_result_budget(self.config, result)
            if (
                self.config._start_identity_explicit
                and result.start_identity_digest != config_contract.start_identity.identity_digest
            ):
                raise ValueError("seed result start identity does not match config")
            if (
                self.config._policy_identity_explicit
                and result.policy_identity_digest != config_contract.policy_identity.identity_digest
            ):
                raise ValueError("seed result policy identity does not match config")
            audit = _audit_with_contract_identity(
                result.attempt_audit
                or AttemptAudit(
                    ram_writes=result.ram_writes,
                    mid_run_loads=result.mid_run_loads,
                    assists=result.assists,
                ),
                result.contract,
            )
            validate_claim(result.contract, audit)
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


def _validate_seed_result_budget(
    config: SeedRobustnessConfig,
    result: SeedAttemptResult,
) -> None:
    if result.frames > config.budget:
        raise ValueError(
            f"frames for seed {result.seed!r} exceed the published frame budget"
        )


def write_seed_robustness_report(
    path: str | Path,
    report: SeedRobustnessReport,
) -> Path:
    """Write a canonical JSON artifact via atomic replace and return its path."""
    if not isinstance(report, SeedRobustnessReport):
        raise TypeError("report must be a SeedRobustnessReport")
    from retro_harness.seed_campaign import atomic_write_json

    return atomic_write_json(path, report.to_record())


__all__ = [
    "SEED_ROBUSTNESS_SCHEMA_VERSION",
    "SeedAttemptResult",
    "SeedRobustnessConfig",
    "SeedRobustnessReport",
    "SeedValue",
    "write_seed_robustness_report",
    "_audit_with_contract_identity",
    "_to_jsonable",
    "_validate_seed_result_budget",
    "_validate_seed_value",
]
