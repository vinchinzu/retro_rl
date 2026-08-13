"""Typed evaluation claims and fail-closed serialized-record validation.

Canonical home under :mod:`retro_harness.benchmark`. Compatibility shims
remain at :mod:`retro_harness.benchmark_claims` and
:mod:`retro_harness.benchmark`.

Maturity: first real-game consumer for fail-closed audits and
``PolicyArtifact`` identity. Resumable seed-campaign publication evidence
remains rr-gbd.33, and publication-ready still requires a second independent
consumer.
"""

from __future__ import annotations

import hashlib
import inspect
import marshal
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from retro_harness.audit import (
    AttemptAudit,
    InterventionClass,
    RuntimeObservationClass,
)
from retro_harness.identity import digest_record as _identity_digest
from retro_harness.model_artifacts import PolicyArtifact, PolicyArtifactError

from retro_harness.benchmark.claims_records import (
    _validate_seed_report_record,
)


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
    if not audit.has_complete_instrumentation:
        errors.append(
            "attempt lacks complete intervention audit instrumentation "
            "(RAM writes, mid-run loads, and assists)"
        )
    if audit.start_identity_digest != contract.start_identity.identity_digest:
        errors.append("start identity digest does not match the contract")
    if audit.policy_identity_digest != contract.policy_identity.identity_digest:
        errors.append("policy identity digest does not match the contract")
    if contract.policy_identity.metadata.get("policy_kind") == "learned":
        artifact_digest = contract.policy_identity.metadata.get(
            "policy_artifact_digest"
        )
        if (
            not isinstance(artifact_digest, str)
            or artifact_digest != contract.policy_identity.identity_digest
        ):
            errors.append("learned policy lacks a matching PolicyArtifact identity")
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

def _normalize_assist_mode(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("assist_mode must be a non-empty string or None")
    return value.strip()


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


def _audit_from_record(value: Any) -> AttemptAudit:
    if isinstance(value, AttemptAudit):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("record audit must be an AttemptAudit or mapping")
    nested = value.get("attempt_audit")
    if isinstance(nested, Mapping):
        value = nested
    return AttemptAudit(
        ram_writes=value.get("ram_writes"),
        mid_run_loads=value.get("mid_run_loads"),
        assists=value.get("assists"),
        start_identity_digest=value.get("start_identity_digest"),
        policy_identity_digest=value.get("policy_identity_digest"),
        runtime_observation_class=value.get("runtime_observation_class"),
        intervention_class=value.get("intervention_class"),
        capabilities=value.get("audit_capabilities"),
    )


def _looks_like_learned_policy(policy: Any) -> bool:
    """Conservatively identify common learned-model interfaces."""
    target = policy if inspect.isclass(policy) else type(policy)
    module = str(getattr(target, "__module__", ""))
    if module.startswith(("torch", "stable_baselines3")):
        return True
    return callable(getattr(policy, "predict", None)) or callable(
        getattr(policy, "state_dict", None)
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
    policy: Any,
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
    policy: Any,
) -> PolicyIdentity:
    """Derive a verifiable identity for the policy implementation.

    The mutable ``policy.name`` remains a display label only.  The identity
    digest covers the implementation's module and qualified name plus source,
    bytecode, or the documented stable module-qualified-name fallback.
    """
    artifact = getattr(policy, "policy_artifact", None)
    if artifact is not None:
        if not isinstance(artifact, PolicyArtifact):
            raise PolicyArtifactError("policy_artifact must be a PolicyArtifact")
        checkpoint_path = getattr(policy, "checkpoint_path", None)
        if checkpoint_path is not None:
            artifact.verify_checkpoint(checkpoint_path)
        return artifact.to_policy_identity(_policy_name(policy))
    if _looks_like_learned_policy(policy):
        raise PolicyArtifactError(
            "learned policies require a weight-bound PolicyArtifact"
        )

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


def _policy_name(policy: Any) -> str:
    value = getattr(policy, "name", getattr(policy, "__name__", policy.__class__.__name__))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return str(value)


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


__all__ = [
    "ClaimValidationError",
    "EvaluationContract",
    "PolicyIdentity",
    "StartIdentity",
    "policy_identity_for",
    "validate_claim",
]
