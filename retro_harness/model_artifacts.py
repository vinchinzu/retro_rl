"""Learned-policy identity manifests and ContractBundle-bound I/O."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from retro_harness.contracts import ContractBundle
from retro_harness.identity import (
    IdentityError,
    canonical_json,
    jsonable,
    sha256_bytes,
    sha256_file,
)
from retro_harness.repo import monorepo_root


class PolicyArtifactError(ValueError):
    """Raised when a learned-policy manifest or its bound files disagree."""


def _artifact_mapping(value: Mapping[str, Any], path: str) -> dict[str, Any]:
    try:
        normalized = jsonable(dict(value), path)
    except IdentityError as exc:
        raise PolicyArtifactError(str(exc)) from exc
    if not isinstance(normalized, dict):  # pragma: no cover - dict input invariant
        raise PolicyArtifactError(f"{path} must be a mapping")
    return normalized


def _artifact_file_digest(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except IdentityError as exc:
        raise PolicyArtifactError(str(exc)) from exc


@dataclass(frozen=True)
class PolicyArtifact:
    """Immutable identity manifest for learned policy weights and contracts."""

    checkpoint_sha256: str
    algorithm: str
    hyperparameters: Mapping[str, Any]
    training_seed: str | int
    observation_schema_digest: str
    action_schema_digest: str
    reward_schema_digest: str
    wrapper_schema_digest: str
    rom_identity_digest: str
    state_identity_digest: str
    core_identity_digest: str
    dependency_lock_sha256: str
    source_commit: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PolicyArtifactError("unsupported PolicyArtifact schema_version")
        for field_name in (
            "checkpoint_sha256",
            "algorithm",
            "observation_schema_digest",
            "action_schema_digest",
            "reward_schema_digest",
            "wrapper_schema_digest",
            "rom_identity_digest",
            "state_identity_digest",
            "core_identity_digest",
            "dependency_lock_sha256",
            "source_commit",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise PolicyArtifactError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        if isinstance(self.training_seed, bool) or not isinstance(
            self.training_seed, (str, int)
        ):
            raise PolicyArtifactError("training_seed must be a string or integer")
        object.__setattr__(
            self,
            "hyperparameters",
            _artifact_mapping(self.hyperparameters, "hyperparameters"),
        )
        object.__setattr__(
            self,
            "metadata",
            _artifact_mapping(self.metadata, "metadata"),
        )

    def _identity_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "algorithm": self.algorithm,
            "hyperparameters": dict(self.hyperparameters),
            "training_seed": self.training_seed,
            "observation_schema_digest": self.observation_schema_digest,
            "action_schema_digest": self.action_schema_digest,
            "reward_schema_digest": self.reward_schema_digest,
            "wrapper_schema_digest": self.wrapper_schema_digest,
            "rom_identity_digest": self.rom_identity_digest,
            "state_identity_digest": self.state_identity_digest,
            "core_identity_digest": self.core_identity_digest,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "source_commit": self.source_commit,
            "metadata": dict(self.metadata),
        }

    @property
    def identity_digest(self) -> str:
        return sha256_bytes(canonical_json(self._identity_record()).encode("utf-8"))

    def to_record(self) -> dict[str, Any]:
        return {**self._identity_record(), "identity_digest": self.identity_digest}

    def to_policy_identity(self, name: str):
        # Imported lazily so benchmark claims may consume PolicyArtifact without
        # making model artifact ownership point back into benchmark.py.
        from retro_harness.benchmark import PolicyIdentity

        return PolicyIdentity(
            name=name,
            digest=self.identity_digest,
            version=f"policy-artifact-v{self.schema_version}",
            source=f"checkpoint:sha256:{self.checkpoint_sha256}",
            metadata={
                "policy_kind": "learned",
                "policy_artifact_digest": self.identity_digest,
                "checkpoint_sha256": self.checkpoint_sha256,
                "observation_schema_digest": self.observation_schema_digest,
                "action_schema_digest": self.action_schema_digest,
                "reward_schema_digest": self.reward_schema_digest,
                "wrapper_schema_digest": self.wrapper_schema_digest,
            },
        )

    def verify_checkpoint(self, checkpoint_path: str | Path) -> None:
        if _artifact_file_digest(checkpoint_path) != self.checkpoint_sha256:
            raise PolicyArtifactError(
                "policy checkpoint digest does not match PolicyArtifact"
            )

    def verify_schema_digests(self, **expected: str) -> None:
        known = {
            "observation": self.observation_schema_digest,
            "action": self.action_schema_digest,
            "reward": self.reward_schema_digest,
            "wrapper": self.wrapper_schema_digest,
        }
        unknown = set(expected) - set(known)
        if unknown:
            raise TypeError(f"unknown schema digest names: {sorted(unknown)}")
        for name, digest in expected.items():
            if digest != known[name]:
                raise PolicyArtifactError(
                    f"{name} schema digest does not match PolicyArtifact"
                )

    def verify_environment_identity_digests(self, **expected: str) -> None:
        known = {
            "rom": self.rom_identity_digest,
            "state": self.state_identity_digest,
            "core": self.core_identity_digest,
        }
        unknown = set(expected) - set(known)
        if unknown:
            raise TypeError(f"unknown environment identity names: {sorted(unknown)}")
        for name, digest in expected.items():
            if digest != known[name]:
                raise PolicyArtifactError(
                    f"{name} identity digest does not match PolicyArtifact"
                )

    def write(self, path: str | Path) -> Path:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(self.to_record(), allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return manifest_path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        dependency_lock_path: str | Path,
        **values: Any,
    ) -> "PolicyArtifact":
        return cls(
            checkpoint_sha256=_artifact_file_digest(checkpoint_path),
            dependency_lock_sha256=_artifact_file_digest(dependency_lock_path),
            **values,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PolicyArtifact":
        if not isinstance(record, Mapping):
            raise TypeError("PolicyArtifact record must be a mapping")
        values = dict(record)
        published_digest = values.pop("identity_digest", None)
        try:
            artifact = cls(**values)
        except TypeError as exc:
            raise PolicyArtifactError("invalid PolicyArtifact record fields") from exc
        if published_digest != artifact.identity_digest:
            raise PolicyArtifactError("PolicyArtifact identity digest mismatch")
        return artifact

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        checkpoint_path: str | Path,
        expected_schema_digests: Mapping[str, str] | None = None,
        expected_environment_identity_digests: Mapping[str, str] | None = None,
    ) -> "PolicyArtifact":
        artifact = cls.from_record(json.loads(Path(path).read_text(encoding="utf-8")))
        artifact.verify_checkpoint(checkpoint_path)
        if expected_schema_digests:
            artifact.verify_schema_digests(**dict(expected_schema_digests))
        if expected_environment_identity_digests:
            artifact.verify_environment_identity_digests(
                **dict(expected_environment_identity_digests)
            )
        return artifact


def policy_artifact_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_suffix(checkpoint.suffix + ".artifact.json")


def current_source_commit(root: str | Path | None = None) -> str:
    cwd = Path(root) if root is not None else monorepo_root()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-source-commit"


def write_policy_artifact(
    checkpoint_path: str | Path,
    contracts: ContractBundle,
    *,
    algorithm: str,
    hyperparameters: Mapping[str, Any],
    training_seed: str | int,
    dependency_lock_path: str | Path | None = None,
    source_commit: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    manifest_path: str | Path | None = None,
) -> PolicyArtifact:
    if not isinstance(contracts, ContractBundle):
        raise TypeError("contracts must be a ContractBundle")
    lock = Path(dependency_lock_path) if dependency_lock_path else monorepo_root() / "uv.lock"
    artifact = PolicyArtifact.from_checkpoint(
        checkpoint_path,
        dependency_lock_path=lock,
        algorithm=algorithm,
        hyperparameters=dict(hyperparameters),
        training_seed=training_seed,
        observation_schema_digest=contracts.observation.identity_digest,
        action_schema_digest=contracts.action.identity_digest,
        reward_schema_digest=contracts.reward.identity_digest,
        wrapper_schema_digest=contracts.wrappers.identity_digest,
        rom_identity_digest=contracts.environment.rom_identity_digest,
        state_identity_digest=contracts.environment.state_identity_digest,
        core_identity_digest=contracts.environment.core_identity_digest,
        source_commit=source_commit or current_source_commit(),
        metadata={
            **dict(metadata or {}),
            "contract_bundle_digest": contracts.identity_digest,
            "environment_contract_digest": contracts.environment.identity_digest,
        },
    )
    artifact.write(manifest_path or policy_artifact_path(checkpoint_path))
    return artifact


def load_policy_artifact(
    checkpoint_path: str | Path,
    contracts: ContractBundle,
    *,
    manifest_path: str | Path | None = None,
) -> PolicyArtifact:
    """Load a checkpoint identity and fail closed on every contract mismatch."""
    artifact = PolicyArtifact.load(
        manifest_path or policy_artifact_path(checkpoint_path),
        checkpoint_path=checkpoint_path,
        expected_schema_digests=contracts.schema_digests,
        expected_environment_identity_digests=(
            contracts.environment_identity_digests
        ),
    )
    if artifact.metadata.get("contract_bundle_digest") != contracts.identity_digest:
        raise PolicyArtifactError("ContractBundle digest does not match PolicyArtifact")
    if (
        artifact.metadata.get("environment_contract_digest")
        != contracts.environment.identity_digest
    ):
        raise PolicyArtifactError(
            "environment contract digest does not match PolicyArtifact"
        )
    return artifact


__all__ = [
    "PolicyArtifact",
    "PolicyArtifactError",
    "current_source_commit",
    "load_policy_artifact",
    "policy_artifact_path",
    "write_policy_artifact",
]
