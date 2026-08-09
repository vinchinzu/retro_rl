"""PolicyArtifact helpers that bind checkpoints to ContractBundle values."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from retro_harness.benchmark import PolicyArtifact, PolicyArtifactError
from retro_harness.contracts import ContractBundle
from retro_harness.repo import monorepo_root


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
    "current_source_commit",
    "load_policy_artifact",
    "policy_artifact_path",
    "write_policy_artifact",
]
