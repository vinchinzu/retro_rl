"""Deterministic branch-rollout batches over certified snapshots (rr-gbd.34).

Given a certified :class:`~retro_harness.snapshot.SnapshotEnvelope` root and a
list of branch controllers, run each branch from a fresh restore of that root.

Design rules:

* Results are **lane-assignment independent**: width 1 (sequential) and width N
  (batched across a pool) produce identical per-branch outcomes and the same
  replay digest.
* A controller exception **isolates** that branch; siblings still complete.
* Accounting is exact: status counts sum to ``branch_count`` and
  ``total_steps`` equals the sum of per-branch ``steps_executed``.
* Roots must be :attr:`~retro_harness.snapshot.SnapshotCertification.CERTIFIED_FULL_ENV`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from retro_harness.emulator_pool import EmulatorPool, EnvFactory
from retro_harness.identity import (
    IdentityError,
    digest_record,
    jsonable,
    require_nonempty,
    sha256_bytes,
)
from retro_harness.runtime import step_env
from retro_harness.snapshot import (
    SnapshotAdapter,
    SnapshotCertification,
    SnapshotEnvelope,
    SnapshotError,
    restore_envelope,
)


ROLLOUT_SCHEMA_VERSION = 1
BranchController = Callable[[Any, int], Any]


class BranchStatus(str, Enum):
    """Terminal status of one branch rollout."""

    OK = "ok"
    CONTROLLER_ERROR = "controller_error"

    @classmethod
    def from_value(cls, value: Any) -> "BranchStatus":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, str):
            raise TypeError("branch status must be a string")
        normalized = value.strip().casefold().replace("-", "_")
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"invalid branch status {value!r}; expected ok or controller_error"
        )


class RolloutError(ValueError):
    """Base error for branch-rollout batch failures."""


@dataclass(frozen=True, slots=True)
class BranchSpec:
    """One hypothetical branch from a certified root envelope.

    ``controller(env, step_index)`` returns the action for that step.  Raising
    any exception marks the branch :attr:`BranchStatus.CONTROLLER_ERROR` and
    does not abort the batch.
    """

    branch_id: str
    max_steps: int
    controller: BranchController

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "branch_id", require_nonempty(self.branch_id, "branch_id")
        )
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 0
        ):
            raise RolloutError("max_steps must be a non-negative int")
        if not callable(self.controller):
            raise RolloutError("controller must be callable")


def branch_from_actions(
    branch_id: str,
    actions: Sequence[Any],
) -> BranchSpec:
    """Build a :class:`BranchSpec` that plays a fixed action sequence."""

    acts = tuple(actions)

    def _controller(env: Any, step_index: int) -> Any:
        del env
        return acts[step_index]

    return BranchSpec(
        branch_id=branch_id,
        max_steps=len(acts),
        controller=_controller,
    )


@dataclass(frozen=True, slots=True)
class RolloutSpec:
    """Batch of branches sharing one certified root snapshot."""

    root: SnapshotEnvelope
    branches: tuple[BranchSpec, ...]
    schema_version: int = ROLLOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.root, SnapshotEnvelope):
            raise RolloutError("root must be a SnapshotEnvelope")
        if self.root.certification is not SnapshotCertification.CERTIFIED_FULL_ENV:
            raise RolloutError(
                "branch rollouts require a CERTIFIED_FULL_ENV root envelope; "
                f"got {self.root.certification.value}"
            )
        if not isinstance(self.branches, tuple):
            object.__setattr__(self, "branches", tuple(self.branches))
        if not self.branches:
            raise RolloutError("RolloutSpec requires at least one branch")
        if any(not isinstance(item, BranchSpec) for item in self.branches):
            raise RolloutError("branches must be BranchSpec instances")
        ids = [item.branch_id for item in self.branches]
        if len(ids) != len(set(ids)):
            raise RolloutError("branch_id values must be unique within a RolloutSpec")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise RolloutError("schema_version must be a positive int")

    def __len__(self) -> int:
        return len(self.branches)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "root_identity": self.root.identity.to_record(),
            "root_certification": self.root.certification.value,
            "branch_ids": [branch.branch_id for branch in self.branches],
            "max_steps": [branch.max_steps for branch in self.branches],
        }


@dataclass(frozen=True, slots=True)
class BranchResult:
    """Outcome of one branch, independent of which lane executed it."""

    branch_id: str
    status: BranchStatus
    steps_executed: int
    trajectory_digest: str
    terminated: bool = False
    truncated: bool = False
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "branch_id", require_nonempty(self.branch_id, "branch_id")
        )
        object.__setattr__(self, "status", BranchStatus.from_value(self.status))
        if (
            isinstance(self.steps_executed, bool)
            or not isinstance(self.steps_executed, int)
            or self.steps_executed < 0
        ):
            raise RolloutError("steps_executed must be a non-negative int")
        object.__setattr__(
            self,
            "trajectory_digest",
            require_nonempty(self.trajectory_digest, "trajectory_digest"),
        )
        if self.status is BranchStatus.CONTROLLER_ERROR:
            if not self.error_type:
                raise RolloutError("controller_error results require error_type")
        elif self.error_type is not None or self.error_message is not None:
            raise RolloutError("ok results must not carry error fields")

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "branch_id": self.branch_id,
            "status": self.status.value,
            "steps_executed": self.steps_executed,
            "trajectory_digest": self.trajectory_digest,
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
        }
        if self.error_type is not None:
            record["error_type"] = self.error_type
        if self.error_message is not None:
            record["error_message"] = self.error_message
        return record


@dataclass(frozen=True, slots=True)
class RolloutAccounting:
    """Exact batch counters (must reconcile with per-branch results)."""

    branch_count: int
    ok_count: int
    controller_error_count: int
    terminated_count: int
    truncated_count: int
    total_steps: int

    def __post_init__(self) -> None:
        for name in (
            "branch_count",
            "ok_count",
            "controller_error_count",
            "terminated_count",
            "truncated_count",
            "total_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RolloutError(f"{name} must be a non-negative int")
        if self.ok_count + self.controller_error_count != self.branch_count:
            raise RolloutError(
                "accounting status counts must sum to branch_count: "
                f"ok={self.ok_count} errors={self.controller_error_count} "
                f"branches={self.branch_count}"
            )
        if self.terminated_count > self.ok_count or self.truncated_count > self.ok_count:
            raise RolloutError(
                "terminated/truncated counts cannot exceed ok_count"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "branch_count": self.branch_count,
            "ok_count": self.ok_count,
            "controller_error_count": self.controller_error_count,
            "terminated_count": self.terminated_count,
            "truncated_count": self.truncated_count,
            "total_steps": self.total_steps,
        }


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Ordered branch outcomes plus accounting and a stable replay digest.

    ``branches`` preserves :class:`RolloutSpec` order.  ``replay_digest`` is
    computed from records sorted by ``branch_id`` so lane assignment and
    batching order cannot change the digest.
    """

    branches: tuple[BranchResult, ...]
    accounting: RolloutAccounting
    replay_digest: str
    width: int
    schema_version: int = ROLLOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.branches, tuple):
            object.__setattr__(self, "branches", tuple(self.branches))
        if not self.branches:
            raise RolloutError("RolloutResult requires at least one branch")
        if any(not isinstance(item, BranchResult) for item in self.branches):
            raise RolloutError("branches must be BranchResult instances")
        if not isinstance(self.accounting, RolloutAccounting):
            raise RolloutError("accounting must be a RolloutAccounting")
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or self.width < 1
        ):
            raise RolloutError("width must be a positive int")
        object.__setattr__(
            self, "replay_digest", require_nonempty(self.replay_digest, "replay_digest")
        )
        if self.accounting.branch_count != len(self.branches):
            raise RolloutError("accounting.branch_count must match len(branches)")
        step_sum = sum(item.steps_executed for item in self.branches)
        if self.accounting.total_steps != step_sum:
            raise RolloutError(
                "accounting.total_steps must equal sum of steps_executed: "
                f"{self.accounting.total_steps} != {step_sum}"
            )
        ok = sum(1 for item in self.branches if item.status is BranchStatus.OK)
        err = sum(
            1 for item in self.branches if item.status is BranchStatus.CONTROLLER_ERROR
        )
        if self.accounting.ok_count != ok or self.accounting.controller_error_count != err:
            raise RolloutError("accounting status counts disagree with branch results")
        term = sum(1 for item in self.branches if item.status is BranchStatus.OK and item.terminated)
        trunc = sum(1 for item in self.branches if item.status is BranchStatus.OK and item.truncated)
        if (
            self.accounting.terminated_count != term
            or self.accounting.truncated_count != trunc
        ):
            raise RolloutError("accounting terminal flags disagree with branch results")

    def result_for(self, branch_id: str) -> BranchResult:
        for item in self.branches:
            if item.branch_id == branch_id:
                return item
        raise KeyError(branch_id)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "width": self.width,
            "replay_digest": self.replay_digest,
            "accounting": self.accounting.to_record(),
            "branches": [item.to_record() for item in self.branches],
        }


def _obs_token(obs: Any) -> Any:
    """Reduce an observation to a stable identity-friendly token."""

    if obs is None or isinstance(obs, (bool, int, str)):
        return obs
    if isinstance(obs, float):
        if obs != obs or obs in (float("inf"), float("-inf")):
            return {"float": "nonfinite", "repr": repr(obs)}
        return obs
    # numpy / array-like
    tobytes = getattr(obs, "tobytes", None)
    shape = getattr(obs, "shape", None)
    if callable(tobytes) and shape is not None:
        try:
            payload = tobytes()
        except Exception:
            payload = None
        if isinstance(payload, (bytes, bytearray)):
            return {
                "shape": [int(x) for x in shape],
                "dtype": str(getattr(obs, "dtype", "unknown")),
                "sha256": sha256_bytes(bytes(payload)),
            }
    try:
        return jsonable(obs)
    except IdentityError:
        return {"repr": repr(obs)[:240]}


def _info_token(info: dict[str, Any]) -> dict[str, Any]:
    """Keep only identity-friendly info fields (drop bulky frames)."""

    out: dict[str, Any] = {}
    for key, value in sorted(info.items()):
        if not isinstance(key, str):
            continue
        if key in {"frame", "rgb", "screenshot", "observation", "obs"}:
            continue
        try:
            out[key] = jsonable(value)
        except IdentityError:
            out[key] = {"repr": repr(value)[:120]}
    return out


def _step_record(
    *,
    step_index: int,
    obs: Any,
    reward: Any,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    reward_token: Any
    try:
        reward_token = jsonable(reward)
    except IdentityError:
        reward_token = {"repr": repr(reward)[:120]}
    return {
        "step_index": step_index,
        "obs": _obs_token(obs),
        "reward": reward_token,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": _info_token(info),
    }


def trajectory_digest_for(step_records: Sequence[dict[str, Any]]) -> str:
    """Stable digest of a branch trajectory (empty steps allowed)."""

    return digest_record(
        "branch-trajectory-v1",
        {"steps": list(step_records)},
    )


def replay_digest_for(results: Sequence[BranchResult]) -> str:
    """Digest of branch results sorted by branch_id (order-independent)."""

    ordered = sorted((item.to_record() for item in results), key=lambda r: r["branch_id"])
    return digest_record(
        "branch-rollout-replay-v1",
        {
            "schema_version": ROLLOUT_SCHEMA_VERSION,
            "branches": ordered,
        },
    )


def accounting_for(results: Sequence[BranchResult]) -> RolloutAccounting:
    """Build exact accounting from per-branch results."""

    items = list(results)
    ok = [item for item in items if item.status is BranchStatus.OK]
    return RolloutAccounting(
        branch_count=len(items),
        ok_count=len(ok),
        controller_error_count=sum(
            1 for item in items if item.status is BranchStatus.CONTROLLER_ERROR
        ),
        terminated_count=sum(1 for item in ok if item.terminated),
        truncated_count=sum(1 for item in ok if item.truncated),
        total_steps=sum(item.steps_executed for item in items),
    )


def _execute_branch(
    env: Any,
    adapter: SnapshotAdapter,
    root: SnapshotEnvelope,
    branch: BranchSpec,
) -> BranchResult:
    """Restore ``root`` onto ``env`` and run one branch to completion."""

    restore_envelope(env, root, adapter)
    step_records: list[dict[str, Any]] = []
    terminated = False
    truncated = False

    for step_index in range(branch.max_steps):
        try:
            action = branch.controller(env, step_index)
        except Exception as exc:  # isolate invalid branch
            return BranchResult(
                branch_id=branch.branch_id,
                status=BranchStatus.CONTROLLER_ERROR,
                steps_executed=step_index,
                trajectory_digest=trajectory_digest_for(step_records),
                terminated=False,
                truncated=False,
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
            )
        try:
            obs, reward, terminated, truncated, info = step_env(env, action)
        except Exception as exc:
            return BranchResult(
                branch_id=branch.branch_id,
                status=BranchStatus.CONTROLLER_ERROR,
                steps_executed=step_index,
                trajectory_digest=trajectory_digest_for(step_records),
                terminated=False,
                truncated=False,
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
            )
        step_records.append(
            _step_record(
                step_index=step_index,
                obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info if isinstance(info, dict) else {},
            )
        )
        if terminated or truncated:
            return BranchResult(
                branch_id=branch.branch_id,
                status=BranchStatus.OK,
                steps_executed=step_index + 1,
                trajectory_digest=trajectory_digest_for(step_records),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

    return BranchResult(
        branch_id=branch.branch_id,
        status=BranchStatus.OK,
        steps_executed=branch.max_steps,
        trajectory_digest=trajectory_digest_for(step_records),
        terminated=False,
        truncated=False,
    )


def _validate_width(width: Any) -> int:
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise RolloutError("width must be a positive int")
    return width


def run_branch_rollouts(
    env_factory: EnvFactory,
    adapter: SnapshotAdapter,
    spec: RolloutSpec,
    *,
    width: int = 1,
) -> RolloutResult:
    """Run ``spec`` branches from its certified root using a pool of ``width``.

    Creates a temporary :class:`~retro_harness.emulator_pool.EmulatorPool` of
    size ``width``, restores the root for each branch, and executes controllers
    with exception isolation.  Width only affects parallelism; outcomes must
    match for any positive width.
    """

    if not isinstance(spec, RolloutSpec):
        raise TypeError("spec must be a RolloutSpec")
    width = _validate_width(width)
    # Cap workers at branch count; empty already rejected by RolloutSpec.
    workers = min(width, len(spec.branches))

    pool = EmulatorPool(
        env_factory,
        num_envs=workers,
        snapshot_adapter=adapter,
    )
    try:
        # Touch env identity path once so adapter digests are warm/valid.
        pool.reset()
        return run_branch_rollouts_on_pool(pool, spec)
    finally:
        pool.close()


def run_branch_rollouts_on_pool(
    pool: EmulatorPool,
    spec: RolloutSpec,
) -> RolloutResult:
    """Run ``spec`` using lanes already owned by ``pool`` (width = pool size).

    Branches are processed in deterministic input order, batched by pool width.
    Within a batch, lanes run concurrently via a private executor so a slow or
    failing controller cannot reorder sibling completion into the result list
    (results are joined in batch order).
    """

    if not isinstance(pool, EmulatorPool):
        raise TypeError("pool must be an EmulatorPool")
    if not isinstance(spec, RolloutSpec):
        raise TypeError("spec must be a RolloutSpec")
    if pool.num_envs < 1:
        raise RolloutError("pool must have at least one lane")

    adapter = pool.snapshot_adapter
    root = spec.root
    # Fail closed on identity before any branch work.
    restore_envelope(pool.envs[0], root, adapter)

    results: list[BranchResult] = []
    width = pool.num_envs
    branches = spec.branches

    for start in range(0, len(branches), width):
        batch = branches[start : start + width]
        lanes = pool.envs[: len(batch)]
        if len(batch) == 1:
            results.append(_execute_branch(lanes[0], adapter, root, batch[0]))
            continue
        with ThreadPoolExecutor(
            max_workers=len(batch),
            thread_name_prefix="retro-rollout",
        ) as executor:
            # map preserves batch order regardless of finish order.
            batch_results = list(
                executor.map(
                    lambda pair: _execute_branch(pair[0], adapter, root, pair[1]),
                    list(zip(lanes, batch)),
                )
            )
        results.extend(batch_results)

    accounting = accounting_for(results)
    digest = replay_digest_for(results)
    return RolloutResult(
        branches=tuple(results),
        accounting=accounting,
        replay_digest=digest,
        width=width,
    )


__all__ = [
    "BranchController",
    "BranchResult",
    "BranchSpec",
    "BranchStatus",
    "ROLLOUT_SCHEMA_VERSION",
    "RolloutAccounting",
    "RolloutError",
    "RolloutResult",
    "RolloutSpec",
    "accounting_for",
    "branch_from_actions",
    "replay_digest_for",
    "run_branch_rollouts",
    "run_branch_rollouts_on_pool",
    "trajectory_digest_for",
]
