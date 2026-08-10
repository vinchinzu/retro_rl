"""Deterministic parallel pool for emulator-backed environments.

Maturity: **fake-tested + real-ROM smoke for certified snapshots** (rr-gbd.32).
Branch-rollout batches remain rr-gbd.34.  Not publication-ready L0 until
certified snapshots have a second independent game consumer and branch
rollouts land.

Two snapshot surfaces:

* :meth:`EmulatorPool.save` / :meth:`load` / :meth:`fork` — raw emulator
  ``get_state``/``set_state`` only via :class:`PoolState`.  **Uncertified**
  (EMULATOR_ONLY).  Does not restore wrapper counters, caches, or RNGs.
* :meth:`EmulatorPool.save_snapshot` / :meth:`load_snapshot` /
  :meth:`fork_snapshot` — :class:`~retro_harness.snapshot.SnapshotEnvelope`
  pools with adapter/schema/core/game identity.  With a full-env
  :class:`~retro_harness.snapshot.SnapshotAdapter`, envelopes are
  CERTIFIED_FULL_ENV and restore wrapper state.  Identity mismatches fail
  **before** mutation.

Stable-retro exposes emulator state through ``env.em``; test doubles may
expose it on the environment.  Each lane owns one environment and is stepped
in fixed caller order; work runs on one long-lived thread per lane.  The pool
never samples actions, auto-resets lanes, or otherwise introduces randomness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

from retro_harness.runtime import reset_env, step_env
from retro_harness.snapshot import (
    EmulatorOnlyAdapter,
    PoolSnapshot,
    SnapshotAdapter,
    SnapshotEnvelope,
    assert_envelope_compatible,
    capture_envelope,
    get_emulator_state,
    restore_envelope,
    set_emulator_state,
)


EnvFactory: TypeAlias = Callable[[], Any]
ResetResult: TypeAlias = tuple[Any, dict[str, Any]]
StepResult: TypeAlias = tuple[Any, Any, bool, bool, dict[str, Any]]


@dataclass(frozen=True)
class PoolState:
    """A per-lane raw emulator snapshot captured by :meth:`EmulatorPool.save`.

    State payloads are copied when a snapshot is created and when they are
    loaded.  This prevents mutable fake states (and any future non-byte state
    backend) from being changed through a live environment after saving.

    This payload is **uncertified** (emulator-only).  Prefer
    :meth:`EmulatorPool.save_snapshot` when wrapper/RNG/cache restoration is
    required.
    """

    states: tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.states)


def _copy_state(state: Any) -> Any:
    """Return an independent copy of an emulator state payload."""

    return deepcopy(state)


def _get_state(env: Any) -> Any:
    return get_emulator_state(env)


def _set_state(env: Any, state: Any) -> None:
    set_emulator_state(env, state)


def _validate_count(value: Any, name: str) -> int:
    """Validate a pool size without accepting bool as an integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive int")
    if value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value


class EmulatorPool:
    """Run a fixed number of independent environments in parallel.

    Args:
        env_factory: Zero-argument factory used once for every lane.
        num_envs: Number of independent environment instances to create.
        size: Optional keyword alias for ``num_envs``.
        snapshot_adapter: Optional full-environment or emulator-only adapter
            used by :meth:`save_snapshot` / :meth:`load_snapshot` /
            :meth:`fork_snapshot`.  Defaults to
            :class:`~retro_harness.snapshot.EmulatorOnlyAdapter` (uncertified).

    ``reset`` returns one normalized ``(observation, info)`` pair per lane.
    ``step`` returns one normalized Gymnasium five-tuple per lane, preserving
    input order even though the calls execute concurrently.

    ``save``/``load``/``fork`` remain the raw emulator-only path
    (:class:`PoolState`, uncertified).  Certified full-env rollouts use the
    ``*_snapshot`` methods and a CERTIFIED_FULL_ENV adapter.
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        num_envs: int | None = None,
        *,
        size: int | None = None,
        snapshot_adapter: SnapshotAdapter | None = None,
    ) -> None:
        if num_envs is not None and size is not None:
            raise TypeError("pass either num_envs or size, not both")
        count = (
            size
            if size is not None
            else (num_envs if num_envs is not None else 1)
        )
        count_name = "size" if size is not None else "num_envs"
        count = _validate_count(count, count_name)

        self._envs: tuple[Any, ...]
        created: list[Any] = []
        try:
            for _ in range(count):
                created.append(env_factory())
        except BaseException:
            for env in created:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            raise

        self._envs = tuple(created)
        self._executor = ThreadPoolExecutor(
            max_workers=count,
            thread_name_prefix="retro-emulator",
        )
        self._closed = False
        self._snapshot_adapter: SnapshotAdapter = (
            snapshot_adapter
            if snapshot_adapter is not None
            else EmulatorOnlyAdapter()
        )

    @property
    def num_envs(self) -> int:
        """Number of independent lanes in the pool."""

        return len(self._envs)

    @property
    def envs(self) -> tuple[Any, ...]:
        """The live lane environments, in deterministic lane order."""

        return self._envs

    @property
    def snapshot_adapter(self) -> SnapshotAdapter:
        """Adapter used for envelope capture/restore."""

        return self._snapshot_adapter

    def __len__(self) -> int:
        return self.num_envs

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("emulator pool is closed")

    def _parallel(self, operation: Callable[[Any], Any]) -> list[Any]:
        self._ensure_open()
        return list(self._executor.map(operation, self._envs))

    def reset(self) -> list[ResetResult]:
        """Reset every lane and return the resulting observation and info."""

        return self._parallel(reset_env)

    def step(self, actions: Sequence[Any]) -> list[StepResult]:
        """Step all lanes concurrently and return results in lane order."""

        self._ensure_open()
        action_list = list(actions)
        if len(action_list) != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} actions, got {len(action_list)}"
            )

        return list(
            self._executor.map(
                lambda pair: step_env(pair[0], pair[1]),
                zip(self._envs, action_list),
            )
        )

    def save(self) -> PoolState:
        """Capture independent **uncertified** emulator-only state snapshots."""

        states = self._parallel(_get_state)
        return PoolState(tuple(states))

    def load(self, snapshot: PoolState) -> None:
        """Load emulator-only state without resetting or updating wrappers."""

        self._ensure_open()
        if not isinstance(snapshot, PoolState):
            raise TypeError("load expects a PoolState returned by save()")
        if len(snapshot) != self.num_envs:
            raise ValueError(
                f"snapshot has {len(snapshot)} states, expected {self.num_envs}"
            )

        list(
            self._executor.map(
                lambda pair: _set_state(pair[0], pair[1]),
                zip(self._envs, snapshot.states),
            )
        )

    def fork(
        self,
        state: Any | None = None,
        *,
        source: int = 0,
    ) -> PoolState:
        """Broadcast an **uncertified** emulator-only branch point to every lane.

        If ``state`` is omitted, the current state of ``source`` is captured.
        ``fork`` intentionally does not call ``env.reset()``; it is the fast
        save/load operation used between rollout branches.  Wrapper state is
        not restored — use :meth:`fork_snapshot` with a full-env adapter.
        """

        self._ensure_open()
        if not 0 <= source < self.num_envs:
            raise IndexError(f"source lane {source} is outside the pool")
        branch_state = (
            _get_state(self._envs[source])
            if state is None
            else _copy_state(state)
        )
        snapshot = PoolState(
            tuple(_copy_state(branch_state) for _ in range(self.num_envs))
        )
        self.load(snapshot)
        return self.save()

    def save_snapshot(self) -> PoolSnapshot:
        """Capture per-lane :class:`SnapshotEnvelope` values via the adapter."""

        adapter = self._snapshot_adapter

        def _capture(env: Any) -> SnapshotEnvelope:
            return capture_envelope(env, adapter)

        return PoolSnapshot(tuple(self._parallel(_capture)))

    def load_snapshot(self, snapshot: PoolSnapshot) -> None:
        """Restore a :class:`PoolSnapshot` after identity checks on every lane.

        Identity is validated for **all** lanes before **any** lane is mutated,
        so a mismatch leaves the pool untouched.
        """

        self._ensure_open()
        if not isinstance(snapshot, PoolSnapshot):
            raise TypeError(
                "load_snapshot expects a PoolSnapshot returned by save_snapshot()"
            )
        if len(snapshot) != self.num_envs:
            raise ValueError(
                f"snapshot has {len(snapshot)} envelopes, "
                f"expected {self.num_envs}"
            )

        adapter = self._snapshot_adapter
        # Phase 1: fail closed before mutation.
        for env, envelope in zip(self._envs, snapshot.envelopes):
            assert_envelope_compatible(env, envelope, adapter)

        # Phase 2: restore (identity already checked; still re-check inside).
        list(
            self._executor.map(
                lambda pair: restore_envelope(pair[0], pair[1], adapter),
                zip(self._envs, snapshot.envelopes),
            )
        )

    def fork_snapshot(
        self,
        envelope: SnapshotEnvelope | None = None,
        *,
        source: int = 0,
    ) -> PoolSnapshot:
        """Broadcast one certified (or emulator-only) envelope to every lane.

        If ``envelope`` is omitted, the current envelope of ``source`` is
        captured.  Identity must match every lane before any mutation.
        """

        self._ensure_open()
        if not 0 <= source < self.num_envs:
            raise IndexError(f"source lane {source} is outside the pool")
        branch = (
            capture_envelope(self._envs[source], self._snapshot_adapter)
            if envelope is None
            else envelope
        )
        if not isinstance(branch, SnapshotEnvelope):
            raise TypeError("fork_snapshot expects a SnapshotEnvelope")
        # Independent envelopes so later lane mutation cannot alias payloads.
        pool = PoolSnapshot(
            tuple(
                SnapshotEnvelope(
                    certification=branch.certification,
                    identity=branch.identity,
                    emulator_state=branch.emulator_state,
                    adapter_state=branch.adapter_state,
                )
                for _ in range(self.num_envs)
            )
        )
        self.load_snapshot(pool)
        return self.save_snapshot()

    def close(self) -> None:
        """Close all environments and release the pool executor."""

        if self._closed:
            return
        try:
            # Consume map so close exceptions are not silently discarded.
            list(
                self._executor.map(
                    lambda env: _close_env(env),
                    self._envs,
                )
            )
        finally:
            self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self) -> EmulatorPool:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


__all__ = [
    "EmulatorPool",
    "EnvFactory",
    "PoolSnapshot",
    "PoolState",
    "ResetResult",
    "SnapshotAdapter",
    "SnapshotEnvelope",
    "StepResult",
]

