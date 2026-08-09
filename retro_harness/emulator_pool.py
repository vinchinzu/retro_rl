"""Deterministic parallel pool for emulator-backed environments.

The pool deliberately knows only the small surface shared by the harness:
``reset()``, ``step()``, and emulator state ``get_state()``/``set_state()``.
Stable-retro exposes the latter through ``env.em``; small test doubles and
other environments may expose them directly on the environment instead.

Snapshots contain emulator state only.  They do not capture wrapper state,
episode bookkeeping, wrapper RNGs, observation caches, or reset results, so
``save``/``load``/``fork`` provide deterministic restoration only for
compatible emulator-only environments where all state relevant to future
behavior is represented by ``get_state``/``set_state``.  They do not claim
full-environment restoration for wrapped environments.

Each lane owns one environment and is stepped in a fixed order from the
caller's point of view.  Work is submitted to one long-lived thread per lane,
so independent emulator calls can run in parallel without requiring an env
factory to be pickleable.  The pool never samples actions, auto-resets lanes,
or otherwise introduces randomness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

from retro_harness.runtime import reset_env, step_env


EnvFactory: TypeAlias = Callable[[], Any]
ResetResult: TypeAlias = tuple[Any, dict[str, Any]]
StepResult: TypeAlias = tuple[Any, Any, bool, bool, dict[str, Any]]


@dataclass(frozen=True)
class PoolState:
    """A per-lane emulator snapshot captured by :meth:`EmulatorPool.save`.

    State payloads are copied when a snapshot is created and when they are
    loaded.  This prevents mutable fake states (and any future non-byte state
    backend) from being changed through a live environment after saving.
    The payload is not a full environment or wrapper snapshot.
    """

    states: tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.states)


def _copy_state(state: Any) -> Any:
    """Return an independent copy of an emulator state payload."""

    return deepcopy(state)


def _state_getter(env: Any) -> Callable[[], Any]:
    """Find the state getter on a stable-retro env or a test double."""

    emulator = getattr(env, "em", None)
    getter = getattr(emulator, "get_state", None)
    if callable(getter):
        return getter

    getter = getattr(env, "get_state", None)
    if callable(getter):
        return getter

    raise TypeError(
        "pool environments must expose get_state() or em.get_state()"
    )


def _state_setter(env: Any) -> Callable[[Any], None]:
    """Find the state setter on a stable-retro env or a test double."""

    emulator = getattr(env, "em", None)
    setter = getattr(emulator, "set_state", None)
    if callable(setter):
        return setter

    setter = getattr(env, "set_state", None)
    if callable(setter):
        return setter

    raise TypeError(
        "pool environments must expose set_state() or em.set_state()"
    )


def _get_state(env: Any) -> Any:
    return _copy_state(_state_getter(env)())


def _set_state(env: Any, state: Any) -> None:
    _state_setter(env)(_copy_state(state))


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

    ``reset`` returns one normalized ``(observation, info)`` pair per lane.
    ``step`` returns one normalized Gymnasium five-tuple per lane, preserving
    input order even though the calls execute concurrently.

    ``save``/``load`` operate on one :class:`PoolState` containing emulator
    state for every lane.  They do not restore arbitrary wrapper state; see
    the module contract.  ``fork`` is the rollout convenience operation: it
    copies one supplied state, or the current state of ``source``, into every
    lane and returns the resulting snapshot.
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        num_envs: int | None = None,
        *,
        size: int | None = None,
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

    @property
    def num_envs(self) -> int:
        """Number of independent lanes in the pool."""

        return len(self._envs)

    @property
    def envs(self) -> tuple[Any, ...]:
        """The live lane environments, in deterministic lane order."""

        return self._envs

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
        """Capture independent emulator-only state snapshots from all lanes."""

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
        """Broadcast an emulator-only branch point to every lane.

        If ``state`` is omitted, the current state of ``source`` is captured.
        ``fork`` intentionally does not call ``env.reset()``; it is the fast
        save/load operation used between rollout branches.
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
    "PoolState",
    "ResetResult",
    "StepResult",
]
