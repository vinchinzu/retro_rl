"""No-ROM tests for the deterministic emulator pool."""

from __future__ import annotations

from threading import Barrier, Lock

import pytest

from retro_harness import EmulatorPool as ExportedEmulatorPool
from retro_harness import PoolState as ExportedPoolState
from retro_harness.emulator_pool import EmulatorPool, PoolState


def test_pool_types_are_exported_from_retro_harness() -> None:
    assert ExportedEmulatorPool is EmulatorPool
    assert ExportedPoolState is PoolState


class _FakeEmulator:
    def __init__(self) -> None:
        self.value = 0

    def get_state(self) -> int:
        return self.value

    def set_state(self, state: int) -> None:
        self.value = state


class _FakeEnv:
    def __init__(self, lane: int = 0) -> None:
        self.lane = lane
        self.em = _FakeEmulator()
        self.reset_count = 0
        self.closed = False

    def reset(self):
        self.reset_count += 1
        self.em.value = 0
        return self.em.value, {"lane": self.lane, "reset": self.reset_count}

    def step(self, action: int):
        self.em.value += action
        return (
            self.em.value,
            action,
            False,
            False,
            {"lane": self.lane, "value": self.em.value},
        )

    def close(self) -> None:
        self.closed = True


def test_pool_steps_lanes_in_order_and_forks_a_common_state() -> None:
    next_lane = iter(range(3))
    pool = EmulatorPool(lambda: _FakeEnv(next(next_lane)), num_envs=3)
    try:
        assert pool.num_envs == 3
        assert pool.reset() == [
            (0, {"lane": 0, "reset": 1}),
            (0, {"lane": 1, "reset": 1}),
            (0, {"lane": 2, "reset": 1}),
        ]
        assert [result[0] for result in pool.step([1, 2, 3])] == [1, 2, 3]

        forked = pool.fork()
        assert forked == PoolState((1, 1, 1))
        assert [result[0] for result in pool.step([4, 5, 6])] == [5, 6, 7]
    finally:
        pool.close()

    assert all(env.closed for env in pool.envs)


def test_save_load_is_deterministic_for_emulator_only_envs() -> None:
    pool = EmulatorPool(_FakeEnv, size=2)
    try:
        pool.reset()
        for env in pool.envs:
            env.em.value = 10
        baseline = pool.save()
        first = pool.step([2, 7])

        pool.step([100, 100])
        pool.load(baseline)
        replay = pool.step([2, 7])

        assert first == replay
        assert pool.save() == PoolState((12, 17))
    finally:
        pool.close()


def test_reset_does_not_accept_an_unsafe_state_argument() -> None:
    pool = EmulatorPool(_FakeEnv)
    try:
        with pytest.raises(TypeError, match="state"):
            pool.reset(state=10)
    finally:
        pool.close()


def test_snapshots_do_not_claim_to_restore_wrapper_state() -> None:
    class WrapperEnv(_FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.wrapper_steps = 0

        def step(self, action: int):
            self.wrapper_steps += 1
            observation, reward, terminated, truncated, info = super().step(action)
            info["wrapper_steps"] = self.wrapper_steps
            return observation, reward, terminated, truncated, info

    pool = EmulatorPool(WrapperEnv)
    try:
        pool.reset()
        baseline = pool.save()
        pool.step([1])
        pool.load(baseline)

        assert pool.envs[0].em.value == 0
        assert pool.envs[0].wrapper_steps == 1
        assert pool.step([1])[0][-1]["wrapper_steps"] == 2
    finally:
        pool.close()


@pytest.mark.parametrize("count_name", ["num_envs", "size"])
@pytest.mark.parametrize("invalid_count", [True, False, 1.0, "2", object()])
def test_count_must_be_a_non_bool_int(
    count_name: str,
    invalid_count: object,
) -> None:
    with pytest.raises(TypeError, match="positive int"):
        EmulatorPool(_FakeEnv, **{count_name: invalid_count})  # type: ignore[arg-type]


@pytest.mark.parametrize("count_name", ["num_envs", "size"])
@pytest.mark.parametrize("invalid_count", [0, -1])
def test_count_must_be_positive(count_name: str, invalid_count: int) -> None:
    with pytest.raises(ValueError, match="positive int"):
        EmulatorPool(_FakeEnv, **{count_name: invalid_count})


def test_step_requires_one_action_per_lane_and_load_requires_pool_state() -> None:
    pool = EmulatorPool(_FakeEnv, 2)
    try:
        with pytest.raises(ValueError, match="expected 2 actions"):
            pool.step([1])
        with pytest.raises(TypeError, match="PoolState"):
            pool.load(b"not-a-pool-state")
        with pytest.raises(ValueError, match="snapshot has 1 states"):
            pool.load(PoolState((0,)))
    finally:
        pool.close()


def test_step_calls_lanes_in_parallel() -> None:
    barrier = Barrier(3)
    lock = Lock()
    entered: list[int] = []

    class BarrierEnv(_FakeEnv):
        def step(self, action: int):
            with lock:
                entered.append(self.lane)
            barrier.wait(timeout=2)
            return super().step(action)

    next_lane = iter(range(3))
    pool = EmulatorPool(lambda: BarrierEnv(next(next_lane)), 3)
    try:
        pool.reset()
        assert [result[0] for result in pool.step([1, 1, 1])] == [1, 1, 1]
        assert sorted(entered) == [0, 1, 2]
    finally:
        pool.close()


def test_closed_pool_rejects_new_work() -> None:
    pool = EmulatorPool(_FakeEnv)
    pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        pool.reset()
