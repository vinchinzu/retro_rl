"""Tests for deterministic branch-rollout batches (rr-gbd.34)."""

from __future__ import annotations

import random

import pytest

from retro_harness import (
    BranchResult as ExportedBranchResult,
    BranchSpec as ExportedBranchSpec,
    BranchStatus as ExportedBranchStatus,
    RolloutResult as ExportedRolloutResult,
    RolloutSpec as ExportedRolloutSpec,
    branch_from_actions as exported_branch_from_actions,
    run_branch_rollouts as exported_run_branch_rollouts,
)
from retro_harness.branch_rollout import (
    BranchResult,
    BranchSpec,
    BranchStatus,
    RolloutAccounting,
    RolloutError,
    RolloutResult,
    RolloutSpec,
    accounting_for,
    branch_from_actions,
    replay_digest_for,
    run_branch_rollouts,
    run_branch_rollouts_on_pool,
)
from retro_harness.emulator_pool import EmulatorPool
from retro_harness.snapshot import (
    AttributeSnapshotAdapter,
    SnapshotCertification,
    capture_envelope,
)


def test_branch_rollout_types_exported_from_retro_harness() -> None:
    assert ExportedBranchSpec is BranchSpec
    assert ExportedBranchResult is BranchResult
    assert ExportedBranchStatus is BranchStatus
    assert ExportedRolloutSpec is RolloutSpec
    assert ExportedRolloutResult is RolloutResult
    assert exported_branch_from_actions is branch_from_actions
    assert exported_run_branch_rollouts is run_branch_rollouts


class _FakeEmulator:
    def __init__(self) -> None:
        self.value = 0

    def get_state(self) -> int:
        return self.value

    def set_state(self, state: int) -> None:
        self.value = state


class _WrappedEnv:
    """Fake wrapped env: emulator + counters + obs cache + RNG."""

    def __init__(self, lane: int = 0, *, seed: int = 0) -> None:
        self.lane = lane
        self.em = _FakeEmulator()
        self.step_count = 0
        self.obs_cache: int | None = None
        self.rng = random.Random(seed)
        self.closed = False
        self.game_id = "fake-game"
        self.core_id = "fake-core"
        self._terminate_at: int | None = None

    def reset(self):
        self.em.value = 0
        self.step_count = 0
        self.obs_cache = 0
        return self.obs_cache, {"lane": self.lane}

    def step(self, action: int):
        noise = self.rng.randint(0, 3)
        self.em.value += int(action) + noise
        self.step_count += 1
        self.obs_cache = self.em.value
        terminated = (
            self._terminate_at is not None and self.step_count >= self._terminate_at
        )
        return (
            self.obs_cache,
            int(action),
            bool(terminated),
            False,
            {
                "lane": self.lane,
                "value": self.em.value,
                "step_count": self.step_count,
                "noise": noise,
            },
        )

    def close(self) -> None:
        self.closed = True


def _full_adapter(
    *,
    game: str = "fake-game",
    core: str = "fake-core",
    adapter_id: str = "tests.BranchRolloutAdapter",
) -> AttributeSnapshotAdapter:
    return AttributeSnapshotAdapter(
        adapter_id=adapter_id,
        attributes=("step_count", "obs_cache", "rng"),
        core_digest=core,
        game_digest=game,
    )


def _certified_root(seed: int = 11):
    adapter = _full_adapter()
    env = _WrappedEnv(seed=seed)
    env.reset()
    env.step(1)
    env.step(2)
    root = capture_envelope(env, adapter)
    env.close()
    assert root.certification is SnapshotCertification.CERTIFIED_FULL_ENV
    return adapter, root


def test_width_1_and_4_identical_independent_of_ordering() -> None:
    adapter, root = _certified_root()
    # Four distinct fixed-action branches.
    branches = (
        branch_from_actions("a", [1, 0, 2, 1]),
        branch_from_actions("b", [3, 3, 1]),
        branch_from_actions("c", [0, 0, 0, 0, 1]),
        branch_from_actions("d", [2, 1]),
    )
    # Permuted order must not change per-branch outcomes / replay digest.
    permuted = (branches[2], branches[0], branches[3], branches[1])

    def factory() -> _WrappedEnv:
        return _WrappedEnv(seed=11)

    spec = RolloutSpec(root=root, branches=branches)
    spec_perm = RolloutSpec(root=root, branches=permuted)

    w1 = run_branch_rollouts(factory, adapter, spec, width=1)
    w4 = run_branch_rollouts(factory, adapter, spec, width=4)
    w1_perm = run_branch_rollouts(factory, adapter, spec_perm, width=1)
    w4_perm = run_branch_rollouts(factory, adapter, spec_perm, width=4)

    assert w1.replay_digest == w4.replay_digest
    assert w1.replay_digest == w1_perm.replay_digest
    assert w1.replay_digest == w4_perm.replay_digest

    for branch_id in ("a", "b", "c", "d"):
        r1 = w1.result_for(branch_id)
        r4 = w4.result_for(branch_id)
        assert r1.status is BranchStatus.OK
        assert r1.to_record() == r4.to_record()
        assert r1.to_record() == w1_perm.result_for(branch_id).to_record()
        assert r1.to_record() == w4_perm.result_for(branch_id).to_record()

    # Spec order preserved in result list; digest ignores order.
    assert [item.branch_id for item in w1.branches] == ["a", "b", "c", "d"]
    assert [item.branch_id for item in w1_perm.branches] == ["c", "a", "d", "b"]


def test_controller_exception_isolates_invalid_branch() -> None:
    adapter, root = _certified_root(seed=3)

    def boom(env: object, step_index: int) -> int:
        del env
        if step_index >= 1:
            raise RuntimeError("bad branch controller")
        return 1

    branches = (
        branch_from_actions("ok_left", [2, 2, 2]),
        BranchSpec(branch_id="bad", max_steps=5, controller=boom),
        branch_from_actions("ok_right", [1, 0, 1, 0]),
    )
    spec = RolloutSpec(root=root, branches=branches)

    def factory() -> _WrappedEnv:
        return _WrappedEnv(seed=3)

    # Width 1 sequential and width 3 concurrent both isolate the bad branch.
    for width in (1, 3):
        result = run_branch_rollouts(factory, adapter, spec, width=width)
        assert result.accounting.branch_count == 3
        assert result.accounting.ok_count == 2
        assert result.accounting.controller_error_count == 1
        assert result.accounting.total_steps == (
            result.result_for("ok_left").steps_executed
            + result.result_for("bad").steps_executed
            + result.result_for("ok_right").steps_executed
        )

        bad = result.result_for("bad")
        assert bad.status is BranchStatus.CONTROLLER_ERROR
        assert bad.error_type == "RuntimeError"
        assert "bad branch" in (bad.error_message or "")
        assert bad.steps_executed == 1  # failed on second controller call

        left = result.result_for("ok_left")
        right = result.result_for("ok_right")
        assert left.status is BranchStatus.OK
        assert right.status is BranchStatus.OK
        assert left.steps_executed == 3
        assert right.steps_executed == 4


def test_accounting_exact_with_early_terminate() -> None:
    adapter, root = _certified_root(seed=5)

    def factory() -> _WrappedEnv:
        env = _WrappedEnv(seed=5)
        return env

    # One branch forces terminate after 2 post-root steps by setting flag mid-run.
    terminate_steps = {"n": 0}

    def terminating_controller(env: _WrappedEnv, step_index: int) -> int:
        # After root restore, arm terminate-at so the 2nd step ends the episode.
        if step_index == 0:
            env._terminate_at = env.step_count + 2
        terminate_steps["n"] += 1
        return 1

    branches = (
        BranchSpec(
            branch_id="term",
            max_steps=10,
            controller=terminating_controller,  # type: ignore[arg-type]
        ),
        branch_from_actions("full", [1, 1, 1]),
    )
    spec = RolloutSpec(root=root, branches=branches)
    result = run_branch_rollouts(factory, adapter, spec, width=2)

    term = result.result_for("term")
    assert term.status is BranchStatus.OK
    assert term.terminated is True
    assert term.steps_executed == 2

    full = result.result_for("full")
    assert full.status is BranchStatus.OK
    assert full.terminated is False
    assert full.steps_executed == 3

    acc = result.accounting
    assert acc == RolloutAccounting(
        branch_count=2,
        ok_count=2,
        controller_error_count=0,
        terminated_count=1,
        truncated_count=0,
        total_steps=5,
    )
    # Recompute and compare.
    assert accounting_for(result.branches) == acc
    assert result.replay_digest == replay_digest_for(result.branches)


def test_uncertified_root_rejected() -> None:
    from retro_harness.snapshot import EmulatorOnlyAdapter

    env = _WrappedEnv(seed=0)
    env.reset()
    uncertified = capture_envelope(env, EmulatorOnlyAdapter())
    env.close()
    with pytest.raises(RolloutError, match="CERTIFIED_FULL_ENV"):
        RolloutSpec(
            root=uncertified,
            branches=(branch_from_actions("x", [1]),),
        )


def test_duplicate_branch_ids_rejected() -> None:
    adapter, root = _certified_root()
    del adapter
    with pytest.raises(RolloutError, match="unique"):
        RolloutSpec(
            root=root,
            branches=(
                branch_from_actions("same", [1]),
                branch_from_actions("same", [2]),
            ),
        )


def test_run_on_existing_pool() -> None:
    adapter, root = _certified_root(seed=7)
    branches = (
        branch_from_actions("p0", [1, 2]),
        branch_from_actions("p1", [3]),
        branch_from_actions("p2", [0, 0, 1]),
    )
    spec = RolloutSpec(root=root, branches=branches)
    pool = EmulatorPool(
        lambda: _WrappedEnv(seed=7),
        num_envs=2,
        snapshot_adapter=adapter,
    )
    try:
        pool.reset()
        # Pollute lanes; rollouts must re-restore root each branch.
        pool.step([9, 9])
        result = run_branch_rollouts_on_pool(pool, spec)
        assert result.width == 2
        assert result.accounting.branch_count == 3
        assert all(item.status is BranchStatus.OK for item in result.branches)
        via_factory = run_branch_rollouts(
            lambda: _WrappedEnv(seed=7),
            adapter,
            spec,
            width=2,
        )
        assert result.replay_digest == via_factory.replay_digest
    finally:
        pool.close()


def test_zero_step_branch_ok() -> None:
    adapter, root = _certified_root()

    def never_called(env: object, step_index: int) -> int:
        del env, step_index
        raise AssertionError("controller must not be called when max_steps=0")

    spec = RolloutSpec(
        root=root,
        branches=(
            BranchSpec(branch_id="empty", max_steps=0, controller=never_called),
        ),
    )
    result = run_branch_rollouts(
        lambda: _WrappedEnv(seed=11),
        adapter,
        spec,
        width=1,
    )
    empty = result.result_for("empty")
    assert empty.status is BranchStatus.OK
    assert empty.steps_executed == 0
    assert result.accounting.total_steps == 0
