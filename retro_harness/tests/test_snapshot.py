"""Tests for certified full-environment snapshots (rr-gbd.32)."""

from __future__ import annotations

import random

import pytest

from retro_harness import (
    AttributeSnapshotAdapter as ExportedAttributeAdapter,
    EmulatorOnlyAdapter as ExportedEmulatorOnly,
    EmulatorPool as ExportedPool,
    PoolSnapshot as ExportedPoolSnapshot,
    SnapshotCertification as ExportedCertification,
    SnapshotEnvelope as ExportedEnvelope,
    SnapshotIdentity as ExportedIdentity,
    SnapshotIdentityMismatch as ExportedMismatch,
)
from retro_harness.emulator_pool import EmulatorPool
from retro_harness.snapshot import (
    AttributeSnapshotAdapter,
    EmulatorOnlyAdapter,
    PoolSnapshot,
    SnapshotCertification,
    SnapshotEnvelope,
    SnapshotError,
    SnapshotIdentity,
    SnapshotIdentityMismatch,
    assert_envelope_compatible,
    capture_envelope,
    restore_envelope,
)


def test_snapshot_types_exported_from_retro_harness() -> None:
    assert ExportedPool is EmulatorPool
    assert ExportedPoolSnapshot is PoolSnapshot
    assert ExportedEnvelope is SnapshotEnvelope
    assert ExportedIdentity is SnapshotIdentity
    assert ExportedCertification is SnapshotCertification
    assert ExportedMismatch is SnapshotIdentityMismatch
    assert ExportedEmulatorOnly is EmulatorOnlyAdapter
    assert ExportedAttributeAdapter is AttributeSnapshotAdapter


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
        return (
            self.obs_cache,
            action,
            False,
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
    adapter_id: str = "tests.WrappedEnvAdapter",
) -> AttributeSnapshotAdapter:
    return AttributeSnapshotAdapter(
        adapter_id=adapter_id,
        attributes=("step_count", "obs_cache", "rng"),
        core_digest=core,
        game_digest=game,
    )


def _trajectory(env: _WrappedEnv, actions: list[int]) -> list[tuple]:
    rows: list[tuple] = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            (
                obs,
                reward,
                terminated,
                truncated,
                info["value"],
                info["step_count"],
                info["noise"],
                env.obs_cache,
                env.step_count,
                env.em.value,
            )
        )
    return rows


def test_certified_full_env_reproduces_100_steps_after_save_load_fork() -> None:
    adapter = _full_adapter()
    actions = [((i * 7) % 5) for i in range(100)]

    pool = EmulatorPool(
        lambda: _WrappedEnv(seed=11),
        num_envs=2,
        snapshot_adapter=adapter,
    )
    try:
        pool.reset()
        # Divergent early steps so fork must fully re-sync wrapper state.
        pool.step([1, 2])
        pool.step([3, 4])

        baseline = pool.save_snapshot()
        assert baseline.certification is SnapshotCertification.CERTIFIED_FULL_ENV
        assert all(env.is_certified_full_env for env in baseline.envelopes)

        first = [
            _trajectory(pool.envs[0], actions),
            _trajectory(pool.envs[1], actions),
        ]

        # Mutate heavily so load must restore counters/cache/RNG, not just emu.
        pool.step([9, 9])
        for env in pool.envs:
            env.step_count += 50
            env.obs_cache = -1
            env.rng.random()
            env.em.value = 999

        pool.load_snapshot(baseline)
        second = [
            _trajectory(pool.envs[0], actions),
            _trajectory(pool.envs[1], actions),
        ]
        assert first == second

        # fork_snapshot from lane 0 must make both lanes match lane-0 trajectory.
        pool.load_snapshot(baseline)
        pool.step([1, 2])  # diverge lanes again
        forked = pool.fork_snapshot(source=0)
        assert len(forked) == 2
        assert forked.envelopes[0].identity == forked.envelopes[1].identity

        traj_a = _trajectory(pool.envs[0], actions)
        traj_b = _trajectory(pool.envs[1], actions)
        assert traj_a == traj_b
    finally:
        pool.close()


def test_mismatched_identity_fails_before_mutation() -> None:
    adapter = _full_adapter(game="game-a", core="core-a")
    env = _WrappedEnv(seed=3)
    env.reset()
    env.step(2)

    envelope = capture_envelope(env, adapter)
    before_emu = env.em.value
    before_steps = env.step_count
    before_cache = env.obs_cache
    before_rng = env.rng.getstate()

    wrong = EmulatorOnlyAdapter(core_digest="core-a", game_digest="game-a")
    with pytest.raises(SnapshotIdentityMismatch, match="adapter_id"):
        restore_envelope(env, envelope, wrong)

    other_game = _full_adapter(game="game-b", core="core-a")
    with pytest.raises(SnapshotIdentityMismatch, match="game identity"):
        restore_envelope(env, envelope, other_game)

    other_core = _full_adapter(game="game-a", core="core-b")
    with pytest.raises(SnapshotIdentityMismatch, match="core identity"):
        restore_envelope(env, envelope, other_core)

    # Mutate identity fields on a copy of the envelope.
    bad_identity = SnapshotIdentity(
        adapter_id=envelope.identity.adapter_id,
        schema_version=envelope.identity.schema_version + 1,
        core_identity_digest=envelope.identity.core_identity_digest,
        game_identity_digest=envelope.identity.game_identity_digest,
    )
    bad_envelope = SnapshotEnvelope(
        certification=envelope.certification,
        identity=bad_identity,
        emulator_state=envelope.emulator_state,
        adapter_state=envelope.adapter_state,
    )
    with pytest.raises(SnapshotIdentityMismatch, match="schema_version"):
        assert_envelope_compatible(env, bad_envelope, adapter)

    assert env.em.value == before_emu
    assert env.step_count == before_steps
    assert env.obs_cache == before_cache
    assert env.rng.getstate() == before_rng


def test_pool_load_snapshot_mismatch_leaves_all_lanes_untouched() -> None:
    adapter = _full_adapter()
    pool = EmulatorPool(
        lambda: _WrappedEnv(seed=1),
        num_envs=2,
        snapshot_adapter=adapter,
    )
    try:
        pool.reset()
        pool.step([1, 1])
        snap = pool.save_snapshot()
        pool.step([5, 6])
        markers = [(env.em.value, env.step_count, env.obs_cache) for env in pool.envs]

        # Corrupt only lane 1 identity so phase-1 check fails.
        bad_identity = SnapshotIdentity(
            adapter_id=snap.envelopes[1].identity.adapter_id,
            schema_version=snap.envelopes[1].identity.schema_version,
            core_identity_digest=snap.envelopes[1].identity.core_identity_digest,
            game_identity_digest="other-game",
        )
        bad = PoolSnapshot(
            (
                snap.envelopes[0],
                SnapshotEnvelope(
                    certification=snap.envelopes[1].certification,
                    identity=bad_identity,
                    emulator_state=snap.envelopes[1].emulator_state,
                    adapter_state=snap.envelopes[1].adapter_state,
                ),
            )
        )
        with pytest.raises(SnapshotIdentityMismatch, match="game identity"):
            pool.load_snapshot(bad)

        assert [
            (env.em.value, env.step_count, env.obs_cache) for env in pool.envs
        ] == markers
    finally:
        pool.close()


def test_raw_emulator_snapshots_remain_supported_but_uncertified() -> None:
    pool = EmulatorPool(_WrappedEnv, size=1)
    try:
        assert isinstance(pool.snapshot_adapter, EmulatorOnlyAdapter)
        pool.reset()
        pool.envs[0].em.value = 11
        pool.envs[0].step_count = 3
        raw = pool.save()
        certified = pool.save_snapshot()
        assert certified.certification is SnapshotCertification.EMULATOR_ONLY
        assert certified.envelopes[0].adapter_state is None

        pool.envs[0].step_count = 99
        pool.envs[0].em.value = 50
        pool.load(raw)
        # Emulator restored; wrapper counter intentionally NOT restored.
        assert pool.envs[0].em.value == 11
        assert pool.envs[0].step_count == 99
    finally:
        pool.close()


def test_certified_envelope_requires_adapter_state() -> None:
    identity = SnapshotIdentity(
        adapter_id="x",
        schema_version=1,
        core_identity_digest="c",
        game_identity_digest="g",
    )
    with pytest.raises(SnapshotError, match="adapter_state"):
        SnapshotEnvelope(
            certification=SnapshotCertification.CERTIFIED_FULL_ENV,
            identity=identity,
            emulator_state=0,
            adapter_state=None,
        )
    with pytest.raises(SnapshotError, match="must not carry adapter_state"):
        SnapshotEnvelope(
            certification=SnapshotCertification.EMULATOR_ONLY,
            identity=identity,
            emulator_state=0,
            adapter_state={"x": 1},
        )


def test_envelope_deepcopy_isolates_from_live_mutation() -> None:
    adapter = _full_adapter()
    env = _WrappedEnv(seed=0)
    env.reset()
    env.em.value = 7
    env.step_count = 3
    env.obs_cache = 7
    envelope = capture_envelope(env, adapter)
    env.em.value = 12345
    env.step_count = 999
    assert envelope.emulator_state == 7
    assert envelope.adapter_state is not None
    assert envelope.adapter_state["step_count"] == 3


def test_uncertified_path_does_not_restore_wrapper_after_snapshot_load() -> None:
    """EmulatorOnlyAdapter envelope restores emu only (regression guard)."""

    adapter = EmulatorOnlyAdapter(core_digest="c", game_digest="g")
    env = _WrappedEnv(seed=2)
    env.reset()
    env.em.value = 17
    env.step_count = 5
    envelope = capture_envelope(env, adapter)
    assert envelope.is_emulator_only

    env.em.value = 99
    env.step_count = 42
    restore_envelope(env, envelope, adapter)
    assert env.em.value == 17
    # Wrapper counter intentionally NOT restored on uncertified path.
    assert env.step_count == 42
