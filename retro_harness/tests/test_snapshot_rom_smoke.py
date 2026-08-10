"""Real stable-retro smoke for certified full-environment snapshots (rr-gbd.32).

Opt-in: ``RETRO_RL_RUN_ROM_SMOKE=1 uv run pytest -m rom_smoke``.
Requires a legally supplied Super Metroid ROM at the usual integration path.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from retro_harness.emulator_pool import EmulatorPool
from retro_harness.env import make_env
from retro_harness.repo import monorepo_root, resolve_game_dir
from retro_harness.snapshot import (
    AttributeSnapshotAdapter,
    SnapshotCertification,
    SnapshotIdentityMismatch,
)


pytestmark = [pytest.mark.rom, pytest.mark.rom_smoke]

GAME_ID = "SuperMetroid-Snes"
CORE_DIGEST = "snes9x-stable-retro"
GAME_DIGEST = "super-metroid-snes-levelstart-or-none"


class _InstrumentedRetroEnv:
    """Thin wrapper: stable-retro env + counters/cache/RNG for full-env cert."""

    def __init__(self, env: object) -> None:
        self._env = env
        self.em = env.em  # type: ignore[attr-defined]
        self.step_count = 0
        self.obs_cache: object | None = None
        self.rng = random.Random(0)
        self.closed = False

    def reset(self):
        obs, info = self._env.reset()  # type: ignore[operator]
        self.step_count = 0
        self.obs_cache = obs
        return obs, info

    def step(self, action):
        # Wrapper RNG affects a Python-side counter (not the core), so restore
        # must put the counter sequence back on the same track.
        bonus = self.rng.randint(0, 2)
        obs, reward, terminated, truncated, info = self._env.step(action)  # type: ignore[operator]
        self.step_count += 1 + bonus
        self.obs_cache = obs
        info = dict(info) if isinstance(info, dict) else {}
        info["wrapper_step_count"] = self.step_count
        info["wrapper_bonus"] = bonus
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


def _rom_available(game_dir: Path) -> bool:
    integration = game_dir / "custom_integrations" / GAME_ID
    for name in ("rom.sfc", "rom.smc", "rom.bin"):
        candidate = integration / name
        if candidate.is_file() or candidate.is_symlink():
            try:
                return candidate.resolve().is_file()
            except OSError:
                return False
    return False


def _pick_state(game_dir: Path) -> str | None:
    integration = game_dir / "custom_integrations" / GAME_ID
    preferred = (
        "LevelStart",
        "dev_kpdr_business",
        "CeresElevator",
    )
    for name in preferred:
        if (integration / f"{name}.state").is_file():
            return name
    states = sorted(integration.glob("*.state"))
    if states:
        return states[0].stem
    return None


def _zero_action(env: _InstrumentedRetroEnv) -> np.ndarray:
    space = getattr(env._env, "action_space", None)
    if space is not None and hasattr(space, "sample"):
        shape = getattr(space, "shape", None)
        if shape:
            return np.zeros(shape, dtype=np.int8)
    return np.zeros(12, dtype=np.int8)


@pytest.fixture(scope="module")
def sm_game_dir() -> Path:
    try:
        game_dir = resolve_game_dir("super_metroid")
    except FileNotFoundError:
        pytest.skip("super_metroid game directory not found")
    if not _rom_available(game_dir):
        # Worktree clones often omit gitignored ROM links; try main monorepo.
        alt = Path("/home/v/01_projects/11_games/retro_rl/snes/super_metroid")
        if alt.is_dir() and _rom_available(alt):
            return alt
        pytest.skip(f"Super Metroid ROM not available under {game_dir}")
    return game_dir


def test_stable_retro_certified_snapshot_replay(sm_game_dir: Path) -> None:
    import stable_retro as retro

    if not hasattr(retro.data.Integrations, "CUSTOM"):
        pytest.skip("stable_retro test stub cannot execute ROM smoke")

    state = _pick_state(sm_game_dir)
    adapter = AttributeSnapshotAdapter(
        adapter_id="retro_harness.tests.InstrumentedRetroEnv",
        attributes=("step_count", "obs_cache", "rng"),
        core_digest=CORE_DIGEST,
        game_digest=GAME_DIGEST,
    )

    def factory() -> _InstrumentedRetroEnv:
        raw = make_env(
            GAME_ID,
            state,
            sm_game_dir,
            render_mode="rgb_array",
        )
        return _InstrumentedRetroEnv(raw)

    # stable-retro allows only one emulator instance per process, so the real
    # consumer is single-lane. Multi-lane certified batches are fake-tested and
    # deferred to multi-process work under rr-gbd.34.
    pool = EmulatorPool(factory, num_envs=1, snapshot_adapter=adapter)
    try:
        pool.reset()
        action = _zero_action(pool.envs[0])
        # Warm-up a few frames so emulator + wrapper diverge from pure boot.
        for _ in range(5):
            pool.step([action])

        baseline = pool.save_snapshot()
        assert baseline.certification is SnapshotCertification.CERTIFIED_FULL_ENV
        assert len(baseline) == 1

        def run_n(n: int) -> list[tuple]:
            rows: list[tuple] = []
            for _ in range(n):
                obs, _rew, term, trunc, info = pool.step([action])[0]
                env = pool.envs[0]
                rows.append(
                    (
                        int(obs.sum()) if hasattr(obs, "sum") else None,
                        bool(term),
                        bool(trunc),
                        int(info["wrapper_step_count"]),
                        int(info["wrapper_bonus"]),
                        env.step_count,
                        env.rng.getstate()[1][0],
                    )
                )
            return rows

        first = run_n(20)

        # Pollute emulator + wrapper state.
        pool.step([action])
        env = pool.envs[0]
        env.step_count += 1000
        env.obs_cache = None
        env.rng.random()
        polluted_steps = env.step_count
        polluted_prefix = env.em.get_state()[:32]

        # Identity mismatch must fail before mutation.
        from retro_harness.snapshot import (
            SnapshotEnvelope,
            SnapshotIdentity,
            PoolSnapshot,
        )

        bad_identity = SnapshotIdentity(
            adapter_id=baseline.envelopes[0].identity.adapter_id,
            schema_version=baseline.envelopes[0].identity.schema_version,
            core_identity_digest=baseline.envelopes[0].identity.core_identity_digest,
            game_identity_digest="wrong-game",
        )
        bad = PoolSnapshot(
            (
                SnapshotEnvelope(
                    certification=baseline.envelopes[0].certification,
                    identity=bad_identity,
                    emulator_state=baseline.envelopes[0].emulator_state,
                    adapter_state=baseline.envelopes[0].adapter_state,
                ),
            )
        )
        with pytest.raises(SnapshotIdentityMismatch, match="game identity"):
            pool.load_snapshot(bad)
        assert env.step_count == polluted_steps
        assert env.em.get_state()[:32] == polluted_prefix

        pool.load_snapshot(baseline)
        second = run_n(20)
        assert first == second

        # fork_snapshot single-lane is identity-preserving re-capture.
        forked = pool.fork_snapshot(source=0)
        assert forked.certification is SnapshotCertification.CERTIFIED_FULL_ENV
        assert forked.envelopes[0].identity.game_identity_digest == GAME_DIGEST

        monorepo_root()  # touch path helper so layout stays wired
    finally:
        pool.close()
