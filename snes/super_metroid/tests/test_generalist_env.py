"""Generalist Gym env smoke (practice ROM + captured pin)."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_metroid.generalist.corpus import load_rows
from super_metroid.paths import SHARED_PRACTICE_ROM

pytestmark = pytest.mark.rom


def test_practice_env_reset_and_terminate() -> None:
    if not SHARED_PRACTICE_ROM.is_file():
        pytest.skip("practice ROM missing")
    rows = load_rows(area="crateria", exclude_ceres=True, dedupe=True)
    ship = [row for row in rows if row.session_id.endswith("/ship")]
    if not ship or not Path(ship[0].state_path).is_file():
        pytest.skip("captured ship pin missing")
    from super_metroid.generalist.env import GeneralistEnv
    from super_metroid.generalist.obs import OBS_DIM

    env = GeneralistEnv(
        rows=ship,
        area="crateria",
        frame_skip=4,
        max_episode_frames=64,
        stall_frames=32,
    )
    try:
        obs, info = env.reset(options={"session_id": ship[0].session_id})
        assert obs.shape == (OBS_DIM,)
        assert info["practice_only"] is True
        assert info["session_id"].endswith("/ship")
        from super_metroid.generalist.obs import N_GRID
        from super_metroid.generalist.solid import editor_rooms_dir

        if editor_rooms_dir() is not None:
            assert float(obs[:N_GRID].max()) > 0.0
        done = False
        last = info
        while not done:
            obs, _reward, terminated, truncated, last = env.step(1)
            done = terminated or truncated
        assert last["reason"] in {"join", "stall", "timeout", "death", "unmapped_room"}
        assert last["frame"] >= 1
    finally:
        env.close()
