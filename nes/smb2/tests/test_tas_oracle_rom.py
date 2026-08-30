"""Live fceumm replay of the vendored SMB2 TAS through 1-1 control."""

from __future__ import annotations

import os

import pytest

from smb2.paths import (
    CONTROL_PROOF_PATH,
    GAME,
    GAME_DIR,
    INTEGRATION_DIR,
    REF_MOVIE_PATH,
    ROM_PATH,
    STATE_ARTIFACTS_DIR,
)
from smb2.ram import is_level1_control, read_snapshot
from smb2.tas_manifest import CheckpointStatus, TASEvidenceManifest
from smb2.tas_oracle import CONTROL_SEARCH_MAX, extract_level1_checkpoints


def _has_real_stable_retro() -> bool:
    try:
        import stable_retro as retro
    except ImportError:
        return False
    return hasattr(getattr(retro, "data", None), "Integrations") and hasattr(
        retro.data.Integrations, "CUSTOM"
    )


pytestmark = [
    pytest.mark.rom,
    pytest.mark.skipif(
        not _has_real_stable_retro()
        or not ROM_PATH.is_file()
        or not (INTEGRATION_DIR / "rom.nes").exists()
        or not REF_MOVIE_PATH.is_file(),
        reason="real stable_retro + SMB2 ROM + TASVideos 1724M FM2 required",
    ),
]


def test_tas_replay_materializes_level1_control(tmp_path) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from retro_harness.env import make_env, read_state_bytes, reset_obs

    manifest = extract_level1_checkpoints(max_frames=CONTROL_SEARCH_MAX)
    by_name = {slot.name: slot for slot in manifest.checkpoints}
    control = by_name["level1_control"]
    start = by_name["level1_start"]
    assert start.status is CheckpointStatus.MATERIALIZED
    assert control.status is CheckpointStatus.MATERIALIZED
    assert start.frame is not None and control.frame is not None
    assert 200 <= start.frame < control.frame <= 360
    state_path = STATE_ARTIFACTS_DIR / "level1_control.state"
    assert state_path.is_file()
    assert CONTROL_PROOF_PATH.is_file()

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        env.em.set_state(read_state_bytes(state_path))
        snap = read_snapshot(env.get_ram(), frame=control.frame)
        assert is_level1_control(snap) is True
        assert snap.player_x == 120
        assert snap.jump_physics != 0
    finally:
        env.close()

    on_disk = TASEvidenceManifest.from_json(manifest.write_json(tmp_path / "out.json"))
    assert on_disk.checkpoints[1].name == "level1_control"
