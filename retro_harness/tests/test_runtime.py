"""Tests for normalized environment and state helpers."""

from __future__ import annotations

from retro_harness.env import GameSpec, read_state_bytes, write_state_bytes
from retro_harness.runtime import reset_env, step_env


class _OldEnv:
    def reset(self):
        return "obs"

    def step(self, action):
        return "next", 2, True, {"action": action}


def test_old_gym_api_is_normalized() -> None:
    env = _OldEnv()

    assert reset_env(env) == ("obs", {})
    assert step_env(env, [1]) == ("next", 2, True, False, {"action": [1]})


def test_game_spec_owns_integration_and_state_paths(tmp_path) -> None:
    game = GameSpec("Demo-Snes", tmp_path / "demo")

    assert (
        game.states_dir
        == (tmp_path / "demo" / "custom_integrations" / "Demo-Snes").resolve()
    )
    assert game.state_path("FirstAction") == game.states_dir / "FirstAction.state"


def test_state_bytes_accept_raw_and_gzip_files(tmp_path) -> None:
    raw_path = tmp_path / "raw.state"
    raw_path.write_bytes(b"raw-state")
    gzip_path = write_state_bytes(tmp_path / "gzip.state", b"gzip-state")

    assert read_state_bytes(raw_path) == b"raw-state"
    assert read_state_bytes(gzip_path) == b"gzip-state"
