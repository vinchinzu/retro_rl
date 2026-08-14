"""Fake-env tests for retro_harness.boot_probe (no ROM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from retro_harness.boot_probe import BootProbeConfig, run_boot_probe
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from retro_harness.ram_state import GameMode, GameState


class _FakeEnv:
    def __init__(self, *, mutate_ram: bool = False) -> None:
        self.ram = np.zeros(8, dtype=np.uint8)
        self.mutate_ram = mutate_ram
        self.actions: list[object] = []
        self.closed = False
        self.obs = np.full((4, 4, 3), 80, dtype=np.uint8)

    def reset(self):
        return self.obs, {}

    def step(self, action):
        self.actions.append(action)
        if self.mutate_ram:
            self.ram = self.ram.copy()
            self.ram[0] = (int(self.ram[0]) + 1) % 256
        return self.obs, 0.0, False, False, {}

    def get_ram(self):
        return self.ram

    def close(self) -> None:
        self.closed = True


def _script(n: int = 8):
    idle = nes_idle_action()

    def frames():
        for _ in range(n):
            yield FrameAction(action=idle, reason="boot")

    return frames


def _parse_playing(ram, frame: int = 0, obs_mean: float | None = None) -> GameState:
    return GameState(frame=frame, mode=GameMode.PLAYING)


def _cfg(
    tmp_path: Path,
    *,
    game: str = "Fake-Nes",
    is_ready=None,
    parse_state=_parse_playing,
    script=None,
    **kwargs,
) -> BootProbeConfig:
    return BootProbeConfig(
        game=game,
        game_dir=tmp_path,
        recordings_dir=tmp_path,
        script=script or _script(),
        parse_state=parse_state,
        is_ready=is_ready,
        **kwargs,
    )


def test_ready_after_min_frame_and_stable_exits_0(tmp_path: Path) -> None:
    env = _FakeEnv()
    cfg = _cfg(
        tmp_path,
        is_ready=lambda ram, mean: True,
        min_frame=2,
        stable_frames=2,
        walk_frames=0,
    )
    assert run_boot_probe(cfg, save=False, env=env) == 0
    assert env.closed
    assert (tmp_path / "boot_level1.png").is_file()


def test_never_ready_exits_1(tmp_path: Path) -> None:
    env = _FakeEnv()
    cfg = _cfg(
        tmp_path,
        is_ready=lambda ram, mean: False,
        min_frame=1,
        stable_frames=2,
        walk_frames=0,
    )
    assert run_boot_probe(cfg, save=False, env=env) == 1
    assert env.closed


def test_motion_check_fails_on_frozen_ram(tmp_path: Path) -> None:
    env = _FakeEnv(mutate_ram=False)
    cfg = _cfg(
        tmp_path,
        is_ready=lambda ram, mean: True,
        min_frame=0,
        stable_frames=1,
        motion_check=True,
        walk_frames=0,
    )
    assert run_boot_probe(cfg, save=False, env=env) == 1
    assert len(env.actions) == 1 + 45  # one ready frame + motion hold


def test_scripted_playing_after_full_script_exits_0(tmp_path: Path) -> None:
    env = _FakeEnv()
    cfg = _cfg(
        tmp_path,
        game="Fake-Snes",
        is_ready=None,
        parse_state=_parse_playing,
        script=_script(4),
    )
    assert run_boot_probe(cfg, save=False, env=env) == 0
    assert len(env.actions) == 4


def test_post_script_frames_issue_extra_actions(tmp_path: Path) -> None:
    env = _FakeEnv()
    cfg = _cfg(
        tmp_path,
        game="Fake-Snes",
        is_ready=None,
        parse_state=_parse_playing,
        script=_script(2),
        post_script_button="RIGHT",
        post_script_frames=3,
    )
    assert run_boot_probe(cfg, save=False, env=env) == 0
    assert len(env.actions) == 5
