"""Shared recording env proxy for SMZ3 probe scripts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from retro_harness.env import reset_obs
from retro_harness.video import FrameVideoWriter


class RecordingEnv:
    """Proxy that writes stepped RGB frames to an MP4 sink."""

    def __init__(
        self,
        env: Any,
        writer: FrameVideoWriter,
        *,
        frame_stride: int = 2,
    ) -> None:
        self._env = env
        self._writer = writer
        self._stride = max(1, frame_stride)
        self.frames_seen = 0
        self.last_obs: np.ndarray | None = None

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        obs, info = reset_obs(self._env)
        self._maybe_write(obs)
        return obs, info

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        self._maybe_write(result[0])
        return result

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()

    def get_ram(self) -> Any:
        return self._env.get_ram()

    @property
    def em(self) -> Any:
        return self._env.em

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _maybe_write(self, obs: Any) -> None:
        rgb = np.asarray(obs)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            rendered = self._env.render()
            if rendered is None:
                return
            rgb = np.asarray(rendered)
        self.last_obs = rgb
        if self.frames_seen % self._stride == 0:
            self._writer.write(rgb)
        self.frames_seen += 1


def configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def probe_frame_size(env: Any) -> tuple[int, int]:
    """Return (height, width) from a fresh env reset observation."""
    probe_obs, _ = env.reset()
    rgb = np.asarray(probe_obs)
    if rgb.ndim != 3:
        rgb = np.asarray(env.render())
    return int(rgb.shape[0]), int(rgb.shape[1])


def wrap_recording(
    env: Any,
    writer: FrameVideoWriter | None,
    *,
    frame_stride: int = 2,
) -> Any:
    if writer is None:
        return env
    return RecordingEnv(env, writer, frame_stride=frame_stride)
