"""Generic ffmpeg-backed recording helpers for oneshot SNES showcases."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from snes_oneshot.recording_footer import render_footer_frame


class FrameSink(Protocol):
    """Accept decorated RGB frames during a replay."""

    def write(self, frame: np.ndarray) -> None: ...


@dataclass(frozen=True)
class FooterLabels:
    """Three text fields rendered on the live recording banner."""

    upper_left: str
    upper_right: str
    lower_left: str


FooterProvider = Callable[[object, list[int], int, float], FooterLabels]


class FrameVideoWriter:
    """Small ffmpeg RGB pipe for silent emulator capture."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        scale: int = 1,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to record showcases")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.scale = max(1, scale)
        self.frames_written = 0
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width * self.scale}x{height * self.scale}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        """Append one HxWx3 RGB frame."""
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg input is closed")
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"unexpected frame shape: {rgb.shape}")
        if self.scale > 1:
            rgb = np.repeat(np.repeat(rgb, self.scale, axis=0), self.scale, axis=1)
        self._proc.stdin.write(rgb.tobytes())
        self.frames_written += 1

    def close(self) -> Path:
        """Finalize the MP4 and raise if ffmpeg failed."""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read() if self._proc.stderr is not None else b""
        code = self._proc.wait()
        if code:
            raise RuntimeError(stderr.decode("utf-8", errors="replace"))
        return self.path


class RecordingSession:
    """Wrap env stepping with footer decoration and optional frame stride."""

    def __init__(
        self,
        env: object,
        *,
        sink: FrameSink,
        footer: FooterProvider,
        fps: float,
        frame_stride: int = 1,
        players: int = 1,
        idle_action: Callable[[], list[int]] | None = None,
    ) -> None:
        self._env = env
        self._sink = sink
        self._footer = footer
        self._fps = fps
        self._frame_stride = max(1, frame_stride)
        self._players = players
        self._idle_action = idle_action
        self.frame = 0

    def capture(self, obs: np.ndarray, action: list[int]) -> None:
        """Decorate one emulator frame and optionally write it to the sink."""
        if self.frame % self._frame_stride != 0:
            return
        labels = self._footer(self._env, action, self.frame, self._fps)
        decorated = render_footer_frame(
            obs,
            upper_left=labels.upper_left,
            upper_right=labels.upper_right,
            lower_left=labels.lower_left,
            action=action,
            players=self._players,
        )
        self._sink.write(decorated)

    def step(self, action: list[int]) -> np.ndarray:
        """Step the emulator and capture the resulting frame."""
        obs, *_rest = self._env.step(action)  # type: ignore[attr-defined]
        rgb = np.asarray(obs)
        self.capture(rgb, action)
        self.frame += 1
        return rgb

    def _default_idle(self) -> list[int]:
        if self._idle_action is not None:
            return self._idle_action()
        from snes_oneshot.actions import idle_action

        return idle_action()

    def idle(self, frames: int) -> np.ndarray:
        """Hold idle input for several frames."""
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        for _ in range(max(frames, 1)):
            obs = self.step(self._default_idle())
        return obs

    def hold(self, action: list[int], frames: int) -> np.ndarray:
        """Repeat one action for several frames."""
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        for _ in range(max(frames, 1)):
            obs = self.step(action)
        return obs
