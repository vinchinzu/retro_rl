"""Small ffmpeg-backed RGB video sink."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import BinaryIO

import numpy as np


class FrameVideoWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: int = 60,
        scale: int = 2,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-vf",
            f"scale=iw*{scale}:ih*{scale}:flags=neighbor",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg did not expose stdin")
        self._stdin: BinaryIO = self._process.stdin
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        self._stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._stdin.closed:
            return
        self._stdin.close()
        result = self._process.wait()
        if result:
            raise RuntimeError(f"ffmpeg exited with status {result}")

    def __enter__(self) -> FrameVideoWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

