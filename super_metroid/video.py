"""Small ffmpeg-backed RGB video sink and helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import BinaryIO, Sequence

import numpy as np


def concat_videos(
    parts: Sequence[str | Path],
    output: str | Path,
    *,
    reencode: bool = False,
) -> dict[str, object]:
    """Concatenate H.264 clips with ffmpeg (stream-copy by default).

    All parts should share resolution/fps when ``reencode`` is false. Returns a
    small report dict; raises ``RuntimeError`` on ffmpeg failure.
    """
    paths = [Path(p).resolve() for p in parts]
    if not paths:
        raise ValueError("concat_videos requires at least one part")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing video part: {path}")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as list_file:
        for path in paths:
            # ffmpeg concat demuxer needs escaped single quotes in paths.
            escaped = str(path).replace("'", r"'\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = Path(list_file.name)

    try:
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if reencode:
            command.extend(
                [
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
        else:
            command.extend(["-c", "copy"])
        command.append(str(out))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"ffmpeg concat failed ({result.returncode}): {stderr or 'no stderr'}"
            )
    finally:
        list_path.unlink(missing_ok=True)

    return {
        "path": str(out),
        "parts": [str(p) for p in paths],
        "reencode": reencode,
        "bytes": out.stat().st_size if out.is_file() else 0,
    }


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

