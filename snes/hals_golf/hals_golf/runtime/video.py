"""FFmpeg-backed frame recorder for golf autoplay / play sessions."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hals_golf.paths import PROJECT_DIR

RECORDINGS_DIR = PROJECT_DIR / "recordings"


def default_video_path(prefix: str = "clear") -> Path:
    """Return a timestamped path under ``hals_golf/recordings/``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RECORDINGS_DIR / f"{prefix}_{stamp}.mp4"


def resolve_video_path(value: str | None, *, prefix: str) -> Path | None:
    """Resolve CLI ``--video`` values.

    ``None`` disables recording. ``"AUTO"`` / empty uses a timestamped default.
    """
    if value is None:
        return None
    if value in {"", "AUTO", "auto", "1", "true", "True"}:
        return default_video_path(prefix)
    return Path(value).expanduser().resolve()


class FrameVideoWriter:
    """Pipe RGB frames into ffmpeg and optionally mux emulator audio."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int = 60,
        scale: int = 1,
        audio_rate: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if scale < 1:
            raise ValueError("scale must be >= 1")
        if audio_rate is not None and audio_rate <= 0:
            raise ValueError("audio_rate must be positive")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; required for --video")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.scale = scale
        self.out_width = width * scale
        self.out_height = height * scale
        self.frames_written = 0
        self.audio_rate = audio_rate
        self.audio_bytes_written = 0
        self._audio_handle = None
        self._audio_path: Path | None = None
        self._video_path = self.path

        if audio_rate is not None:
            video_temp = tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                prefix=f".{self.path.stem}.",
                suffix=f".video{self.path.suffix}",
                delete=False,
            )
            video_temp.close()
            self._video_path = Path(video_temp.name)
            audio_temp = tempfile.NamedTemporaryFile(
                dir=self.path.parent,
                prefix=f".{self.path.stem}.",
                suffix=".s16le",
                delete=False,
            )
            self._audio_handle = audio_temp
            self._audio_path = Path(audio_temp.name)

        suffix = self.path.suffix.lower()
        if suffix in {".ogv", ".ogg"}:
            video_output = [
                "-an",
                "-c:v",
                "libtheora",
                "-q:v",
                "7",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "ogg",
            ]
        else:
            video_output = [
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
                f"{self.out_width}x{self.out_height}",
                "-r",
                str(fps),
                "-i",
                "-",
                *video_output,
                str(self._video_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray, *, audio: Any = None) -> None:
        """Append one RGB frame (H×W×3) and its stereo s16le audio."""
        if self._proc.stdin is None:
            raise RuntimeError("video writer stdin is closed")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB frame, got {frame.shape}")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"frame size {frame.shape[1]}x{frame.shape[0]} != "
                f"{self.width}x{self.height}"
            )
        rgb = np.asarray(frame, dtype=np.uint8)
        if audio is not None:
            if self._audio_handle is None:
                raise RuntimeError("audio supplied without an audio_rate")
            audio_bytes = bytes(audio)
            self._audio_handle.write(audio_bytes)
            self.audio_bytes_written += len(audio_bytes)
        if self.scale > 1:
            rgb = np.repeat(np.repeat(rgb, self.scale, axis=0), self.scale, axis=1)
        self._proc.stdin.write(rgb.tobytes())
        self.frames_written += 1

    def close(self) -> Path:
        """Finalize the recording and return its path."""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr = b""
        if self._proc.stderr is not None:
            stderr = self._proc.stderr.read()
        code = self._proc.wait()
        if code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            self._cleanup_temporary_files()
            raise RuntimeError(
                f"ffmpeg failed ({code}) writing {self.path}: {detail}"
            )

        if self._audio_handle is not None:
            self._audio_handle.close()
            self._audio_handle = None
            try:
                if self.audio_bytes_written:
                    self._mux_audio()
                else:
                    shutil.move(self._video_path, self.path)
            finally:
                self._cleanup_temporary_files()
        return self.path

    def _mux_audio(self) -> None:
        """Mux the captured stereo s16le stream with the encoded video."""
        assert self.audio_rate is not None
        assert self._audio_path is not None
        suffix = self.path.suffix.lower()
        if suffix in {".ogv", ".ogg"}:
            audio_output = [
                "-c:v",
                "copy",
                "-c:a",
                "libvorbis",
                "-q:a",
                "5",
                "-f",
                "ogg",
            ]
        else:
            audio_output = [
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
            ]
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self._video_path),
                "-f",
                "s16le",
                "-ar",
                str(self.audio_rate),
                "-ac",
                "2",
                "-i",
                str(self._audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                *audio_output,
                str(self.path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg failed ({result.returncode}) muxing {self.path}: {detail}"
            )

    def _cleanup_temporary_files(self) -> None:
        for path in (self._video_path, self._audio_path):
            if path is not None and path != self.path:
                path.unlink(missing_ok=True)

    def __enter__(self) -> FrameVideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()
