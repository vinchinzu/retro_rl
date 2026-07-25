"""Tests for MP4 frame recording helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from hals_golf.runtime.video import (
    FrameVideoWriter,
    default_video_path,
    resolve_video_path,
)


def test_resolve_video_path_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hals_golf.runtime.video.RECORDINGS_DIR",
        tmp_path / "recordings",
    )
    path = resolve_video_path("AUTO", prefix="clear")
    assert path is not None
    assert path.parent == tmp_path / "recordings"
    assert path.name.startswith("clear_")
    assert path.suffix == ".mp4"
    assert resolve_video_path(None, prefix="clear") is None


def test_default_video_path_uses_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hals_golf.runtime.video.RECORDINGS_DIR",
        tmp_path / "recordings",
    )
    path = default_video_path("play")
    assert "play_" in path.name


def test_frame_video_writer_encodes_mp4(tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    height, width = 32, 48
    with FrameVideoWriter(out, width=width, height=height, fps=30, scale=2) as writer:
        for i in range(15):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 10
            frame[:, :, 1] = 40
            frame[:, :, 2] = 200 - i * 5
            writer.write(frame)
    assert out.exists()
    assert out.stat().st_size > 500
    assert writer.frames_written == 15


def test_frame_video_writer_encodes_ogv_with_audio(tmp_path: Path) -> None:
    out = tmp_path / "clip.ogv"
    height, width = 32, 48
    audio_rate = 32_000
    fps = 30
    samples_per_frame = audio_rate // fps
    audio = np.zeros((samples_per_frame, 2), dtype=np.int16)
    with FrameVideoWriter(
        out,
        width=width,
        height=height,
        fps=fps,
        audio_rate=audio_rate,
    ) as writer:
        for i in range(15):
            frame = np.full((height, width, 3), i * 10, dtype=np.uint8)
            writer.write(frame, audio=audio)

    codecs = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(out),
        ],
        text=True,
    ).splitlines()
    assert codecs == ["theora", "vorbis"]
    assert writer.audio_bytes_written > 0
