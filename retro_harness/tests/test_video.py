"""Tests for the shared high-level video recorder."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from retro_harness.actions import buttons
from retro_harness.video import (
    FrameVideoWriter,
    VideoCaptureConfig,
    VideoRecorder,
    format_snes_buttons,
    probe_video_evidence,
    render_button_footer,
    should_capture_frame,
)


def test_should_capture_immediate() -> None:
    config = VideoCaptureConfig(audio=False, footer=False)
    write, started = should_capture_frame(
        frame=0, room_id=1, config=config, recording_started=False
    )
    assert write and started


def test_should_capture_frame_gate() -> None:
    config = VideoCaptureConfig(start_frame=10, audio=False, footer=False)
    write, started = should_capture_frame(
        frame=9, room_id=None, config=config, recording_started=False
    )
    assert not write and not started
    write, started = should_capture_frame(
        frame=10, room_id=None, config=config, recording_started=False
    )
    assert write and started
    write, started = should_capture_frame(
        frame=11, room_id=None, config=config, recording_started=True
    )
    assert write and started


def test_should_capture_room_latch() -> None:
    config = VideoCaptureConfig(start_room_id=0x91F8, audio=False, footer=False)
    write, started = should_capture_frame(
        frame=100, room_id=0xDF45, config=config, recording_started=False
    )
    assert not write and not started
    write, started = should_capture_frame(
        frame=200, room_id=0x91F8, config=config, recording_started=False
    )
    assert write and started
    # Stay open after leaving the trigger room.
    write, started = should_capture_frame(
        frame=300, room_id=0x92FD, config=config, recording_started=True
    )
    assert write and started


def test_format_snes_buttons() -> None:
    action = buttons("RIGHT", "B", "A")
    label = format_snes_buttons(action)
    assert "RIGHT" in label
    assert "B" in label
    assert "A" in label
    assert format_snes_buttons(None) == "---"


def test_render_button_footer_grows_height() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    action = buttons("LEFT")
    out = render_button_footer(frame, action=action, frame=12, fps=60)
    assert out.shape == (32 + 16, 48, 3)


def test_frame_video_writer_encodes_mp4(tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    height, width = 32, 48
    with FrameVideoWriter(
        out,
        width=width,
        height=height,
        fps=30,
        scale=2,
        crf=23,
        preset="ultrafast",
        footer=True,
    ) as writer:
        for i in range(12):
            frame = np.full((height, width, 3), i * 10, dtype=np.uint8)
            writer.write(frame, action=buttons("A"), frame_index=i)
    assert out.exists()
    assert out.stat().st_size > 400
    assert writer.frames == 12
    # Footer extends height before scale: (32+16)*2 = 96
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(out),
        ],
        text=True,
    ).strip()
    assert probe == "96,96"


def test_video_recorder_start_gate_and_audio(tmp_path: Path) -> None:
    out = tmp_path / "gated.mp4"
    height, width = 24, 32
    config = VideoCaptureConfig(
        fps=30,
        scale=1,
        crf=28,
        preset="ultrafast",
        audio=True,
        footer=False,
        start_frame=5,
    )
    audio_rate = 3000
    samples = audio_rate // config.fps
    with VideoRecorder(
        out,
        width=width,
        height=height,
        config=config,
        audio_rate=audio_rate,
    ) as rec:
        for i in range(12):
            frame = np.full((height, width, 3), i * 8, dtype=np.uint8)
            tone = np.zeros((samples, 2), dtype=np.int16)
            tone[:, 0] = 1000
            rec.write(frame, audio=tone, frame_index=i)
    # Frames 0..4 gated out → 7 written (5..11)
    assert rec.frames == 7
    codecs = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "csv=p=0",
            str(out),
        ],
        text=True,
    ).splitlines()
    # csv=p=0 with codec_name,codec_type → "h264,video" / "aac,audio"
    assert any(line.endswith(",video") for line in codecs)
    assert any(line.endswith(",audio") for line in codecs)
    assert probe_video_evidence(out, expected_frames=7)["frame_count_matches"] is True


def test_high_quality_preset() -> None:
    cfg = VideoCaptureConfig.high_quality(start_room_id=1)
    assert cfg.scale == 3
    assert cfg.crf == 15
    assert cfg.preset == "slow"
    assert cfg.start_room_id == 1
