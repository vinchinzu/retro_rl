"""Unit tests for YouTube reference helpers (no VOD required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "tools"
sys.path.insert(0, str(TOOLS))

import yt_ref_lib as lib  # noqa: E402


def test_video_id_from_url() -> None:
    assert lib.video_id_from_url("https://youtu.be/TFsGVxQReMw") == "TFsGVxQReMw"
    assert lib.video_id_from_url("https://youtu.be/TFsGVxQReMw?si=abc") == "TFsGVxQReMw"
    assert (
        lib.video_id_from_url("https://www.youtube.com/watch?v=TFsGVxQReMw&t=10")
        == "TFsGVxQReMw"
    )


def test_parse_time_token() -> None:
    assert lib.parse_time_token(90) == 90.0
    assert lib.parse_time_token("90") == 90.0
    assert lib.parse_time_token("1:30") == 90.0
    assert lib.parse_time_token("0:22:26") == 1346.0
    assert lib.parse_time_token("1h2m3s") == 3723.0
    assert lib.parse_time_token("2m3s") == 123.0
    with pytest.raises(ValueError):
        lib.parse_time_token("not-a-time")


def test_hold_intervals_and_duty() -> None:
    events = [
        {"vod_s": 1.0, "button": "Right", "edge": "down"},
        {"vod_s": 3.5, "button": "Right", "edge": "up"},
        {"vod_s": 2.0, "button": "B", "edge": "down"},
        {"vod_s": 4.0, "button": "B", "edge": "up"},
    ]
    holds = lib.hold_intervals(events)
    by_btn = {h["button"]: h for h in holds}
    assert by_btn["Right"]["dur_s"] == 2.5
    assert by_btn["B"]["dur_s"] == 2.0

    frames = [
        {"lit": ["Right", "B"]},
        {"lit": ["Right"]},
        {"lit": []},
        {"lit": ["B"]},
    ]
    duty = lib.duty_cycle(frames, ["Left", "Right", "B"])
    assert duty["Right"] == 0.5
    assert duty["B"] == 0.5


def test_default_ref_id() -> None:
    assert lib.DEFAULT_REF_ID == "TFsGVxQReMw"
    ws = lib.RefWorkspace.resolve(None)
    assert ws.video_id == "TFsGVxQReMw"
