"""Unit tests for bubble_save_practice.diagnose_trace (no emulator)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe"
    / "bubble_save_practice.py"
)


@pytest.fixture(scope="module")
def bsp():
    spec = importlib.util.spec_from_file_location("bubble_save_practice", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(
    frame: int,
    *,
    pose: int = 1,
    x: int = 50,
    y: int = 395,
    room: int = 0xACB3,
    buttons: list[str] | None = None,
) -> dict:
    return {
        "frame": frame,
        "x": x,
        "y": y,
        "room": room,
        "room_hex": f"0x{room:04X}",
        "pose": pose,
        "buttons": buttons or [],
    }


def test_empty(bsp) -> None:
    d = bsp.diagnose_trace([])
    assert d["ok"] is False
    assert d["grade"] == "EMPTY"


def test_still_in_save(bsp) -> None:
    rows = [
        _row(i, pose=1, x=100 + i, y=152, room=0xB0DD, buttons=["RIGHT"])
        for i in range(30)
    ]
    d = bsp.diagnose_trace(rows)
    assert d["grade"] == "RED"
    assert any("Save" in f or "0xB0DD" in f for f in d["failures"])


def test_early_walljump(bsp) -> None:
    rows: list[dict] = []
    # Approach with A pressed early
    for i in range(10):
        rows.append(
            _row(i, pose=25, x=200 + i, y=300, buttons=["RIGHT", "A"] if i >= 5 else ["RIGHT"])
        )
    # Latch without A (missed — A was early)
    for i in range(10, 18):
        rows.append(_row(i, pose=132, x=260, y=280, buttons=["LEFT"]))
    # Fall off
    for i in range(18, 30):
        rows.append(_row(i, pose=25, x=250, y=300 + (i - 18) * 5, buttons=[]))
    d = bsp.diagnose_trace(rows)
    assert d["n_early"] >= 1
    assert d["windows"][0]["grade"] == "EARLY"
    assert isinstance(d["windows"][0]["frames_off"], int)
    assert d["windows"][0]["frames_off"] < 0


def test_on_time_walljump(bsp) -> None:
    rows: list[dict] = []
    for i in range(5):
        rows.append(_row(i, pose=25, x=240 + i, y=290, buttons=["RIGHT", "B", "A"]))
    # Latch + A immediately
    for i in range(5, 12):
        btns = ["LEFT", "A"] if i <= 7 else ["RIGHT", "A"]
        rows.append(_row(i, pose=132 if i < 10 else 25, x=260, y=250 - (i - 5) * 8, buttons=btns))
    d = bsp.diagnose_trace(rows)
    assert d["n_on"] >= 1
    w = next(w for w in d["windows"] if w["grade"] == "ON")
    assert w["frames_off"] is not None
    assert 0 <= int(w["frames_off"]) <= bsp.IDEAL_MAX


def test_late_after_latch(bsp) -> None:
    rows: list[dict] = []
    for i in range(5):
        rows.append(_row(i, pose=25, x=250, y=300, buttons=["RIGHT"]))
    # Latch, no A
    for i in range(5, 12):
        rows.append(_row(i, pose=132, x=260, y=290, buttons=["LEFT"]))
    # Latch ends, then late A
    for i in range(12, 20):
        btns = ["A"] if i == 15 else []
        rows.append(_row(i, pose=25, x=255, y=310, buttons=btns))
    d = bsp.diagnose_trace(rows)
    assert d["n_late"] >= 1
    assert d["windows"][0]["grade"] == "LATE"
    assert int(d["windows"][0]["frames_off"]) > 0


def test_phase_d_green(bsp) -> None:
    rows: list[dict] = []
    for i in range(20):
        rows.append(
            _row(
                i,
                pose=132 if 5 <= i < 10 else 25,
                x=200 + i * 8,
                y=300 - i * 10,
                buttons=["LEFT", "A"] if 5 <= i < 8 else ["RIGHT", "A"],
            )
        )
    # Ensure we hit Phase D band
    rows.append(_row(20, pose=25, x=320, y=150, buttons=["RIGHT"]))
    d = bsp.diagnose_trace(rows)
    assert d["phase_d"] is True
    assert d["grade"] == "GREEN"
    assert d["ok"] is True


def test_format_diagnosis_includes_frames_off(bsp) -> None:
    rows = [
        _row(0, pose=132, x=260, y=280, buttons=["LEFT", "A"]),
        _row(1, pose=132, x=260, y=270, buttons=["LEFT", "A"]),
        _row(2, pose=25, x=270, y=250, buttons=["RIGHT", "A"]),
    ]
    d = bsp.diagnose_trace(rows)
    text = bsp.format_diagnosis(d, take="take01.json")
    assert "frames_off" in text
    assert "WJ1" in text or "latch" in text.lower()
