"""Unit tests for human recording skill chunk extraction."""

from __future__ import annotations

from smb.scripts.parse_human_recording import (
    extract_jump_skills,
    extract_stage_skills,
    parse_recording,
)


def _row(i: int, *, stage: str, x: int, y: int, in_air: bool, world: int = 3, level: int = 0):
    return {
        "frame": i,
        "world": world,
        "level": level,
        "stage": stage,
        "x": x,
        "y": y,
        "xs": 40,
        "ys": -2 if in_air else 0,
        "player_state": 8,
        "oper_mode": 1,
        "timer": 400,
        "lives": 2,
        "area_pointer": 194,
        "in_air": in_air,
    }


def test_extract_jump_and_stage_skills():
    frames: list[list[int]] = []
    trace: list[dict] = []
    for i in range(10):
        frames.append([1, 0, 0, 0, 0, 0, 0, 1, 0])
        trace.append(_row(i, stage="4-1", x=40 + i * 2, y=176, in_air=False))
    for i in range(10, 25):
        frames.append([1, 0, 0, 0, 0, 0, 0, 1, 1 if i < 18 else 0])
        trace.append(
            _row(i, stage="4-1", x=40 + i * 2, y=160, in_air=True)
        )
    for i in range(25, 30):
        frames.append([1, 0, 0, 0, 0, 0, 0, 1, 0])
        trace.append(_row(i, stage="4-1", x=40 + i * 2, y=176, in_air=False))
    for i in range(30, 35):
        frames.append([0] * 9)
        trace.append(
            _row(i, stage="4-2", x=40, y=176, in_air=False, world=3, level=1)
        )

    jumps = extract_jump_skills(frames, trace, min_air=4)
    assert len(jumps) == 1
    assert jumps[0]["takeoff"] == 10
    assert jumps[0]["land"] == 25
    assert jumps[0]["hillclimb_window"]["start"] >= 0
    assert "segments" in jumps[0]

    stages = extract_stage_skills(frames, trace)
    assert len(stages) == 2
    assert stages[0]["stage"] == "4-1"
    assert stages[1]["stage"] == "4-2"


def test_parse_recording_roundtrip(tmp_path):
    frames = [[1, 0, 0, 0, 0, 0, 0, 1, 1]] * 20
    trace = [
        _row(i, stage="4-1", x=100 + i, y=170 if 5 <= i < 15 else 176, in_air=5 <= i < 15)
        for i in range(20)
    ]
    # force A rising at 5
    for i in range(20):
        frames[i] = [1, 0, 0, 0, 0, 0, 0, 1, 1 if 5 <= i < 12 else 0]
    path = tmp_path / "rec.json"
    path.write_text(
        __import__("json").dumps(
            {
                "format": "smb_human_nes9",
                "name": "unit",
                "frames": frames,
                "trace": trace,
                "handoff": "4-1",
            }
        ),
        encoding="utf-8",
    )
    report = parse_recording(path)
    assert report["total_frames"] == 20
    assert report["counts"]["jump"] >= 1
