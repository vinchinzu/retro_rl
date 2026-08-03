"""Tests for shared human task recording helpers."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.task_recording import (
    RecordedTask,
    coalesce_action_runs,
    coalesce_windows,
    pressed_buttons,
    stasis_windows,
    summarize_position_trace,
)


def test_pressed_buttons_snes_default() -> None:
    action = [0] * 12
    action[0] = 1  # B
    action[7] = 1  # RIGHT
    assert pressed_buttons(action) == ["B", "RIGHT"]


def test_coalesce_windows_and_runs() -> None:
    assert coalesce_windows([1, 2, 3, 10, 11]) == [
        {"start": 1, "end": 3, "length": 3},
        {"start": 10, "end": 11, "length": 2},
    ]
    frames = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # RIGHT
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0] * 12,
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # B+RIGHT
    ]
    runs = coalesce_action_runs(frames)
    assert runs[0]["buttons"] == ["RIGHT"]
    assert runs[0]["length"] == 2
    assert runs[1]["buttons"] == ["B", "RIGHT"]


def test_stasis_and_summary() -> None:
    trace = []
    for i in range(50):
        trace.append(
            {
                "frame": i,
                "x": 100,
                "y": 200,
                "room": 0xA788,
                "buttons": ["RIGHT"],
            }
        )
    wins = stasis_windows(trace, min_length=40)
    assert len(wins) == 1
    assert wins[0]["length"] >= 40

    frames = [[0] * 12 for _ in range(50)]
    summary = summarize_position_trace(frames=frames, trace=trace, room_key="room")
    assert summary["frame_count"] == 50
    assert summary["transitions"][0]["room"] == 0xA788


def test_recorded_task_roundtrip(tmp_path: Path) -> None:
    task = RecordedTask(name="demo", start_state="foo")
    task.append_frame([0] * 12, trace_row={"frame": 0, "x": 1, "y": 2, "room": 1})
    task.append_frame([1] + [0] * 11, trace_row={"frame": 1, "x": 2, "y": 2, "room": 1})
    task.end_state_data = b"fake-state"
    path = tmp_path / "demo.json"
    task.save(path, end_state_paths=[tmp_path / "demo_end.state"])
    assert path.is_file()
    assert (tmp_path / "demo_end.state").is_file()
    loaded = RecordedTask.load(path)
    assert loaded.name == "demo"
    assert len(loaded.frames) == 2
    assert loaded.trace[1]["x"] == 2
    assert loaded.end_state_data == b"fake-state"
    data = json.loads(path.read_text())
    assert data["frame_count"] == 2
