"""Pause-menu freeze cut (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.human_tape.cut_pause import (
    apply_kept_ranges,
    cut_pause_tape,
    find_phase_runs,
    find_trailing_idle,
    remap_frame,
    spans_to_kept_ranges,
)


def test_find_phase_and_remap() -> None:
    trace = []
    for i in range(100):
        phase = "pause_or_inventory" if 20 <= i < 50 else "ordinary_gameplay"
        trace.append({"frame": i, "phase": phase, "room_hex": "0x9CB3", "x": 1, "y": 2})
    spans = find_phase_runs(trace, min_frames=10)
    assert len(spans) == 1
    assert spans[0].start == 20 and spans[0].end == 49 and spans[0].frames == 30

    frames = [[0] * 12 for _ in range(100)]
    # last 15 idle after input at 84
    for i in range(85):
        frames[i][7] = 1  # RIGHT
    trail = find_trailing_idle(frames, trace, min_frames=10)
    assert trail is not None
    assert trail.start == 85

    kept = spans_to_kept_ranges(100, spans)
    assert remap_frame(19, kept) == 19
    assert remap_frame(20, kept) is None
    assert remap_frame(50, kept) == 20


def test_cut_pause_tape_in_place(tmp_path: Path) -> None:
    n = 80
    frames = [[0] * 12 for _ in range(n)]
    for i in range(n):
        if i < 70:
            frames[i][7] = 1
    trace = []
    for i in range(n):
        phase = "pause_or_inventory" if 30 <= i < 55 else "ordinary_gameplay"
        trace.append(
            {
                "frame": i,
                "phase": phase,
                "room": 0x9CB3,
                "room_hex": "0x9CB3",
                "x": 10,
                "y": 20,
                "items": 0x1004,
                "supers": 0 if i < 60 else 5,
                "buttons": ["RIGHT"] if i < 70 else [],
            }
        )
    task = tmp_path / "take.json"
    task.write_text(
        json.dumps(
            {
                "name": "take",
                "start_state": "scratch/x.state",
                "frames": frames,
                "trace": trace,
                "frame_count": n,
                "metadata": {"power_on": False},
            }
        ),
        encoding="utf-8",
    )
    # Anchors index
    (tmp_path / "take_anchors.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {"kind": "boot", "frame": 0, "room": "0x9CB3"},
                    {"kind": "room_enter", "frame": 40, "room": "0x9CB3"},  # inside pause
                    {"kind": "end", "frame": 79, "room": "0x9CB3"},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = cut_pause_tape(
        task,
        write=True,
        in_place=True,
        archive_first=False,
        materialize=False,
        min_pause_frames=10,
        min_trailing_idle=5,
    )
    assert report.cut_frames > 0
    data = json.loads(task.read_text())
    assert data["frame_count"] == report.frames_after
    assert len(data["frames"]) == report.frames_after
    # No pause phases left
    assert not any(r.get("phase") == "pause_or_inventory" for r in data["trace"])
    # Anchors: mid-pause dropped, boot+end remapped
    idx = json.loads((tmp_path / "take_anchors.json").read_text())
    kinds = [a["kind"] for a in idx["anchors"]]
    assert "boot" in kinds
    assert "end" in kinds
    assert "room_enter" not in kinds  # was inside pause
