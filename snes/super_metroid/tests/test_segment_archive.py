"""Immutable segment archive when reusing guided_human --name."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.human_tape.segment_archive import (
    archive_existing_take,
    list_archived_tapes,
    list_segment_ids,
    next_segment_id,
    segments_dir_for,
)


def _write_take(path: Path, *, frames: int = 10, items: str = "0x0000") -> None:
    task = {
        "name": path.stem,
        "frame_count": frames,
        "frames": [[0] * 12 for _ in range(frames)],
        "trace": [{"frame": i, "room": 0x91F8} for i in range(frames)],
        "start_state": "power_on",
        "metadata": {
            "power_on": True,
            "end_fingerprint": {
                "kind": "end",
                "frame": frames - 1,
                "room": "0x91F8",
                "items": items,
            },
        },
    }
    path.write_text(json.dumps(task), encoding="utf-8")
    anchors = {
        "task": path.stem,
        "anchors": [
            {
                "kind": "boot",
                "frame": 0,
                "room": "0x91F8",
                "path": str(path.parent / "boot.state"),
            }
        ],
    }
    path.with_name(path.stem + "_anchors.json").write_text(
        json.dumps(anchors), encoding="utf-8"
    )


def test_archive_existing_take_creates_segment(tmp_path: Path) -> None:
    task = tmp_path / "full_start_v1.json"
    _write_take(task, frames=50, items="0x1004")
    hops_dir = tmp_path / "full_start_v1_hops"
    hops_dir.mkdir()
    (hops_dir / "hop_00_Landing.json").write_text(
        json.dumps({"frames": [[0] * 12], "frame_count": 1, "meta": {}}),
        encoding="utf-8",
    )

    dest = archive_existing_take(task)
    assert dest is not None
    assert dest.name == "s0"
    assert (dest / "tape.json").is_file()
    assert (dest / "anchors.json").is_file()
    assert (dest / "join.json").is_file()
    assert (dest / "hops" / "hop_00_Landing.json").is_file()
    join = json.loads((dest / "join.json").read_text(encoding="utf-8"))
    assert join["segment_id"] == 0
    assert join["frame_count"] == 50
    assert join["hop_bodies"] == 1
    assert join["end_fingerprint"]["items"] == "0x1004"

    reg = json.loads((segments_dir_for(task) / "registry.json").read_text())
    assert len(reg["segments"]) == 1
    assert reg["segments"][0]["id"] == 0
    assert reg["segments"][0]["hop_bodies"] == 1

    # Live task still present (caller overwrites later)
    assert task.is_file()
    assert next_segment_id(segments_dir_for(task)) == 1


def test_archive_skips_empty(tmp_path: Path) -> None:
    task = tmp_path / "empty.json"
    task.write_text(json.dumps({"name": "empty", "frames": []}), encoding="utf-8")
    assert archive_existing_take(task) is None


def test_archive_second_segment(tmp_path: Path) -> None:
    task = tmp_path / "run.json"
    _write_take(task, frames=20)
    assert archive_existing_take(task) is not None
    _write_take(task, frames=30, items="0x1105")
    dest2 = archive_existing_take(task)
    assert dest2 is not None
    assert dest2.name == "s1"
    assert list_segment_ids(segments_dir_for(task)) == [0, 1]
    rows = list_archived_tapes(task)
    assert len(rows) == 2
    assert rows[1]["frame_count"] == 30
