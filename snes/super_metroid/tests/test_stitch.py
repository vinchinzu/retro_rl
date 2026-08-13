"""Multi-session anchor stitch + PB table (no ROM)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from super_metroid.human_tape.stitch import (
    find_join,
    fmt_time,
    format_pb_table,
    orphan_prefix_anchors,
    sessions_from_boots,
    stitch_task_anchors,
)


def _write_state(path: Path, mtime: float) -> None:
    path.write_bytes(b"not-a-real-state")
    os.utime(path, (mtime, mtime))


def test_orphan_prefix_and_join(tmp_path: Path) -> None:
    anchors = tmp_path / "take_anchors"
    anchors.mkdir()
    base = 1_700_000_000.0
    # Session 0: power-on → Big Pink
    for i, name in enumerate(
        (
            "f011382_boot_0xDF45.state",
            "f020503_enter_0x91F8_0x91F8.state",
            "f025026_items_0x0004_0x9E9F.state",
            "f049424_enter_0x9D19_0x9D19.state",
        )
    ):
        _write_state(anchors / name, base + i)

    # Session 1 (current take index): Big Pink → Varia
    take_files = [
        ("f000000_boot_0x9D19.state", 0, "boot", 0x9D19, "0x1004"),
        ("f001000_items_0x1105_0xA6E2.state", 1000, "item_delta", 0xA6E2, "0x1105"),
        ("f001100_end_0xA6E2.state", 1100, "end", 0xA6E2, "0x1105"),
    ]
    take_index = {
        "task": "take",
        "anchors_dir": str(anchors),
        "anchors": [],
    }
    for j, (name, frame, kind, room, items) in enumerate(take_files):
        path = anchors / name
        _write_state(path, base + 100 + j)
        take_index["anchors"].append(
            {
                "kind": kind,
                "frame": frame,
                "room": f"0x{room:04X}",
                "room_id": room,
                "items": items,
                "path": str(path),
            }
        )

    task = {
        "name": "take",
        "frames": [[0] * 12 for _ in range(1101)],
        "trace": [],
        "frame_count": 1101,
        "start_state": "scratch/x.state",
        "metadata": {},
    }
    task_path = tmp_path / "take.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    (tmp_path / "take_anchors.json").write_text(
        json.dumps(take_index), encoding="utf-8"
    )

    orphans = orphan_prefix_anchors(anchors, take_index["anchors"])
    assert len(orphans) == 4
    join_room, join_frame, _ = find_join(orphans, take_index["anchors"])
    assert join_room == 0x9D19
    assert join_frame == 49424

    report = stitch_task_anchors(task_path)
    assert report.join_frame == 49424
    assert report.total_frames == 49424 + 1100  # end event
    items = [m for m in report.milestones if m["kind"] == "item"]
    assert any(m["frame"] == 49424 + 1000 for m in items)
    table = format_pb_table(report)
    assert "FULL RUN PB TABLE" in table
    assert fmt_time(49424 + 1000) in table


def test_three_sessions_do_not_interleave(tmp_path: Path) -> None:
    """Power-on → Pink, Pink → Varia, Varia → Bubble chain without frame soup."""
    anchors = tmp_path / "full_anchors"
    anchors.mkdir()
    base = 1_700_000_000.0

    s0 = [
        ("f011382_boot_0xDF45.state", 11382, "boot", 0xDF45, None),
        ("f025026_items_0x0004_0x9E9F.state", 25026, "items", 0x9E9F, 0x0004),
        ("f049424_enter_0x9D19_0x9D19.state", 49424, "enter", 0x9D19, None),
    ]
    s1 = [
        ("f000000_boot_0x9D19.state", 0, "boot", 0x9D19, None),
        ("f038638_items_0x1105_0xA6E2.state", 38638, "items", 0xA6E2, 0x1105),
        ("f039252_end_0xA6E2.state", 39252, "end", 0xA6E2, None),
    ]
    s2 = [
        ("f000000_boot_0xA6E2.state", 0, "boot", 0xA6E2, None),
        ("f010405_enter_0xACB3_0xACB3.state", 10405, "enter", 0xACB3, None),
        ("f013091_end_0xACB3.state", 13091, "end", 0xACB3, None),
    ]

    t = base
    for group in (s0, s1, s2):
        for name, _fr, _k, _r, _i in group:
            _write_state(anchors / name, t)
            t += 1

    # Index is latest take only (Varia → Bubble), with items filled.
    take_index = {
        "task": "full",
        "anchors_dir": str(anchors),
        "anchors": [
            {
                "kind": "boot",
                "frame": 0,
                "room": "0xA6E2",
                "room_id": 0xA6E2,
                "items": "0x1105",
                "path": str(anchors / "f000000_boot_0xA6E2.state"),
            },
            {
                "kind": "room_enter",
                "frame": 10405,
                "room": "0xACB3",
                "room_id": 0xACB3,
                "items": "0x1105",
                "path": str(anchors / "f010405_enter_0xACB3_0xACB3.state"),
            },
            {
                "kind": "end",
                "frame": 13091,
                "room": "0xACB3",
                "room_id": 0xACB3,
                "items": "0x1105",
                "path": str(anchors / "f013091_end_0xACB3.state"),
            },
        ],
    }
    task_path = tmp_path / "full.json"
    task_path.write_text(
        json.dumps(
            {
                "name": "full",
                "frames": [],
                "trace": [],
                "frame_count": 13092,
                "start_state": "scratch/x.state",
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "full_anchors.json").write_text(
        json.dumps(take_index), encoding="utf-8"
    )

    disk = []
    for p in anchors.glob("f*.state"):
        from super_metroid.human_tape.stitch import list_anchor_state_files

        break
    rows = list_anchor_state_files(anchors)
    sessions = sessions_from_boots(rows)
    assert len(sessions) == 3

    report = stitch_task_anchors(task_path)
    # join S0→S1 at pink 49424; S1→S2 at varia item 38638
    # Bubble end full = 49424 + 38638 + 13091
    assert report.total_frames == 49424 + 38638 + 13091

    morph = next(m for m in report.milestones if m["label"] == "Morph")
    varia = next(
        m
        for m in report.milestones
        if m["kind"] == "item" and m.get("items") == "0x1105"
    )
    assert morph["frame"] < varia["frame"]
    assert varia["frame"] == 49424 + 38638
    assert any(
        m.get("items") == "0x1105"
        for m in report.milestones
        if m["kind"] == "end"
    )
    assert any(n.startswith("sessions=3") for n in report.notes)
    assert any(n.startswith("s0:") for n in report.notes)


def test_fmt_time() -> None:
    assert fmt_time(0) == "0:00.000"
    assert fmt_time(60) == "0:01.000"
    assert fmt_time(38638).startswith("10:")
