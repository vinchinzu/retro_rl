"""Any% RTA clock from Ceres first control (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.human_tape.rta_clock import (
    CERES_ELEVATOR_ROOM,
    find_ceres_zero_frame,
    fmt_time,
    resolve_rta_clock,
)


def test_fmt_time() -> None:
    assert fmt_time(0) == "0:00.000"
    assert fmt_time(60) == "0:01.000"
    assert fmt_time(10830) == "3:00.500"


def test_find_ceres_zero_prefers_boot() -> None:
    anchors = [
        {"kind": "boot", "frame": 100, "room": "0x0000"},
        {"kind": "boot", "frame": 11483, "room": "0xDF45"},
        {"kind": "room_enter", "frame": 12000, "room": "0xDF45"},
    ]
    assert find_ceres_zero_frame(anchors) == 11483


def test_resolve_rta_clock_from_segments(tmp_path: Path) -> None:
    task = tmp_path / "full_start_v1.json"
    task.write_text(
        json.dumps(
            {
                "name": "full_start_v1",
                "start_state": "scratch/full_start_v1_morph.state",
                "frame_count": 100,
                "frames": [[0] * 12] * 100,
                "trace": [],
                "metadata": {
                    "power_on": False,
                    "end_fingerprint": {
                        "frame": 99,
                        "room": "0x9804",
                        "items": "0x1004",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Older junk segment (should be ignored once power_on exists)
    s0 = tmp_path / "full_start_v1_segments" / "s0"
    s0.mkdir(parents=True)
    (s0 / "join.json").write_text(
        json.dumps(
            {
                "power_on": False,
                "start_state": "scratch/full_start_v1_varia.state",
                "frame_count": 5000,
                "end_fingerprint": {"frame": 4999, "room": "0xACB3"},
            }
        ),
        encoding="utf-8",
    )
    # Power-on → morph
    s1 = tmp_path / "full_start_v1_segments" / "s1"
    s1.mkdir(parents=True)
    (s1 / "join.json").write_text(
        json.dumps(
            {
                "power_on": True,
                "start_state": "power_on",
                "frame_count": 25956,
                "end_fingerprint": {
                    "frame": 25955,
                    "room": "0x9E9F",
                    "items": "0x0004",
                },
            }
        ),
        encoding="utf-8",
    )
    (s1 / "anchors.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "kind": "boot",
                        "frame": 11483,
                        "room": f"0x{CERES_ELEVATOR_ROOM:04X}",
                        "items": "0x0000",
                    },
                    {
                        "kind": "item_delta",
                        "frame": 25342,
                        "room": "0x9E9F",
                        "items": "0x0004",
                    },
                    {
                        "kind": "end",
                        "frame": 25955,
                        "room": "0x9E9F",
                        "items": "0x0004",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    # Next session offset = power-on span only (live not included)
    clk = resolve_rta_clock(task, include_live_tape=False)
    assert clk.ceres_zero_local == 11483
    assert clk.offset_frames == 25955 - 11483
    assert clk.full_frames(10830) == (25955 - 11483) + 10830
    assert fmt_time(clk.offset_frames) == "4:01.200"

    # Post-hoc fold includes live morph→bombs tape
    clk2 = resolve_rta_clock(task, include_live_tape=True)
    assert clk2.offset_frames == (25955 - 11483) + 99


def test_resolve_rta_clock_dedupes_seam_retakes(tmp_path: Path) -> None:
    """Re-archiving the same bomb→supers pin must not triple-count RTA."""
    task = tmp_path / "full_start_v1.json"
    task.write_text(
        json.dumps(
            {
                "name": "full_start_v1",
                "start_state": "scratch/full_start_v1_supers.state",
                "frame_count": 100,
                "frames": [[0] * 12] * 100,
                "trace": [],
                "metadata": {
                    "power_on": False,
                    "end_fingerprint": {
                        "frame": 99,
                        "room": "0xA447",
                        "items": "0x1004",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    segs = tmp_path / "full_start_v1_segments"
    # power-on → morph
    s1 = segs / "s1"
    s1.mkdir(parents=True)
    (s1 / "join.json").write_text(
        json.dumps(
            {
                "power_on": True,
                "start_state": "power_on",
                "frame_count": 25956,
                "end_fingerprint": {
                    "frame": 25955,
                    "room": "0x9E9F",
                    "items": "0x0004",
                },
            }
        ),
        encoding="utf-8",
    )
    (s1 / "anchors.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "kind": "boot",
                        "frame": 11483,
                        "room": f"0x{CERES_ELEVATOR_ROOM:04X}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # morph → bombs
    s2 = segs / "s2"
    s2.mkdir(parents=True)
    (s2 / "join.json").write_text(
        json.dumps(
            {
                "power_on": False,
                "start_state": "scratch/full_start_v1_morph.state",
                "frame_count": 11336,
                "end_fingerprint": {
                    "frame": 11335,
                    "room": "0x9804",
                    "items": "0x1004",
                },
            }
        ),
        encoding="utf-8",
    )
    # Three identical bomb→supers retakes (same start pin + end room/items/len)
    for sid in (4, 5, 6):
        sdir = segs / f"s{sid}"
        sdir.mkdir(parents=True)
        (sdir / "join.json").write_text(
            json.dumps(
                {
                    "power_on": False,
                    "start_state": "scratch/full_start_v1_bomb.state",
                    "frame_count": 20507,
                    "end_fingerprint": {
                        "frame": 20506,
                        "room": "0x9B5B",
                        "room_id": 39771,
                        "items": "0x1004",
                        "cut_pause": True,
                    },
                }
            ),
            encoding="utf-8",
        )

    clk = resolve_rta_clock(task, include_live_tape=False)
    # s1 ceres span + s2 + one supers seam only
    expect = (25955 - 11483) + 11335 + 20506
    assert clk.offset_frames == expect
    assert any("retake" in n for n in clk.notes)

    clk_live = resolve_rta_clock(task, include_live_tape=True)
    assert clk_live.offset_frames == expect + 99
