"""Unit tests for hop-replay resolve/match/check (no emulator by default)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_metroid.human_tape import fingerprint, write_gzip_state
from super_metroid.human_tape_replay import (
    check_hop_green,
    frame_action,
    match_anchor,
    propose_trace_midpoints,
    resolve_anchor_path,
    resolve_hop_slice,
)


def _synthetic_task(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Small task: room A (0xDE4D) then B (0xDE7A) with live anchors."""
    room_a = 0xDE4D
    room_b = 0xDE7A
    n = 40
    frames = [[0] * 12 for _ in range(n)]
    # room A for frames 0..24, room B for 25..39
    trace = []
    for i in range(n):
        room = room_a if i < 25 else room_b
        x = 20 + i if room == room_a else 118
        y = 138 if room == room_a else 224
        if i == 24:
            x, y = 118, 224
        trace.append(
            {
                "frame": i,
                "room": room,
                "room_hex": f"0x{room:04X}",
                "x": x,
                "y": y,
                "pose": 2,
                "items": 0x732F,
                "energy": 499,
                "buttons": [],
            }
        )

    anchors_dir = tmp_path / "demo_anchors"
    anchors_dir.mkdir()
    # boot at 0, room_enter A at frame 5 (after hop start 0 — door settle)
    boot_path = anchors_dir / "f000000_boot_0xDE4D.state"
    enter_path = anchors_dir / "f000005_enter_0xDE4D_0xDE4D.state"
    write_gzip_state(boot_path, b"boot-state")
    write_gzip_state(enter_path, b"enter-state")
    anchors = [
        fingerprint(
            frame=0,
            room_id=room_a,
            x=20,
            y=138,
            pose=2,
            items=0x732F,
            kind="boot",
            path=str(boot_path),
        ),
        fingerprint(
            frame=5,
            room_id=room_a,
            x=25,
            y=138,
            pose=2,
            items=0x732F,
            kind="room_enter",
            path=str(enter_path),
        ),
    ]
    anchors_index = {
        "task": "demo",
        "anchors_dir": str(anchors_dir),
        "count": len(anchors),
        "anchors": anchors,
    }
    idx_path = tmp_path / "demo_anchors.json"
    idx_path.write_text(json.dumps(anchors_index, indent=2) + "\n", encoding="utf-8")

    task = {
        "name": "demo",
        "frames": frames,
        "trace": trace,
        "frame_count": n,
        "start_state": "scratch/x.state",
        "recorded_at": "t",
        "metadata": {},
    }
    # Name so load_anchors_index finds demo_anchors.json
    task_path = tmp_path / "demo.json"
    # anchors index must be stem_anchors.json → rename
    task_path = tmp_path / "demo.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    # extract_tape convention: <stem>_anchors.json
    (tmp_path / "demo_anchors.json").write_text(
        json.dumps(anchors_index, indent=2) + "\n", encoding="utf-8"
    )
    return task_path, enter_path, anchors_index


def test_match_anchor_prefers_room_enter_near_start(tmp_path: Path) -> None:
    _task, enter_path, idx = _synthetic_task(tmp_path)
    # Hop starts at 0; enter at 5 is after start but same room — preferred over nothing
    hit = match_anchor(idx, 0, 0xDE4D, task_path=_task)
    assert hit is not None
    assert hit["kind"] == "room_enter"
    assert hit["frame"] == 5
    assert Path(hit["path"]) == enter_path.resolve()

    # At-or-before: target 10 still prefers enter@5 over boot@0 (later + room_enter)
    hit2 = match_anchor(idx, 10, 0xDE4D, task_path=_task)
    assert hit2 is not None
    assert hit2["frame"] == 5


def test_match_anchor_at_or_before(tmp_path: Path) -> None:
    _task, _enter, idx = _synthetic_task(tmp_path)
    # enter@5 is still inside settle_window from target 3 → room_enter wins
    hit = match_anchor(idx, 3, 0xDE4D, task_path=_task, settle_window=256)
    assert hit is not None
    assert hit["kind"] == "room_enter"
    assert hit["frame"] == 5
    # With settle_window=0, only at-or-before remains → boot@0
    hit2 = match_anchor(idx, 3, 0xDE4D, task_path=_task, settle_window=0)
    assert hit2 is not None
    assert hit2["kind"] == "boot"
    assert hit2["frame"] == 0


def test_resolve_hop_slice_by_index(tmp_path: Path) -> None:
    task_path, enter_path, _idx = _synthetic_task(tmp_path)
    info = resolve_hop_slice(task_path, hop_index=0, leave_extra=1)
    assert info["start_room"] == 0xDE4D
    assert info["leave_room"] == 0xDE7A
    assert info["start_index"] == 0
    assert info["end_index"] == 25  # 24 + leave_extra 1
    assert info["end_xy"] == [118, 224]
    assert info["anchor_path"] is not None
    assert Path(info["anchor_path"]) == enter_path.resolve() or info["anchor_frame"] in (
        0,
        5,
    )
    # replay starts after anchor dump frame
    assert info["replay_start"] >= 0
    assert info["steps"] > 0


def test_resolve_hop_slice_defaults_to_settled(tmp_path: Path) -> None:
    """Settled start matches materialize body bounds (skip door_transition)."""
    room_a, room_b = 0xDE4D, 0xDE7A
    n = 30
    frames = [[0] * 12 for _ in range(n)]
    trace = []
    for i in range(n):
        if i < 10:
            room, door, phase = room_a, 0, "ordinary_gameplay"
        elif i < 14:
            room, door, phase = room_b, 1, "ROOM_TRANSITION"
        else:
            room, door, phase = room_b, 0, "ordinary_gameplay"
        trace.append(
            {
                "frame": i,
                "room": room,
                "x": 10 + i,
                "y": 100,
                "pose": 2,
                "items": 0x4,
                "door_transition": door,
                "phase": phase,
            }
        )
    task_path = tmp_path / "settle_demo.json"
    task_path.write_text(
        json.dumps(
            {
                "name": "settle_demo",
                "frames": frames,
                "trace": trace,
                "frame_count": n,
            }
        ),
        encoding="utf-8",
    )
    settled = resolve_hop_slice(task_path, hop_index=1, leave_extra=0, settle=True)
    raw = resolve_hop_slice(task_path, hop_index=1, leave_extra=0, settle=False)
    assert raw["start_index"] == 10  # room-change edge
    assert settled["start_index"] == 14  # first ordinary
    assert settled["hop"]["settled_entry"] is True
    assert settled["hop"]["raw_start_index"] == 10


def test_resolve_hop_slice_by_room(tmp_path: Path) -> None:
    task_path, _, _ = _synthetic_task(tmp_path)
    info = resolve_hop_slice(
        task_path, room=0xDE4D, to_room=0xDE7A, leave_extra=1
    )
    assert info["hop_index"] == 0
    assert info["leave_room"] == 0xDE7A


def test_resolve_hop_slice_from_frame(tmp_path: Path) -> None:
    task_path, _, _ = _synthetic_task(tmp_path)
    info = resolve_hop_slice(
        task_path, from_frame=5, frames_count=20, leave_extra=0
    )
    assert info["start_index"] == 5
    assert info["end_index"] == 24
    assert info["start_room"] == 0xDE4D


def test_resolve_anchor_path_basename_fallback(tmp_path: Path) -> None:
    task_path, enter_path, idx = _synthetic_task(tmp_path)
    # Break absolute path — only basename remains
    broken = dict(idx["anchors"][1])
    broken["path"] = f"/nonexistent/elsewhere/{enter_path.name}"
    resolved = resolve_anchor_path(broken, anchors_index=idx, task_path=task_path)
    assert resolved is not None
    assert resolved == enter_path.resolve()


def test_check_hop_green_room_and_xy() -> None:
    result = {
        "room_id": 0xDE7A,
        "xy": [120, 220],
        "pose": 41,
        "phase": "ORDINARY_GAMEPLAY",
        "game_state": 8,
    }
    ok = check_hop_green(result, 0xDE7A, [118, 224], xy_tol=24)
    assert ok["ok"] is True
    assert ok["room_ok"] is True
    assert ok["xy_ok"] is True

    bad_room = check_hop_green(result, 0xDE4D, [118, 224], xy_tol=24)
    assert bad_room["ok"] is False
    assert bad_room["room_ok"] is False

    bad_xy = check_hop_green(result, 0xDE7A, [0, 0], xy_tol=4)
    assert bad_xy["ok"] is False
    assert bad_xy["xy_ok"] is False


def test_check_hop_green_dual() -> None:
    good = {"room_id": 0xDE7A, "xy": [118, 224]}
    bad = {"room_id": 0xDE4D, "xy": [118, 224]}
    d = check_hop_green([good, good], 0xDE7A, [118, 224], dual=True)
    assert d["ok"] is True
    d2 = check_hop_green([good, bad], 0xDE7A, [118, 224], dual=True)
    assert d2["ok"] is False
    assert len(d2["runs"]) == 2


def test_frame_action_shape() -> None:
    a = frame_action([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert a.dtype == "int8"
    assert a.shape == (12,)


def test_propose_trace_midpoints_floor_and_combat() -> None:
    """Offline parser finds floor land + combat pose without emulator."""
    room = 0xDBCD
    trace = []
    # enter high y=139, drop to y=459, freeze pose 138, walk to leave
    for i in range(0, 40):
        y = 139 + min(i * 10, 320)  # climb toward 459
        pose = 2
        if i == 25:
            pose = 138
        if i > 25:
            pose = 2 if i % 2 == 0 else 10
        trace.append(
            {
                "frame": i,
                "room": room,
                "room_hex": f"0x{room:04X}",
                "x": 100 + i,
                "y": y,
                "pose": pose,
                "vy": 0 if y >= 450 else 2,
                "energy": 499 - (30 if i == 30 else 0),
                "buttons": [],
            }
        )
    # last row near leave
    trace[-1]["x"] = 140
    trace[-1]["y"] = 489
    # min_gap always enforced (no kind-based bypass); use small gap so
    # floor_land / combat_pose / pre_leave can all appear in this short hop.
    cands = propose_trace_midpoints(
        trace, 0, len(trace) - 1, end_xy=[140, 489], min_gap=1
    )
    kinds = {c["kind"] for c in cands}
    assert "floor_land" in kinds or "combat_pose" in kinds
    assert any(c["kind"] == "combat_pose" for c in cands)
    assert any(c["kind"] == "pre_leave" for c in cands)
    # indices sorted and spaced by min_gap
    idxs = [c["index"] for c in cands]
    assert idxs == sorted(idxs)
    for a, b in zip(idxs, idxs[1:]):
        assert b - a >= 1


def test_propose_trace_midpoints_empty() -> None:
    assert propose_trace_midpoints([], 0, 10) == []
    assert propose_trace_midpoints([{"room": 1, "x": 0, "y": 0, "pose": 1}], 5, 3) == []


def test_resolve_real_g4_metadata_if_present() -> None:
    """Optional: resolve Escape1 against real anchors JSON (no emulator)."""
    candidates = [
        Path("snes/super_metroid/tasks/g4_tourian_human_mb.json"),
        Path(
            "/home/v/01_projects/11_games/retro_rl/"
            "snes/super_metroid/tasks/g4_tourian_human_mb.json"
        ),
    ]
    task = next((p for p in candidates if p.is_file()), None)
    if task is None:
        pytest.skip("g4_tourian_human_mb.json not on disk (gitignored tasks/)")
    # Default settle=True matches materialize body / room_enter pin.
    info = resolve_hop_slice(task, hop_index=1, leave_extra=1)
    assert info["start_room"] == 0xDE4D
    assert info["leave_room"] == 0xDE7A
    assert info["start_index"] == 10924  # settled ordinary (was raw 10805)
    assert info["hop"]["raw_start_index"] == 10805
    assert info["end_index"] == 11354  # 11353 + 1
    assert info["end_xy"] == [118, 224]
    # Live enter dump at settled start
    assert info["anchor_frame"] == 10924
    assert info["anchor_path"] is not None
    assert Path(info["anchor_path"]).is_file()
    assert info["replay_start"] == 10925

    raw = resolve_hop_slice(task, hop_index=1, leave_extra=1, settle=False)
    assert raw["start_index"] == 10805
