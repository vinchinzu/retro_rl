"""Unit tests for human tape anchors + hop extract (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.human_tape import (
    AnchorRecorder,
    build_room_hops,
    default_skill_groups,
    extract_tape,
    fingerprint,
    fingerprint_from_trace_row,
    hop_items_int,
    settle_room_hops,
    verify_end_against_trace,
    write_gzip_state,
)


def test_fingerprint_and_verify() -> None:
    fp = fingerprint(
        frame=100,
        room_id=0xCFC9,
        x=317,
        y=1963,
        pose=2,
        items=0x7125,
        beams=0x1007,
        kind="end",
    )
    assert fp["room"] == "0xCFC9"
    assert fp["grapple"] is True
    assert fp["gravity"] is True
    trace = [
        {
            "frame": 100,
            "room": 0xCFC9,
            "x": 317,
            "y": 1963,
            "pose": 2,
            "items": 0x7125,
        }
    ]
    assert verify_end_against_trace(fp, trace)["ok"] is True
    bad = dict(fp)
    bad["xy"] = [0, 0]
    assert verify_end_against_trace(bad, trace)["ok"] is False


def test_build_room_hops_and_skills() -> None:
    trace = []
    # two rooms
    for i in range(10):
        trace.append(
            {
                "frame": i,
                "room": 0xA322,
                "x": 70 + i,
                "y": 1419,
                "pose": 2,
                "items": 0x3125,
            }
        )
    for i in range(10, 25):
        trace.append(
            {
                "frame": i,
                "room": 0xA2F7,
                "x": 18,
                "y": 1419,
                "pose": 10,
                "items": 0x3125,
            }
        )
    hops = build_room_hops(trace, room_names={0xA322: "Caterpillar", 0xA2F7: "Hellway"})
    assert len(hops) == 2
    assert hops[0]["name"] == "Caterpillar"
    assert hops[0]["dwell"] == 10
    assert hops[1]["room"] == "0xA2F7"
    assert hops[1]["end_frame"] == 24
    skills = default_skill_groups(hops)
    assert len(skills) >= 1


def test_settle_room_hops_moves_start_past_door_transition() -> None:
    """Settle clock matches room_enter: skip leading door_transition frames."""
    trace: list[dict] = []
    # Room A: already ordinary (boot-like)
    for i in range(5):
        trace.append(
            {
                "frame": i,
                "room": 0xA322,
                "x": 70 + i,
                "y": 1419,
                "pose": 2,
                "items": 0x3125,
                "door_transition": 0,
                "phase": "ORDINARY_GAMEPLAY",
            }
        )
    # Room B: first 4 frames still door_transition, then ordinary
    transition_n = 4
    for i in range(5, 5 + transition_n):
        trace.append(
            {
                "frame": i,
                "room": 0xA2F7,
                "x": 10,
                "y": 1419,
                "pose": 0,
                "items": 0x3125,
                "door_transition": 1,
                "phase": "ROOM_TRANSITION",
            }
        )
    settled_start = 5 + transition_n  # index 9
    for i in range(settled_start, 20):
        trace.append(
            {
                "frame": i,
                "room": 0xA2F7,
                "x": 18 + (i - settled_start),
                "y": 1419,
                "pose": 10,
                "items": 0x3125,
                "energy": 299,
                "door_transition": 0,
                "phase": "ordinary_gameplay",
            }
        )

    raw = build_room_hops(
        trace, room_names={0xA322: "Caterpillar", 0xA2F7: "Hellway"}
    )
    assert len(raw) == 2
    assert raw[1]["start_index"] == 5  # room-change leading edge
    assert raw[1]["dwell"] == 15
    raw_start_b = raw[1]["start_index"]
    raw_xy_b = list(raw[1]["xy"])

    settled = settle_room_hops(raw, trace)
    # Raw hops must be unchanged (settle deep-copies).
    assert raw[1]["start_index"] == raw_start_b
    assert raw[1]["xy"] == raw_xy_b
    assert "settled_entry" not in raw[1]
    assert "raw_start_index" not in raw[1]

    # Boot hop already ordinary: leave indices, mark settled.
    assert settled[0]["start_index"] == 0
    assert settled[0]["raw_start_index"] == 0
    assert settled[0]["settled_entry"] is True
    assert settled[0]["transition_frames"] == 0
    assert settled[0]["dwell"] == 5

    # Room B: start moves past transition frames.
    assert settled[1]["raw_start_index"] == 5
    assert settled[1]["start_index"] == settled_start
    assert settled[1]["frame"] == settled_start
    assert settled[1]["transition_frames"] == transition_n
    assert settled[1]["settled_entry"] is True
    assert settled[1]["dwell"] == 20 - settled_start  # end_index 19
    assert settled[1]["xy"] == [18, 1419]
    assert settled[1]["pose"] == 10
    assert settled[1]["end_index"] == raw[1]["end_index"]  # end span unchanged


def test_hop_items_int_parses_hex_and_int() -> None:
    assert hop_items_int({"items": 0x3125}) == 0x3125
    assert hop_items_int({"items": "0x7125"}) == 0x7125
    assert hop_items_int({"end_items": "0xABCD"}) == 0xABCD
    assert hop_items_int({"items": None, "end_items": 99}) == 99
    assert hop_items_int({"items": "0x7125"}, key="end_items") is None
    assert hop_items_int({}) is None


def test_anchor_recorder_writes_gzip(tmp_path: Path) -> None:
    class _Em:
        def get_state(self) -> bytes:
            return b"state-bytes-here"

    class _Env:
        em = _Em()

    class _Phase:
        name = "ORDINARY_GAMEPLAY"
        value = "ordinary_gameplay"

    class _DoorPhase:
        name = "DOOR_TRANSITION"
        value = "door_transition"

    class _St:
        room_id = 0xA322
        samus_x = 70
        samus_y = 1419
        pose = 2
        collected_items = 0x3125
        collected_beams = 0x1007
        health = 299
        door_transition = 0
        phase = _Phase()

    rec = AnchorRecorder(task_name="demo", anchors_dir=tmp_path / "anchors")
    env = _Env()
    st = _St()
    a0 = rec.on_frame(env=env, st=st, frame=0)
    assert len(a0) == 1
    assert a0[0]["kind"] == "boot"
    assert Path(a0[0]["path"]).is_file()

    # Door phase in new room must NOT consume the enter (regression).
    st.room_id = 0xA2F7
    st.samus_x = 18
    st.door_transition = 1
    st.phase = _DoorPhase()
    assert rec.on_frame(env=env, st=st, frame=9) == []
    assert rec._last_room == 0xA322  # unchanged through transition

    st.door_transition = 0
    st.phase = _Phase()
    a1 = rec.on_frame(env=env, st=st, frame=10)
    assert any(a["kind"] == "room_enter" for a in a1)

    st.collected_items = 0x7125
    a2 = rec.on_frame(env=env, st=st, frame=20)
    assert any(a["kind"] == "item_delta" for a in a2)
    assert a2[-1]["grapple"] is True

    pin = rec.manual_pin(env=env, st=st, frame=21)
    assert pin is not None and pin["kind"] == "manual"

    idx = rec.write_index(tmp_path / "demo_anchors.json")
    data = json.loads(idx.read_text())
    assert data["count"] == 4


def test_extract_tape_offline(tmp_path: Path) -> None:
    trace = [
        {
            "frame": 0,
            "room": 0xA322,
            "room_hex": "0xA322",
            "x": 70,
            "y": 1419,
            "pose": 2,
            "items": 0x3125,
            "energy": 299,
        },
        {
            "frame": 5,
            "room": 0xCFC9,
            "room_hex": "0xCFC9",
            "x": 317,
            "y": 1963,
            "pose": 2,
            "items": 0x7125,
            "energy": 399,
        },
    ]
    task = {
        "name": "demo",
        "frames": [[0] * 12, [0] * 12],
        "trace": [
            {**trace[0], "buttons": []},
            # fill middle frames so hop end_frame indexing works
            *[{**trace[0], "frame": i, "buttons": []} for i in range(1, 5)],
            {**trace[1], "buttons": []},
        ],
        "start_state": "scratch/x.state",
        "metadata": {
            "end_fingerprint": fingerprint(
                frame=5,
                room_id=0xCFC9,
                x=317,
                y=1963,
                pose=2,
                items=0x7125,
                kind="end",
            )
        },
        "frame_count": 6,
        "recorded_at": "t",
    }
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    board = extract_tape(path, room_names={0xA322: "Cat", 0xCFC9: "Main"})
    assert board["frame_count"] == 6
    assert len(board["room_hops"]) == 2
    assert board["end_verify"]["ok"] is True
    assert fingerprint_from_trace_row(task["trace"][-1])["room"] == "0xCFC9"


def test_write_gzip_state_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.state"
    write_gzip_state(p, b"abc123")
    import gzip

    with gzip.open(p, "rb") as gz:
        assert gz.read() == b"abc123"
