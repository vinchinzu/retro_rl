"""s23 Attic / Bowling tape adapters (no ROM)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from super_metroid.room_adapter import search_live_adapter as live_adapter
from super_metroid.splice.errors import PreflightError
from super_metroid.splice.tapes import (
    ATTIC_ROOM,
    ATTIC_TASK_ID,
    BOWLING_INTERNAL_MAX_FRAMES,
    BOWLING_PLANNED_DWELL,
    BOWLING_ROOM,
    BOWLING_TASK_ID,
    GRAVITY_ROOM,
    MAIN_SHAFT_ROOM,
    RECOVERY,
    WEST_OCEAN_ROOM,
    load_s23_tape_candidates,
    project_live,
    recover_live,
    resolve_segment_dir,
    search_live_adapter,
)

ITEMS = 0x3105
ATTIC_START = 100
ATTIC_END = 180
BOWLING_START = 2000
BOWLING_DWELL = BOWLING_PLANNED_DWELL
BOWLING_END = BOWLING_START + BOWLING_DWELL - 1


def _hop(
    index: int,
    room: int,
    *,
    start: int,
    end: int,
    name: str,
    xy: tuple[int, int],
    end_xy: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "start_index": start,
        "end_index": end,
        "frame": start,
        "end_frame": end,
        "dwell": end - start + 1,
        "room": f"0x{room:04X}",
        "room_id": room,
        "name": name,
        "items": f"0x{ITEMS:04X}",
        "xy": list(xy),
        "end_xy": list(end_xy or xy),
        "pose": 1,
    }


def _s23_hops() -> list[dict[str, Any]]:
    return [
        _hop(0, MAIN_SHAFT_ROOM, start=0, end=99, name="Main Shaft", xy=(1135, 80)),
        _hop(1, ATTIC_ROOM, start=ATTIC_START, end=ATTIC_END, name="Attic", xy=(40, 120), end_xy=(700, 120)),
        _hop(2, WEST_OCEAN_ROOM, start=181, end=400, name="West Ocean", xy=(60, 200)),
        _hop(3, 0x9461, start=401, end=800, name="Pancakes and Wavers", xy=(80, 180)),
        _hop(4, 0x968F, start=801, end=1999, name="Homing Geemer", xy=(90, 160)),
        _hop(
            5,
            BOWLING_ROOM,
            start=BOWLING_START,
            end=BOWLING_END,
            name="Bowling Alley",
            xy=(40, 180),
            end_xy=(420, 180),
        ),
        _hop(6, GRAVITY_ROOM, start=BOWLING_END + 1, end=BOWLING_END + 80, name="Gravity Suit Room", xy=(80, 160)),
    ]


def _write_s23(
    root: Path,
    *,
    hops: list[dict[str, Any]] | None = None,
    write_tape: bool = True,
    empty_tape: bool = False,
    write_anchors: bool = True,
    write_extract: bool = True,
    tape_payload: dict[str, Any] | None = None,
    anchors: list[dict[str, Any]] | None = None,
) -> Path:
    sdir = root / "s23"
    sdir.mkdir(parents=True, exist_ok=True)
    hops = _s23_hops() if hops is None else hops
    attic_pin = sdir / "attic.state"
    bowl_pin = sdir / "bowling.state"
    main_pin = sdir / "main.state"
    attic_pin.write_bytes(b"attic-enter")
    bowl_pin.write_bytes(b"bowling-enter")
    main_pin.write_bytes(b"main-enter")
    if write_tape:
        if empty_tape:
            (sdir / "tape.json").write_bytes(b"")
        else:
            payload = tape_payload or {
                "name": "s23",
                "frame_count": BOWLING_END + 81,
                "frames": [],
            }
            (sdir / "tape.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if write_extract:
        (sdir / "extract.json").write_text(
            json.dumps({"room_hops": hops}) + "\n", encoding="utf-8"
        )
    if write_anchors:
        rows = anchors
        if rows is None:
            rows = [
                {
                    "kind": "room_enter",
                    "frame": ATTIC_START,
                    "room": f"0x{ATTIC_ROOM:04X}",
                    "room_id": ATTIC_ROOM,
                    "path": str(attic_pin.resolve()),
                    "xy": [40, 120],
                },
                {
                    "kind": "room_enter",
                    "frame": BOWLING_START,
                    "room": f"0x{BOWLING_ROOM:04X}",
                    "room_id": BOWLING_ROOM,
                    "path": str(bowl_pin.resolve()),
                    "xy": [40, 180],
                },
            ]
        (sdir / "anchors.json").write_text(
            json.dumps({"anchors_dir": str(sdir), "anchors": rows}) + "\n",
            encoding="utf-8",
        )
    return sdir


def _walk_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_walk_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_walk_strings(item))
    return out


def test_missing_tape_fails_closed(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path, write_tape=False)
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(sdir)
    missing = exc.value.details.get("missing") or []
    assert any("tape" in str(label) for label in missing)
    assert exc.value.code == "preflight.missing"
    assert exc.value.details.get("segment") == "s23"
    empty = _write_s23(tmp_path / "empty", empty_tape=True)
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(empty)
    assert any("tape" in str(label) for label in (exc.value.details.get("missing") or []))


def test_missing_default_s23_fails_closed(tmp_path: Path) -> None:
    sdir = tmp_path / "full_start_v1_segments" / "s23"
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(sdir)
    missing = exc.value.details.get("missing") or []
    assert any("tape" in str(label) for label in missing)
    assert any("anchors" in str(label) for label in missing)
    assert any("extract" in str(label) for label in missing)
    assert resolve_segment_dir(tmp_path / "full_start_v1_segments").name == "s23"


def test_missing_attic_or_bowling_hop_fails_closed(tmp_path: Path) -> None:
    hops = [h for h in _s23_hops() if h["room_id"] != ATTIC_ROOM]
    sdir = _write_s23(tmp_path, hops=hops)
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(sdir)
    missing = exc.value.details.get("missing") or []
    assert any("0xCA52" in str(label) for label in missing)
    hops = [h for h in _s23_hops() if h["room_id"] != BOWLING_ROOM]
    sdir = _write_s23(tmp_path / "no-bowl", hops=hops)
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(sdir)
    missing = exc.value.details.get("missing") or []
    assert any("0xC98E" in str(label) for label in missing)


def test_other_room_enter_pin_fails_closed(tmp_path: Path) -> None:
    sdir = _write_s23(
        tmp_path,
        anchors=[
            {
                "kind": "room_enter",
                "frame": 0,
                "room": f"0x{MAIN_SHAFT_ROOM:04X}",
                "room_id": MAIN_SHAFT_ROOM,
                "path": str((tmp_path / "s23" / "main.state").resolve()),
                "xy": [1135, 80],
            }
        ],
    )
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(sdir)
    missing = exc.value.details.get("missing") or []
    assert any("enter_pin" in str(label) for label in missing)
    assert any("0xCA52" in str(label) or "0xC98E" in str(label) for label in missing)


def test_hop_span_and_successor_fail_closed(tmp_path: Path) -> None:
    hops = _s23_hops()
    for hop in hops:
        if hop["room_id"] == ATTIC_ROOM:
            for key in ("start_index", "end_index", "end_frame", "dwell", "frame"):
                hop.pop(key, None)
    sdir = _write_s23(tmp_path, hops=hops)
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(sdir)
    missing = exc.value.details.get("missing") or []
    assert any("span" in str(label) and "0xCA52" in str(label) for label in missing)
    hops = [h for h in _s23_hops() if h["room_id"] != GRAVITY_ROOM]
    sdir = _write_s23(tmp_path / "no-grav", hops=hops)
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(sdir)
    missing = exc.value.details.get("missing") or []
    assert any("0xCE40" in str(label) for label in missing)
    hops = [h for h in _s23_hops() if h["room_id"] != WEST_OCEAN_ROOM]
    sdir = _write_s23(tmp_path / "no-wo", hops=hops)
    with pytest.raises(PreflightError) as exc:
        load_s23_tape_candidates(sdir)
    missing = exc.value.details.get("missing") or []
    assert any("0x93FE" in str(label) for label in missing)


def test_repo_relative_paths(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    attic, bowling = load_s23_tape_candidates(sdir)
    tape_digest = hashlib.sha256((sdir / "tape.json").read_bytes()).hexdigest()
    assert attic.artifact.tape_digest == tape_digest
    assert bowling.edge.tape_digest == tape_digest
    for cand in (attic, bowling):
        payload = cand.to_dict()
        assert cand.edge.tape_path is not None
        assert not Path(cand.edge.tape_path).is_absolute()
        assert not cand.edge.tape_path.startswith("/")
        assert cand.edge.entry.state_path is not None
        assert not Path(cand.edge.entry.state_path).is_absolute()
        for text in _walk_strings(payload):
            if "/" in text or text.endswith(".state") or text.endswith(".json"):
                assert not text.startswith("/"), text
                assert not Path(text).is_absolute() or text.startswith("snes/"), text


def test_attic_and_bowling_task_ids(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    attic, bowling = load_s23_tape_candidates(sdir)
    assert attic.task_id == ATTIC_TASK_ID
    assert bowling.task_id == BOWLING_TASK_ID
    assert attic.room_id == ATTIC_ROOM == 0xCA52
    assert attic.next_room_id == WEST_OCEAN_ROOM == 0x93FE
    assert bowling.room_id == BOWLING_ROOM == 0xC98E
    assert bowling.next_room_id == GRAVITY_ROOM == 0xCE40
    assert attic.artifact.kind == "tape"
    assert bowling.artifact.kind == "tape"
    assert attic.candidate_id.startswith("tape:")
    assert bowling.edge.selected_map()["scaffold"].startswith("tape:")
    assert attic.edge.segment == "s23"
    assert bowling.edge.segment == "s23"
    ids = {attic.task_id, bowling.task_id}
    assert MAIN_SHAFT_ROOM not in {attic.room_id, bowling.room_id}
    assert ids == {ATTIC_TASK_ID, BOWLING_TASK_ID}


def test_skips_main_shaft_serial_hop(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    attic, bowling = load_s23_tape_candidates(sdir)
    assert attic.edge.predecessor_room_id == MAIN_SHAFT_ROOM
    rooms = {attic.room_id, bowling.room_id}
    assert MAIN_SHAFT_ROOM not in rooms
    assert "Main Shaft" in " ".join(attic.source_notes)
    assert all("rr-kw8t" in " ".join(c.source_notes) or "serial" in " ".join(c.source_notes).lower() for c in (attic, bowling))


def test_bowling_internal_split_one_external_contract(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    attic, bowling = load_s23_tape_candidates(sdir)
    assert attic.task_id == ATTIC_TASK_ID
    assert len(attic.internal_slices) == 1
    assert attic.internal_slices[0].frame_start == ATTIC_START
    assert attic.internal_slices[0].frame_end == ATTIC_END
    assert bowling.task_id == BOWLING_TASK_ID
    assert bowling.contract.task_id == BOWLING_TASK_ID
    assert bowling.contract.next_room_id == GRAVITY_ROOM
    assert bowling.contract.natural_entry is True
    assert bowling.edge.next_room_id == GRAVITY_ROOM
    assert bowling.edge.frame_end - bowling.edge.frame_start + 1 == BOWLING_DWELL
    assert len(bowling.internal_slices) > 1
    assert BOWLING_DWELL > BOWLING_INTERNAL_MAX_FRAMES
    assert len(bowling.internal_slices) == 3
    assert [s.slice_id for s in bowling.internal_slices] == [
        "bowling:entry",
        "bowling:mid",
        "bowling:leave",
    ]
    assert all(s.room_id == BOWLING_ROOM for s in bowling.internal_slices)
    slice_ids = {s.slice_id for s in bowling.internal_slices}
    assert BOWLING_TASK_ID not in slice_ids
    assert bowling.artifact.task_id == BOWLING_TASK_ID
    covered = sum(s.frame_end - s.frame_start + 1 for s in bowling.internal_slices)
    assert covered == BOWLING_DWELL
    assert bowling.internal_slices[0].frame_start == BOWLING_START
    assert bowling.internal_slices[-1].frame_end == BOWLING_END
    for window in bowling.internal_slices:
        assert window.frame_end - window.frame_start + 1 <= BOWLING_INTERNAL_MAX_FRAMES


def test_bounded_projection_and_search_live_adapter_recovery(tmp_path: Path) -> None:
    sdir = _write_s23(tmp_path)
    attic, bowling = load_s23_tape_candidates(sdir)
    assert search_live_adapter is live_adapter
    assert recover_live.__name__ == "recover_live"
    assert attic.recovery == RECOVERY == "search_live_adapter"
    assert bowling.recovery == "search_live_adapter"
    assert "search_live_adapter" in attic.artifact.action_reasons
    hit = project_live(attic, {"room_id": ATTIC_ROOM, "x": 40, "y": 120, "frame": ATTIC_START})
    assert hit.within_bound
    assert hit.recovery is None
    miss_room = project_live(attic, {"room_id": MAIN_SHAFT_ROOM, "x": 40, "y": 120})
    assert not miss_room.within_bound
    assert miss_room.recovery == "search_live_adapter"
    miss_xy = project_live(bowling, {"room_id": BOWLING_ROOM, "x": 900, "y": 900})
    assert not miss_xy.within_bound
    assert miss_xy.recovery == "search_live_adapter"
    miss_frame = project_live(
        bowling,
        {"room_id": BOWLING_ROOM, "x": 40, "y": 180, "frame": 0},
    )
    assert not miss_frame.within_bound
    assert miss_frame.recovery == "search_live_adapter"
    miss_no_xy = project_live(attic, {"room_id": ATTIC_ROOM})
    assert not miss_no_xy.within_bound
    assert miss_no_xy.recovery == "search_live_adapter"
    assert bowling.adapter_config.max_depth == 4
    assert bowling.adapter_config.beam_width == 8
