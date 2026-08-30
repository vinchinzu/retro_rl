"""Phase 0 splice preflight (no ROM)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from super_metroid.human_tape.rta_clock import CERES_ELEVATOR_ROOM
from super_metroid.splice import (
    PreflightError,
    file_digest,
    repo_relative,
    run_preflight,
)
from super_metroid.splice.preflight import format_preflight_summary


def _write_seg(
    root: Path,
    sid: int,
    *,
    power_on: bool,
    start: str,
    end_room: str,
    end_items: str,
    hops: list[dict],
    anchors: list[dict] | None = None,
    tape_bytes: bytes | None = None,
    write_tape: bool = True,
    bodies: dict[str, bytes] | None = None,
) -> Path:
    sdir = root / f"s{sid}"
    sdir.mkdir(parents=True, exist_ok=True)
    end_frame = int(hops[-1]["end_frame"]) if hops else 10
    join = {
        "power_on": power_on,
        "start_state": start,
        "frame_count": end_frame + 1,
        "end_fingerprint": {
            "frame": end_frame,
            "room": end_room,
            "items": end_items,
        },
    }
    (sdir / "join.json").write_text(json.dumps(join) + "\n", encoding="utf-8")
    if write_tape:
        payload = tape_bytes if tape_bytes is not None else json.dumps(
            {"name": f"s{sid}", "frame_count": end_frame + 1, "frames": []}
        ).encode("utf-8") + b"\n"
        (sdir / "tape.json").write_bytes(payload)
    (sdir / "extract.json").write_text(
        json.dumps({"room_hops": hops}) + "\n", encoding="utf-8"
    )
    if anchors is not None:
        (sdir / "anchors.json").write_text(
            json.dumps({"anchors": anchors, "anchors_dir": str(sdir)}) + "\n",
            encoding="utf-8",
        )
    if bodies:
        hops_dir = sdir / "hops"
        hops_dir.mkdir(exist_ok=True)
        for name, blob in bodies.items():
            (hops_dir / name).write_bytes(blob)
    return sdir / "tape.json"


def _task(tmp_path: Path, stem: str = "take") -> tuple[Path, Path]:
    segs = tmp_path / f"{stem}_segments"
    segs.mkdir()
    task = tmp_path / f"{stem}.json"
    task.write_text("{}", encoding="utf-8")
    return task, segs


def test_file_digest_missing_or_empty_is_none(tmp_path: Path) -> None:
    missing = tmp_path / "nope.bin"
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert file_digest(None) is None
    assert file_digest(missing) is None
    assert file_digest(empty) is None
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"abc")
    assert file_digest(blob) == hashlib.sha256(b"abc").hexdigest()


def test_repo_relative_never_returns_host_absolute(tmp_path: Path) -> None:
    pin = tmp_path / "pin.state"
    pin.write_bytes(b"x")
    rel = repo_relative(pin)
    assert rel is not None
    assert not Path(rel).is_absolute()
    assert not rel.startswith("/")
    already = repo_relative("tasks/full_start_v1_segments/s1/tape.json")
    assert already == "tasks/full_start_v1_segments/s1/tape.json"


def test_missing_tape(tmp_path: Path) -> None:
    task, segs = _task(tmp_path)
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0x9E9F",
        end_items="0x0004",
        hops=[
            {
                "index": 0,
                "start_index": 100,
                "frame": 100,
                "room": "0xDF45",
                "room_id": CERES_ELEVATOR_ROOM,
                "name": "Ceres Elevator",
                "items": "0x0000",
                "end_frame": 200,
                "dwell": 101,
            }
        ],
        anchors=[],
        write_tape=False,
    )
    report = run_preflight(
        task,
        include_live=False,
        policy_dir=tmp_path / "policies",
        bank_path=tmp_path / "no-bank.json",
        rom_path=tmp_path / "no.rom",
        repo_root=tmp_path,
    )
    assert any("tape" in label for label in report.selected_missing)
    assert report.segments
    assert report.segments[0].tape.exists is False
    with pytest.raises(PreflightError) as exc:
        run_preflight(
            task,
            include_live=False,
            policy_dir=tmp_path / "policies",
            bank_path=tmp_path / "no-bank.json",
            rom_path=tmp_path / "no.rom",
            repo_root=tmp_path,
            strict=True,
        )
    assert "tape" in str(exc.value.details.get("missing"))


def test_absolute_path_rewritten(tmp_path: Path) -> None:
    task, segs = _task(tmp_path)
    pin = segs / "s1" / "climb.state"
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_bytes(b"pin-bytes")
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0x9E9F",
        end_items="0x0004",
        hops=[
            {
                "index": 0,
                "start_index": 201,
                "frame": 201,
                "room": "0x96BA",
                "room_id": 0x96BA,
                "name": "The Climb",
                "items": "0x0000",
                "end_frame": 400,
                "dwell": 200,
            }
        ],
        anchors=[
            {
                "kind": "room_enter",
                "frame": 201,
                "room": "0x96BA",
                "room_id": 0x96BA,
                "path": str(pin.resolve()),
            }
        ],
    )
    report = run_preflight(
        task,
        include_live=False,
        policy_dir=tmp_path / "policies",
        bank_path=tmp_path / "no-bank.json",
        rom_path=tmp_path / "no.rom",
        repo_root=tmp_path,
    )
    assert report.hops
    enter = report.hops[0].enter_pin
    assert enter is not None
    assert not Path(enter).is_absolute()
    assert not enter.startswith("/")
    board_hop = report.board["hops"][0]
    assert board_hop.get("anchor_path")
    assert not str(board_hop["anchor_path"]).startswith("/")
    assert not Path(str(board_hop["tape"])).is_absolute()


def test_duplicate_hop_keys(tmp_path: Path) -> None:
    task, segs = _task(tmp_path)
    hops = [
        {
            "index": 0,
            "start_index": 10,
            "frame": 10,
            "room": "0x91F8",
            "room_id": 0x91F8,
            "name": "Landing Site",
            "items": "0x0004",
            "end_frame": 20,
            "dwell": 11,
        },
        {
            "index": 1,
            "start_index": 21,
            "frame": 21,
            "room": "0x92FD",
            "room_id": 0x92FD,
            "name": "Parlor",
            "items": "0x0004",
            "end_frame": 40,
            "dwell": 20,
        },
        {
            "index": 2,
            "start_index": 41,
            "frame": 41,
            "room": "0x9F64",
            "room_id": 0x9F64,
            "name": "Crateria Tube",
            "items": "0x0004",
            "end_frame": 50,
            "dwell": 10,
        },
        {
            "index": 3,
            "start_index": 51,
            "frame": 51,
            "room": "0x91F8",
            "room_id": 0x91F8,
            "name": "Landing Site",
            "items": "0x0004",
            "end_frame": 60,
            "dwell": 10,
        },
        {
            "index": 4,
            "start_index": 61,
            "frame": 61,
            "room": "0x92FD",
            "room_id": 0x92FD,
            "name": "Parlor",
            "items": "0x0004",
            "end_frame": 80,
            "dwell": 20,
        },
        {
            "index": 5,
            "start_index": 81,
            "frame": 81,
            "room": "0x9F64",
            "room_id": 0x9F64,
            "name": "Crateria Tube",
            "items": "0x0004",
            "end_frame": 90,
            "dwell": 10,
        },
    ]
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0x9F64",
        end_items="0x0004",
        hops=hops,
        anchors=[],
    )
    report = run_preflight(
        task,
        include_live=False,
        policy_dir=tmp_path / "policies",
        bank_path=tmp_path / "no-bank.json",
        rom_path=tmp_path / "no.rom",
        repo_root=tmp_path,
    )
    parlor = [h.hop_key for h in report.hops if h.room_id == 0x92FD]
    assert len(parlor) == 2
    assert parlor[0] == parlor[1]
    assert parlor[0] in report.duplicate_hop_keys


def test_invalid_rooms_flagged(tmp_path: Path) -> None:
    task, segs = _task(tmp_path)
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0x91F8",
        end_items="0x0000",
        hops=[
            {
                "index": 0,
                "start_index": 1,
                "frame": 1,
                "room": "0x0000",
                "room_id": 0x0000,
                "name": "boot",
                "items": "0x0000",
                "end_frame": 5,
                "dwell": 5,
            },
            {
                "index": 1,
                "start_index": 6,
                "frame": 6,
                "room": "0x5555",
                "room_id": 0x5555,
                "name": "unsettled",
                "items": "0x0000",
                "end_frame": 10,
                "dwell": 5,
            },
        ],
        anchors=[],
    )
    report = run_preflight(
        task,
        include_live=False,
        policy_dir=tmp_path / "policies",
        bank_path=tmp_path / "no-bank.json",
        rom_path=tmp_path / "no.rom",
        repo_root=tmp_path,
    )
    rooms = {h.room_id: h for h in report.hops}
    assert rooms[0x0000].invalid_room
    assert rooms[0x5555].invalid_room
    edge = report.first_uncovered_edge
    assert edge is not None
    assert edge["room_id"] in {0x0000, 0x5555}
    assert any("invalid_room" in r for r in edge["reasons"])


def test_digest_matches_sha256_of_bytes(tmp_path: Path) -> None:
    task, segs = _task(tmp_path)
    payload = b'{"name": "s1", "frames": [1, 2, 3]}\n'
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0xDF45",
        end_items="0x0000",
        hops=[
            {
                "index": 0,
                "start_index": 0,
                "frame": 0,
                "room": "0xDF45",
                "room_id": CERES_ELEVATOR_ROOM,
                "name": "Ceres Elevator",
                "items": "0x0000",
                "end_frame": 10,
                "dwell": 11,
            }
        ],
        anchors=[],
        tape_bytes=payload,
    )
    report = run_preflight(
        task,
        include_live=False,
        policy_dir=tmp_path / "policies",
        bank_path=tmp_path / "no-bank.json",
        rom_path=tmp_path / "no.rom",
        repo_root=tmp_path,
    )
    assert report.segments[0].tape.digest == hashlib.sha256(payload).hexdigest()


def test_first_uncovered_edge_missing_enter_pin(tmp_path: Path) -> None:
    task, segs = _task(tmp_path)
    pin = segs / "s1" / "ceres.state"
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_bytes(b"ceres-pin")
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0x96BA",
        end_items="0x0000",
        hops=[
            {
                "index": 0,
                "start_index": 0,
                "frame": 0,
                "room": "0xDF45",
                "room_id": CERES_ELEVATOR_ROOM,
                "name": "Ceres Elevator",
                "items": "0x0000",
                "end_frame": 100,
                "dwell": 101,
            },
            {
                "index": 1,
                "start_index": 101,
                "frame": 101,
                "room": "0x96BA",
                "room_id": 0x96BA,
                "name": "The Climb",
                "items": "0x0000",
                "end_frame": 400,
                "dwell": 300,
            },
        ],
        anchors=[
            {
                "kind": "boot",
                "frame": 0,
                "room": "0xDF45",
                "room_id": CERES_ELEVATOR_ROOM,
                "path": str(pin),
            }
        ],
    )
    report = run_preflight(
        task,
        include_live=False,
        policy_dir=tmp_path / "policies",
        bank_path=tmp_path / "no-bank.json",
        rom_path=tmp_path / "no.rom",
        repo_root=tmp_path,
    )
    edge = report.first_uncovered_edge
    assert edge is not None
    assert edge["room_id"] == 0x96BA
    assert "missing_enter_pin" in edge["reasons"]
    assert report.hops[0].enter_pin_digest == hashlib.sha256(b"ceres-pin").hexdigest()


def test_gravity_path_human_oracle_only(tmp_path: Path) -> None:
    task, segs = _task(tmp_path)
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0xDF45",
        end_items="0x0000",
        hops=[
            {
                "index": 0,
                "start_index": 0,
                "frame": 0,
                "room": "0xDF45",
                "room_id": CERES_ELEVATOR_ROOM,
                "name": "Ceres Elevator",
                "items": "0x0000",
                "end_frame": 10,
                "dwell": 11,
            }
        ],
        anchors=[],
    )
    report = run_preflight(
        task,
        include_live=False,
        policy_dir=tmp_path / "policies",
        bank_path=tmp_path / "no-bank.json",
        rom_path=tmp_path / "no.rom",
        repo_root=tmp_path,
    )
    oracle = report.gravity_path_human
    assert oracle["name"] == "gravity_path_human"
    assert oracle["role"] == "oracle_only"
    assert "s23" in oracle["prefer"]
    summary = format_preflight_summary(report)
    assert "oracle_only" in summary
    assert "gravity_path_human" in summary
    dumped = json.dumps(report.to_dict())
    assert "oracle_only" in dumped


def test_impossible_inventory_without_dump(tmp_path: Path) -> None:
    task, segs = _task(tmp_path)
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0x91F8",
        end_items="0x0000",
        hops=[
            {
                "index": 0,
                "start_index": 0,
                "frame": 0,
                "room": "0x9E9F",
                "room_id": 0x9E9F,
                "name": "Morph",
                "items": "0x0004",
                "end_frame": 20,
                "dwell": 21,
            },
            {
                "index": 1,
                "start_index": 21,
                "frame": 21,
                "room": "0x91F8",
                "room_id": 0x91F8,
                "name": "Landing Site",
                "items": "0x0000",
                "end_frame": 40,
                "dwell": 20,
            },
        ],
        anchors=[],
    )
    report = run_preflight(
        task,
        include_live=False,
        policy_dir=tmp_path / "policies",
        bank_path=tmp_path / "no-bank.json",
        rom_path=tmp_path / "no.rom",
        repo_root=tmp_path,
    )
    assert report.impossible_inventory
    row = report.impossible_inventory[0]
    assert row.from_items == "0x0004"
    assert row.to_items == "0x0000"
    assert row.lost_bits == "0x0004"
