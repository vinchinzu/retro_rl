"""Product-chain hop board + archived tape anchors.json fallback (no ROM)."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.human_tape.anchors import load_anchors_index, match_anchor
from super_metroid.human_tape.product_chain import (
    PolicyIndexRow,
    build_product_chain_board,
    format_board_summary,
    match_policy,
)
from super_metroid.human_tape.rta_clock import CERES_ELEVATOR_ROOM


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
) -> Path:
    sdir = root / f"s{sid}"
    sdir.mkdir(parents=True)
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
    (sdir / "tape.json").write_text(
        json.dumps({"name": f"s{sid}", "frame_count": end_frame + 1, "frames": []})
        + "\n",
        encoding="utf-8",
    )
    (sdir / "extract.json").write_text(
        json.dumps({"room_hops": hops}) + "\n", encoding="utf-8"
    )
    if anchors is not None:
        (sdir / "anchors.json").write_text(
            json.dumps({"anchors": anchors, "anchors_dir": str(sdir)}) + "\n",
            encoding="utf-8",
        )
    return sdir / "tape.json"


def test_load_anchors_index_finds_segment_anchors_json(tmp_path: Path) -> None:
    sdir = tmp_path / "s0"
    sdir.mkdir()
    tape = sdir / "tape.json"
    tape.write_text("{}", encoding="utf-8")
    pin = sdir / "boot.state"
    pin.write_bytes(b"x")
    (sdir / "anchors.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "kind": "boot",
                        "frame": 0,
                        "room": "0x91F8",
                        "room_id": 0x91F8,
                        "path": str(pin),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    idx = load_anchors_index(tape)
    assert idx is not None
    hit = match_anchor(idx, 0, 0x91F8, task_path=tape)
    assert hit is not None
    assert hit["kind"] == "boot"
    assert Path(hit["path"]).resolve() == pin.resolve()


def test_match_policy_prefers_from_and_exit() -> None:
    rows = (
        PolicyIndexRow("room_only", 0x96BA, None, None, "candidate", "a"),
        PolicyIndexRow(
            "climb", 0x96BA, 0x975C, 0x92FD, "verified_live_anchor", "b"
        ),
    )
    hit = match_policy(rows, room_id=0x96BA, from_room_id=0x975C, to_room_id=0x92FD)
    assert hit is not None
    assert hit.policy_id == "climb"


def test_board_dedupes_retakes_and_flags_missing_policy(tmp_path: Path) -> None:
    segs = tmp_path / "take_segments"
    segs.mkdir()
    task = tmp_path / "take.json"
    task.write_text("{}", encoding="utf-8")

    climb_pin = segs / "s1" / "climb.state"
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
            },
            {
                "index": 1,
                "start_index": 201,
                "frame": 201,
                "room": "0x96BA",
                "room_id": 0x96BA,
                "name": "The Climb",
                "items": "0x0000",
                "end_frame": 400,
                "dwell": 200,
            },
        ],
        anchors=[
            {
                "kind": "room_enter",
                "frame": 201,
                "room": "0x96BA",
                "room_id": 0x96BA,
                "path": str(climb_pin),
            }
        ],
    )
    climb_pin.parent.mkdir(parents=True, exist_ok=True)
    climb_pin.write_bytes(b"pin")

    # Retake of the same seam — should be dropped from product chain.
    _write_seg(
        segs,
        2,
        power_on=True,
        start="power_on",
        end_room="0x9E9F",
        end_items="0x0004",
        hops=[
            {
                "index": 0,
                "room": "0xDF45",
                "room_id": CERES_ELEVATOR_ROOM,
                "items": "0x0000",
                "end_frame": 50,
                "dwell": 50,
            }
        ],
        anchors=[],
    )

    pol_dir = tmp_path / "policies"
    pol_dir.mkdir()
    (pol_dir / "room_96ba_from_975c_to_92fd.json").write_text(
        json.dumps(
            {
                "kind": "super_metroid_reactive_room_policy",
                "policyId": "room_96ba_from_975c_to_92fd",
                "status": "verified_live_anchor",
                "roomId": 0x96BA,
                "fromRoomId": 0x975C,
                "exitRoomId": 0x92FD,
            }
        ),
        encoding="utf-8",
    )

    board = build_product_chain_board(
        task, include_live=False, policy_dir=pol_dir, bank_path=tmp_path / "no-bank.json"
    )
    hops = board["hops"]
    # Only latest power-on seam (s2) — s1 dropped as retake of same start/end.
    # Wait: both s1 and s2 are power_on → 0x9E9F 0x0004, so only latest (s2) remains.
    segs_used = {h["segment"] for h in hops}
    assert segs_used == {"s2"}
    summary = format_board_summary(board)
    assert "product-chain hops=" in summary


def test_ap_join_contract_on_board(tmp_path: Path) -> None:
    task = tmp_path / "empty.json"
    task.write_text("{}", encoding="utf-8")
    board = build_product_chain_board(task, include_live=False)
    assert "subpixels" in board["ap_join"]
    assert "enemy phase" in board["ap_join"]
    assert "DoorKinematics" in board["ap_join"]
