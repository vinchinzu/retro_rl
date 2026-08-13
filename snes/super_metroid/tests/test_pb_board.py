"""Persistent PB board: samples, avg/sd, re-ingest merge (no ROM)."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.human_tape.pb_board import (
    PbBoard,
    format_pb_board_table,
    ingest_rooms,
    materialize_pb_board,
    rooms_to_hop_samples,
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
    end_frame: int,
    rooms: list[dict],
    rta_exclude: bool = False,
    ceres_boot: bool = False,
) -> None:
    sdir = root / f"s{sid}"
    sdir.mkdir(parents=True)
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
    if rta_exclude:
        join["rta_exclude"] = True
        join["reason"] = "test_exclude"
    (sdir / "join.json").write_text(json.dumps(join) + "\n", encoding="utf-8")
    (sdir / "run_timing.json").write_text(
        json.dumps({"rooms": rooms, "total_frames": end_frame + 1}) + "\n",
        encoding="utf-8",
    )
    if ceres_boot:
        (sdir / "anchors.json").write_text(
            json.dumps(
                {
                    "anchors": [
                        {
                            "kind": "boot",
                            "frame": 100,
                            "room": f"0x{CERES_ELEVATOR_ROOM:04X}",
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )


def test_rooms_to_hop_samples_keys() -> None:
    rooms = [
        {
            "room_id": 0x9D19,
            "name": "Big Pink",
            "dwell_frames": 100,
            "dest_room_id": 0x9E52,
            "entry_frame": 0,
        },
        {
            "room_id": 0x9E52,
            "name": "Green Hill Zone",
            "dwell_frames": 50,
            "dest_room_id": 0x9FBA,
            "entry_frame": 120,
        },
    ]
    samples = rooms_to_hop_samples(rooms, source="live", items=0x1004)
    assert len(samples) == 2
    assert samples[0]["hop_key"].startswith("0x9D19:start->0x9E52:0x1004")
    assert samples[1]["hop_key"].startswith("0x9E52:0x9D19->0x9FBA:0x1004")


def test_board_merge_and_stats(tmp_path: Path) -> None:
    board = PbBoard(task="t")
    rooms_a = [
        {
            "room_id": 0xA253,
            "name": "Red Tower",
            "dwell_frames": 1200,
            "dest_room_id": 0xA3DD,
            "entry_frame": 0,
        }
    ]
    rooms_b = [
        {
            "room_id": 0xA253,
            "name": "Red Tower",
            "dwell_frames": 1000,
            "dest_room_id": 0xA3DD,
            "entry_frame": 0,
        }
    ]
    assert ingest_rooms(board, rooms_a, source="s1", items=0x1004) == 1
    assert ingest_rooms(board, rooms_b, source="s2", items=0x1004) == 1
    # re-ingest same segment → no new sample
    assert ingest_rooms(board, rooms_a, source="s1", items=0x1004) == 0

    key = next(iter(board.hops))
    st = board.hop_stats(key)
    assert st is not None
    assert st.n == 2
    assert st.pb == 1000
    assert abs(st.avg - 1100) < 1e-6
    assert st.sd > 0

    path = tmp_path / "t_pb_board.json"
    board.save(path)
    loaded = PbBoard.load(path)
    assert loaded.hop_stats(key) is not None
    assert loaded.hop_stats(key).pb == 1000


def test_materialize_pb_board_product_chain(tmp_path: Path) -> None:
    task = tmp_path / "full_start_v1.json"
    segs = tmp_path / "full_start_v1_segments"
    # power-on with ceres at f100, end f500 → span 400
    _write_seg(
        segs,
        1,
        power_on=True,
        start="power_on",
        end_room="0x9E9F",
        end_items="0x0004",
        end_frame=500,
        ceres_boot=True,
        rooms=[
            {
                "room_id": CERES_ELEVATOR_ROOM,
                "name": "Ceres Elevator Room",
                "dwell_frames": 50,
                "dest_room_id": 0x91F8,
                "entry_frame": 100,
            },
            {
                "room_id": 0x91F8,
                "name": "Landing Site",
                "dwell_frames": 80,
                "dest_room_id": 0x92FD,
                "entry_frame": 200,
            },
        ],
    )
    # morph → bombs
    _write_seg(
        segs,
        2,
        power_on=False,
        start="scratch/morph.state",
        end_room="0x9804",
        end_items="0x1004",
        end_frame=300,
        rooms=[
            {
                "room_id": 0x9E9F,
                "name": "Morph Ball Room",
                "dwell_frames": 90,
                "dest_room_id": 0x9F11,
                "entry_frame": 0,
            }
        ],
    )
    # two identical supers retakes — stats keep both; product chain keeps latest
    for sid in (4, 5):
        _write_seg(
            segs,
            sid,
            power_on=False,
            start="scratch/bomb.state",
            end_room="0x9B5B",
            end_items="0x1004",
            end_frame=200,
            rooms=[
                {
                    "room_id": 0x9804,
                    "name": "Bomb Torizo Room",
                    "dwell_frames": 100 + sid,  # slight diff so both samples stay
                    "dest_room_id": 0x9D19,
                    "entry_frame": 0,
                }
            ],
        )

    # live supers → spazer
    task.write_text(
        json.dumps(
            {
                "name": "full_start_v1",
                "start_state": "scratch/supers.state",
                "frame_count": 150,
                "frames": [[0] * 12] * 150,
                "trace": [],
                "metadata": {
                    "end_fingerprint": {
                        "frame": 149,
                        "room": "0xA447",
                        "items": "0x1004",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "full_start_v1_run_timing.json").write_text(
        json.dumps(
            {
                "total_frames": 150,
                "rooms": [
                    {
                        "room_id": 0x9B5B,
                        "name": "Spore Spawn Super Room",
                        "dwell_frames": 60,
                        "dest_room_id": 0xA0A4,
                        "entry_frame": 0,
                    },
                    {
                        "room_id": 0xA447,
                        "name": "Spazer Room",
                        "dwell_frames": 40,
                        "dest_room_id": None,
                        "entry_frame": 100,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    board, timeline, total, table = materialize_pb_board(
        task, write=True, print_table=False
    )
    # product: s1 span 400 + s2 300 + s5 200 (latest supers) + live 150
    assert total == 400 + 300 + 200 + 150
    assert any(r["source"] == "s5" for r in timeline)
    assert not any(r["source"] == "s4" for r in timeline)  # retake dropped from product
    # history still has both bomb-room samples
    bt_keys = [k for k, h in board.hops.items() if h.get("room_id") == 0x9804]
    assert bt_keys
    st = board.hop_stats(bt_keys[0])
    assert st is not None and st.n == 2

    assert "PB BOARD" in table
    assert "AVG" in table
    assert (tmp_path / "full_start_v1_pb_board.json").is_file()

    # second materialize is merge-compatible (no sample explosion)
    n1 = sum(len(h.get("samples") or []) for h in board.hops.values())
    board2, _, total2, _ = materialize_pb_board(task, write=True, print_table=False)
    n2 = sum(len(h.get("samples") or []) for h in board2.hops.values())
    assert n2 == n1
    assert total2 == total

    table2 = format_pb_board_table(board2, timeline, total_frames=total2)
    assert "★" in table2 or "✓" in table2
    assert "SEGMENTS" in table2
    assert "EST" in table2
    assert "Ceres Elevator" in table2
    assert "Mother Brain" in table2 or "Morph Ball Room" in table2
    assert "KPDR ANY%" in table2
    assert "Ceres Station" in table2
    assert "Best +/-" in table2
