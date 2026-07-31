"""Tests for easiest-first room work queue and entry door resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_metroid.paths import ROOM_PROBLEMS_PATH
from super_metroid.rooms.entry_bootstrap import (
    build_entry_door_map,
    doorway_spawn,
    resolve_entry_door,
)
from super_metroid.rooms.room_graph import load_problem_catalog
from super_metroid.rooms.segment_contract import (
    EntryContract,
    resolve_entry_door_ptr,
)
from super_metroid.rooms.work_queue import (
    build_work_queue,
    difficulty_score,
    export_work_queue,
    work_queue_to_csv_rows,
    work_queue_to_markdown,
)


pytestmark = pytest.mark.skipif(
    not ROOM_PROBLEMS_PATH.is_file(),
    reason="maps/room_problems.json not generated",
)


def test_difficulty_score_orders_ready_before_unstarted_easy() -> None:
    catalog = load_problem_catalog(ROOM_PROBLEMS_PATH)
    problems = catalog["problems"]
    ready = next(p for p in problems if p["practice"]["status"] == "ready")
    easy_open = next(
        p
        for p in problems
        if p["tier"] == "easy"
        and p["practice"]["status"] == "unstarted"
        and int(p["queue"]) == 1
    )
    boss = next(p for p in problems if p["tier"] == "boss_late")
    assert difficulty_score(ready) < difficulty_score(easy_open)
    assert difficulty_score(easy_open) < difficulty_score(boss)


def test_work_queue_is_sorted_easiest_first_and_has_progress() -> None:
    payload = build_work_queue()
    rows = payload["problems"]
    assert len(rows) >= 200
    scores = [int(row["difficultyScore"]) for row in rows]
    assert scores == sorted(scores)
    ranks = [int(row["rank"]) for row in rows]
    assert ranks == list(range(1, len(rows) + 1))
    summary = payload["summary"]
    assert summary["problemCount"] == len(rows)
    assert "easyAndStandardReady" in summary["percentComplete"]
    assert summary["workFocus"]["bossDeferred"] >= 1
    # Top ranks should not be bosses.
    assert all(int(row["queue"]) < 4 for row in rows[:10])


def test_work_queue_export_writes_artifacts(tmp_path: Path) -> None:
    json_out = tmp_path / "queue.json"
    csv_out = tmp_path / "queue.csv"
    md_out = tmp_path / "queue.md"
    payload = export_work_queue(
        json_output=json_out,
        csv_output=csv_out,
        md_output=md_out,
    )
    assert json_out.is_file()
    assert csv_out.is_file()
    assert md_out.is_file()
    assert "easiest first" in md_out.read_text(encoding="utf-8").lower()
    csv_rows = work_queue_to_csv_rows(payload)
    assert csv_rows
    assert "problemId" in csv_rows[0]
    assert "Percent complete" in work_queue_to_markdown(payload)


def test_entry_door_map_covers_easy_queue() -> None:
    door_map = build_entry_door_map()
    assert len(door_map) > 400
    catalog = load_problem_catalog(ROOM_PROBLEMS_PATH)
    easy = [p for p in catalog["problems"] if int(p.get("queue", 3)) == 1]
    assert easy
    missing = [
        p["problemId"]
        for p in easy
        if resolve_entry_door_ptr(p) is None and resolve_entry_door(p, door_map) is None
    ]
    assert missing == []


def test_resolve_entry_door_ptr_matches_connection_map() -> None:
    door_map = build_entry_door_map()
    catalog = load_problem_catalog(ROOM_PROBLEMS_PATH)
    checked = 0
    for problem in catalog["problems"]:
        if int(problem.get("queue", 3)) != 1:
            continue
        ptr = resolve_entry_door_ptr(problem)
        mapped = resolve_entry_door(problem, door_map)
        if ptr is None and mapped is None:
            continue
        assert ptr == mapped, problem["problemId"]
        checked += 1
        if checked >= 20:
            break
    assert checked >= 10


def test_doorway_spawn_insets_from_right_door() -> None:
    problem = {
        "problemId": "test_right",
        "entry": {
            "endpoint": {
                "orientation": "right",
                "block": [15, 7],
            }
        },
        "geometry": {"widthBlocks": 16, "heightBlocks": 16},
    }
    spawn = doorway_spawn(problem, warp_x=256, warp_y=139, inset_px=56)
    # Just inside right door, facing into the room (left).
    assert spawn["x"] < 256 - 40
    assert spawn["face"] == "left"
    assert spawn["doorOrientation"] == "right"
    assert spawn["pose"] == 2


def test_doorway_spawn_insets_from_left_and_vertical() -> None:
    left = {
        "problemId": "test_left",
        "entry": {"endpoint": {"orientation": "left", "block": [0, 7]}},
        "geometry": {"widthBlocks": 16, "heightBlocks": 16},
    }
    spawn_l = doorway_spawn(left, warp_x=16, warp_y=120, inset_px=56)
    assert spawn_l["face"] == "right"
    assert spawn_l["x"] > 16
    assert spawn_l["pose"] == 1

    up = {
        "problemId": "test_up",
        "entry": {"endpoint": {"orientation": "up", "block": [7, 0]}},
        "geometry": {"widthBlocks": 16, "heightBlocks": 16},
    }
    spawn_u = doorway_spawn(up, warp_x=120, warp_y=16, inset_px=56)
    assert spawn_u["y"] > 16
    assert spawn_u["doorOrientation"] == "up"

    down = {
        "problemId": "test_down",
        "entry": {"endpoint": {"orientation": "down", "block": [7, 15]}},
        "geometry": {"widthBlocks": 16, "heightBlocks": 16},
    }
    spawn_d = doorway_spawn(down, warp_x=120, warp_y=240, inset_px=56)
    assert spawn_d["y"] < 240
    assert spawn_d["doorOrientation"] == "down"


def test_doorway_spawn_rejects_missing_orientation() -> None:
    with pytest.raises(ValueError, match="orientation"):
        doorway_spawn(
            {"problemId": "bad", "entry": {"endpoint": {}}, "geometry": {}},
            warp_x=100,
            warp_y=100,
        )


def test_entry_contract_round_trip() -> None:
    catalog = load_problem_catalog(ROOM_PROBLEMS_PATH)
    problem = next(
        p
        for p in catalog["problems"]
        if p.get("entry") is not None and int(p.get("queue", 3)) == 1
    )
    contract = EntryContract.from_problem(
        problem,
        door_ptr=0x8F52,
        spawn={
            "x": 192,
            "y": 120,
            "pose": 2,
            "face": "left",
            "insetPx": 56,
            "doorBlock": [15, 7],
            "warpSample": {"x": 256, "y": 139},
        },
        boot_idle_frames=12,
    )
    restored = EntryContract.from_dict(contract.to_dict())
    assert restored is not None
    assert restored.door_ptr == 0x8F52
    assert restored.spawn_x == 192
    assert restored.boot_idle_frames == 12
    assert restored.kind == "doorway_natural"
    assert restored.exit_travel_direction is not None or problem.get("exit") is None
