from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from super_metroid.paths import FULL_ROOM_GRAPH_PATH, ROOM_PROBLEMS_PATH
from super_metroid.rooms.room_graph import (
    PhysicalConnection,
    PhysicalEndpoint,
    load_problem_catalog,
    problem_by_id,
    shortest_room_path,
)
from super_metroid.rooms.room_practice import (
    _expand_steps,
    _objective_progress_failure,
    _scaffold_frame_budget,
    promote_room_policy,
    scaffold_room_policy,
)


def _endpoint(room_id: int, *, requires: tuple[str, ...] = ()) -> PhysicalEndpoint:
    return PhysicalEndpoint(
        room_id=room_id,
        logical_room_id=room_id,
        node_id=1,
        node_name=f"room {room_id} door",
        position="left",
        orientation="right",
        subtype="blue",
        block=(0, 7),
        requires=requires,
        local_requirements=(),
        impossible_exit=False,
    )


def test_forward_connection_stays_one_way() -> None:
    connection = PhysicalConnection(
        connection_id="one_way",
        connection_type="StoryMarker",
        description="test transition",
        direction="Forward",
        first=_endpoint(1),
        second=_endpoint(2),
    )

    edges = connection.directed_edges()

    assert len(edges) == 1
    assert edges[0]["source"]["roomId"] == 1
    assert edges[0]["target"]["roomId"] == 2


def test_shortest_path_respects_capability_gate_and_direction() -> None:
    graph = {
        "edges": [
            {
                "edgeId": "one_to_two",
                "source": {"roomId": 1},
                "target": {"roomId": 2},
                "requires": ["super_missiles"],
                "impossible": False,
            },
            {
                "edgeId": "two_to_three",
                "source": {"roomId": 2},
                "target": {"roomId": 3},
                "requires": [],
                "impossible": False,
            },
        ]
    }

    assert shortest_room_path(graph, 1, 3) is None
    path = shortest_room_path(graph, 1, 3, {"Super Missiles"})
    assert path is not None
    assert [edge["edgeId"] for edge in path] == ["one_to_two", "two_to_three"]
    assert shortest_room_path(graph, 3, 1, {"super_missiles"}) is None


def test_compact_room_policy_repeats_nested_spans() -> None:
    spans = tuple(
        _expand_steps(
            [
                {
                    "label": "cycle",
                    "repeat": 2,
                    "steps": [
                        {"buttons": ["left", "a"], "frames": 4},
                        {"buttons": [], "frames": 1},
                    ],
                }
            ]
        )
    )

    assert [span.buttons for span in spans] == [
        ("LEFT", "A"),
        (),
        ("LEFT", "A"),
        (),
    ]
    assert [span.frames for span in spans] == [4, 1, 4, 1]


def test_problem_lookup_rejects_unknown_id(tmp_path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"problems": [{"problemId": "known"}]}),
        encoding="utf-8",
    )
    catalog = load_problem_catalog(path)

    assert problem_by_id(catalog, "known")["problemId"] == "known"
    with pytest.raises(KeyError):
        problem_by_id(catalog, "missing")


def test_scaffold_policy_is_explicitly_unverified(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.json"
    policy_path = tmp_path / "policy.json"
    catalog_path.write_text(
        json.dumps(
            {
                "problems": [
                    {
                        "problemId": "room_test",
                        "roomId": 1,
                        "roomIdHex": "0x0001",
                        "roomName": "Test Room",
                        "objective": "traverse_to_exit",
                        "entry": None,
                        "exit": {"endpoint": {"orientation": "left"}},
                        "staticPlan": {
                            "status": "planned_static",
                            "pathBlocks": 8,
                            "waypointsBlocks": [[4, 7], [0, 7]],
                        },
                        "practice": {
                            "stateFile": "state.state",
                            "policyFile": "policy.json",
                            "reportFile": "report.json",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = scaffold_room_policy(
        "room_test",
        catalog_path=catalog_path,
        output_path=policy_path,
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert result["orientationHint"] == "left"
    assert policy["status"] == "generated_unverified"
    labels = [step["label"] for step in policy["steps"]]
    assert "open_exit_door" in labels
    assert "enter_exit_door" in labels
    open_step = next(s for s in policy["steps"] if s["label"] == "open_exit_door")
    enter_step = next(s for s in policy["steps"] if s["label"] == "enter_exit_door")
    assert open_step["buttons"] == ["LEFT", "X"]
    # Traverse enter uses approach buttons (jump-run) for door-ledge clearance.
    assert enter_step["buttons"] == ["LEFT", "A", "B"]
    assert policy.get("entryContract", {}).get("kind") == "doorway_natural"
    assert "frameBudget" in result
    # Plan-driven approach (not orientation-specific LEFT magic).
    approach = next(s for s in policy["steps"] if s["label"] == "coarse_exit_approach")
    assert approach["frames"] == result["frameBudget"]["traverse_approach"]


def test_scaffold_same_door_return_uses_path_budget(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.json"
    policy_path = tmp_path / "policy.json"
    catalog_path.write_text(
        json.dumps(
            {
                "problems": [
                    {
                        "problemId": "room_station",
                        "roomId": 2,
                        "roomIdHex": "0x0002",
                        "roomName": "Save Room",
                        "objective": "visit_station_and_return",
                        "entry": {
                            "sourceRoomId": 1,
                            "sourceRoomIdHex": "0x0001",
                            "doorPtr": 0x8F52,
                            "doorPtrHex": "0x8F52",
                            "endpoint": {
                                "orientation": "right",
                                "block": [15, 7],
                            },
                        },
                        "exit": {
                            "targetRoomId": 1,
                            "endpoint": {"orientation": "right"},
                        },
                        "staticPlan": {
                            "status": "planned_static",
                            "pathBlocks": 7,
                        },
                        "practice": {
                            "stateFile": "s.state",
                            "policyFile": "p.json",
                            "reportFile": "r.json",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = scaffold_room_policy(
        "room_station",
        catalog_path=catalog_path,
        output_path=policy_path,
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert result["sameDoorReturn"] is True
    labels = [s["label"] for s in policy["steps"]]
    assert "deeper_into_room" in labels
    deeper = next(s for s in policy["steps"] if s["label"] == "deeper_into_room")
    assert deeper["frames"] == result["frameBudget"]["into"]
    assert policy["entryContract"]["doorPtrHex"] == "0x8F52"
    assert policy["entryContract"]["sameDoorReturn"] is True


def test_scaffold_through_station_uses_traverse_not_same_door(tmp_path) -> None:
    """visit_station naming with a far-door exit must not reverse-out scaffold."""
    catalog_path = tmp_path / "catalog.json"
    policy_path = tmp_path / "policy.json"
    catalog_path.write_text(
        json.dumps(
            {
                "problems": [
                    {
                        "problemId": "room_through_station",
                        "roomId": 2,
                        "roomIdHex": "0x0002",
                        "roomName": "Draygon Save Room",
                        "objective": "visit_station_and_return",
                        "entry": {
                            "sourceRoomId": 9,
                            "sourceRoomIdHex": "0x0009",
                            "doorPtr": 0xA930,
                            "endpoint": {
                                "orientation": "right",
                                "block": [15, 7],
                            },
                        },
                        "exit": {
                            "targetRoomId": 8,
                            "endpoint": {"orientation": "left"},
                        },
                        "staticPlan": {
                            "status": "planned_static",
                            "pathBlocks": 12,
                        },
                        "practice": {
                            "stateFile": "s.state",
                            "policyFile": "p.json",
                            "reportFile": "r.json",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = scaffold_room_policy(
        "room_through_station",
        catalog_path=catalog_path,
        output_path=policy_path,
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert result["sameDoorReturn"] is False
    labels = [s["label"] for s in policy["steps"]]
    assert "coarse_exit_approach" in labels
    assert "deeper_into_room" not in labels
    assert policy["entryContract"]["sameDoorReturn"] is False
    assert policy["entryContract"]["exitTravelDirection"] == "LEFT"


def test_scaffold_frame_budget_scales_with_path_blocks() -> None:
    short = _scaffold_frame_budget({"staticPlan": {"pathBlocks": 3}})
    long = _scaffold_frame_budget({"staticPlan": {"pathBlocks": 40}})
    assert short["approach"] < long["approach"]
    assert short["into"] <= long["into"]
    assert short["enter"] == long["enter"]


def test_promote_requires_green_report_matching_sha(tmp_path) -> None:
    import hashlib

    state_path = tmp_path / "room.state"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    state_path.write_bytes(b"fake-state")
    policy = {
        "schemaVersion": 2,
        "problemId": "room_promote",
        "status": "generated_unverified",
        "steps": [{"label": "idle", "buttons": [], "frames": 1}],
    }
    policy_path.write_text(json.dumps(policy) + "\n", encoding="utf-8")
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    report_path.write_text(
        json.dumps(
            {
                "problemId": "room_promote",
                "success": True,
                "state": {"sha256": state_sha},
                "policy": {"sha256": policy_sha},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "problems": [
                    {
                        "problemId": "room_promote",
                        "roomId": 1,
                        "practice": {
                            "stateFile": str(state_path.relative_to(tmp_path)),
                            "policyFile": str(policy_path.relative_to(tmp_path)),
                            "reportFile": str(report_path.relative_to(tmp_path)),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # promote resolves paths via GAME_DIR — monkeypatch by writing under GAME_DIR
    # is heavy; call with absolute paths by patching _problem_paths via full files
    # under GAME_DIR-style relative paths. Use direct promote with catalog that
    # points at absolute-ish files by patching GAME_DIR usage.
    from super_metroid.rooms import room_practice as rp

    original = rp.GAME_DIR
    try:
        rp.GAME_DIR = tmp_path
        # practice paths are relative to GAME_DIR
        catalog_path.write_text(
            json.dumps(
                {
                    "problems": [
                        {
                            "problemId": "room_promote",
                            "roomId": 1,
                            "practice": {
                                "stateFile": "room.state",
                                "policyFile": "policy.json",
                                "reportFile": "report.json",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        # Recompute policy sha after catalog rewrite is unrelated; re-write report
        # after ensuring policy bytes are final.
        policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        report_path.write_text(
            json.dumps(
                {
                    "problemId": "room_promote",
                    "success": True,
                    "state": {"sha256": state_sha},
                    "policy": {"sha256": policy_sha},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = promote_room_policy(
            "room_promote",
            catalog_path=catalog_path,
            report_path=report_path,
        )
        assert result["promoted"] is True
        updated = json.loads(policy_path.read_text(encoding="utf-8"))
        assert updated["status"] == "verified_development_state"
        assert "promotedAt" in updated

        # Mismatched sha blocks re-promote after editing policy without re-run.
        updated["steps"] = [{"label": "idle", "buttons": [], "frames": 2}]
        updated["status"] = "generated_unverified"
        del updated["promotedAt"]
        policy_path.write_text(json.dumps(updated) + "\n", encoding="utf-8")
        blocked = promote_room_policy(
            "room_promote",
            catalog_path=catalog_path,
            report_path=report_path,
        )
        assert blocked["promoted"] is False
        assert "sha256" in blocked["error"]
    finally:
        rp.GAME_DIR = original


def test_collect_objective_requires_item_capacity_change() -> None:
    problem = {
        "objective": "collect_and_return",
        "items": [{"name": "Missile (Chozo)"}],
    }
    start = SimpleNamespace(
        max_missiles=0,
        max_super_missiles=0,
        max_power_bombs=0,
        max_health=99,
        max_reserve_health=0,
        collected_items=4,
        collected_beams=0,
    )
    missed = SimpleNamespace(**vars(start))
    collected = SimpleNamespace(**{**vars(start), "max_missiles": 5})

    assert "max_missiles did not increase" in _objective_progress_failure(
        problem, start, missed
    )
    assert _objective_progress_failure(problem, start, collected) is None


@pytest.mark.skipif(
    not FULL_ROOM_GRAPH_PATH.is_file() or not ROOM_PROBLEMS_PATH.is_file(),
    reason="generated external-reference catalog is not present",
)
def test_generated_catalog_has_complete_reference_topology() -> None:
    graph = json.loads(FULL_ROOM_GRAPH_PATH.read_text(encoding="utf-8"))
    catalog = load_problem_catalog(ROOM_PROBLEMS_PATH)

    assert graph["summary"] == {
        "roomCount": 262,
        "vanillaReferenceRoomCount": 261,
        "editorOnlyRoomCount": 1,
        "physicalConnectionCount": 300,
        "directedEdgeCount": 583,
        "editorPhysicalComponentCount": 2,
        "vanillaPhysicalComponentCount": 1,
        "isolatedEditorRoomIds": ["0xB3E1"],
        "directionCounts": {"Bidirectional": 283, "Forward": 17},
        "connectionTypeCounts": {
            "Elevator": 7,
            "HorizontalDoor": 254,
            "HorizontalMorphTunnel": 1,
            "StoryMarker": 1,
            "VerticalDoor": 25,
            "VerticalSandpit": 12,
        },
        "completionAnchorCount": 23,
        "completionLegCount": 22,
        "completionTopologyGapCount": 0,
    }
    assert catalog["summary"]["problemCount"] == 262
    assert catalog["summary"]["queueCounts"] == {
        "0": 3,
        "1": 67,
        "2": 38,
        "3": 143,
        "4": 11,
    }
    assert catalog["summary"]["staticPlanStatusCounts"] == {
        "planned_static": 157,
        "unavailable": 1,
        "unresolved": 104,
    }
    assert len({problem["roomId"] for problem in catalog["problems"]}) == 262
    assert all(
        leg["status"] == "planned" for leg in graph["completionSequence"]["legs"]
    )
    assert [anchor["id"] for anchor in graph["completionSequence"]["anchors"]][6:9] == [
        "spore_spawn",
        "spore_spawn_supers",
        "early_power_bombs",
    ]
