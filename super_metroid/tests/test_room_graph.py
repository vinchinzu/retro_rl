from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from super_metroid.paths import FULL_ROOM_GRAPH_PATH, ROOM_PROBLEMS_PATH
from super_metroid.room_graph import (
    PhysicalConnection,
    PhysicalEndpoint,
    load_problem_catalog,
    problem_by_id,
    shortest_room_path,
)
from super_metroid.room_practice import (
    _expand_steps,
    _objective_progress_failure,
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
                        "roomIdHex": "0x0001",
                        "roomName": "Test Room",
                        "objective": "traverse_to_exit",
                        "entry": None,
                        "exit": {"endpoint": {"orientation": "left"}},
                        "staticPlan": {
                            "status": "planned_static",
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
    assert policy["steps"][-1]["buttons"] == ["LEFT", "B", "X"]


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
