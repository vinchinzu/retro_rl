from __future__ import annotations

import json

from super_metroid.map_planning import EditorNavigationGraph
from super_metroid.routes.spore_spawn_route import (
    POST_TORIZO_CAPABILITIES,
    POST_TORIZO_ROUTE_PATCHES,
    POST_TORIZO_TO_SPORE_SPAWN,
)


_ROOMS = (
    (0x92FD, "Parlor"),
    (0x990D, "Terminator"),
    (0x99BD, "Green Pirates"),
    (0x9969, "Lower Mushrooms"),
    (0x9938, "Green Elevator"),
    (0x9AD9, "Main Shaft"),
    (0x9BC8, "Early Supers"),
    (0x9CB3, "Dachora"),
    (0x9D19, "Big Pink"),
    (0x9D9C, "Spore Kihunters"),
    (0x9DC7, "Spore Spawn"),
    (0x9B5B, "Spore Super"),
)

_EDITOR_EDGES = (
    (0x92FD, 0x990D, None),
    (0x990D, 0x99BD, None),
    (0x99BD, 0x9969, None),
    (0x9969, 0x9938, None),
    (0x9938, 0x9AD9, None),
    # The editor only exports the reverse sides of these two doors.
    (0x9BC8, 0x9AD9, None),
    (0x9CB3, 0x9AD9, None),
    (0x9CB3, 0x9D19, None),
    (0x9D19, 0x9D9C, None),
    # The forward editor edge currently omits its green-door ability.
    (0x9D9C, 0x9DC7, None),
    (0x9DC7, 0x9B5B, None),
)


def _graph(tmp_path):
    payload = {
        "nodes": [
            {
                "roomId": room_id,
                "name": name,
                "areaName": "test",
                "handle": name.lower().replace(" ", "_"),
                "mapX": index,
                "mapY": 0,
                "widthScreens": 1,
                "heightScreens": 1,
            }
            for index, (room_id, name) in enumerate(_ROOMS)
        ],
        "edges": [
            {
                "fromRoomId": source,
                "toRoomId": target,
                "direction": "Right",
                "isElevator": False,
                "doorCapColor": None,
                "requiredAbility": required,
            }
            for source, target, required in _EDITOR_EDGES
        ],
    }
    path = tmp_path / "nav_graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return EditorNavigationGraph.load(path)


def test_editor_route_is_contiguous_and_reaches_post_spore_room(tmp_path) -> None:
    graph = _graph(tmp_path).add_patches(POST_TORIZO_ROUTE_PATCHES)

    planned = graph.plan_legs(
        POST_TORIZO_TO_SPORE_SPAWN,
        initial_capabilities=POST_TORIZO_CAPABILITIES,
    )

    assert [item.leg.target_id for item in planned] == [
        0x990D,
        0x99BD,
        0x9969,
        0x9938,
        0x9AD9,
        0x9CB3,
        0x9D19,
        0x9D9C,
        0x9DC7,
        0x9B5B,
    ]
    assert "super_missiles" not in planned[-1].capabilities_after
    assert "spore_spawn_defeated" in planned[-1].capabilities_after
    assert all(item.edge.verification == "planned" for item in planned)


def test_route_does_not_require_early_supers_or_mockball(tmp_path) -> None:
    graph = _graph(tmp_path).add_patches(POST_TORIZO_ROUTE_PATCHES)

    planned = graph.plan_legs(
        POST_TORIZO_TO_SPORE_SPAWN,
        initial_capabilities=frozenset({"morph_ball", "bombs", "missiles"}),
    )

    assert all(item.leg.target_id != 0x9BC8 for item in planned)
    assert all("mockball" not in item.effective_requires for item in planned)
    assert all("super_missiles" not in item.effective_requires for item in planned)


def test_missing_directed_editor_edges_are_explicit_patches(tmp_path) -> None:
    base = _graph(tmp_path)

    assert base.edge_for(0x9AD9, 0x9BC8) is None
    assert base.edge_for(0x9AD9, 0x9CB3) is None

    patched = base.add_patches(POST_TORIZO_ROUTE_PATCHES)
    assert patched.edge_for(0x9AD9, 0x9BC8) is None
    assert patched.edge_for(0x9AD9, 0x9CB3).provenance == "explicit_route_patch"


def test_shortest_path_respects_editor_ability_gates(tmp_path) -> None:
    graph = _graph(tmp_path).add_patches(POST_TORIZO_ROUTE_PATCHES)

    assert graph.shortest_path(0x9AD9, 0x9CB3) is None
    path = graph.shortest_path(
        0x9AD9,
        0x9CB3,
        frozenset({"missiles"}),
    )
    assert path is not None
    assert path[0].provenance == "explicit_route_patch"
