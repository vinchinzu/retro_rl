from __future__ import annotations

from super_metroid.progression import (
    EARLY_GAME_GRAPH,
    START_TO_MORPH_GRAPH,
    START_TO_SPORE_SPAWN_GRAPH,
)


def test_start_to_morph_room_path_is_connected() -> None:
    path = START_TO_MORPH_GRAPH.shortest_path(0x91F8, 0x9E9F)

    assert path is not None
    assert [edge.target_room_id for edge in path] == [
        0x92FD,
        0x96BA,
        0x975C,
        0x97B5,
        0x9E9F,
    ]


def test_morph_capability_gates_construction_zone() -> None:
    assert START_TO_MORPH_GRAPH.shortest_path(0x9E9F, 0x9F11) is None
    path = START_TO_MORPH_GRAPH.shortest_path(
        0x9E9F,
        0x9F11,
        frozenset({"morph_ball"}),
    )
    assert path is not None
    assert path[0].edge_id == "morph_to_construction"


def test_graph_export_is_json_ready() -> None:
    payload = START_TO_MORPH_GRAPH.to_dict()

    assert payload["graphId"] == "start_to_morph"
    assert len(payload["rooms"]) == 13
    assert len(payload["edges"]) == 17


def test_early_graph_covers_two_missiles_and_torizo_return() -> None:
    path = EARLY_GAME_GRAPH.shortest_path(
        0x9E9F,
        0x9804,
        frozenset({"morph_ball", "missiles"}),
    )

    assert path is not None
    assert any(edge.target_room_id == 0x9F64 for edge in EARLY_GAME_GRAPH.edges)
    assert path[-1].edge_id == "flyway_to_torizo"
    assert {
        milestone.milestone_id for milestone in EARLY_GAME_GRAPH.milestones
    }.issuperset(
        {
            "first_missiles",
            "blue_brinstar_missiles",
            "bombs",
            "bomb_torizo_clear",
        }
    )


def test_bomb_torizo_exit_requires_defeat_capability() -> None:
    assert (
        EARLY_GAME_GRAPH.shortest_path(
            0x9804,
            0x9879,
            frozenset({"bombs"}),
        )
        is None
    )
    path = EARLY_GAME_GRAPH.shortest_path(
        0x9804,
        0x9879,
        frozenset({"bombs", "bomb_torizo_defeated"}),
    )
    assert path is not None
    assert path[0].edge_id == "torizo_to_flyway"


def test_spore_graph_matches_editor_planned_room_sequence() -> None:
    path = START_TO_SPORE_SPAWN_GRAPH.shortest_path(
        0x92FD,
        0x9B5B,
        frozenset(
            {
                "morph_ball",
                "missiles",
                "bombs",
                "bomb_torizo_defeated",
                "spore_spawn_defeated",
            }
        ),
    )

    assert path is not None
    assert [edge.target_room_id for edge in path] == [
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


def test_spore_exit_requires_natural_defeat_capability() -> None:
    assert (
        START_TO_SPORE_SPAWN_GRAPH.shortest_path(
            0x9DC7,
            0x9B5B,
            frozenset({"missiles"}),
        )
        is None
    )
