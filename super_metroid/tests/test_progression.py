from __future__ import annotations

from super_metroid.progression import (
    EARLY_GAME_GRAPH,
    START_TO_MORPH_GRAPH,
    START_TO_RED_TOWER_GRAPH,
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


def test_k1_graph_super_room_to_red_tower() -> None:
    path = START_TO_RED_TOWER_GRAPH.shortest_path(
        0x9B5B,
        0xA253,
        frozenset(
            {
                "morph_ball",
                "bombs",
                "missiles",
                "super_missiles",
            }
        ),
    )
    assert path is not None
    assert [edge.target_room_id for edge in path] == [
        0xA0A4,
        0x9D19,
        0x9E52,
        0x9FBA,
        0xA253,
    ]
    assert all(edge.verification == "continuous" for edge in path)


def test_k2_graph_red_tower_to_bat() -> None:
    from super_metroid.progression import START_TO_BAT_GRAPH

    path = START_TO_BAT_GRAPH.shortest_path(
        0xA253,
        0xA3DD,
        frozenset(
            {
                "morph_ball",
                "bombs",
                "missiles",
                "super_missiles",
            }
        ),
    )
    assert path is not None
    assert [edge.target_room_id for edge in path] == [0xA3DD]
    assert all(edge.verification == "continuous" for edge in path)


def test_k2_graph_bat_to_below_spazer() -> None:
    from super_metroid.progression import START_TO_BELOW_SPAZER_GRAPH

    path = START_TO_BELOW_SPAZER_GRAPH.shortest_path(
        0xA3DD,
        0xA408,
        frozenset(
            {
                "morph_ball",
                "bombs",
                "missiles",
                "super_missiles",
            }
        ),
    )
    assert path is not None
    assert [edge.target_room_id for edge in path] == [0xA408]
    assert all(edge.verification == "continuous" for edge in path)


def test_k2_graph_below_spazer_to_warehouse() -> None:
    from super_metroid.progression import START_TO_WAREHOUSE_GRAPH

    caps = frozenset(
        {
            "morph_ball",
            "bombs",
            "missiles",
            "super_missiles",
        }
    )
    path = START_TO_WAREHOUSE_GRAPH.shortest_path(0xA408, 0xA6A1, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [
        0xCF54,
        0xCEFB,
        0xCF80,
        0xA6A1,
    ]
    assert all(edge.verification == "continuous" for edge in path)


def test_k2_graph_warehouse_to_hijump() -> None:
    from super_metroid.progression import START_TO_HIJUMP_GRAPH

    caps = frozenset({"morph_ball", "bombs", "missiles", "super_missiles"})
    path = START_TO_HIJUMP_GRAPH.shortest_path(0xA6A1, 0xA9E5, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [0xA7DE, 0xAA41, 0xA9E5]


def test_k2_graph_hijump_return_to_kraid() -> None:
    from super_metroid.progression import START_TO_KRAID_GRAPH

    caps = frozenset(
        {"morph_ball", "bombs", "missiles", "super_missiles", "hi_jump"}
    )
    path = START_TO_KRAID_GRAPH.shortest_path(0xA9E5, 0xA59F, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [
        0xAA41,
        0xA7DE,
        0xA6A1,
        0xA471,
        0xA4DA,
        0xA521,
        0xA56B,
        0xA59F,
    ]


def test_k3_graph_kraid_to_varia() -> None:
    from super_metroid.progression import START_TO_VARIA_GRAPH

    caps = frozenset(
        {"morph_ball", "bombs", "missiles", "super_missiles", "hi_jump"}
    )
    path = START_TO_VARIA_GRAPH.shortest_path(0xA59F, 0xA6E2, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [0xA6E2]
