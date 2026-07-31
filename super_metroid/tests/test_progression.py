from __future__ import annotations

from dataclasses import replace

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
    assert all(edge.verification == "continuous" for edge in path)


def test_k3_graph_kraid_to_varia() -> None:
    from super_metroid.progression import START_TO_VARIA_GRAPH

    caps = frozenset(
        {"morph_ball", "bombs", "missiles", "super_missiles", "hi_jump"}
    )
    path = START_TO_VARIA_GRAPH.shortest_path(0xA59F, 0xA6E2, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [0xA6E2]
    assert all(edge.verification == "continuous" for edge in path)
    summary = START_TO_VARIA_GRAPH.path_verification(0xA6A1, 0xA6E2, caps)
    assert summary["reachable"] is True
    assert summary["all_continuous"] is True
    assert summary["blocking"] is None


def test_suggest_next_hops_prefers_continuous() -> None:
    from super_metroid.progression import START_TO_VARIA_GRAPH

    caps = frozenset(
        {"morph_ball", "bombs", "missiles", "super_missiles", "hi_jump"}
    )
    hops = START_TO_VARIA_GRAPH.suggest_next_hops(0xA59F, capabilities=caps)
    assert hops
    assert hops[0].edge_id == "kraid_to_varia"
    assert hops[0].verification == "continuous"
    empty = START_TO_VARIA_GRAPH.suggest_next_hops(0xA6E2, capabilities=caps)
    assert empty == ()


def test_k4_graph_varia_to_speed_scaffold() -> None:
    from super_metroid.progression import START_TO_SPEED_GRAPH

    caps = frozenset(
        {
            "morph_ball",
            "bombs",
            "missiles",
            "super_missiles",
            "hi_jump",
            "varia_suit",
        }
    )
    path = START_TO_SPEED_GRAPH.shortest_path(0xA6E2, 0xAD1B, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [
        0xA59F,  # kraid
        0xA56B,  # eye
        0xA521,  # baby
        0xA4DA,  # kihunter
        0xA471,  # zeela
        0xA6A1,  # warehouse
        0xA7DE,  # business
        0xB167,  # frog save
        0xB106,  # frog speedway
        0xAF72,  # farm
        0xACB3,  # bubble
        0xB07A,  # bat cave
        0xACF0,  # speed hall
        0xAD1B,  # speed
    ]
    assert path[0].edge_id == "varia_to_kraid"
    assert path[0].verification == "controller_dev"
    # First non-continuous block is the reverse eye hop until pure reverse exists.
    summary = START_TO_SPEED_GRAPH.path_verification(0xA6E2, 0xAD1B, caps)
    assert summary["reachable"] is True
    assert summary["all_continuous"] is False
    assert summary["blocking"] == "varia_to_kraid"

    hops = START_TO_SPEED_GRAPH.suggest_next_hops(0xA6E2, capabilities=caps)
    assert hops
    assert hops[0].edge_id == "varia_to_kraid"

    bubble = START_TO_SPEED_GRAPH.shortest_path(0xA7DE, 0xACB3, caps)
    assert bubble is not None
    assert [e.target_room_id for e in bubble] == [0xB167, 0xB106, 0xAF72, 0xACB3]

    wave = START_TO_SPEED_GRAPH.shortest_path(0xACB3, 0xADDE, caps)
    assert wave is not None
    assert [e.target_room_id for e in wave] == [0xAD5E, 0xADAD, 0xADDE]

    ice = START_TO_SPEED_GRAPH.shortest_path(0xA7DE, 0xA890, caps)
    assert ice is not None
    assert [e.target_room_id for e in ice] == [0xA815, 0xA865, 0xA8B9, 0xA890]


def test_k4_graph_locks_varia_return_edge_contract() -> None:
    from super_metroid.progression import START_TO_SPEED_GRAPH

    caps = frozenset(
        {
            "morph_ball",
            "bombs",
            "missiles",
            "super_missiles",
            "hi_jump",
            "varia_suit",
        }
    )
    path = START_TO_SPEED_GRAPH.shortest_path(0xA6E2, 0xA7DE, caps)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "varia_to_kraid",
        "kraid_to_eye_return",
        "eye_to_baby_return",
        "baby_to_kihunter_return",
        "kihunter_to_zeela_return",
        "zeela_to_warehouse_return",
        "warehouse_to_business",
    ]
    assert [edge.target_room_id for edge in path] == [
        0xA59F,
        0xA56B,
        0xA521,
        0xA4DA,
        0xA471,
        0xA6A1,
        0xA7DE,
    ]
    assert [edge.verification for edge in path] == [
        "controller_dev",
        "unverified",
        "unverified",
        "unverified",
        "unverified",
        "unverified",
        "continuous",
    ]


def test_k4_graph_keeps_kraid_to_eye_unverified_until_pure_green() -> None:
    from super_metroid.progression import START_TO_SPEED_GRAPH

    edge = next(
        edge
        for edge in START_TO_SPEED_GRAPH.edges
        if edge.edge_id == "kraid_to_eye_return"
    )

    assert edge.verification == "unverified"


def test_k4_reverse_business_path_is_not_continuous_ready_with_eye_unverified() -> None:
    from super_metroid.progression import START_TO_SPEED_GRAPH

    caps = frozenset(
        {
            "morph_ball",
            "bombs",
            "missiles",
            "super_missiles",
            "hi_jump",
            "varia_suit",
        }
    )
    summary = START_TO_SPEED_GRAPH.path_verification(0xA6E2, 0xA7DE, caps)

    assert summary["reachable"] is True
    assert summary["all_continuous"] is False
    assert summary["blocking"] == "varia_to_kraid"
    eye = next(
        edge
        for edge in summary["edges"]
        if edge["edgeId"] == "kraid_to_eye_return"
    )
    assert eye["verification"] == "unverified"


def test_k4_reverse_fixture_promotes_eye_only_after_varia_hop_is_continuous() -> None:
    from super_metroid.progression import RoomProgressionGraph, START_TO_SPEED_GRAPH

    caps = frozenset(
        {
            "morph_ball",
            "bombs",
            "missiles",
            "super_missiles",
            "hi_jump",
            "varia_suit",
        }
    )
    path = START_TO_SPEED_GRAPH.shortest_path(0xA6E2, 0xA56B, caps)
    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "varia_to_kraid",
        "kraid_to_eye_return",
    ]

    rooms = tuple(
        START_TO_SPEED_GRAPH.rooms[room_id]
        for room_id in (0xA6E2, 0xA59F, 0xA56B)
    )
    fixture = RoomProgressionGraph(
        rooms,
        (
            replace(path[0], verification="continuous"),
            path[1],
        ),
        (),
        graph_id="k4_reverse_eye_fixture",
    )

    blocked = fixture.path_verification(0xA6E2, 0xA56B, caps)
    assert blocked["all_continuous"] is False
    assert blocked["blocking"] == "kraid_to_eye_return"

    promoted_fixture = RoomProgressionGraph(
        rooms,
        (
            replace(path[0], verification="continuous"),
            replace(path[1], verification="continuous"),
        ),
        (),
        graph_id="k4_reverse_eye_promoted_fixture",
    )
    unblocked = promoted_fixture.path_verification(0xA6E2, 0xA56B, caps)
    assert unblocked["reachable"] is True
    assert unblocked["all_continuous"] is True
    assert unblocked["blocking"] is None


def test_k4_kraid_next_hop_does_not_rank_eye_return_as_continuous() -> None:
    from super_metroid.progression import START_TO_SPEED_GRAPH

    hops = START_TO_SPEED_GRAPH.suggest_next_hops(
        0xA59F,
        capabilities=frozenset(
            {
                "morph_ball",
                "bombs",
                "missiles",
                "super_missiles",
                "hi_jump",
                "varia_suit",
            }
        ),
    )

    eye = next(edge for edge in hops if edge.edge_id == "kraid_to_eye_return")
    assert hops[0].verification == "continuous"
    assert hops.index(eye) > 0
    assert eye.verification == "unverified"


def test_path_verification_blocks_first_non_continuous_edge_on_local_k4_fixture() -> None:
    """Blocking is the first path edge not marked continuous, not merely unverified."""
    from super_metroid.progression import DoorEdge, RoomNode, RoomProgressionGraph

    rooms = tuple(
        RoomNode(room_id=room_id, name=str(room_id), area="test")
        for room_id in (1, 2, 3)
    )
    varia_to_kraid = DoorEdge(
        edge_id="varia_to_kraid",
        source_room_id=1,
        target_room_id=2,
        exit_direction="left",
        entry_direction="right",
        verification="controller_dev",
    )
    kraid_to_eye_return = DoorEdge(
        edge_id="kraid_to_eye_return",
        source_room_id=2,
        target_room_id=3,
        exit_direction="left",
        entry_direction="right",
        verification="unverified",
    )

    scaffold = RoomProgressionGraph(
        rooms,
        (varia_to_kraid, kraid_to_eye_return),
        (),
        graph_id="local_k4_scaffold",
    )
    summary = scaffold.path_verification(1, 3)
    assert summary["all_continuous"] is False
    assert summary["blocking"] == "varia_to_kraid"

    raised_varia = RoomProgressionGraph(
        rooms,
        (replace(varia_to_kraid, verification="continuous"), kraid_to_eye_return),
        (),
        graph_id="local_k4_varia_continuous",
    )
    summary = raised_varia.path_verification(1, 3)
    assert summary["all_continuous"] is False
    assert summary["blocking"] == "kraid_to_eye_return"


def test_path_verification_rejects_any_unverified_reverse_edge() -> None:
    from super_metroid.progression import DoorEdge, RoomNode, RoomProgressionGraph

    rooms = tuple(
        RoomNode(room_id=room_id, name=str(room_id), area="test")
        for room_id in (1, 2, 3)
    )
    graph = RoomProgressionGraph(
        rooms,
        (
            DoorEdge("first", 1, 2, "right", "left", verification="continuous"),
            DoorEdge("reverse_unverified", 2, 3, "left", "right"),
        ),
        (),
        graph_id="unverified_reverse_guard",
    )

    summary = graph.path_verification(1, 3)

    assert summary["reachable"] is True
    assert summary["all_continuous"] is False
    assert summary["blocking"] == "reverse_unverified"


def test_suggest_next_hops_ranks_continuous_controller_dev_then_unverified() -> None:
    from super_metroid.progression import DoorEdge, RoomNode, RoomProgressionGraph

    rooms = tuple(
        RoomNode(room_id=room_id, name=str(room_id), area="test")
        for room_id in (1, 2, 3, 4)
    )
    graph = RoomProgressionGraph(
        rooms,
        (
            DoorEdge("unverified_hop", 1, 2, "right", "left"),
            DoorEdge(
                "controller_hop",
                1,
                3,
                "right",
                "left",
                verification="controller_dev",
            ),
            DoorEdge(
                "continuous_hop",
                1,
                4,
                "right",
                "left",
                verification="continuous",
            ),
        ),
        (),
        graph_id="verification_ranking_guard",
    )

    hops = graph.suggest_next_hops(1)

    assert [edge.verification for edge in hops] == [
        "continuous",
        "controller_dev",
        "unverified",
    ]


def test_capabilities_from_state_maps_varia_loadout() -> None:
    from types import SimpleNamespace

    from super_metroid.progression import capabilities_from_state
    from super_metroid.ram import (
        BOMBS_MASK,
        HI_JUMP_MASK,
        MORPH_BALL_MASK,
        VARIA_MASK,
    )

    state = SimpleNamespace(
        collected_items=MORPH_BALL_MASK | BOMBS_MASK | HI_JUMP_MASK | VARIA_MASK,
        max_missiles=15,
        max_super_missiles=5,
        max_power_bombs=0,
    )
    caps = capabilities_from_state(state)  # type: ignore[arg-type]
    assert "varia_suit" in caps
    assert "hi_jump" in caps
    assert "super_missiles" in caps
    assert "power_bombs" not in caps
