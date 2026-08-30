from __future__ import annotations

from dataclasses import replace

from super_metroid.progression import (
    EARLY_GAME_GRAPH,
    MORPH_GRAPH,
    RED_TOWER_GRAPH,
    SPORE_GRAPH,
)


def test_morph_room_path_is_connected() -> None:
    path = MORPH_GRAPH.shortest_path(0x91F8, 0x9E9F)

    assert path is not None
    assert [edge.target_room_id for edge in path] == [
        0x92FD,
        0x96BA,
        0x975C,
        0x97B5,
        0x9E9F,
    ]


def test_morph_capability_gates_construction_zone() -> None:
    assert MORPH_GRAPH.shortest_path(0x9E9F, 0x9F11) is None
    path = MORPH_GRAPH.shortest_path(
        0x9E9F,
        0x9F11,
        frozenset({"morph_ball"}),
    )
    assert path is not None
    assert path[0].edge_id == "morph_to_construction"


def test_graph_export_is_json_ready() -> None:
    payload = MORPH_GRAPH.to_dict()

    assert payload["graphId"] == "morph"
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
    path = SPORE_GRAPH.shortest_path(
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
        SPORE_GRAPH.shortest_path(
            0x9DC7,
            0x9B5B,
            frozenset({"missiles"}),
        )
        is None
    )


def test_k1_graph_super_room_to_red_tower() -> None:
    path = RED_TOWER_GRAPH.shortest_path(
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
    from super_metroid.progression import BAT_GRAPH

    path = BAT_GRAPH.shortest_path(
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
    from super_metroid.progression import BELOW_SPAZER_GRAPH

    path = BELOW_SPAZER_GRAPH.shortest_path(
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
    from super_metroid.progression import WAREHOUSE_GRAPH

    caps = frozenset(
        {
            "morph_ball",
            "bombs",
            "missiles",
            "super_missiles",
        }
    )
    path = WAREHOUSE_GRAPH.shortest_path(0xA408, 0xA6A1, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [
        0xCF54,
        0xCEFB,
        0xCF80,
        0xA6A1,
    ]
    assert all(edge.verification == "continuous" for edge in path)


def test_k2_graph_warehouse_to_hijump() -> None:
    from super_metroid.progression import HIJUMP_GRAPH

    caps = frozenset({"morph_ball", "bombs", "missiles", "super_missiles"})
    path = HIJUMP_GRAPH.shortest_path(0xA6A1, 0xA9E5, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [0xA7DE, 0xAA41, 0xA9E5]


def test_k2_graph_hijump_return_to_kraid() -> None:
    from super_metroid.progression import KRAID_GRAPH

    caps = frozenset({"morph_ball", "bombs", "missiles", "super_missiles", "hi_jump"})
    path = KRAID_GRAPH.shortest_path(0xA9E5, 0xA59F, caps)
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
    from super_metroid.progression import VARIA_GRAPH

    caps = frozenset({"morph_ball", "bombs", "missiles", "super_missiles", "hi_jump"})
    path = VARIA_GRAPH.shortest_path(0xA59F, 0xA6E2, caps)
    assert path is not None
    assert [edge.target_room_id for edge in path] == [0xA6E2]
    assert all(edge.verification == "continuous" for edge in path)
    summary = VARIA_GRAPH.path_verification(0xA6A1, 0xA6E2, caps)
    assert summary["reachable"] is True
    assert summary["all_continuous"] is True
    assert summary["blocking"] is None


def test_suggest_next_hops_prefers_continuous() -> None:
    from super_metroid.progression import VARIA_GRAPH

    caps = frozenset({"morph_ball", "bombs", "missiles", "super_missiles", "hi_jump"})
    hops = VARIA_GRAPH.suggest_next_hops(0xA59F, capabilities=caps)
    assert hops
    assert hops[0].edge_id == "kraid_to_varia"
    assert hops[0].verification == "continuous"
    empty = VARIA_GRAPH.suggest_next_hops(0xA6E2, capabilities=caps)
    assert empty == ()


def test_k4_graph_varia_to_speed_scaffold() -> None:
    from super_metroid.progression import SPEED_GRAPH

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
    path = SPEED_GRAPH.shortest_path(0xA6E2, 0xAD1B, caps)
    assert path is not None
    # First Bubble visit is Cathedral climb (Frog Speedway needs Speed).
    assert [edge.target_room_id for edge in path] == [
        0xA59F,  # kraid
        0xA56B,  # eye
        0xA521,  # baby
        0xA4DA,  # kihunter
        0xA471,  # zeela
        0xA6A1,  # warehouse
        0xA7DE,  # business
        0xA7B3,  # cathedral entrance
        0xA788,  # cathedral
        0xAFA3,  # rising tide
        0xACB3,  # bubble
        0xB07A,  # bat cave
        0xACF0,  # speed hall
        0xAD1B,  # speed
    ]
    assert path[0].edge_id == "varia_to_kraid"
    assert path[0].verification == "continuous"
    # Return spine continuous through Speed collect (STATUS-promoted tip).
    summary = SPEED_GRAPH.path_verification(0xA6E2, 0xAD1B, caps)
    assert summary["reachable"] is True
    assert summary["all_continuous"] is True
    assert summary["blocking"] is None

    hops = SPEED_GRAPH.suggest_next_hops(0xA6E2, capabilities=caps)
    assert hops
    assert hops[0].edge_id == "varia_to_kraid"

    bubble = SPEED_GRAPH.shortest_path(0xA7DE, 0xACB3, caps)
    assert bubble is not None
    assert [e.target_room_id for e in bubble] == [0xA7B3, 0xA788, 0xAFA3, 0xACB3]
    assert [e.edge_id for e in bubble] == [
        "business_to_cathedral_entrance",
        "cathedral_entrance_to_cathedral",
        "cathedral_to_rising_tide",
        "rising_tide_to_bubble",
    ]

    wave = SPEED_GRAPH.shortest_path(0xACB3, 0xADDE, caps)
    assert wave is not None
    assert [e.target_room_id for e in wave] == [0xAD5E, 0xADAD, 0xADDE]

    ice = SPEED_GRAPH.shortest_path(0xA7DE, 0xA890, caps)
    assert ice is not None
    assert [e.target_room_id for e in ice] == [0xA815, 0xA865, 0xA8B9, 0xA890]


def test_k4_graph_locks_varia_return_edge_contract() -> None:
    from super_metroid.progression import SPEED_GRAPH

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
    path = SPEED_GRAPH.shortest_path(0xA6E2, 0xA7DE, caps)

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
        "continuous",
        "continuous",
        "continuous",
        "continuous",
        "continuous",
        "continuous",
        "continuous",
    ]


def test_k4_reverse_path_is_continuous_after_business_tip_acceptance() -> None:
    from super_metroid.progression import SPEED_GRAPH

    edges = {edge.edge_id: edge for edge in SPEED_GRAPH.edges}

    assert edges["kihunter_to_zeela_return"].verification == "continuous"
    assert edges["zeela_to_warehouse_return"].verification == "continuous"
    assert [
        edges[edge_id].verification
        for edge_id in (
            "kraid_to_eye_return",
            "eye_to_baby_return",
            "baby_to_kihunter_return",
        )
    ] == ["continuous", "continuous", "continuous"]

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
    summary = SPEED_GRAPH.path_verification(0xA59F, 0xA6A1, caps)

    assert summary["reachable"] is True
    assert summary["all_continuous"] is True
    assert summary["blocking"] is None
    assert [edge["verification"] for edge in summary["edges"]] == [
        "continuous",
        "continuous",
        "continuous",
        "continuous",
        "continuous",
    ]


def test_k4_graph_locks_kraid_to_eye_continuous_after_business_tip() -> None:
    from super_metroid.progression import SPEED_GRAPH

    edge = next(
        edge
        for edge in SPEED_GRAPH.edges
        if edge.edge_id == "kraid_to_eye_return"
    )

    assert edge.verification == "continuous"


def test_k4_reverse_business_path_is_continuous_ready() -> None:
    from super_metroid.progression import SPEED_GRAPH

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
    summary = SPEED_GRAPH.path_verification(0xA6E2, 0xA7DE, caps)

    assert summary["reachable"] is True
    assert summary["all_continuous"] is True
    assert summary["blocking"] is None
    eye = next(
        edge for edge in summary["edges"] if edge["edgeId"] == "kraid_to_eye_return"
    )
    assert eye["verification"] == "continuous"


def test_k4_reverse_fixture_blocks_only_when_downgraded() -> None:
    from super_metroid.progression import RoomProgressionGraph, SPEED_GRAPH

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
    path = SPEED_GRAPH.shortest_path(0xA6E2, 0xA56B, caps)
    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "varia_to_kraid",
        "kraid_to_eye_return",
    ]

    rooms = tuple(
        SPEED_GRAPH.rooms[room_id] for room_id in (0xA6E2, 0xA59F, 0xA56B)
    )
    fixture = RoomProgressionGraph(
        rooms,
        (
            path[0],
            replace(path[1], verification="controller_dev"),
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


def test_k4_kraid_return_hop_is_ranked_continuous() -> None:
    from super_metroid.progression import SPEED_GRAPH

    hops = SPEED_GRAPH.suggest_next_hops(
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
    assert hops.index(eye) == 0
    assert eye.verification == "continuous"


def test_morph_spine_edges_join_morph_graph() -> None:
    """Morph seed DoorEdges from early spine match MORPH_GRAPH; full Ceres bulk stays hand list."""
    from super_metroid.progression import MORPH_GRAPH
    from super_metroid.routes.kpdr.early_spine import (
        MORPH_DOOR_EDGES,
        MORPH_SPINE,
        continuous_edges_from_morph_spine,
    )

    assert MORPH_GRAPH.edges == MORPH_DOOR_EDGES
    assert len(MORPH_SPINE) == 9
    hop_ids = [h.hop_id for h in MORPH_SPINE]
    assert hop_ids[0] == "first_ceres_control"
    assert hop_ids[-1] == "morph_ball"

    by_id = {edge.edge_id: edge for edge in MORPH_GRAPH.edges}
    for edge in continuous_edges_from_morph_spine():
        graph_edge = by_id[edge.edge_id]
        assert graph_edge.source_room_id == edge.source_room_id
        assert graph_edge.target_room_id == edge.target_room_id
        assert graph_edge.policy_id == edge.policy_id
        assert graph_edge.verification == "continuous"
    # Multi-room Ceres bulk intermediates are hand edges, not play-hop emits.
    assert "ceres_elevator_to_falling" in by_id
    assert "ceres_elevator_to_falling" not in {
        e.edge_id for e in continuous_edges_from_morph_spine()
    }


def test_early_post_morph_spines_shape() -> None:
    """Bombs/Spore/Supers play spines: ordered SpineHop rows; edges stay in data.py."""
    from super_metroid.routes.kpdr.early_post_morph import (
        BOMBS_SPINE,
        SPORE_SPINE,
        SUPERS_SPINE,
        TWO_MISSILES,
        CONSTRUCTION_RETURN,
        ELEVATOR_RETURN,
        PIT_TO_POST_TORIZO,
    )

    assert [h.hop_id for h in BOMBS_SPINE] == [
        "two_missile_detour",
        "construction_return",
        "elevator_return",
        "pit_natural_entry",
        "pit_to_post_torizo",
    ]
    assert all(h.tip_id == "bombs" for h in BOMBS_SPINE)
    assert all(not h.use_transition_split for h in BOMBS_SPINE)
    # Policy JSON paths unchanged (hash-pinned product path).
    assert TWO_MISSILES.filename == "two_missile_detour.json"
    assert CONSTRUCTION_RETURN.filename == "construction_to_elevator.json"
    assert ELEVATOR_RETURN.filename == "elevator_to_pit.json"
    assert PIT_TO_POST_TORIZO.filename == "pit_to_post_torizo.json"

    assert [h.hop_id for h in SPORE_SPINE] == [
        "parlor_to_main_shaft",
        "main_shaft_to_spore_exit",
    ]
    assert all(h.tip_id == "spore" for h in SPORE_SPINE)
    assert SPORE_SPINE[0].to_room == 0x9AD9
    assert SPORE_SPINE[1].to_room == 0x9B5B

    assert [h.hop_id for h in SUPERS_SPINE] == ["spore_supers_collected"]
    assert SUPERS_SPINE[0].tip_id == "supers"
    assert SUPERS_SPINE[0].from_room == SUPERS_SPINE[0].to_room == 0x9B5B


def test_super_plus_continuous_door_edges_come_from_spine() -> None:
    """Super+ product door edges are generated from SpineHop door meta."""
    from super_metroid.progression import SPEED_GRAPH
    from super_metroid.routes.kpdr.spine import (
        POST_SUPERS_SPINE,
        continuous_edges_from_spine,
    )

    spine_edges = continuous_edges_from_spine()
    assert len(spine_edges) >= 30
    # In-room milestones + hop replays / multi-room reverses do not emit edges.
    non_emitting = {h.hop_id for h in POST_SUPERS_SPINE if not h.emits_door_edge}
    assert non_emitting == {
        "big_pink_main",
        "hijump_collected",
        "warehouse_to_business_return",
        "speed_return_to_bubble",  # multi-room reverse; not a single door
        "ice_business_to_warehouse",  # replays kraid-tip business_to_warehouse
        "phantoon_fight",  # in-room; boss bit, no door
        # Gravity scratch: only Basement→Main emits a product door.
        "ws_main_to_attic",
        "attic_to_west_ocean",
        "west_ocean_to_pancakes",
        "pancakes_to_homing_geemer",
        "homing_geemer_to_bowling",
        "bowling_to_gravity",
        "gravity_collect",
    }

    by_id = {edge.edge_id: edge for edge in SPEED_GRAPH.edges}
    for edge in spine_edges:
        graph_edge = by_id[edge.edge_id]
        assert graph_edge is edge or (
            graph_edge.source_room_id == edge.source_room_id
            and graph_edge.target_room_id == edge.target_room_id
            and graph_edge.exit_direction == edge.exit_direction
            and graph_edge.entry_direction == edge.entry_direction
            and graph_edge.requires == edge.requires
            and graph_edge.policy_id == edge.policy_id
            and graph_edge.verification == edge.verification
        )
        assert edge.verification == "continuous"

    # Historical edge_id overrides (hop_id may differ for splits).
    assert by_id["super_room_to_farming"].source_room_id == 0x9B5B
    assert by_id["varia_to_kraid"].verification == "continuous"
    assert "warehouse_to_business" in by_id


def test_path_verification_blocks_first_non_continuous_edge_on_local_k4_fixture() -> (
    None
):
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
