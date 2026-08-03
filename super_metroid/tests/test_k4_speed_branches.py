from __future__ import annotations

from super_metroid.progression import START_TO_SPEED_GRAPH


CAPS = frozenset(
    {
        "morph_ball",
        "bombs",
        "missiles",
        "super_missiles",
        "hi_jump",
        "varia_suit",
    }
)


def test_k4_business_to_bubble_edge_contract() -> None:
    """No-Speed first Bubble visit is Cathedral climb (not Frog Speedway)."""
    path = START_TO_SPEED_GRAPH.shortest_path(0xA7DE, 0xACB3, CAPS)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "business_to_cathedral_entrance",
        "cathedral_entrance_to_cathedral",
        "cathedral_to_rising_tide",
        "rising_tide_to_bubble",
    ]
    # Pure-first Cathedral stack; continuous tip still Frog Save only.
    assert all(
        edge.verification in ("controller_dev", "pure", "unverified", "continuous")
        for edge in path
    )


def test_k4_bubble_to_wave_edge_contract() -> None:
    path = START_TO_SPEED_GRAPH.shortest_path(0xACB3, 0xADDE, CAPS)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "bubble_to_single_chamber",
        "single_to_double_chamber",
        "double_chamber_to_wave",
    ]
    assert [edge.verification for edge in path] == [
        "unverified",
        "unverified",
        "unverified",
    ]


def test_k4_business_to_ice_edge_contract() -> None:
    path = START_TO_SPEED_GRAPH.shortest_path(0xA7DE, 0xA890, CAPS)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "business_to_ice_gate",
        "ice_gate_to_tutorial",
        "ice_tutorial_to_snake",
        "ice_snake_to_ice",
    ]
    assert [edge.verification for edge in path] == [
        "unverified",
        "unverified",
        "unverified",
        "unverified",
    ]


def test_k4_branch_path_verification_blocks_first_unverified_edge() -> None:
    wave = START_TO_SPEED_GRAPH.path_verification(0xA7DE, 0xADDE, CAPS)
    ice = START_TO_SPEED_GRAPH.path_verification(0xA7DE, 0xA890, CAPS)

    assert wave["reachable"] is True
    assert wave["all_continuous"] is False
    # First Bubble path is Cathedral; first non-continuous hop blocks wave path.
    assert wave["blocking"] == "business_to_cathedral_entrance"
    assert ice["reachable"] is True
    assert ice["all_continuous"] is False
    assert ice["blocking"] == "business_to_ice_gate"


def test_k4_speed_path_includes_farm_and_speed_hall_hops() -> None:
    path = START_TO_SPEED_GRAPH.shortest_path(0xACB3, 0xAD1B, CAPS)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "bubble_to_bat_cave",
        "bat_cave_to_speed_hall",
        "speed_hall_to_speed",
    ]
