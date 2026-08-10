from __future__ import annotations

from super_metroid.progression import SPEED_GRAPH


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
    path = SPEED_GRAPH.shortest_path(0xA7DE, 0xACB3, CAPS)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "business_to_cathedral_entrance",
        "cathedral_entrance_to_cathedral",
        "cathedral_to_rising_tide",
        "rising_tide_to_bubble",
    ]
    # Dual integrity bat_cave promoted Cathedral path to continuous.
    assert all(edge.verification == "continuous" for edge in path)


def test_k4_bubble_to_wave_edge_contract() -> None:
    path = SPEED_GRAPH.shortest_path(0xACB3, 0xADDE, CAPS)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "bubble_to_single_chamber",
        "single_to_double_chamber",
        "double_chamber_to_wave",
    ]
    # Wave continuous tip (K4.10 dual) promotes branch edges.
    assert all(edge.verification == "continuous" for edge in path)


def test_k4_business_to_ice_edge_contract() -> None:
    # Without Speed: graph still has Tutorial return path (no Acid Boost Blocks).
    # Outbound Ice hops are spine-emitted continuous (tip ``ice`` compose).
    path = SPEED_GRAPH.shortest_path(0xA7DE, 0xA890, CAPS)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "business_to_ice_gate",
        "ice_gate_to_tutorial",
        "ice_tutorial_to_snake",
        "ice_snake_to_ice",
    ]
    assert [edge.verification for edge in path] == [
        "continuous",  # spine tip ice (rr-dbu.7 compose)
        "unverified",
        "unverified",
        "continuous",  # spine tip ice
    ]

    # With Speed: tape entry Gate → Acid → Snake (pure dual GREEN + compose).
    caps_speed = CAPS | frozenset({"speed_booster"})
    path_s = SPEED_GRAPH.shortest_path(0xA7DE, 0xA890, caps_speed)
    assert path_s is not None
    assert [edge.edge_id for edge in path_s] == [
        "business_to_ice_gate",
        "ice_gate_to_acid",
        "ice_acid_to_snake",
        "ice_snake_to_ice",
    ]
    assert all(edge.verification == "continuous" for edge in path_s)


def test_k4_branch_path_verification_blocks_first_unverified_edge() -> None:
    wave = SPEED_GRAPH.path_verification(0xA7DE, 0xADDE, CAPS)
    ice = SPEED_GRAPH.path_verification(0xA7DE, 0xA890, CAPS)
    speed = SPEED_GRAPH.path_verification(0xACB3, 0xAD1B, CAPS)

    assert wave["reachable"] is True
    # Business→…→Wave is continuous (default tip).
    assert wave["all_continuous"] is True
    assert wave["blocking"] is None
    assert ice["reachable"] is True
    # Without Speed, path uses Tutorial return edges (unverified) — blocks.
    assert ice["all_continuous"] is False
    assert ice["blocking"] == "ice_gate_to_tutorial"
    # Speed path: all outbound Ice edges continuous (compose wiring).
    caps_speed = CAPS | frozenset({"speed_booster"})
    ice_speed = SPEED_GRAPH.path_verification(0xA7DE, 0xA890, caps_speed)
    assert ice_speed["reachable"] is True
    assert ice_speed["all_continuous"] is True
    assert ice_speed["blocking"] is None
    # Speed Hall + collect are continuous spine product edges.
    assert speed["reachable"] is True
    assert speed["all_continuous"] is True


def test_k4_speed_path_includes_farm_and_speed_hall_hops() -> None:
    path = SPEED_GRAPH.shortest_path(0xACB3, 0xAD1B, CAPS)

    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "bubble_to_bat_cave",
        "bat_cave_to_speed_hall",
        "speed_hall_to_speed",
    ]
    # Dual integrity bat_cave + pure-green Speed hops on spine → continuous.
    assert path[0].verification == "continuous"
    assert path[1].verification == "continuous"
    assert path[2].verification == "continuous"
