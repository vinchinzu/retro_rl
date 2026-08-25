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


def test_ice_to_moat_path_uses_k5_k6_spine() -> None:
    """Post-Ice Alpha PB + Moat hops are spine-emitted (compose wiring)."""
    caps_speed = CAPS | frozenset({"speed_booster"})
    to_pb = SPEED_GRAPH.shortest_path(0xA890, 0xA3AE, caps_speed)
    assert to_pb is not None
    assert [edge.edge_id for edge in to_pb] == [
        "ice_to_snake",
        "ice_snake_to_tutorial",
        "ice_tutorial_to_gate",
        "ice_gate_to_business",
        "business_to_warehouse",
        "warehouse_to_east",
        "east_to_glass",
        "glass_to_west",
        "west_to_below",
        "below_to_bat",
        "bat_to_red",
        "red_to_hellway",
        "hellway_to_caterpillar",
        "caterpillar_to_alpha_pb",
    ]
    assert all(edge.verification == "continuous" for edge in to_pb)
    to_ocean = SPEED_GRAPH.shortest_path(0xA3AE, 0x93FE, caps_speed)
    assert to_ocean is not None
    assert [edge.edge_id for edge in to_ocean] == [
        "alpha_pb_to_caterpillar",
        "caterpillar_to_elevator",
        "elevator_to_kihunter",
        "kihunter_to_moat",
        "moat_cross",
    ]
    assert all(edge.verification == "continuous" for edge in to_ocean)
    # Reverse door used by play_moat_cross Moat standing setup (not a hop split).
    reverse = SPEED_GRAPH.edge_for(0x95FF, 0x948C)
    assert reverse is not None
    assert reverse.edge_id == "moat_to_kihunter"
    summary = SPEED_GRAPH.path_verification(0xA890, 0x93FE, caps_speed)
    assert summary["reachable"] is True
    assert summary["all_continuous"] is True
    assert summary["blocking"] is None


def test_west_ocean_to_ws_path_is_spine_hop() -> None:
    """West Ocean → WS Entrance is the ws-tip over-ocean spark hop."""
    caps_speed = CAPS | frozenset({"speed_booster"})
    path = SPEED_GRAPH.shortest_path(0x93FE, 0xCA08, caps_speed)
    assert path is not None
    assert [edge.edge_id for edge in path] == ["west_ocean_to_ws"]
    assert all(edge.verification == "continuous" for edge in path)
    summary = SPEED_GRAPH.path_verification(0x93FE, 0xCA08, caps_speed)
    assert summary["reachable"] is True
    assert summary["all_continuous"] is True
    assert summary["blocking"] is None


def test_ws_entrance_to_phantoon_room_is_spine_path() -> None:
    """Unpowered ship interior hops; fight is in-room (no DoorEdge)."""
    caps = CAPS | frozenset({"speed_booster"})
    path = SPEED_GRAPH.shortest_path(0xCA08, 0xCD13, caps)
    assert path is not None
    assert [edge.edge_id for edge in path] == [
        "ws_entrance_to_main",
        "ws_main_to_basement",
        "ws_basement_to_phantoon",
    ]
    assert all(edge.verification == "continuous" for edge in path)
    assert SPEED_GRAPH.edge_for(0xCD13, 0xCD13) is None
    leave = SPEED_GRAPH.edge_for(0xCD13, 0xCC6F)
    assert leave is not None
    assert leave.edge_id == "phantoon_loot_exit"
    assert leave.exit_direction == "left"
    assert leave.entry_direction == "right"
    assert leave.verification == "continuous"


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
