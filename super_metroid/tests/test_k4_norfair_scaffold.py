from __future__ import annotations

from collections.abc import Callable

from super_metroid.progression import START_TO_SPEED_GRAPH
from super_metroid.routes.kpdr import k4_norfair


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


def test_k4_norfair_segment_callables_are_importable() -> None:
    segments: tuple[Callable[..., object], ...] = (
        k4_norfair.play_business_to_frog_save,
        k4_norfair.play_frog_save_to_business,
        k4_norfair.play_business_to_cathedral_entrance,
        k4_norfair.play_cathedral_entrance_to_cathedral,
        k4_norfair.play_cathedral_to_rising_tide,
        k4_norfair.play_rising_tide_to_bubble,
        k4_norfair.play_frog_save_to_speedway,
        k4_norfair.play_speedway_to_farm,
        k4_norfair.play_farm_to_bubble,
    )

    assert all(callable(segment) for segment in segments)


def test_cathedral_climb_segments_are_registered() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("business_to_cathedral_entrance")
        is k4_norfair.play_business_to_cathedral_entrance
    )
    assert get_segment("frog_save_to_business") is k4_norfair.play_frog_save_to_business


def test_business_to_frog_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert get_segment("business_to_frog_save") is k4_norfair.play_business_to_frog_save


def test_business_to_cathedral_entrance_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("business_to_cathedral_entrance")
        is k4_norfair.play_business_to_cathedral_entrance
    )


def test_cathedral_entrance_to_cathedral_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("cathedral_entrance_to_cathedral")
        is k4_norfair.play_cathedral_entrance_to_cathedral
    )


def test_frog_save_to_speedway_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("frog_save_to_speedway")
        is k4_norfair.play_frog_save_to_speedway
    )


def test_speedway_to_farm_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert get_segment("speedway_to_farm") is k4_norfair.play_speedway_to_farm


def test_k4_norfair_constants_match_graph_path() -> None:
    """No-Speed first Bubble visit is Cathedral climb (not Frog Speedway)."""
    path = START_TO_SPEED_GRAPH.shortest_path(
        k4_norfair.ROOM_BUSINESS,
        k4_norfair.ROOM_BUBBLE,
        CAPS,
    )

    assert path is not None
    assert [(edge.source_room_id, edge.target_room_id) for edge in path] == [
        (k4_norfair.ROOM_BUSINESS, 0xA7B3),  # cathedral entrance
        (0xA7B3, 0xA788),  # cathedral
        (0xA788, 0xAFA3),  # rising tide
        (0xAFA3, k4_norfair.ROOM_BUBBLE),
    ]
    assert [edge.edge_id for edge in path] == [
        "business_to_cathedral_entrance",
        "cathedral_entrance_to_cathedral",
        "cathedral_to_rising_tide",
        "rising_tide_to_bubble",
    ]


def test_k4_norfair_key_rooms_match_route_contract() -> None:
    assert k4_norfair.ROOM_BUSINESS == 0xA7DE
    assert k4_norfair.ROOM_BUBBLE == 0xACB3
    assert k4_norfair.ROOM_SPEED == 0xAD1B
    assert k4_norfair.ROOM_CATHEDRAL_ENTRANCE == 0xA7B3
    assert k4_norfair.ROOM_CATHEDRAL == 0xA788
    assert k4_norfair.ROOM_RISING_TIDE == 0xAFA3
