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
        k4_norfair.play_frog_save_to_speedway,
        k4_norfair.play_speedway_to_farm,
        k4_norfair.play_farm_to_bubble,
    )

    assert all(callable(segment) for segment in segments)


def test_business_to_frog_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert get_segment("business_to_frog_save") is k4_norfair.play_business_to_frog_save


def test_frog_save_to_speedway_is_registered_for_pure_segment_use() -> None:
    from super_metroid.routes.kpdr import get_segment

    assert (
        get_segment("frog_save_to_speedway")
        is k4_norfair.play_frog_save_to_speedway
    )


def test_k4_norfair_constants_match_graph_path() -> None:
    path = START_TO_SPEED_GRAPH.shortest_path(
        k4_norfair.ROOM_BUSINESS,
        k4_norfair.ROOM_BUBBLE,
        CAPS,
    )

    assert path is not None
    assert [(edge.source_room_id, edge.target_room_id) for edge in path] == [
        (k4_norfair.ROOM_BUSINESS, k4_norfair.ROOM_FROG_SAVE),
        (k4_norfair.ROOM_FROG_SAVE, k4_norfair.ROOM_FROG_SPEEDWAY),
        (k4_norfair.ROOM_FROG_SPEEDWAY, k4_norfair.ROOM_UPPER_NORFAIR_FARM),
        (k4_norfair.ROOM_UPPER_NORFAIR_FARM, k4_norfair.ROOM_BUBBLE),
    ]


def test_k4_norfair_key_rooms_match_route_contract() -> None:
    assert k4_norfair.ROOM_BUSINESS == 0xA7DE
    assert k4_norfair.ROOM_BUBBLE == 0xACB3
    assert k4_norfair.ROOM_SPEED == 0xAD1B
