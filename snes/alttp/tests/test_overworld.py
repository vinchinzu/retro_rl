"""Unit tests for light-world castle routing."""

from __future__ import annotations

from alttp.overworld import (
    direction_to_screen,
    next_direction_to_hyrule_castle,
    next_screen_in_path,
    shortest_screen_path,
)
from alttp.ram import (
    HYRULE_CASTLE_SCREEN,
    LINKS_HOUSE_SCREEN,
    AlttpSnapshot,
)


def _snap(**kwargs: object) -> AlttpSnapshot:
    base = dict(
        game_mode=0x09,
        submodule=0x00,
        room_id=0,
        indoors=False,
        screen_id=LINKS_HOUSE_SCREEN,
        link_x=2394,
        link_y=3000,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
    )
    base.update(kwargs)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def test_shortest_path_links_house_to_castle() -> None:
    path = shortest_screen_path(LINKS_HOUSE_SCREEN, HYRULE_CASTLE_SCREEN)
    assert path[0] == LINKS_HOUSE_SCREEN
    assert path[-1] == HYRULE_CASTLE_SCREEN
    assert 0x24 in path  # north field


def test_next_screen_steps_north_from_links_house() -> None:
    assert next_screen_in_path(LINKS_HOUSE_SCREEN, HYRULE_CASTLE_SCREEN) == 0x24


def test_direction_helpers() -> None:
    assert direction_to_screen(0x2C, 0x24) == "UP"
    assert direction_to_screen(0x24, 0x1C) == "UP"
    assert direction_to_screen(0x1C, 0x1B) == "LEFT"


def test_porch_escape_then_north() -> None:
    # Still too far left of the porch corridor → move right after clearing south.
    assert (
        next_direction_to_hyrule_castle(_snap(link_x=2300, link_y=3000)) == "RIGHT"
    )
    assert next_direction_to_hyrule_castle(_snap(link_x=2394, link_y=3000)) == "UP"
