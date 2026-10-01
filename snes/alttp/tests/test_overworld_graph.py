"""Overworld screen graph. No ROM."""

from __future__ import annotations

from alttp.overworld_graph import OverworldGraph, screen_neighbors
from alttp.ram import SANCTUARY_SCREEN


def test_light_and_dark_screens_stay_in_their_world() -> None:
    assert set(screen_neighbors(0x00)) == {0x01, 0x08}
    assert set(screen_neighbors(0x3F)) == {0x37, 0x3E}
    assert set(screen_neighbors(0x40)) == {0x41, 0x48}
    assert set(screen_neighbors(0x7F)) == {0x77, 0x7E}
    assert all(n < 0x40 for n in screen_neighbors(0x1E))
    assert all(n >= 0x40 for n in screen_neighbors(0x5E))
    assert 0x1E not in screen_neighbors(0x5E)


def test_sanctuary_to_eastern_candidate_screen_topology() -> None:
    graph = OverworldGraph.load()
    eastern = graph.screen_of("Eastern Palace")
    assert eastern == 0x1E
    assert graph.screen_of("Sanctuary") == SANCTUARY_SCREEN == 0x13
    assert graph.screen_of("Swamp Palace (Entrance)") == 0x7B
    assert graph.screen_of("Ice Palace (Entrance)") is None

    # This only checks the candidate 8×8 screen topology. It says nothing
    # about tile collisions or the 2×2 screen area currently loaded in-game.
    path = graph.path("Sanctuary", eastern)
    assert path[0] == "Sanctuary"
    assert path[-1] == "screen:1E"
    assert "Light World" not in path
    screens = [int(node.split(":")[1], 16) for node in path if node.startswith("screen:")]
    assert screens[0] == 0x13
    assert screens[-1] == 0x1E
    assert all(screen < 0x40 for screen in screens)
    for left, right in zip(screens, screens[1:]):
        assert right in graph.neighbors[left]

    kinds = {row.connection_type for row in graph.entrances}
    assert {"Door", "Drop"} <= kinds
    assert any(
        row.origin == "Sanctuary Exit" and row.destination == "Light World"
        for row in graph.entrances
    )
    assert any(
        row.origin == "Eastern Palace Exit" and row.destination == "Light World"
        for row in graph.entrances
    )
    assert any(
        row.origin == "Top of Pyramid" and row.destination == "East Dark World"
        for row in graph.entrances
    )
