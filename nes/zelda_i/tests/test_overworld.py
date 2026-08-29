from __future__ import annotations

from zelda_i.overworld import (
    LEVEL1_PATH_SCREENS,
    LEVEL2_DOOR_SCREENS,
    LEVEL2_PATH_SCREENS,
    NODE_START,
    NODE_SWORD_CAVE,
    SCREEN_START,
    build_early_route_graph,
    neighbor_screens,
    node_id_for_screen,
    screen_to_grid,
)


def test_start_screen_grid() -> None:
    col, row = screen_to_grid(SCREEN_START)
    assert (col, row) == (7, 7)
    neighbors = neighbor_screens(SCREEN_START)
    assert neighbors["north"] == 0x67
    assert neighbors["south"] is None


def test_early_graph_has_sword_cave_portal() -> None:
    graph = build_early_route_graph()
    assert NODE_SWORD_CAVE in graph.nodes
    edge = graph.edge_for(NODE_START, NODE_SWORD_CAVE)
    assert edge is not None
    assert edge.verification == "observed"


def test_level1_path_screens_chain() -> None:
    assert LEVEL1_PATH_SCREENS[0] == 0x77
    assert LEVEL1_PATH_SCREENS[-1] == 0x37
    # consecutive screens are grid neighbors
    for a, b in zip(LEVEL1_PATH_SCREENS, LEVEL1_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values()


def test_level2_door_path_screens_in_graph() -> None:
    """Door path 0x37→…→0x3C is fully node/edge-covered (planned beyond prefix)."""
    graph = build_early_route_graph()
    assert LEVEL2_DOOR_SCREENS[0] == 0x37
    assert LEVEL2_DOOR_SCREENS[-1] == 0x3C
    assert LEVEL2_DOOR_SCREENS == (
        0x37,
        0x38,
        0x48,
        0x58,
        0x59,
        0x5A,
        0x5B,
        0x5C,
        0x5D,
        0x4D,
        0x4C,
        0x3C,
    )
    for screen in LEVEL2_DOOR_SCREENS:
        assert node_id_for_screen(screen) in graph.nodes

    # Shared prefix hops stay observed (Clean 0x4A walk).
    prefix_pairs = set(zip(LEVEL2_PATH_SCREENS, LEVEL2_PATH_SCREENS[1:]))
    for a, b in zip(LEVEL2_DOOR_SCREENS, LEVEL2_DOOR_SCREENS[1:]):
        edge = graph.edge_for(node_id_for_screen(a), node_id_for_screen(b))
        assert edge is not None, f"missing edge {a:02X}->{b:02X}"
        if (a, b) in prefix_pairs:
            assert edge.verification == "observed"
            assert edge.meta.get("segment") == "to_level2_prefix"
        else:
            assert edge.verification == "planned"
            assert edge.provenance == "probe_geometry"
            assert edge.meta.get("segment") == "to_level2_door"

    # Maze hop is tagged (controller still open).
    maze = graph.edge_for("ow_5c", "ow_5d")
    assert maze is not None
    assert maze.meta.get("requires_maze") is True
