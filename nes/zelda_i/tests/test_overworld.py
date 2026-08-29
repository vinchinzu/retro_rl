from __future__ import annotations

import numpy as np

from zelda_i.overworld.graph import (
    LEVEL1_PATH_SCREENS,
    NODE_START,
    NODE_SWORD_CAVE,
    SCREEN_START,
    build_early_route_graph,
    neighbor_screens,
    screen_to_grid,
)
from zelda_i.overworld.nav import level1_entrance_success
from zelda_i.ram import ADDR_LEVEL, ADDR_MODE, ADDR_SCREEN, ADDR_SWORD, PLAY_MODE


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
    for a, b in zip(LEVEL1_PATH_SCREENS, LEVEL1_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values()


def test_level1_entrance_success_requires_dungeon() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_SWORD] = 1
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = 0x73
    assert level1_entrance_success(ram, require_dungeon=True)
    ram[ADDR_LEVEL] = 0
    ram[ADDR_SCREEN] = 0x37
    assert not level1_entrance_success(ram, require_dungeon=True)
