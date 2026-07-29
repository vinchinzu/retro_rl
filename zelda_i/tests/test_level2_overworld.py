from __future__ import annotations

import numpy as np

from zelda_i.level2_overworld import (
    LEVEL2_PATH_SCREENS,
    OverworldToLevel2Controller,
    PostTriforceSettleController,
    level2_path_prefix_success,
    post_triforce_overworld_ready,
)
from zelda_i.overworld import (
    LEVEL2_PATH_HOPS,
    LEVEL2_PATH_SCREENS as OW_LEVEL2_SCREENS,
    NODE_LEVEL1_COMPLETE,
    NODE_LEVEL1_EXIT_OVERWORLD,
    NODE_LEVEL2_PATH_4A,
    build_early_route_graph,
    level2_path_prefix_route_plan,
    neighbor_screens,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    SCREEN_LEVEL1_ENTRANCE,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_LEVEL1_ENTRANCE)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 125)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x01)
    return ram


def test_level2_path_screens_chain() -> None:
    assert LEVEL2_PATH_SCREENS[0] == 0x37
    assert LEVEL2_PATH_SCREENS[-1] == 0x4A
    assert LEVEL2_PATH_SCREENS == OW_LEVEL2_SCREENS
    assert len(LEVEL2_PATH_HOPS) == len(LEVEL2_PATH_SCREENS) - 1
    assert all(
        hop.target == screen
        for hop, screen in zip(LEVEL2_PATH_HOPS, LEVEL2_PATH_SCREENS[1:])
    )
    for a, b in zip(LEVEL2_PATH_SCREENS, LEVEL2_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values()


def test_post_triforce_overworld_ready() -> None:
    assert post_triforce_overworld_ready(
        _ram(screen=SCREEN_LEVEL1_ENTRANCE, triforce=0x01, level=0)
    )
    assert not post_triforce_overworld_ready(
        _ram(screen=SCREEN_LEVEL1_ENTRANCE, triforce=0x00)
    )
    assert not post_triforce_overworld_ready(_ram(level=1, triforce=0x01))


def test_level2_path_prefix_success() -> None:
    assert level2_path_prefix_success(
        _ram(screen=0x4A, triforce=0x01, sword=1, level=0)
    )
    assert not level2_path_prefix_success(
        _ram(screen=0x37, triforce=0x01, sword=1)
    )


def test_settle_controller_accepts_overworld() -> None:
    ctrl = PostTriforceSettleController()
    from zelda_i.ram import read_snapshot

    snap = read_snapshot(
        _ram(mode=PLAY_MODE, level=0, screen=SCREEN_LEVEL1_ENTRANCE, triforce=0x01)
    )
    ctrl.step(snap)
    assert ctrl.success
    assert ctrl.phase.name == "DONE"


def test_nav_controller_reports_hop() -> None:
    ctrl = OverworldToLevel2Controller()
    assert ctrl.hop_index == 0
    rep = ctrl.report()
    assert rep["hop"]["target"] == 0x38


def test_level2_path_prefix_route_plan() -> None:
    planned = level2_path_prefix_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL2_PATH_4A
    assert "triforce_shard_1" in planned[-1].capabilities_before
    assert any(leg.leg.leg_id == "settle_post_triforce_overworld" for leg in planned)


def test_graph_has_level2_prefix_edge() -> None:
    graph = build_early_route_graph()
    assert NODE_LEVEL1_EXIT_OVERWORLD in graph.nodes
    assert NODE_LEVEL2_PATH_4A in graph.nodes
    edge = graph.edge_for(NODE_LEVEL1_COMPLETE, NODE_LEVEL1_EXIT_OVERWORLD)
    assert edge is not None
    assert edge.verification == "observed"
    edge2 = graph.edge_for(NODE_LEVEL1_EXIT_OVERWORLD, NODE_LEVEL2_PATH_4A)
    assert edge2 is not None
    assert edge2.verification == "observed"
    # Grid hops along the verified walk prefix are promoted to observed.
    hop_edge = graph.edge_for("ow_37", "ow_38")
    assert hop_edge is not None
    assert hop_edge.verification == "observed"
    assert hop_edge.meta.get("segment") == "to_level2_prefix"
