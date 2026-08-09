"""Offline capability path tests for SM rando early graph."""

from __future__ import annotations

from sm_rando.logic_graph import (
    N_BOMBS,
    N_MORPH,
    N_SHIP,
    N_VARIA,
    PLACEMENT_OVERLAY_A,
    PLACEMENT_OVERLAY_B,
    path_with_capabilities,
    plan_with_placement,
    plan_to_varia,
)


def test_ship_to_morph_open() -> None:
    path = path_with_capabilities(N_SHIP, N_MORPH, frozenset())
    assert path is not None
    assert path[0].edge_id == "ship_to_morph"


def test_bombs_need_morph() -> None:
    assert path_with_capabilities(N_SHIP, N_BOMBS, frozenset()) is None
    path = path_with_capabilities(N_SHIP, N_BOMBS, frozenset({"morph_ball"}))
    assert path is not None
    assert any(e.edge_id == "parlor_to_bombs" for e in path)


def test_varia_tip_needs_kit() -> None:
    assert plan_to_varia(frozenset({"morph_ball"})) is None
    path = plan_to_varia(frozenset({"morph_ball", "missiles", "bombs"}))
    assert path is not None
    assert path[-1].target_id == N_VARIA


def test_fixture_placement_overlays_choose_different_valid_plans() -> None:
    plan_a = plan_with_placement(PLACEMENT_OVERLAY_A)
    plan_b = plan_with_placement(PLACEMENT_OVERLAY_B)
    assert plan_a is not None
    assert plan_b is not None
    assert [edge.edge_id for edge in plan_a] == [
        "sm_fixture_to_bombs_check",
        "sm_fixture_bombs_to_goal",
    ]
    assert [edge.edge_id for edge in plan_b] == [
        "sm_fixture_to_missiles_check",
        "sm_fixture_missiles_to_goal",
    ]
    assert plan_a != plan_b
