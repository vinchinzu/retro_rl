"""Offline capability path tests for the early SMZ3 graph."""

from __future__ import annotations

from smz3.route_graph import (
    COARSE_GRAPH,
    EARLY_GRAPH,
    N_LANDING,
    N_LINKS_HOUSE_CHEST,
    N_LINKS_HOUSE_OW,
    N_PORTAL_SETTLED,
    N_RED_DOOR,
    PLACEMENT_OVERLAY_A,
    PLACEMENT_OVERLAY_B,
    SM_BOMBS,
    Z3_BOMBS,
    path_with_capabilities,
    plan_with_placement,
    plan_early_legs,
)
from retro_harness.adventure import Has
from smz3.quest import early_path_summary, resolve_stop


def test_red_door_requires_missiles() -> None:
    without = path_with_capabilities(N_LANDING, N_RED_DOOR, frozenset())
    assert without is None
    with_m = path_with_capabilities(N_LANDING, N_RED_DOOR, frozenset({"missiles"}))
    assert with_m is not None
    assert any(e.edge_id == "parlor_to_red_door" for e in with_m)
    assert with_m[-1].requires == frozenset({"missiles"})


def test_path_to_chest_with_missiles() -> None:
    path = path_with_capabilities(
        N_LANDING, N_LINKS_HOUSE_CHEST, frozenset({"missiles"})
    )
    assert path is not None
    ids = [e.edge_id for e in path]
    assert "landing_to_parlor" in ids
    assert "portal_sm_to_z3" in ids
    assert "open_links_house_chest" in ids


def test_plan_early_legs_to_house_ow() -> None:
    legs = plan_early_legs(
        stop_node=N_LINKS_HOUSE_OW, initial_capabilities=frozenset({"missiles"})
    )
    assert legs[-1].leg.target_id == N_LINKS_HOUSE_OW
    assert all(isinstance(pl.capabilities_after, frozenset) for pl in legs)


def test_plan_without_missiles_fails_at_red_door() -> None:
    try:
        plan_early_legs(stop_node=N_RED_DOOR, initial_capabilities=frozenset())
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "missiles" in str(exc)


def test_graphs_validate() -> None:
    assert N_PORTAL_SETTLED in EARLY_GRAPH.nodes
    assert N_LINKS_HOUSE_CHEST in COARSE_GRAPH.nodes
    # Coarse direct edge exists.
    assert COARSE_GRAPH.edge_for(N_PORTAL_SETTLED, N_LINKS_HOUSE_OW) is not None


def test_resolve_stop_and_summary() -> None:
    assert resolve_stop("portal") == "portal"
    assert resolve_stop(N_LINKS_HOUSE_CHEST) == "links_house_chest"
    summary = early_path_summary("links_house_chest", with_missiles=True)
    assert summary["path_edge_ids"] is not None
    assert "landing_to_parlor" in summary["path_edge_ids"]
    blocked = early_path_summary("red_door", with_missiles=False)
    assert blocked["path_edge_ids"] is None


def test_fixture_placement_overlays_choose_different_valid_plans() -> None:
    plan_a = plan_with_placement(PLACEMENT_OVERLAY_A)
    plan_b = plan_with_placement(PLACEMENT_OVERLAY_B)
    assert plan_a is not None
    assert plan_b is not None
    assert [edge.edge_id for edge in plan_a] == [
        "smz3_fixture_to_sm_bombs_check",
        "smz3_fixture_sm_bombs_to_goal",
    ]
    assert [edge.edge_id for edge in plan_b] == [
        "smz3_fixture_to_z3_bombs_check",
        "smz3_fixture_z3_bombs_to_goal",
    ]
    assert plan_a != plan_b


def test_world_namespaces_do_not_alias_bombs() -> None:
    assert Has(Z3_BOMBS).satisfied_by({SM_BOMBS}) is False
