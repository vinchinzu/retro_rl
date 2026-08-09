"""Unit tests for retro_harness.adventure route graphs."""

from __future__ import annotations

import pytest

from retro_harness.adventure import inventory_aware_path as facade_inventory_aware_path
from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    ProgressionMilestone,
    RouteGraph,
    RouteLeg,
    RoutePatch,
    apply_milestones,
    inventory_aware_path,
    normalize_capability,
    shortest_path,
)


def _nodes(*ids: str) -> list[GraphNode]:
    return [GraphNode(node_id=i, name=i) for i in ids]


def test_normalize_capability() -> None:
    assert normalize_capability("Super Missiles") == "super_missiles"
    assert normalize_capability("wooden-sword") == "wooden_sword"
    assert normalize_capability("missile") == "missiles"
    assert normalize_capability("maru_mari") == "morph_ball"


def test_shortest_path_respects_capability_gate() -> None:
    edges = (
        GraphEdge("a", "b", requires=frozenset({"sword"})),
        GraphEdge("b", "c"),
    )
    assert shortest_path(edges, "a", "c") is None
    path = shortest_path(edges, "a", "c", {"sword"})
    assert path is not None
    assert [e.source_id for e in path] == ["a", "b"]


def test_inventory_aware_path_collects_edge_capability() -> None:
    graph = RouteGraph(
        _nodes("start", "item", "goal"),
        (
            GraphEdge("start", "item", edge_id="collect_sword", acquires={"sword"}),
            GraphEdge(
                "item",
                "goal",
                edge_id="open_gate",
                requires={"sword"},
            ),
        ),
    )

    assert graph.shortest_path("start", "goal") is None
    path = inventory_aware_path(graph.edges, "start", "goal")
    assert path is not None
    assert [edge.edge_id for edge in path] == ["collect_sword", "open_gate"]
    assert path[0].acquires == frozenset({"sword"})


def test_inventory_aware_path_is_exported_from_facade() -> None:
    assert facade_inventory_aware_path is inventory_aware_path


def test_inventory_aware_path_uses_dijkstra_cost_and_deterministic_ties() -> None:
    edges = (
        GraphEdge("start", "slow", edge_id="slow_first", cost=4),
        GraphEdge("slow", "goal", edge_id="slow_goal", cost=4),
        GraphEdge("start", "z", edge_id="z_first", cost=1),
        GraphEdge("z", "goal", edge_id="z_goal", cost=1),
        GraphEdge("start", "a", edge_id="a_first", cost=1),
        GraphEdge("a", "goal", edge_id="a_goal", cost=1),
    )
    expected = ["a_first", "a_goal"]
    path = inventory_aware_path(edges, "start", "goal")
    reversed_path = inventory_aware_path(reversed(edges), "start", "goal")
    assert path is not None
    assert reversed_path is not None
    assert [edge.edge_id for edge in path] == expected
    assert [edge.edge_id for edge in reversed_path] == expected


def test_inventory_aware_path_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        inventory_aware_path(
            (GraphEdge("a", "b", cost=-1),),
            "a",
            "b",
        )


def test_inventory_aware_path_rejects_non_finite_costs() -> None:
    for cost in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            inventory_aware_path(
                (GraphEdge("a", "b", cost=cost),),
                "a",
                "b",
            )


def test_route_graph_plan_legs_acquires() -> None:
    graph = RouteGraph(
        _nodes("start", "cave", "exit"),
        (
            GraphEdge("start", "cave", direction="in"),
            GraphEdge("cave", "exit", direction="out", requires=frozenset({"sword"})),
        ),
    )
    planned = graph.plan_legs(
        (
            RouteLeg("enter", "start", "cave", acquires=frozenset({"sword"})),
            RouteLeg("leave", "cave", "exit"),
        ),
        initial_capabilities=frozenset(),
    )
    assert planned[0].capabilities_after == frozenset({"sword"})
    assert planned[1].capabilities_before == frozenset({"sword"})


def test_route_graph_plan_legs_applies_edge_acquires() -> None:
    graph = RouteGraph(
        _nodes("start", "item", "goal"),
        (
            GraphEdge("start", "item", acquires={"sword"}),
            GraphEdge("item", "goal", requires={"sword"}),
        ),
    )
    planned = graph.plan_legs(
        (RouteLeg("collect", "start", "item"), RouteLeg("open", "item", "goal")),
    )

    assert planned[0].capabilities_after == frozenset({"sword"})
    assert planned[1].capabilities_before == frozenset({"sword"})


def test_route_graph_rejects_missing_capability() -> None:
    graph = RouteGraph(
        _nodes("a", "b"),
        (GraphEdge("a", "b", requires=frozenset({"bombs"})),),
    )
    with pytest.raises(ValueError, match="missing capabilities"):
        graph.plan_legs(
            (RouteLeg("hop", "a", "b"),),
            initial_capabilities=frozenset(),
        )


def test_add_patches() -> None:
    graph = RouteGraph(_nodes("a", "b", "c"), (GraphEdge("a", "b"),))
    patched = graph.add_patches(
        (RoutePatch("b", "c", direction="east", support="manual"),)
    )
    path = patched.shortest_path("a", "c")
    assert path is not None
    assert len(path) == 2


def test_apply_milestones_acquires() -> None:
    caps, ordered = apply_milestones(
        (
            ProgressionMilestone("m1", "Morph", acquires=frozenset({"morph_ball"})),
            ProgressionMilestone(
                "m2",
                "Missiles",
                requires=frozenset({"morph_ball"}),
                acquires=frozenset({"missiles"}),
            ),
        )
    )
    assert caps == frozenset({"morph_ball", "missiles"})
    assert len(ordered) == 2


def test_route_patch_meta_preserved() -> None:
    graph = RouteGraph(_nodes("a", "b"), ())
    patched = graph.add_patches(
        (
            RoutePatch(
                "a",
                "b",
                direction="Right",
                requires=frozenset({"missile"}),
                acquires=frozenset({"bomb"}),
                support="manual",
                meta={"doorCapColor": "red"},
            ),
        )
    )
    edge = patched.edge_for("a", "b")
    assert edge is not None
    assert edge.requires == frozenset({"missiles"})
    assert edge.acquires == frozenset({"bombs"})
    assert edge.meta["doorCapColor"] == "red"
    assert edge.meta["support"] == "manual"
