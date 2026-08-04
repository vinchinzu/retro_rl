"""Unit tests for retro_harness.adventure route graphs."""

from __future__ import annotations

import pytest

from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    ProgressionMilestone,
    RouteGraph,
    RouteLeg,
    RoutePatch,
    apply_milestones,
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
                support="manual",
                meta={"doorCapColor": "red"},
            ),
        )
    )
    edge = patched.edge_for("a", "b")
    assert edge is not None
    assert edge.requires == frozenset({"missiles"})
    assert edge.meta["doorCapColor"] == "red"
    assert edge.meta["support"] == "manual"
