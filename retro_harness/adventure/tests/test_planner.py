"""Golden fixtures for the bounded explainable planner."""

from __future__ import annotations

from retro_harness.adventure import (
    GraphEdge,
    PlanRequest,
    PlanStatus,
    inventory_aware_path,
    plan,
)


def test_plan_output_is_byte_identical_under_reordered_edges() -> None:
    edges = (
        GraphEdge("start", "slow", edge_id="slow", cost=4),
        GraphEdge("slow", "goal", edge_id="slow-goal", cost=4),
        GraphEdge("start", "item", edge_id="collect", acquires={"morph"}),
        GraphEdge("item", "goal", edge_id="open", requires={"morph"}),
    )
    forward = plan(PlanRequest(edges, "start", "goal"))
    reverse = plan(PlanRequest(reversed(edges), "start", "goal"))

    assert forward.status is PlanStatus.FOUND
    assert [edge.edge_id for edge in forward.path] == ["collect", "open"]
    assert forward.to_json().encode("utf-8") == reverse.to_json().encode("utf-8")


def test_default_500_expansion_gate_is_explicit() -> None:
    edges = tuple(
        GraphEdge(index, index + 1, edge_id=f"edge-{index:03d}")
        for index in range(600)
    )
    result = plan(PlanRequest(edges, 0, 600))

    assert result.status is PlanStatus.BUDGET_EXHAUSTED
    assert result.expanded_count == 500
    assert result.path == ()
    assert result.total_cost is None
    assert result.final_progression.node == 500


def test_dominated_progression_states_are_pruned_golden_fixture() -> None:
    edges = (
        GraphEdge("start", "hub", edge_id="a-rich", acquires={"boots"}),
        GraphEdge("start", "hub", edge_id="b-poor"),
        GraphEdge("hub", "goal", edge_id="c-goal", requires={"boots"}),
    )
    result = plan(PlanRequest(edges, "start", "goal"))

    assert result.status is PlanStatus.FOUND
    assert [edge.edge_id for edge in result.path] == ["a-rich", "c-goal"]
    assert result.total_cost == 2.0
    assert result.dominated_pruned == 1
    assert result.final_progression.capabilities == frozenset({"boots"})


def test_unreachable_result_reports_stable_frontier_blockers() -> None:
    result = plan(
        PlanRequest(
            (GraphEdge("start", "goal", edge_id="bomb-door", requires={"bombs"}),),
            "start",
            "goal",
        )
    )

    assert result.status is PlanStatus.UNREACHABLE
    assert result.expanded_count == 1
    assert [blocker.edge_id for blocker in result.frontier_blockers] == ["bomb-door"]
    assert result.frontier_blockers[0].to_record()["requires"] == ["bombs"]


def test_inventory_aware_path_remains_compatible_with_bounded_planner() -> None:
    edges = (
        GraphEdge("start", "item", edge_id="collect", acquires={"morph"}),
        GraphEdge("item", "goal", edge_id="open", requires={"morph"}),
    )
    expected = plan(PlanRequest(edges, "start", "goal"))

    assert inventory_aware_path(edges, "start", "goal") == expected.path
