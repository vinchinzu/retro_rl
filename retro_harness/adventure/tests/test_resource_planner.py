"""Resource feasibility, risk cost, and bounded-search fixtures."""

from __future__ import annotations

from retro_harness.adventure import (
    EdgeResourceProfile,
    GraphEdge,
    PlanBudget,
    PlanRequest,
    PlanStatus,
    ResourceKind,
    ResourcePlanRequest,
    ResourceSpec,
    RiskCostModel,
    reliability_from_outcomes,
    resource_plan,
)
from retro_harness.solver import SkillOutcome, SkillOutcomeStatus


def _outcome(edge_id: str, status: SkillOutcomeStatus, frames: int) -> SkillOutcome:
    return SkillOutcome(
        edge_id=edge_id,
        skill_id=f"skill:{edge_id}",
        status=status,
        frames=frames,
        start_observation_digest="before",
        end_observation_digest="after",
        observed_capability_delta=frozenset(),
        observed_resource_delta={},
    )


def test_key_door_and_missile_fixture_tracks_resource_trajectory() -> None:
    edges = (
        GraphEdge("start", "door", edge_id="locked-shortcut", cost=0.1),
        GraphEdge("start", "cache", edge_id="collect-key"),
        GraphEdge("cache", "door", edge_id="unlock-door"),
        GraphEdge("door", "goal", edge_id="missile-gate"),
    )
    request = ResourcePlanRequest(
        PlanRequest(edges, "start", "goal"),
        resources=(
            ResourceSpec("keys", maximum=1),
            ResourceSpec("missiles", maximum=10),
        ),
        initial_resources={"keys": 0, "missiles": 7},
        profiles=(
            EdgeResourceProfile("locked-shortcut", consumes={"keys": 1}),
            EdgeResourceProfile("collect-key", produces={"keys": 1}),
            EdgeResourceProfile("unlock-door", consumes={"keys": 1}),
            EdgeResourceProfile("missile-gate", consumes={"missiles": 5}),
        ),
    )

    result = resource_plan(request)

    assert result.status is PlanStatus.FOUND
    assert [edge.edge_id for edge in result.path] == [
        "collect-key",
        "unlock-door",
        "missile-gate",
    ]
    assert result.resource_trajectory == (
        {"keys": 0.0, "missiles": 7.0},
        {"keys": 1.0, "missiles": 7.0},
        {"keys": 0.0, "missiles": 7.0},
        {"keys": 0.0, "missiles": 2.0},
    )
    assert result.risk_adjusted_cost == result.total_cost


def test_resource_failure_reports_typed_blocker() -> None:
    edge = GraphEdge("start", "goal", edge_id="missile-gate")
    result = resource_plan(
        ResourcePlanRequest(
            PlanRequest((edge,), "start", "goal"),
            resources=(ResourceSpec("missiles", maximum=10),),
            initial_resources={"missiles": 4},
            profiles=(EdgeResourceProfile("missile-gate", consumes={"missiles": 5}),),
        )
    )

    assert result.status is PlanStatus.UNREACHABLE
    assert result.resource_blockers[0].resource == "missiles"
    assert result.resource_blockers[0].required == 5
    assert result.resource_blockers[0].available == 4


def test_observed_skill_outcomes_change_route_risk_ranking() -> None:
    edges = (
        GraphEdge("start", "goal", edge_id="cheap-risky", cost=1),
        GraphEdge("start", "goal", edge_id="safe", cost=2),
    )
    outcomes = (
        _outcome("cheap-risky", SkillOutcomeStatus.RETRYABLE_FAILURE, 2),
        _outcome("cheap-risky", SkillOutcomeStatus.TERMINAL_FAILURE, 3),
        _outcome("safe", SkillOutcomeStatus.SUCCESS, 5),
        _outcome("safe", SkillOutcomeStatus.SUCCESS, 7),
    )
    reliability = reliability_from_outcomes(outcomes)
    result = resource_plan(
        ResourcePlanRequest(
            PlanRequest(edges, "start", "goal"),
            resources=(
                ResourceSpec("health", minimum=1, maximum=99, kind=ResourceKind.SAFETY),
            ),
            initial_resources={"health": 99},
            reliability=reliability,
            cost_model=RiskCostModel(failure_weight=10),
        )
    )

    assert [edge.edge_id for edge in result.path] == ["safe"]
    assert {item.edge_id: item.success_probability for item in reliability} == {
        "cheap-risky": 0.25,
        "safe": 0.75,
    }


def test_resource_search_honors_base_expansion_budget() -> None:
    edges = tuple(
        GraphEdge(index, index + 1, edge_id=f"edge-{index}")
        for index in range(10)
    )
    result = resource_plan(
        ResourcePlanRequest(
            PlanRequest(edges, 0, 10, budget=PlanBudget(max_expansions=3)),
            resources=(ResourceSpec("keys", maximum=1),),
            initial_resources={"keys": 0},
        )
    )

    assert result.status is PlanStatus.BUDGET_EXHAUSTED
    assert result.expanded_count == 3
