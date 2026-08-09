"""Evidence-gated executable edge binding tests."""

from __future__ import annotations

import pytest

from retro_harness.adventure import (
    BindingCatalog,
    EdgeEvidence,
    ExecutionReadiness,
    GraphEdge,
    GraphNode,
    PlanRequest,
    PlanStatus,
    PromotionPolicy,
    RouteGraph,
    SkillBinding,
    plan,
)


def _binding(edge_id: str) -> SkillBinding:
    return SkillBinding(
        edge_id=edge_id,
        skill_id=f"skill:{edge_id}",
        dispatch_key=f"dispatch:{edge_id}",
        entry_contract_digest=f"entry:{edge_id}",
        exit_contract_digest=f"exit:{edge_id}",
    )


def _evidence(
    binding: SkillBinding,
    readiness: ExecutionReadiness,
    *,
    entry_digest: str = "natural-entry-state",
) -> EdgeEvidence:
    values = {
        "edge_id": binding.edge_id,
        "binding_digest": binding.identity_digest,
        "readiness": readiness,
        "target_entry_observation_digest": entry_digest,
        "target_exit_observation_digest": f"exit-observation:{binding.edge_id}",
        "attempts": 3,
        "successes": 3,
    }
    if readiness >= ExecutionReadiness.NATURAL_ENTRY:
        values.update(
            {
                "predecessor_edge_id": "predecessor",
                "predecessor_exit_observation_digest": entry_digest,
            }
        )
    return EdgeEvidence(**values)


def test_route_graph_rejects_duplicate_edge_ids() -> None:
    with pytest.raises(ValueError, match="duplicate edge ID"):
        RouteGraph(
            (GraphNode("a"), GraphNode("b"), GraphNode("c")),
            (
                GraphEdge("a", "b", edge_id="duplicate"),
                GraphEdge("a", "c", edge_id="duplicate"),
            ),
        )


def test_parallel_edges_promote_independently_by_edge_id() -> None:
    first = _binding("door-safe")
    second = _binding("door-fast")
    policy = PromotionPolicy(minimum_attempts=2, minimum_success_rate=1.0)
    promoted = policy.promote(
        first,
        _evidence(first, ExecutionReadiness.NATURAL_ENTRY),
    )
    catalog = BindingCatalog((promoted, second))

    assert catalog.binding_for("door-safe").readiness is ExecutionReadiness.NATURAL_ENTRY
    assert catalog.binding_for("door-fast").readiness is ExecutionReadiness.SCAFFOLD
    assert promoted.evidence_digest


def test_natural_entry_promotion_requires_digest_linked_predecessor() -> None:
    binding = _binding("target")
    with pytest.raises(ValueError, match="must match target entry"):
        EdgeEvidence(
            edge_id=binding.edge_id,
            binding_digest=binding.identity_digest,
            readiness=ExecutionReadiness.NATURAL_ENTRY,
            target_entry_observation_digest="target-entry",
            target_exit_observation_digest="target-exit",
            predecessor_edge_id="previous",
            predecessor_exit_observation_digest="different-state",
            attempts=1,
            successes=1,
        )


def test_publication_planner_excludes_below_natural_entry() -> None:
    isolated = _binding("a-isolated")
    natural = _binding("b-natural")
    policy = PromotionPolicy()
    isolated = policy.promote(
        isolated,
        _evidence(isolated, ExecutionReadiness.ISOLATED),
    )
    natural = policy.promote(
        natural,
        _evidence(natural, ExecutionReadiness.NATURAL_ENTRY),
    )
    edges = (
        GraphEdge("start", "goal", edge_id="a-isolated", cost=1),
        GraphEdge("start", "goal", edge_id="b-natural", cost=2),
    )
    publication_edges = BindingCatalog((isolated, natural)).publication_edges(edges)
    result = plan(PlanRequest(publication_edges, "start", "goal"))

    assert [edge.edge_id for edge in publication_edges] == ["b-natural"]
    assert result.status is PlanStatus.FOUND
    assert [edge.edge_id for edge in result.path] == ["b-natural"]


def test_readiness_does_not_accept_bare_verification_strings() -> None:
    with pytest.raises(TypeError, match="ExecutionReadiness"):
        SkillBinding(
            edge_id="edge",
            skill_id="skill",
            dispatch_key="dispatch",
            entry_contract_digest="entry",
            exit_contract_digest="exit",
            readiness="natural-entry",  # type: ignore[arg-type]
        )


def test_promotion_rejects_evidence_for_another_binding() -> None:
    first = _binding("first")
    second = _binding("second")
    with pytest.raises(ValueError, match="edge_id"):
        PromotionPolicy().promote(
            first,
            _evidence(second, ExecutionReadiness.NATURAL_ENTRY),
        )
