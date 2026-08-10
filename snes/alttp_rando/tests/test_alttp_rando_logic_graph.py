"""Offline capability path tests for ALTTP rando early graph."""

from __future__ import annotations

from alttp_rando.logic_graph import (
    EARLY_GRAPH,
    N_EASTERN_BOW,
    N_LINKS_HOUSE,
    N_SANCTUARY,
    N_UNCLE,
    PLACEMENT_OVERLAY_A,
    PLACEMENT_OVERLAY_B,
    path_with_capabilities,
    plan_with_placement,
    plan_to_eastern_bow,
)
from alttp_rando.solver_bindings import (
    HOUSE_TO_UNCLE_SPEC,
    build_early_binding_catalog,
    load_house_to_uncle_evidence,
    play_house_to_uncle,
)
from retro_harness.adventure.bindings import ExecutionReadiness


def test_house_to_uncle_open() -> None:
    path = path_with_capabilities(N_LINKS_HOUSE, N_UNCLE, frozenset())
    assert path is not None
    assert path[0].edge_id == "house_to_uncle"


def test_house_to_uncle_graph_verification_is_natural_entry() -> None:
    edge = next(e for e in EARLY_GRAPH.edges if e.edge_id == "house_to_uncle")
    assert edge.verification == "natural_entry"
    assert edge.source_id == N_LINKS_HOUSE
    assert edge.target_id == N_UNCLE


def test_house_to_uncle_has_digest_checked_natural_entry_binding() -> None:
    binding = build_early_binding_catalog().binding_for("house_to_uncle")
    assert binding is not None
    assert binding.readiness is ExecutionReadiness.NATURAL_ENTRY
    assert binding.skill_id == HOUSE_TO_UNCLE_SPEC.skill_id
    assert binding.dispatch_key.endswith(":play_house_to_uncle")
    assert binding.evidence_digest == load_house_to_uncle_evidence().identity_digest
    assert callable(play_house_to_uncle)


def test_sanctuary_needs_lamp_scaffold() -> None:
    assert path_with_capabilities(
        N_LINKS_HOUSE, N_SANCTUARY, frozenset({"sword"})
    ) is None
    path = path_with_capabilities(
        N_LINKS_HOUSE, N_SANCTUARY, frozenset({"sword", "lamp"})
    )
    assert path is not None
    assert path[-1].target_id == N_SANCTUARY


def test_eastern_bow_tip() -> None:
    path = plan_to_eastern_bow(frozenset({"sword", "lamp"}))
    assert path is not None
    assert path[-1].target_id == N_EASTERN_BOW


def test_fixture_placement_overlays_choose_different_valid_plans() -> None:
    plan_a = plan_with_placement(PLACEMENT_OVERLAY_A)
    plan_b = plan_with_placement(PLACEMENT_OVERLAY_B)
    assert plan_a is not None
    assert plan_b is not None
    assert [edge.edge_id for edge in plan_a] == [
        "z3_fixture_to_sword_check",
        "z3_fixture_sword_to_goal",
    ]
    assert [edge.edge_id for edge in plan_b] == [
        "z3_fixture_to_lamp_check",
        "z3_fixture_lamp_to_goal",
    ]
    assert plan_a != plan_b
