"""Focused tests for the bounded namespaced progression model."""

from __future__ import annotations

import json

import pytest

from retro_harness.adventure import (
    AllOf,
    AnyOf,
    CapabilityId,
    GraphEdge,
    GraphNode,
    Has,
    ItemCheck,
    ProgressionState,
    RouteGraph,
    SeedPlacement,
)


def test_capability_ids_are_qualified_and_immutable() -> None:
    capability = CapabilityId("SM", "bombs")
    assert capability.qualified == "sm:bombs"
    assert str(capability) == "sm:bombs"
    with pytest.raises(AttributeError):
        capability.name = "missiles"  # type: ignore[misc]


def test_requirements_are_monotonic_and_namespace_aware() -> None:
    sm_bombs = CapabilityId("sm", "bombs")
    z3_bombs = CapabilityId("z3", "bombs")
    sm_morph = CapabilityId("sm", "morph_ball")

    assert Has(sm_bombs).satisfied_by({sm_bombs})
    assert not Has(z3_bombs).satisfied_by({sm_bombs})
    requirement = AllOf(Has(sm_bombs), Has(sm_morph))
    assert not requirement.satisfied_by({sm_bombs})
    assert requirement.satisfied_by({sm_bombs, sm_morph})
    assert AnyOf(Has(z3_bombs), Has(sm_bombs)).satisfied_by({sm_bombs})


def test_any_of_serializes_in_canonical_order() -> None:
    bombs = Has(CapabilityId("sm", "bombs"))
    morph = Has(CapabilityId("sm", "morph_ball"))
    left = AnyOf(bombs, morph)
    right = AnyOf(morph, bombs)
    expected = '{"anyOf":[{"has":"sm:bombs"},{"has":"sm:morph_ball"}]}'
    assert left.to_json() == expected
    assert right.to_json() == expected
    assert json.loads(left.to_json()) == json.loads(right.to_json())


def test_collecting_an_item_check_is_immutable_and_not_repeatable() -> None:
    bombs = CapabilityId("sm", "bombs")
    check = ItemCheck("bomb_check", "item_room")
    placement = SeedPlacement("bomb_check", bombs)
    before = ProgressionState("item_room")
    after = before.collect(check, placement)

    assert before.capabilities == frozenset()
    assert after.capabilities == frozenset({bombs})
    assert after.collected_checks == frozenset({"bomb_check"})
    with pytest.raises(ValueError, match="already been collected"):
        after.collect(check, placement)


def test_progression_state_accepts_legacy_single_game_strings() -> None:
    legacy = ProgressionState("room", frozenset({"missiles"}))
    assert legacy.satisfies("missiles")
    assert legacy.legacy_capabilities() == frozenset({"missiles"})

    qualified = ProgressionState.from_legacy("room", "sm", {"missiles"})
    assert qualified.satisfies(Has(CapabilityId("sm", "missiles")))
    assert qualified.legacy_capabilities("sm") == frozenset({"missiles"})


def test_route_graph_progression_plan_collects_overlay_item() -> None:
    bombs = CapabilityId("sm", "bombs")
    graph = RouteGraph(
        (
            GraphNode("start"),
            GraphNode("item"),
            GraphNode("goal"),
        ),
        (
            GraphEdge("start", "item", edge_id="to_item"),
            GraphEdge("item", "goal", edge_id="to_goal", requires=Has(bombs)),
        ),
        (ItemCheck("bombs", "item"),),
    )
    result = graph.progression_plan(
        "start",
        "goal",
        SeedPlacement("bombs", bombs),
    )
    assert result is not None
    path, state = result
    assert [edge.edge_id for edge in path] == ["to_item", "to_goal"]
    assert state.capabilities == frozenset({bombs})
    assert state.collected_checks == frozenset({"bombs"})
