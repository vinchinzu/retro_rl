"""Thin route composer: bind existing controllers to L1/L2 legs (no ROM)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from zelda_i.route_composer import (
    ControllerBinding,
    RouteSession,
    describe_named_route,
    legs_for_named_route,
    optional_door_graph_bindings,
    resolve_factory,
    session_for_named_route,
)
from zelda_i.route_legs import level1_complete_route_legs, level2_path_prefix_route_legs


class _FakeController:
    def __init__(self) -> None:
        self.success = False
        self.failed = False


def test_plan_sequences_fake_controller_table() -> None:
    session = RouteSession(route_id="fake_sword")
    session.bind(
        (
            ControllerBinding("enter_sword_cave", "fake.Sword", "in_cave"),
            ControllerBinding(
                "take_wooden_sword_and_exit", "fake.Sword", "has_sword_on_start"
            ),
        )
    )
    planned = session.plan(("enter_sword_cave", "take_wooden_sword_and_exit"))
    assert [binding.factory for binding in planned] == ["fake.Sword", "fake.Sword"]
    assert [binding.stop_predicate for binding in planned] == [
        "in_cave",
        "has_sword_on_start",
    ]

    registry = {"fake.Sword": _FakeController}
    controllers = session.instantiate(registry)
    assert len(controllers) == 2
    assert all(isinstance(ctl, _FakeController) for ctl in controllers)

    payload = session.describe()
    json.dumps(payload)
    assert payload["route_id"] == "fake_sword"
    assert payload["bound_count"] == 2
    assert payload["unbound"] == []


def test_plan_matches_route_leg_objects_and_reports_unbound() -> None:
    session = RouteSession()
    session.bind((ControllerBinding("enter_sword_cave", "fake.Sword", "in_cave"),))
    legs = (
        SimpleNamespace(leg_id="enter_sword_cave"),
        SimpleNamespace(leg=SimpleNamespace(leg_id="mystery_edge")),
    )
    planned = session.plan(legs)
    assert [binding.edge_or_leg_id for binding in planned] == ["enter_sword_cave"]
    desc = session.describe()
    assert desc["unbound"] == ["mystery_edge"]
    assert desc["unbound_count"] == 1


def test_level1_complete_binds_every_published_leg() -> None:
    session = session_for_named_route("level1_complete")
    desc = session.describe()
    assert desc["route_id"] == "zelda_level1_complete"
    assert desc["unbound_count"] == 0
    ids = [row["edge_or_leg_id"] for row in desc["bindings"]]
    expected = [leg.leg_id for leg in level1_complete_route_legs()]
    assert ids == expected
    assert "complete_level1_eagle" in ids
    assert "enter_sword_cave" in ids
    factories = {row["factory"] for row in desc["bindings"]}
    assert "zelda_i.sword_cave.SwordCaveController" in factories
    assert "zelda_i.overworld_nav.OverworldToLevel1Controller" in factories
    assert "level1_complete_chain" in factories


def test_level2_prefix_binds_settle_and_walk() -> None:
    session = session_for_named_route("level2_prefix")
    desc = session.describe()
    assert desc["route_id"] == "zelda_level2_path_prefix"
    assert desc["unbound_count"] == 0
    ids = [row["edge_or_leg_id"] for row in desc["bindings"]]
    assert ids == [leg.leg_id for leg in level2_path_prefix_route_legs()]
    assert "settle_post_triforce_overworld" in ids
    assert "walk_level2_path_prefix" in ids
    by_id = {row["edge_or_leg_id"]: row for row in desc["bindings"]}
    assert (
        by_id["settle_post_triforce_overworld"]["factory"]
        == "zelda_i.level2_overworld.PostTriforceSettleController"
    )
    assert (
        by_id["walk_level2_path_prefix"]["stop_predicate"]
        == "level2_path_prefix_success"
    )


def test_level2_door_path_is_bound_without_path_geometry() -> None:
    desc = describe_named_route("to_level2")
    assert desc["route_id"] == "zelda_level2_door_path"
    assert desc["unbound_count"] == 0
    ids = [row["edge_or_leg_id"] for row in desc["bindings"]]
    assert "walk_level2_door_path" in ids
    assert "enter_level2_dungeon" in ids
    json.dumps(desc)


def test_level5_complete_binds_later_legs() -> None:
    session = session_for_named_route("level5_complete")
    desc = session.describe()
    assert desc["route_id"] == "zelda_level5_complete"
    assert desc["unbound_count"] == 0
    ids = [row["edge_or_leg_id"] for row in desc["bindings"]]
    assert ids[0] == "level5_hills_to_door"
    assert "level5_whistle" in ids
    assert ids[-1] == "level5_triforce"


def test_level9_fixture_route_is_bound_not_eligible() -> None:
    desc = describe_named_route("ganon")
    assert desc["route_id"] == "zelda_level9_ganon"
    assert desc["unbound_count"] == 0
    assert "level9_41_to_31" in [row["edge_or_leg_id"] for row in desc["bindings"]]


def test_legs_for_named_route_resolves_aliases() -> None:
    legs = legs_for_named_route("sword")
    assert [leg.leg_id for leg in legs] == [
        "enter_sword_cave",
        "take_wooden_sword_and_exit",
    ]


def test_optional_door_graph_bind_is_skippable() -> None:
    # Sibling door_graph.bind is optional; composer must still plan L1/L2.
    bindings = optional_door_graph_bindings()
    assert isinstance(bindings, tuple)
    desc = describe_named_route("level1")
    assert desc["bound_count"] > 0
    assert desc["unbound_count"] == 0


def test_resolve_factory_registry_and_dotted_path() -> None:
    assert resolve_factory("fake", {"fake": _FakeController}) is _FakeController
    from zelda_i.sword_cave import SwordCaveController

    assert resolve_factory("zelda_i.sword_cave.SwordCaveController") is SwordCaveController


def test_compose_report_cli_is_plan_only_and_gates_ineligible_start() -> None:
    from zelda_i.scripts.compose_named_route import compose_report

    payload = compose_report(route="sword")
    assert payload["would_run"] is True
    assert payload["plan"]["unbound_count"] == 0
    json.dumps(payload)

    blocked = compose_report(
        route="sword",
        from_state="Level9BeforeGanonReconFixture",
    )
    assert blocked["would_run"] is False
    assert blocked["blocked_by"] == "from_state_not_route_eligible"
    assert blocked["from_state_eligibility"]["eligible"] is False
