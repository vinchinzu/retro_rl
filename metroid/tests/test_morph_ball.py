from __future__ import annotations

from metroid.brinstar import (
    EARLY_BRINSTAR_GRAPH,
    EARLY_MILESTONES,
    NODE_EAST_DOOR,
    NODE_FIRST_MISSILES,
    NODE_MORPH,
    NODE_START,
    missiles_route_legs,
    morph_route_legs,
    validate_early_milestones,
)
from metroid.first_missiles import FirstMissilesController, MissilesPhase
from metroid.morph_ball import MorphBallController, MorphPhase
from metroid.routes import ROUTE_FIRST_MISSILES, ROUTE_MORPH_BALL, get_route


def test_early_graph_path_to_morph() -> None:
    path = EARLY_BRINSTAR_GRAPH.shortest_path(NODE_START, NODE_MORPH)
    assert path is not None
    assert path[-1].target_id == NODE_MORPH


def test_early_graph_path_to_east_door() -> None:
    path = EARLY_BRINSTAR_GRAPH.shortest_path(NODE_START, NODE_EAST_DOOR)
    assert path is not None
    assert path[-1].target_id == NODE_EAST_DOOR


def test_morph_route_legs_plan() -> None:
    planned = EARLY_BRINSTAR_GRAPH.plan_legs(morph_route_legs())
    assert planned[-1].capabilities_after == frozenset({"morph_ball"})
    assert planned[-1].leg.target_id == NODE_MORPH


def test_missiles_route_legs_plan() -> None:
    planned = EARLY_BRINSTAR_GRAPH.plan_legs(
        missiles_route_legs(),
        initial_capabilities=frozenset({"morph_ball"}),
    )
    assert planned[-1].capabilities_after == frozenset({"morph_ball", "missiles"})
    assert planned[-1].leg.target_id == NODE_FIRST_MISSILES


def test_milestones_apply() -> None:
    caps = validate_early_milestones()
    assert "morph_ball" in caps
    assert "missiles" in caps
    assert EARLY_MILESTONES[1].acquires == frozenset({"morph_ball"})
    assert EARLY_MILESTONES[2].acquires == frozenset({"missiles"})


def test_route_registry() -> None:
    route = get_route("morph")
    assert route.route_id == ROUTE_MORPH_BALL.route_id
    assert route.milestones[-1].stop_predicate == "is_morph_obtained"
    missiles = get_route("missiles")
    assert missiles.route_id == ROUTE_FIRST_MISSILES.route_id
    assert missiles.milestones[-1].stop_predicate == "is_missiles_obtained"


def test_controller_starts_on_align() -> None:
    ctrl = MorphBallController()
    assert ctrl.phase is MorphPhase.ALIGN
    assert ctrl.success is False


def test_missiles_controller_starts_on_morph_exit() -> None:
    ctrl = FirstMissilesController()
    assert ctrl.phase is MissilesPhase.MORPH_EXIT
    assert ctrl.success is False
    corridor = FirstMissilesController(start_from_corridor=True)
    corridor.reset()
    assert corridor.phase is MissilesPhase.EAST_CORRIDOR


def test_missiles_frontier_is_terminal_but_not_success() -> None:
    ctrl = FirstMissilesController(phase=MissilesPhase.FRONTIER)
    assert ctrl.terminal is True
    assert ctrl.success is False


def test_missiles_reset_clears_span_state() -> None:
    ctrl = FirstMissilesController(
        phase=MissilesPhase.THIRD_DOOR,
        frames=500,
        phase_frames=80,
        span_index=4,
        span_progress=3,
        last_x=99,
        stable_x_frames=2,
        shaft_variant="natural",
        notes=["probe"],
    )
    ctrl.reset()
    assert ctrl.phase is MissilesPhase.MORPH_EXIT
    assert ctrl.frames == 0
    assert ctrl.span_index == 0
    assert ctrl.span_progress == 0
    assert ctrl.last_x is None
    assert ctrl.stable_x_frames == 0
    assert ctrl.shaft_variant is None
    assert ctrl.notes == []
