"""Tests for named routes and shared nav helpers."""

from __future__ import annotations

import pytest

from retro_harness.adventure.nav import (
    Waypoint,
    WaypointFollower,
    direction_toward,
    reached_waypoint,
)
from retro_harness.adventure.routes import (
    NamedRoute,
    RouteMilestone,
    RouteRegistry,
    get_named_route,
    get_route,
    list_routes,
)


def test_named_route_registry() -> None:
    route = NamedRoute(
        route_id="demo",
        display_name="Demo",
        milestones=(RouteMilestone("a", "n1", "A", "pred_a"),),
    )
    reg: dict[str, NamedRoute] = {}
    from retro_harness.adventure.routes import register_routes

    register_routes(reg, route, "alias")
    assert get_named_route(reg, "DEMO").route_id == "demo"
    assert get_route(reg, "alias").display_name == "Demo"  # compat alias
    assert [r.route_id for r in list_routes(reg)] == ["demo"]
    with pytest.raises(KeyError):
        get_named_route(reg, "missing")


def test_route_registry_class() -> None:
    reg = RouteRegistry()
    reg.register(
        NamedRoute("r1", "One", (RouteMilestone("m", "n", "L", "p"),)),
        "one",
    )
    assert reg.get("one").route_id == "r1"
    assert len(reg.list()) == 1


def test_waypoint_follower() -> None:
    follower = WaypointFollower(
        (
            Waypoint(10, 10, tolerance=2),
            Waypoint(50, 10, tolerance=2),
        )
    )
    assert direction_toward(0, 10, 10, 10) == "RIGHT"
    assert follower.step(0, 10) == "RIGHT"
    assert follower.step(10, 10) == "RIGHT"  # advanced to second
    assert not follower.done
    assert follower.step(50, 10) is None
    assert follower.done
    assert reached_waypoint(50, 10, Waypoint(50, 10, tolerance=0))
