"""Named routes for NES Metroid full-run scaffolding."""

from __future__ import annotations

from retro_harness.adventure.routes import (
    NamedRoute,
    RouteMilestone,
    get_route as _get_route,
    list_routes as _list_routes,
)
from metroid.brinstar import (
    NODE_EAST_DOOR,
    NODE_FIRST_MISSILES,
    NODE_MORPH,
    NODE_START,
    missiles_route_legs,
    morph_route_legs,
)

ROUTE_MORPH_BALL = NamedRoute(
    route_id="metroid_morph_ball",
    display_name="Maru Mari (Morph Ball)",
    description=(
        "From Brinstar start (3,14), traverse west through (2,14) into (1,14) "
        "and collect Maru Mari (equipment bit 0x10)."
    ),
    milestones=(
        RouteMilestone(
            "brinstar_start",
            NODE_START,
            "Brinstar start",
            "is_level1_ready",
        ),
        RouteMilestone(
            "morph_obtained",
            NODE_MORPH,
            "Morph Ball collected",
            "is_morph_obtained",
        ),
    ),
)

ROUTE_FIRST_MISSILES = NamedRoute(
    route_id="metroid_first_missiles",
    display_name="First missiles",
    description=(
        "After Maru Mari, return toward start, traverse east corridor "
        "(3,14)→(5,14), open the blue door / shaft route, and collect the "
        "first missile expansion ($687A > 0). Morph return + door still WIP."
    ),
    milestones=(
        RouteMilestone(
            "morph_obtained",
            NODE_MORPH,
            "Morph Ball held",
            "is_morph_obtained",
        ),
        RouteMilestone(
            "east_door",
            NODE_EAST_DOOR,
            "East corridor door cell (5,14)",
            "at_east_door",
        ),
        RouteMilestone(
            "missiles_obtained",
            NODE_FIRST_MISSILES,
            "First missiles",
            "is_missiles_obtained",
        ),
    ),
)

ROUTE_REGISTRY: dict[str, NamedRoute] = {
    ROUTE_MORPH_BALL.route_id: ROUTE_MORPH_BALL,
    "morph": ROUTE_MORPH_BALL,
    "morph_ball": ROUTE_MORPH_BALL,
    "maru_mari": ROUTE_MORPH_BALL,
    ROUTE_FIRST_MISSILES.route_id: ROUTE_FIRST_MISSILES,
    "missiles": ROUTE_FIRST_MISSILES,
    "first_missiles": ROUTE_FIRST_MISSILES,
}


def get_route(route_id: str) -> NamedRoute:
    return _get_route(ROUTE_REGISTRY, route_id)


def list_routes() -> list[NamedRoute]:
    return _list_routes(ROUTE_REGISTRY)


MORPH_BALL_LEGS = morph_route_legs()
FIRST_MISSILES_LEGS = missiles_route_legs()
