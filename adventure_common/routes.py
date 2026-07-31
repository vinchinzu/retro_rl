"""Named route catalogs shared by adventure games.

A :class:`NamedRoute` is a human/tooling catalog of milestones with string
stop-predicate names. Capability-aware planning stays on
:class:`~adventure_common.graph.RouteGraph` / :class:`~adventure_common.graph.ProgressionMilestone`;
this module only owns identity, labels, and registry lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RouteMilestone:
    """One labeled checkpoint on a named route."""

    milestone_id: str
    node_id: str
    label: str
    stop_predicate: str
    """Machine name of the stop check (documented in STATUS / segment code)."""


@dataclass(frozen=True)
class NamedRoute:
    """Ordered milestones forming a published adventure route."""

    route_id: str
    display_name: str
    milestones: tuple[RouteMilestone, ...]
    description: str = ""


def register_routes(
    registry: dict[str, NamedRoute],
    route: NamedRoute,
    *aliases: str,
) -> None:
    """Insert ``route`` under its id and optional aliases."""
    registry[route.route_id] = route
    for alias in aliases:
        registry[alias] = route


def get_route(registry: dict[str, NamedRoute], route_id: str) -> NamedRoute:
    """Look up a route by id or alias (case-insensitive on the key)."""
    key = route_id.strip().lower()
    # Prefer exact key, then lowercased (registries usually store lowercase aliases).
    if key in registry:
        return registry[key]
    if route_id in registry:
        return registry[route_id]
    available = sorted({r.route_id for r in registry.values()})
    raise KeyError(f"Unknown route {route_id!r}. Available: {available}")


def list_routes(registry: dict[str, NamedRoute]) -> list[NamedRoute]:
    """Return deduplicated routes in first-seen registry order."""
    seen: set[str] = set()
    out: list[NamedRoute] = []
    for route in registry.values():
        if route.route_id not in seen:
            seen.add(route.route_id)
            out.append(route)
    return out


@dataclass
class RouteRegistry:
    """Mutable named-route registry with get/list helpers."""

    _routes: dict[str, NamedRoute] = field(default_factory=dict)

    def register(self, route: NamedRoute, *aliases: str) -> None:
        register_routes(self._routes, route, *aliases)

    def get(self, route_id: str) -> NamedRoute:
        return get_route(self._routes, route_id)

    def list(self) -> list[NamedRoute]:
        return list_routes(self._routes)

    @property
    def raw(self) -> dict[str, NamedRoute]:
        return self._routes
