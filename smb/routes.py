"""Exit-route definitions for Super Mario Bros.

Routes are ordered lists of *exits* (levels / segments). The warp any% route
is the near-term video target (8 exits). The all-exits route lists all 32
main-game stages so the same stitch/render pipeline can grow into a full
100% / 32-exit showcase without a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExitSegment:
    """One exit on a route (one level, or one multi-area segment)."""

    exit_id: str
    """Human id, e.g. ``1-1`` or ``8-4``."""

    segment_id: str
    """Machine id used by practice leaderboard / platformer_common, e.g. ``smb_1_1``."""

    label: str = ""
    """Overlay label; defaults to ``exit_id``."""

    world: int = 0
    level: int = 0
    """1-indexed world/level for all-exits bookkeeping (0 if N/A)."""

    def display(self) -> str:
        return self.label or self.exit_id


@dataclass(frozen=True)
class ExitRoute:
    """Ordered exit list forming a showcase or speedrun route."""

    route_id: str
    display_name: str
    exits: tuple[ExitSegment, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.exits:
            raise ValueError(f"route {self.route_id!r} has no exits")


def _warp_exits() -> tuple[ExitSegment, ...]:
    return (
        ExitSegment("1-1", "smb_1_1", "1-1", 1, 1),
        ExitSegment("1-2", "smb_1_2", "1-2 (→W4)", 1, 2),
        ExitSegment("4-1", "smb_4_1", "4-1", 4, 1),
        ExitSegment("4-2", "smb_4_2", "4-2 (→W8)", 4, 2),
        ExitSegment("8-1", "smb_8_1", "8-1", 8, 1),
        ExitSegment("8-2", "smb_8_2", "8-2", 8, 2),
        ExitSegment("8-3", "smb_8_3", "8-3", 8, 3),
        ExitSegment("8-4", "smb_8_4", "8-4", 8, 4),
    )


def _all_32_exits() -> tuple[ExitSegment, ...]:
    exits: list[ExitSegment] = []
    for world in range(1, 9):
        for level in range(1, 5):
            exit_id = f"{world}-{level}"
            exits.append(
                ExitSegment(
                    exit_id=exit_id,
                    segment_id=f"smb_{world}_{level}",
                    label=exit_id,
                    world=world,
                    level=level,
                )
            )
    return tuple(exits)


ROUTE_WARP_ANY_PERCENT = ExitRoute(
    route_id="smb_warp_any_percent",
    display_name="Super Mario Bros Any% (Warp → 8 Exit)",
    description=(
        "Classic warp route: 1-1, 1-2 warp to 4, 4-1, 4-2 warp to 8, "
        "then 8-1 through 8-4. Eight exits total."
    ),
    exits=_warp_exits(),
)

ROUTE_ALL_EXITS = ExitRoute(
    route_id="smb_all_exits",
    display_name="Super Mario Bros All 32 Exits",
    description=(
        "Every main-game stage 1-1 through 8-4 (32 exits). "
        "Sources fill in as full-level bests are recorded; missing exits skip."
    ),
    exits=_all_32_exits(),
)

ROUTE_REGISTRY: dict[str, ExitRoute] = {
    ROUTE_WARP_ANY_PERCENT.route_id: ROUTE_WARP_ANY_PERCENT,
    "smb_any_percent": ROUTE_WARP_ANY_PERCENT,
    "smb_any": ROUTE_WARP_ANY_PERCENT,
    "warp": ROUTE_WARP_ANY_PERCENT,
    "warp8": ROUTE_WARP_ANY_PERCENT,
    ROUTE_ALL_EXITS.route_id: ROUTE_ALL_EXITS,
    "all_exits": ROUTE_ALL_EXITS,
    "32": ROUTE_ALL_EXITS,
    "smb_100": ROUTE_ALL_EXITS,
}


def get_route(route_id: str) -> ExitRoute:
    key = route_id.strip().lower()
    if key not in ROUTE_REGISTRY:
        available = sorted(set(r.route_id for r in ROUTE_REGISTRY.values()))
        raise KeyError(f"Unknown route {route_id!r}. Available: {available}")
    return ROUTE_REGISTRY[key]


def list_routes() -> list[ExitRoute]:
    seen: set[str] = set()
    out: list[ExitRoute] = []
    for route in ROUTE_REGISTRY.values():
        if route.route_id not in seen:
            seen.add(route.route_id)
            out.append(route)
    return out
