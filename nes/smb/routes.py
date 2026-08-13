"""Exit-route definitions and completion contracts for Super Mario Bros.

Routes are ordered lists of *exits* (levels / segments). The warp any% route
is the near-term video target (8 exits). The all-exits route lists all 32
main-game stages so the same stitch/render pipeline can grow into a full
100% / 32-exit showcase without a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExitDestination:
    """A legal successor state for an exit.

    Worlds and levels use the human-facing one-indexed values here, so route
    declarations are readable.  :meth:`matches` accepts an SMB RAM snapshot
    (or any object with the same fields) whose values are zero-indexed.
    """

    world: int
    level: int
    ending: bool = False
    label: str = ""

    def matches(self, snap: Any) -> bool:
        if int(getattr(snap, "world")) != self.world - 1:
            return False
        # Prefer LevelNumber ($075C) so 1-2 / 4-2 underground (AreaNumber++)
        # is not treated as the next stage. Hand-built snaps without
        # dash_level / level_number keep using AreaNumber.
        level = getattr(snap, "dash_level", None)
        if level is None:
            raw = getattr(snap, "level_number", None)
            level = raw if raw is not None else getattr(snap, "level")
        if int(level) != self.level - 1:
            return False
        return not self.ending or int(getattr(snap, "oper_mode")) == 2


@dataclass(frozen=True)
class ExitSegment:
    """One exit on a route (one level, or one multi-area segment)."""

    exit_id: str
    """Human id, e.g. ``1-1`` or ``8-4``."""

    segment_id: str
    """Machine id used by practice leaderboard / retro_harness.platformer, e.g. ``smb_1_1``."""

    label: str = ""
    """Overlay label; defaults to ``exit_id``."""

    world: int = 0
    level: int = 0
    """1-indexed world/level for all-exits bookkeeping (0 if N/A)."""

    successors: tuple[ExitDestination, ...] = ()
    """Legal successor states; warp and normal exits can differ."""

    policy_id: str = ""
    """Stable policy registry key.  Empty means no controller is registered."""

    def display(self) -> str:
        return self.label or self.exit_id

    def accepts_successor(self, snap: Any) -> bool:
        """Return whether ``snap`` is a declared post-exit state."""
        return any(destination.matches(snap) for destination in self.successors)


def _destination_after_normal_exit(world: int, level: int) -> ExitDestination:
    """Return the normal stage successor for one human-indexed stage."""
    if (world, level) == (8, 4):
        return ExitDestination(8, 4, ending=True, label="ending")
    if level < 4:
        return ExitDestination(world, level + 1)
    return ExitDestination(world + 1, 1)


def _stage(
    exit_id: str,
    *,
    label: str = "",
    successors: tuple[ExitDestination, ...] | None = None,
    policy_id: str | None = None,
) -> ExitSegment:
    """Build a stage declaration with its normal successor by default."""
    world, level = (int(part) for part in exit_id.split("-", 1))
    return ExitSegment(
        exit_id=exit_id,
        segment_id=f"smb_{world}_{level}",
        label=label or exit_id,
        world=world,
        level=level,
        successors=successors
        if successors is not None
        else (_destination_after_normal_exit(world, level),),
        policy_id=policy_id or f"smb_{world}_{level}",
    )


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
        _stage("1-1"),
        _stage(
            "1-2",
            label="1-2 (→W4)",
            successors=(ExitDestination(4, 1, label="warp_world_4"),),
            policy_id="smb_1_2_warp",
        ),
        _stage("4-1"),
        _stage(
            "4-2",
            label="4-2 (→W8)",
            successors=(ExitDestination(8, 1, label="warp_world_8"),),
            policy_id="smb_4_2_warp",
        ),
        _stage("8-1"),
        _stage("8-2"),
        _stage("8-3"),
        _stage("8-4"),
    )


def _all_32_exits() -> tuple[ExitSegment, ...]:
    exits: list[ExitSegment] = []
    for world in range(1, 9):
        for level in range(1, 5):
            exit_id = f"{world}-{level}"
            exits.append(_stage(exit_id))
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
