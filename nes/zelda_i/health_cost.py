"""Measured heart/bomb/key cost on top of Zelda I geometry hops.

Geometry stays in hop tables. This module only encodes the measured L2
door-path breakpoint (STATUS 2026-07-29): Clean dies on ``0x5C`` from
heart starvation, not a missing screen transition. Not a planner rewrite
and not a Clean re-route.

RAM ``ADDR_HEALTH`` (``HeartValues``): high nibble = containers − 1,
low nibble = whole hearts (full when low == high; **not** ``0xF``).
Bombs / keys travel with the resource state but have no measured
L2-door cost.

``walk_level2_door_path`` already carries
``constraints=("planned_not_clean", "requires_heart_management")`` in
``route_legs.level2_door_path_route_legs``. Consume that name here; do
not edit ``route_legs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Consumed from RouteLeg.constraints on walk_level2_door_path.
HEART_MANAGEMENT_CONSTRAINT = "requires_heart_management"

PATH_L2_DOOR = "level2_door_path"
PATH_L2_PREFIX = "level2_path_prefix"

# Aliases used by NamedRoute / graph meta — same measured corridor.
_L2_DOOR_PATH_IDS = frozenset(
    {
        PATH_L2_DOOR,
        "zelda_level2_door_path",
        "to_level2_door",
        "level2_door",
        "walk_level2_door_path",
    }
)

L2_DOOR_PATH_EVIDENCE = (
    "docs/STATUS.md measured door-path breakpoint 2026-07-29"
)
L2_DOOR_DEATH_SCREEN = 0x5C
# Post-L1 inventory on the measured probe (4 containers, 3 filled on first hop).
L2_DOOR_PATH_CONTAINERS = 4
L2_DOOR_PATH_START_FILLED = 3


@dataclass(frozen=True)
class ResourceState:
    """Hearts match RAM; bombs/keys are inventory counts."""

    containers: int
    filled: int
    bombs: int = 0
    keys: int = 0

    def __post_init__(self) -> None:
        if self.containers < 1:
            raise ValueError("containers must be >= 1")
        if self.filled < 0 or self.bombs < 0 or self.keys < 0:
            raise ValueError("resource counts must be non-negative")
        object.__setattr__(self, "filled", min(int(self.filled), int(self.containers)))

    @property
    def health_byte(self) -> int:
        return encode_health(self.containers, self.filled)

    @classmethod
    def from_health_byte(
        cls,
        health: int,
        *,
        bombs: int = 0,
        keys: int = 0,
    ) -> ResourceState:
        containers, filled = decode_health(health)
        return cls(containers=containers, filled=filled, bombs=bombs, keys=keys)


@dataclass(frozen=True)
class HopCost:
    """Cost to *arrive* on ``screen``. Geometry hops are not rewritten."""

    screen: int
    hearts_lost: int
    notes: str = ""
    evidence: str = ""


# Hearts on arrival from Level1ExitOverworld, Clean input, 2/2 identical fail.
L2_DOOR_PATH_HOP_COSTS: tuple[HopCost, ...] = (
    HopCost(0x38, 0, "3/4 on arrival", L2_DOOR_PATH_EVIDENCE),
    HopCost(0x48, 0, "3/4 on arrival", L2_DOOR_PATH_EVIDENCE),
    HopCost(0x58, 1, "2/4 on arrival (−1 on 0x48/58)", L2_DOOR_PATH_EVIDENCE),
    HopCost(0x59, 0, "2/4 on arrival", L2_DOOR_PATH_EVIDENCE),
    HopCost(0x5A, 1, "1/4 on arrival (−1 on 0x59/5A)", L2_DOOR_PATH_EVIDENCE),
    HopCost(0x5B, 0, "1/4 on arrival", L2_DOOR_PATH_EVIDENCE),
    HopCost(
        0x5C,
        1,
        "0/4 DEATH on arrival (−1 on 0x5B/5C); maze never reached",
        L2_DOOR_PATH_EVIDENCE,
    ),
)


def encode_health(containers: int, filled: int) -> int:
    """Pack ``HeartValues``: high = containers−1, low = whole hearts (0-based)."""
    if containers < 1:
        raise ValueError("containers must be >= 1")
    n = (int(containers) - 1) & 0x0F
    filled = max(0, int(filled))
    lo = n if filled >= containers else max(0, filled - 1) & 0x0F
    return (n << 4) | lo


def decode_health(health: int) -> tuple[int, int]:
    """Return ``(containers, filled_count)``. Full when low nibble == high."""
    containers = ((int(health) >> 4) & 0x0F) + 1
    raw = int(health) & 0x0F
    filled = containers if raw >= (containers - 1) else raw + 1
    return containers, filled


def resources_from_snapshot(snap: object) -> ResourceState:
    """Build a resource state from a ``ZeldaSnapshot`` (or duck-typed snap)."""
    containers, filled = decode_health(int(getattr(snap, "health")))
    return ResourceState(
        containers=containers,
        filled=filled,
        bombs=int(getattr(snap, "bombs", 0)),
        keys=int(getattr(snap, "keys", 0)),
    )


def _as_state(
    start_hearts: int | ResourceState,
    *,
    containers: int = L2_DOOR_PATH_CONTAINERS,
) -> ResourceState:
    if isinstance(start_hearts, ResourceState):
        return start_hearts
    return ResourceState(containers=containers, filled=int(start_hearts))


def simulate_corridor(
    start_hearts: int | ResourceState,
    hops: Iterable[HopCost] | None = None,
) -> tuple[bool, int | None, ResourceState]:
    """Apply measured hop costs. Death when filled hearts reach 0.

    Returns ``(survives, death_screen | None, remaining)``. Geometry after
    the last given hop is not simulated — the L2 maze is unmeasured because
    Clean never arrives there alive from 3/4.
    """
    state = _as_state(start_hearts)
    sequence = tuple(L2_DOOR_PATH_HOP_COSTS if hops is None else hops)
    if state.filled <= 0:
        return False, None, state
    for hop in sequence:
        lost = max(0, int(hop.hearts_lost))
        nxt = state.filled - lost
        if nxt <= 0:
            dead = ResourceState(
                containers=state.containers,
                filled=0,
                bombs=state.bombs,
                keys=state.keys,
            )
            return False, int(hop.screen), dead
        state = ResourceState(
            containers=state.containers,
            filled=nxt,
            bombs=state.bombs,
            keys=state.keys,
        )
    return True, None, state


def hop_costs_for_path(path_id: str) -> tuple[HopCost, ...]:
    if path_id in _L2_DOOR_PATH_IDS:
        return L2_DOOR_PATH_HOP_COSTS
    return ()


def route_leg_needs_heart_management(leg: object) -> bool:
    """True when a RouteLeg already declares ``requires_heart_management``."""
    constraints = getattr(leg, "constraints", ())
    return HEART_MANAGEMENT_CONSTRAINT in tuple(constraints)


def requires_assist_or_farm(
    path_id: str,
    start_hearts: int | ResourceState | None = None,
) -> bool:
    """True for the L2 door path at the measured 3–4 heart start.

    Extra farmed hearts are unmeasured; this still is not a Clean re-route.
    The 0x4A prefix is heart-stable and does not require assist/farm.
    """
    if path_id not in _L2_DOOR_PATH_IDS:
        return False
    if start_hearts is None:
        return True
    filled = _as_state(start_hearts).filled
    return filled <= L2_DOOR_PATH_CONTAINERS


__all__ = [
    "HEART_MANAGEMENT_CONSTRAINT",
    "L2_DOOR_DEATH_SCREEN",
    "L2_DOOR_PATH_CONTAINERS",
    "L2_DOOR_PATH_EVIDENCE",
    "L2_DOOR_PATH_HOP_COSTS",
    "L2_DOOR_PATH_START_FILLED",
    "PATH_L2_DOOR",
    "PATH_L2_PREFIX",
    "HopCost",
    "ResourceState",
    "decode_health",
    "encode_health",
    "hop_costs_for_path",
    "requires_assist_or_farm",
    "resources_from_snapshot",
    "route_leg_needs_heart_management",
    "simulate_corridor",
]
