"""Room progression types, verification ranks, and capability helpers.

Graph nodes/edges/milestones are pure data; pathfinding lives in
:mod:`super_metroid.progression.graph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    normalize_capability,
)
from super_metroid.ram import (
    BOMBS_MASK,
    HI_JUMP_MASK,
    MORPH_BALL_MASK,
    VARIA_MASK,
    SuperMetroidState,
)

# Shared verification rank (higher = more proven). One table for path_summary,
# pure_gate, path_verification, and suggest ranking helpers.
VERIFICATION_RANK: dict[str, int] = {
    "planned": 0,
    "unverified": 1,
    "controller_dev": 2,
    "continuous": 3,
}


def capabilities_from_state(state: SuperMetroidState) -> frozenset[str]:
    """Map live RAM inventory/ammo into progression capability tokens.

    Covers the KPDR spine through Varia. Beam/speed/PB tokens expand as those
    continuous tips land (do not invent equipment the state does not hold).
    """
    caps: set[str] = set()
    items = state.collected_items
    if items & MORPH_BALL_MASK:
        caps.add("morph_ball")
    if items & BOMBS_MASK:
        caps.add("bombs")
    if items & HI_JUMP_MASK:
        caps.add("hi_jump")
    if items & VARIA_MASK:
        caps.add("varia_suit")
    if state.max_missiles > 0:
        caps.add("missiles")
    if state.max_super_missiles > 0:
        caps.add("super_missiles")
    if state.max_power_bombs > 0:
        caps.add("power_bombs")
    # Boss/event defeat bits used as graph gates on early spine.
    # boss_bits layout is area-indexed; Spore/Torizo flags are not always
    # mirrored here — callers that need them can union extra tokens.
    return frozenset(caps)


@dataclass(frozen=True)
class RoomNode:
    room_id: int
    name: str
    area: str
    tags: frozenset[str] = field(default_factory=frozenset)

    def as_graph_node(self) -> GraphNode:
        return GraphNode(
            node_id=self.room_id,
            name=self.name,
            area=self.area,
            tags=self.tags,
        )


@dataclass(frozen=True)
class DoorEdge:
    edge_id: str
    source_room_id: int
    target_room_id: int
    exit_direction: str
    entry_direction: str
    requires: frozenset[str] = field(default_factory=frozenset)
    policy_id: str = ""
    verification: str = "unverified"

    def __post_init__(self) -> None:
        if self.requires:
            object.__setattr__(
                self,
                "requires",
                frozenset(normalize_capability(v) for v in self.requires),
            )

    def as_graph_edge(self) -> GraphEdge:
        return GraphEdge(
            source_id=self.source_room_id,
            target_id=self.target_room_id,
            edge_id=self.edge_id,
            direction=self.exit_direction,
            requires=self.requires,
            verification=self.verification,
            provenance="progression",
            meta={
                "entryDirection": self.entry_direction,
                "policyId": self.policy_id,
            },
        )


@dataclass(frozen=True)
class ProgressCondition:
    """SM-specific stop predicate over live RAM (not shared catalog text)."""

    room_id: int | None = None
    game_states: frozenset[int] = field(default_factory=frozenset)
    collected_items_mask: int = 0
    minimum_ammo_capacities: tuple[int, int, int] = (0, 0, 0)

    def matches(self, state: SuperMetroidState) -> bool:
        if self.room_id is not None and state.room_id != self.room_id:
            return False
        if self.game_states and state.game_state not in self.game_states:
            return False
        if self.collected_items_mask:
            if (
                state.collected_items & self.collected_items_mask
                != self.collected_items_mask
            ):
                return False
        actual = (
            state.max_missiles,
            state.max_super_missiles,
            state.max_power_bombs,
        )
        return all(
            value >= minimum
            for value, minimum in zip(actual, self.minimum_ammo_capacities)
        )


@dataclass(frozen=True)
class ProgressionMilestone:
    """Runtime milestone with SM RAM condition + capability bookkeeping."""

    milestone_id: str
    label: str
    condition: ProgressCondition
    requires: frozenset[str] = field(default_factory=frozenset)
    acquires: frozenset[str] = field(default_factory=frozenset)
    timeout_frames: int = 0
    policy_id: str = ""

    def __post_init__(self) -> None:
        if self.requires:
            object.__setattr__(
                self,
                "requires",
                frozenset(normalize_capability(v) for v in self.requires),
            )
        if self.acquires:
            object.__setattr__(
                self,
                "acquires",
                frozenset(normalize_capability(v) for v in self.acquires),
            )


@dataclass(frozen=True)
class ObservedTransition:
    """Live room hop for continuous reports (stable JSON field names).

    Optional ``leave_kinematics`` / ``entry_kinematics`` capture Samus
    speed, position, pose, and door ptr at the last controllable frame
    before the transition and at room-id change (spawn). Needed for TAS /
    speedrun door tech that depends on entry speed and alignment.
    """

    frame: int
    source_room_id: int
    target_room_id: int
    edge_id: str | None
    leave_kinematics: dict[str, object] | None = None
    entry_kinematics: dict[str, object] | None = None

    @property
    def source_id(self) -> int:
        return self.source_room_id

    @property
    def target_id(self) -> int:
        return self.target_room_id

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "frame": self.frame,
            "source_room_id": self.source_room_id,
            "target_room_id": self.target_room_id,
            "edge_id": self.edge_id,
        }
        if self.leave_kinematics is not None:
            payload["leave_kinematics"] = self.leave_kinematics
        if self.entry_kinematics is not None:
            payload["entry_kinematics"] = self.entry_kinematics
        return payload
