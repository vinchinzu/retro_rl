"""Shared Super+ spine types — SpineHop and TipSegment.

:mod:`super_metroid.routes.kpdr.spine` re-exports these for public callers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from super_metroid.progression.types import DoorEdge
from super_metroid.routes.runtime import RouteSession

__all__ = [
    "SpineHop",
    "TipSegment",
    "BASE_CAPS",
    "HJ_CAPS",
    "K4_CAPS",
]


# Capability tokens matching progression/data.py Super+ continuous edges.
BASE_CAPS = frozenset({"morph_ball", "bombs", "missiles", "super_missiles"})
HJ_CAPS = BASE_CAPS | frozenset({"hi_jump"})
K4_CAPS = HJ_CAPS | frozenset({"varia_suit"})

# Private aliases used by hop table construction (same values).
_BASE_CAPS = BASE_CAPS
_HJ_CAPS = HJ_CAPS
_K4_CAPS = K4_CAPS


@dataclass(frozen=True)
class SpineHop:
    """One ordered continuous hop; tip_id tags the tip that owns this leg.

    DoorEdge product meta (when ``exit_direction`` is set) is the source of truth
    for continuous product graph edges. Omit door meta for in-room milestones and
    for hops that reuse an earlier edge (e.g. Warehouse→Business return).
    """

    hop_id: str
    """Split id / catalog split id (alias: :attr:`split_id`)."""

    play: Callable[[RouteSession], Any]
    from_room: int
    to_room: int
    room_label: str
    tip_id: str
    """Tip that *ends* after its hop group (delta from parent) is played."""

    use_transition_split: bool = True
    after: Callable[[RouteSession], None] | None = None
    # --- DoorEdge generation (None exit_direction ⇒ no generated DoorEdge) ---
    exit_direction: str | None = None
    entry_direction: str = ""
    requires: frozenset[str] = field(default_factory=frozenset)
    policy_id: str = ""
    verification: str = "continuous"
    edge_id: str | None = None
    """Graph ``DoorEdge.edge_id``; defaults to ``hop_id`` when emitting."""

    def __post_init__(self) -> None:
        if self.requires:
            object.__setattr__(self, "requires", frozenset(self.requires))

    @property
    def split_id(self) -> str:
        """Historical name for :attr:`hop_id` (reports / hop tables)."""
        return self.hop_id

    @property
    def emits_door_edge(self) -> bool:
        return self.exit_direction is not None

    def as_door_edge(self) -> DoorEdge:
        if self.exit_direction is None:
            raise ValueError(f"Spine hop {self.hop_id!r} does not emit a DoorEdge")
        return DoorEdge(
            edge_id=self.edge_id or self.hop_id,
            source_room_id=self.from_room,
            target_room_id=self.to_room,
            exit_direction=self.exit_direction,
            entry_direction=self.entry_direction,
            requires=self.requires,
            policy_id=self.policy_id,
            verification=self.verification,
        )


@dataclass(frozen=True)
class TipSegment:
    """Tip-level metadata for generating :class:`~super_metroid.routes.tips.TipSpec` rows.

    Parent chain + hop deltas come from the spine; report fields live here.
    ``graph_id`` matches :attr:`RoomProgressionGraph.graph_id` (resolved in
    ``hops.py`` after progression graphs are built — avoids import cycles).

    ``parent_tip_id`` for the first Super+ tip is ``\"supers\"`` (early chain),
    not ``None``.
    """

    tip_id: str
    parent_tip_id: str | None
    """``None`` → compose on top of Supers continuous play."""

    graph_id: str
    kind: str
    success_outcome: str
    route_label: str
    source_policy: str
    timing_source: str
    entry_condition_key: str
    ordinary_condition_key: str
    require_hi_jump: bool = False
    require_varia: bool = False

