"""Segment adapter contracts (practice / tests only).

**Not the continuous tip spine.** Live continuous composition is
:class:`~super_metroid.routes.tips.TipSpec` +
:class:`~super_metroid.routes.kpdr.spine.SpineHop` +
:func:`~super_metroid.routes.tips.play_hops` (see ``docs/ARCHITECTURE.md``).

This module is a thin facade for tests and optional adapters:

- ``policy.PolicySegment`` + ``play_policy`` (hash-pinned JSON)
- ``routes.kpdr.registry.KPDR_SEGMENTS`` (pure controller callables)
- ``progression`` milestones / conditions (consumed by adapters elsewhere)

Continuous power-on tips execute hops **only** via
:func:`~super_metroid.routes.tips.play_hops`. There is no second hop
executor in this package.

Boss fights use the same surface via
:class:`super_metroid.combat.protocol.BossSegment` (see
``docs/BOSS_PIPELINE.md``). Practice doorway bootstrap uses
:mod:`super_metroid.rooms.segment_contract` (``EntryContract``) — a
different product track; do not mix the two.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from super_metroid.policy import (
    PolicySegment,
    SegmentEvidence,
    StateRequirement,
    play_policy,
)
from super_metroid.routes.runtime import ControllerSession


@runtime_checkable
class Segment(Protocol):
    """Uniform play unit for practice adapters and policy segments."""

    @property
    def id(self) -> str: ...

    def play(self, session: ControllerSession) -> Any: ...


@dataclass(frozen=True)
class ControllerSegment:
    """Adapter: pure controller callable → Segment contract."""

    segment_id: str
    play_fn: Callable[[ControllerSession], Any]
    entry_room: int | None = None
    exit_room: int | None = None
    entry_items_mask: int = 0
    exit_items_mask: int = 0
    label: str = ""

    @property
    def id(self) -> str:
        return self.segment_id

    def play(self, session: ControllerSession) -> Any:
        if self.entry_room is not None and session.state.room_id != self.entry_room:
            raise RuntimeError(
                f"{self.segment_id}: expected entry room 0x{self.entry_room:04X}, "
                f"got 0x{session.state.room_id:04X}"
            )
        if self.entry_items_mask and (
            session.state.collected_items & self.entry_items_mask != self.entry_items_mask
        ):
            raise RuntimeError(
                f"{self.segment_id}: missing entry items "
                f"0x{self.entry_items_mask:04X} (have "
                f"0x{session.state.collected_items:04X})"
            )
        result = self.play_fn(session)
        if self.exit_room is not None and session.state.room_id != self.exit_room:
            raise RuntimeError(
                f"{self.segment_id}: expected exit room 0x{self.exit_room:04X}, "
                f"got 0x{session.state.room_id:04X}"
            )
        if self.exit_items_mask and (
            session.state.collected_items & self.exit_items_mask != self.exit_items_mask
        ):
            raise RuntimeError(
                f"{self.segment_id}: missing exit items "
                f"0x{self.exit_items_mask:04X} (have "
                f"0x{session.state.collected_items:04X})"
            )
        return result


@dataclass(frozen=True)
class PolicySegmentAdapter:
    """Adapter: hash-pinned :class:`PolicySegment` → Segment contract."""

    segment: PolicySegment

    @property
    def id(self) -> str:
        return self.segment.segment_id

    @property
    def entry(self) -> StateRequirement:
        return self.segment.entry

    @property
    def exit(self) -> StateRequirement:
        return self.segment.exit

    def play(self, session: ControllerSession) -> SegmentEvidence:
        # PolicySession is a subset of ControllerSession / RouteSession.
        return play_policy(session, self.segment)  # type: ignore[arg-type]


def segment_from_kpdr(
    segment_id: str,
    *,
    entry_room: int | None = None,
    exit_room: int | None = None,
    entry_items_mask: int = 0,
    exit_items_mask: int = 0,
) -> ControllerSegment:
    """Build a :class:`ControllerSegment` from the KPDR registry."""
    from super_metroid.routes.kpdr.registry import get_segment

    return ControllerSegment(
        segment_id=segment_id,
        play_fn=get_segment(segment_id),
        entry_room=entry_room,
        exit_room=exit_room,
        entry_items_mask=entry_items_mask,
        exit_items_mask=exit_items_mask,
    )


__all__ = [
    "Segment",
    "ControllerSegment",
    "PolicySegmentAdapter",
    "segment_from_kpdr",
]
