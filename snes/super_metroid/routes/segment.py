"""Segment / hop adapter contracts (practice / tests only).

**Not the continuous tip spine.** Live continuous composition is
:class:`~super_metroid.routes.tips.TipSpec` +
:class:`~super_metroid.routes.kpdr.spine.SpineHop` +
:func:`~super_metroid.routes.tips.play_hops` (see ``docs/ARCHITECTURE.md``).

This module is a thin facade for tests and optional adapters:

- ``policy.PolicySegment`` + ``play_policy`` (hash-pinned JSON)
- ``routes.kpdr.registry.KPDR_SEGMENTS`` (pure controller callables)
- ``progression`` milestones / conditions

Continuous power-on tips do **not** route through ``HopExecutor``.

Boss fights use the same surface via
:class:`super_metroid.combat.protocol.BossSegment` (see
``docs/BOSS_PIPELINE.md``). Practice doorway bootstrap uses
:mod:`super_metroid.rooms.segment_contract` (``EntryContract``) — a
different product track; do not mix the two.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from super_metroid.policy import (
    PolicySegment,
    SegmentEvidence,
    StateRequirement,
    play_policy,
)
from super_metroid.progression import (
    ProgressCondition,
    ProgressionMilestone,
    RoomProgressionGraph,
)
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import (
    ContinuousRunReport,
    ControllerSession,
    RouteSession,
)


@runtime_checkable
class Segment(Protocol):
    """Uniform play unit for continuous hops and policy segments."""

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


@dataclass(frozen=True)
class HopResult:
    """Evidence for one continuous :class:`~super_metroid.routes.continuous.RouteHop`."""

    hop_id: str
    start_frame: int
    end_frame: int
    from_room: int
    to_room: int
    final_room: int
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "hopId": self.hop_id,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "fromRoom": self.from_room,
            "toRoom": self.to_room,
            "finalRoom": self.final_room,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass
class HopExecutor:
    """Run continuous RouteHop rows with room asserts (natural-entry harness)."""

    session: RouteSession
    splits: list[Any] = field(default_factory=list)
    results: list[HopResult] = field(default_factory=list)

    def execute_hop(self, hop: Any) -> HopResult:
        """Execute one RouteHop-like object (``split_id``, ``play``, rooms)."""
        start = self.session.frame
        from_room = int(hop.from_room)
        to_room = int(hop.to_room)
        hop_id = str(getattr(hop, "split_id", None) or getattr(hop, "hop_id", hop))
        try:
            hop.play(self.session)
            if getattr(hop, "after", None) is not None:
                hop.after(self.session)
            final_room = self.session.state.room_id
            ok = final_room == to_room
            detail = "" if ok else (
                f"expected 0x{to_room:04X}, got 0x{final_room:04X}"
            )
            if not ok:
                raise RuntimeError(f"{hop_id}: {detail}")
            # Optional split bookkeeping when continuous helpers are available.
            use_transition = bool(getattr(hop, "use_transition_split", True))
            if use_transition:
                from super_metroid.routes.runtime import split_for_transition

                self.splits.append(
                    split_for_transition(
                        self.session.transitions,
                        hop_id,
                        from_room,
                        to_room,
                    )
                )
            else:
                from super_metroid.routes.runtime import Split

                self.splits.append(Split(hop_id, self.session.frame, final_room))
            result = HopResult(
                hop_id=hop_id,
                start_frame=start,
                end_frame=self.session.frame,
                from_room=from_room,
                to_room=to_room,
                final_room=final_room,
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface as hop evidence
            result = HopResult(
                hop_id=hop_id,
                start_frame=start,
                end_frame=self.session.frame,
                from_room=from_room,
                to_room=to_room,
                final_room=self.session.state.room_id,
                ok=False,
                detail=str(exc),
            )
            self.results.append(result)
            raise
        self.results.append(result)
        return result

    def execute_hops(self, hops: Sequence[Any]) -> list[HopResult]:
        out: list[HopResult] = []
        for hop in hops:
            out.append(self.execute_hop(hop))
        return out


class ContinuousSession:
    """Thin compositor facade over tip runners and hop execution.

    Preferred agent surface for continuous work:

    - ``run_to(tip_id)`` — full power-on through a registered tip
    - ``execute_hop`` / hop tables — attach natural-entry legs
    - ``current_state`` / ``progress_vector`` — efficient status
    - ``verify_milestone`` — progression graph conditions
    """

    def __init__(
        self,
        *,
        tip: str | None = None,
        session: RouteSession | None = None,
        graph: RoomProgressionGraph | None = None,
    ) -> None:
        self.tip = tip
        self._session = session
        self._graph = graph
        self._hop_executor: HopExecutor | None = None

    @property
    def session(self) -> RouteSession | None:
        return self._session

    def bind(self, session: RouteSession, *, graph: RoomProgressionGraph | None = None) -> None:
        """Attach a live continuous :class:`RouteSession` (e.g. mid-run)."""
        self._session = session
        if graph is not None:
            self._graph = graph
        self._hop_executor = HopExecutor(session=session)

    def current_state(self) -> SuperMetroidState:
        if self._session is None:
            raise RuntimeError("ContinuousSession has no bound RouteSession")
        return self._session.state

    def progress_vector(self) -> tuple[int, ...]:
        return self.current_state().progress_vector()

    def verify_milestone(self, milestone: ProgressionMilestone) -> bool:
        return milestone.condition.matches(self.current_state())

    def verify_condition(self, condition: ProgressCondition) -> bool:
        return condition.matches(self.current_state())

    def execute_hop(self, hop: Any) -> HopResult:
        if self._session is None:
            raise RuntimeError("ContinuousSession has no bound RouteSession")
        if self._hop_executor is None:
            self._hop_executor = HopExecutor(session=self._session)
        return self._hop_executor.execute_hop(hop)

    def execute_hops(self, hops: Sequence[Any]) -> list[HopResult]:
        return [self.execute_hop(hop) for hop in hops]

    def run_to(
        self,
        tip_id: str | None = None,
        **kwargs: Any,
    ) -> ContinuousRunReport:
        """Power-on once through a named continuous tip (delegates to continuous.run_to)."""
        from super_metroid.routes.continuous import run_to

        target = tip_id if tip_id is not None else self.tip
        if target is None:
            from super_metroid.routes.catalog import DEFAULT_CONTINUOUS_TIP

            target = DEFAULT_CONTINUOUS_TIP
        return run_to(target, **kwargs)


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
    "HopResult",
    "HopExecutor",
    "ContinuousSession",
    "segment_from_kpdr",
]
