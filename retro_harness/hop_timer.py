"""Generic hop / location-transition timer engine.

Three games share the same settle → leave → complete state machine
(Super Metroid rooms, NES Metroid map cells, Zelda I screens). Game modules
supply location keys, settle predicates, and visit builders; this module owns
the duplicated open-visit bookkeeping.

Frame unit is always **emulator frames** (one ``env.step`` = one frame).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

Loc = TypeVar("Loc", bound=Hashable)
VisitT = TypeVar("VisitT")
DiscT = TypeVar("DiscT")
SnapT = TypeVar("SnapT")


class BaseDiscontinuityReason(str, Enum):
    """Shared discontinuity labels; games may extend with their own Enum."""

    FRAME_REGRESSION = "frame_regression"
    BOOT_OR_MENU = "boot_or_menu"
    LOCATION_JUMP = "location_jump"
    DEATH = "death"
    SESSION_END = "session_end"
    RESET = "reset"
    LOAD = "load"


@dataclass
class OpenHop(Generic[Loc]):
    """In-progress visit being tracked by :class:`HopTimer`."""

    location: Loc
    source: Loc | None
    entry_frame: int
    leave_frame: int | None = None
    in_transition: bool = False
    # Game-specific inventory / leave metadata (mutated in place).
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HopFrame(Generic[Loc]):
    """Normalized one-frame sample for the hop state machine.

    ``status`` values:
      * ``settled`` — controllable play at ``location``
      * ``transition`` — non-settled play that should mark leave
      * ``ignore`` — non-settled but still dwelling (e.g. hit freeze)
      * ``abandon`` — force abandon with ``abandon_reason``
    """

    frame: int
    location: Loc
    status: str  # settled | transition | ignore | abandon
    abandon_reason: str = ""
    abandon_detail: str = ""
    # Inventory / context merged into the open visit while settled.
    context: Mapping[str, Any] = field(default_factory=dict)
    # Metadata captured at leave / updated during transition.
    leave_meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class HopTimer(Generic[Loc, VisitT, DiscT]):
    """Incremental location-transition detector and hop timer.

    Feed one :class:`HopFrame` per emulator frame via :meth:`observe_frame`.
    Completed hops accumulate in :attr:`visits`.
    """

    # Build a completed visit when a hop closes.
    # Args: open_hop, dest, leave_frame, exit_frame, sequence_index,
    #       open_meta, dest_context.
    make_visit: Callable[
        [OpenHop[Loc], Loc, int, int, int, Mapping[str, Any], Mapping[str, Any]],
        VisitT,
    ]
    # Build a discontinuity event (frame, reason, location, detail).
    make_discontinuity: Callable[[int, str, Loc, str], DiscT]
    # Whether a settled location change without leave is a seamless hop.
    seamless_allowed: Callable[[Loc, Loc], bool] = field(
        default=lambda _a, _b: False
    )
    # Reasons that clear ``_ever_settled`` after abandon.
    reset_ever_settled_reasons: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                BaseDiscontinuityReason.BOOT_OR_MENU.value,
                BaseDiscontinuityReason.FRAME_REGRESSION.value,
                BaseDiscontinuityReason.RESET.value,
                BaseDiscontinuityReason.DEATH.value,
                BaseDiscontinuityReason.LOAD.value,
                "death_or_game_over",
                "death_or_reset",
            }
        )
    )
    # Zero / null location used when no open visit for discontinuity payload.
    null_location: Loc | None = None
    # Jump reason label when settled location changes without leave/seamless.
    jump_reason: str = BaseDiscontinuityReason.LOCATION_JUMP.value
    # Context keys captured at leave; cleared when a bounce cancels leave.
    leave_context_keys: frozenset[str] = field(default_factory=frozenset)

    visits: list[VisitT] = field(default_factory=list)
    discontinuities: list[DiscT] = field(default_factory=list)
    _open: OpenHop[Loc] | None = field(default=None, repr=False)
    _last_frame: int | None = field(default=None, repr=False)
    _ever_settled: bool = field(default=False, repr=False)

    def observe_frame(self, hop: HopFrame[Loc]) -> VisitT | None:
        """Ingest one normalized frame. Return a completed visit if one closed."""
        if self._last_frame is not None and hop.frame < self._last_frame:
            self._abandon(
                hop.frame,
                BaseDiscontinuityReason.FRAME_REGRESSION.value,
                hop.location,
                f"frame {hop.frame} < previous {self._last_frame}",
            )
            # Fall through: may re-anchor if settled after load.

        completed: VisitT | None = None
        if hop.status == "abandon":
            if hop.abandon_reason in {
                BaseDiscontinuityReason.BOOT_OR_MENU.value,
                "boot_or_menu",
            }:
                if self._open is not None or self._ever_settled:
                    self._abandon(
                        hop.frame,
                        hop.abandon_reason,
                        hop.location,
                        hop.abandon_detail,
                    )
            elif hop.abandon_reason:
                # Death / ending: abandon only when a visit is open
                # (matches Super Metroid / Zelda; Metroid also when ever_settled
                # via abandon_if_ever_settled flag in detail prefix).
                abandon_if_ever = hop.abandon_detail.startswith("ever:")
                detail = (
                    hop.abandon_detail[5:]
                    if abandon_if_ever
                    else hop.abandon_detail
                )
                if self._open is not None or (
                    abandon_if_ever and self._ever_settled
                ):
                    self._abandon(
                        hop.frame, hop.abandon_reason, hop.location, detail
                    )
            completed = None
        elif hop.status == "settled":
            completed = self._on_settled(hop)
        elif hop.status == "ignore":
            completed = None
        elif hop.status == "transition":
            if self._open is not None and not self._open.in_transition:
                self._mark_leave(hop)
            elif self._open is not None and self._open.in_transition:
                # Fill leave fields that were zero/missing at the leave frame.
                for key, value in hop.leave_meta.items():
                    if value and not self._open.context.get(key):
                        self._open.context[key] = value
            completed = None
        else:
            raise ValueError(f"unknown HopFrame status: {hop.status!r}")

        self._last_frame = hop.frame
        return completed

    def observe_many(self, frames: Iterable[HopFrame[Loc]]) -> list[VisitT]:
        newly: list[VisitT] = []
        for hop in frames:
            visit = self.observe_frame(hop)
            if visit is not None:
                newly.append(visit)
        return newly

    def finalize(self, *, frame: int | None = None) -> None:
        """End the session without inventing a synthetic exit hop."""
        if self._open is None:
            return
        end_frame = frame if frame is not None else (self._last_frame or 0)
        loc = self._open.location
        self._abandon(
            end_frame,
            BaseDiscontinuityReason.SESSION_END.value,
            loc,
            "session finalized with open visit",
        )

    def report_base(
        self,
        *,
        kind: str,
        timing_semantics: Mapping[str, Any],
        source: str = "",
        extra: Mapping[str, Any] | None = None,
        open_visit_payload: Mapping[str, Any] | None = None,
        visit_to_dict: Callable[[VisitT], dict[str, Any]],
        disc_to_dict: Callable[[DiscT], dict[str, Any]],
        totals: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        """JSON-serializable timing session artifact skeleton."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": kind,
            "timing_unit": "emulator_frames",
            "timing_semantics": dict(timing_semantics),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "visit_count": len(self.visits),
            "discontinuity_count": len(self.discontinuities),
            "visits": [visit_to_dict(v) for v in self.visits],
            "discontinuities": [disc_to_dict(d) for d in self.discontinuities],
            "open_visit": open_visit_payload,
        }
        if totals:
            payload.update(totals)
        if extra:
            payload["extra"] = dict(extra)
        return payload

    # --- internals ---------------------------------------------------------

    def _on_settled(self, hop: HopFrame[Loc]) -> VisitT | None:
        self._ever_settled = True
        loc = hop.location

        if self._open is None:
            self._open = OpenHop(
                location=loc,
                source=None,
                entry_frame=hop.frame,
                context=dict(hop.context),
            )
            return None

        if not self._open.in_transition:
            if loc == self._open.location:
                self._open.context.update(hop.context)
                return None
            if self.seamless_allowed(self._open.location, loc):
                leave_frame = hop.frame
                return self._complete(
                    hop,
                    leave_frame=leave_frame,
                    leave_meta_override={"seamless": True},
                )
            # Settled jump without leave/seamless → discontinuity + re-anchor.
            self._abandon(
                hop.frame,
                self.jump_reason,
                loc,
                (
                    f"{self._open.location!r} -> {loc!r} while settled "
                    "(no transition phase)"
                ),
            )
            self._open = OpenHop(
                location=loc,
                source=None,
                entry_frame=hop.frame,
                context=dict(hop.context),
            )
            return None

        # Completing a transition.
        if loc == self._open.location:
            self._open.in_transition = False
            self._open.leave_frame = None
            for key in self.leave_context_keys:
                self._open.context.pop(key, None)
            self._open.context.update(hop.context)
            return None

        leave_frame = self._open.leave_frame
        if leave_frame is None:
            leave_frame = max(self._open.entry_frame, hop.frame - 1)
        return self._complete(hop, leave_frame=leave_frame)

    def _complete(
        self,
        hop: HopFrame[Loc],
        *,
        leave_frame: int,
        leave_meta_override: Mapping[str, Any] | None = None,
    ) -> VisitT:
        assert self._open is not None
        dest = hop.location
        leave_meta = dict(self._open.context)
        if leave_meta_override:
            leave_meta.update(leave_meta_override)
        visit = self.make_visit(
            self._open,
            dest,
            leave_frame,
            hop.frame,
            len(self.visits),
            leave_meta,
            hop.context,
        )
        self.visits.append(visit)
        self._open = OpenHop(
            location=dest,
            source=self._open.location,
            entry_frame=hop.frame,
            context=dict(hop.context),
        )
        return visit

    def _mark_leave(self, hop: HopFrame[Loc]) -> None:
        assert self._open is not None
        self._open.in_transition = True
        self._open.leave_frame = hop.frame
        self._open.context.update(hop.leave_meta)

    def _abandon(
        self,
        frame: int,
        reason: str,
        location: Loc,
        detail: str,
    ) -> None:
        disc_loc = (
            self._open.location
            if self._open is not None
            else (location if location is not None else self.null_location)  # type: ignore[arg-type]
        )
        if disc_loc is None and self.null_location is not None:
            disc_loc = self.null_location
        if self._open is not None:
            self.discontinuities.append(
                self.make_discontinuity(frame, reason, disc_loc, detail)  # type: ignore[arg-type]
            )
        elif reason != BaseDiscontinuityReason.SESSION_END.value:
            self.discontinuities.append(
                self.make_discontinuity(frame, reason, disc_loc, detail)  # type: ignore[arg-type]
            )
        self._open = None
        if reason in self.reset_ever_settled_reasons:
            self._ever_settled = False


def snapshots_from_json_mapping(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    from_mapping: Callable[[Mapping[str, Any]], SnapT],
) -> list[SnapT]:
    """Parse an offline fixture (list of samples or ``{"samples": [...]}``)."""
    if isinstance(data, Mapping):
        samples = data.get("samples", data.get("frames", []))
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("expected samples list in mapping fixture")
    else:
        samples = data
    return [from_mapping(item) for item in samples]


def rank_by_field(
    visits: Sequence[Any],
    *,
    key: str,
    allowed: frozenset[str],
    to_dict: Callable[[Any], dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return visits sorted by a timing field descending."""
    if key not in allowed:
        raise ValueError(f"key must be one of {sorted(allowed)}")

    def as_dict(v: Any) -> dict[str, Any]:
        if isinstance(v, Mapping):
            return dict(v)
        if to_dict is not None:
            return to_dict(v)
        return v.to_dict()  # type: ignore[no-any-return]

    ranked = sorted(
        (as_dict(v) for v in visits), key=lambda d: d[key], reverse=True
    )
    if limit is not None:
        ranked = ranked[:limit]
    return ranked


__all__ = [
    "BaseDiscontinuityReason",
    "HopFrame",
    "HopTimer",
    "OpenHop",
    "rank_by_field",
    "snapshots_from_json_mapping",
]
