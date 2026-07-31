"""Vanilla Super Metroid room/door timing for stable-retro sessions.

Detects *confirmed* room transitions from project-native RAM signals
(:class:`~super_metroid.ram.SuperMetroidState`) and records per-hop timing in
**emulator frames** (one ``env.step`` = one frame).

This is **not** a practice-hack timer. The Super Metroid practice ROM exposes
IGT/lag/door-lag counters in high WRAM that stock ROMs do not; this module
never reads those fields and does not claim to reproduce them.

Confirmation rules (settle):
  A room is settled when ``phase == ordinary_gameplay``, ``game_state == 8``,
  ``door_transition == 0``, and ``room_id != 0``.

A completed hop is emitted only when ordinary gameplay settles in a *new*
room after a transition (or after an explicit leave of ordinary gameplay).
Boot/menu, soft-reset, frame rewinds, and room jumps without a transition
phase are treated as discontinuities and do not produce timing records.

State machine: :class:`retro_harness.hop_timer.HopTimer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from retro_harness.hop_timer import (
    HopFrame,
    HopTimer,
    OpenHop,
    rank_by_field,
    snapshots_from_json_mapping,
)
from super_metroid.ram import GameplayPhase, SuperMetroidState, phase_for_game_state

# Game state 8 is ordinary controllable gameplay (see docs/ram_map.md).
_ORDINARY_GAME_STATE = 8


class DiscontinuityReason(str, Enum):
    """Why an in-progress visit was abandoned without a timing record."""

    FRAME_REGRESSION = "frame_regression"
    BOOT_OR_MENU = "boot_or_menu"
    ROOM_JUMP = "room_jump"
    DEATH_OR_GAME_OVER = "death_or_game_over"
    ENDING_OR_CREDITS = "ending_or_credits"
    SESSION_END = "session_end"
    RESET = "reset"


@dataclass(frozen=True)
class TimingSnapshot:
    """Minimal frame sample for room timing (synthetic or from live state)."""

    frame: int
    room_id: int
    area_index: int = 0
    game_state: int = _ORDINARY_GAME_STATE
    door_transition: int = 0
    transition_direction: int = 0
    collected_items: int = 0
    collected_beams: int = 0
    phase: GameplayPhase | None = None

    def resolved_phase(self) -> GameplayPhase:
        if self.phase is not None:
            return self.phase
        return phase_for_game_state(self.game_state, self.door_transition)

    @classmethod
    def from_state(cls, state: SuperMetroidState) -> TimingSnapshot:
        return cls(
            frame=state.frame,
            room_id=state.room_id,
            area_index=state.area_index,
            game_state=state.game_state,
            door_transition=state.door_transition,
            transition_direction=state.transition_direction,
            collected_items=state.collected_items,
            collected_beams=state.collected_beams,
            phase=state.phase,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TimingSnapshot:
        phase_raw = data.get("phase")
        phase: GameplayPhase | None = None
        if isinstance(phase_raw, GameplayPhase):
            phase = phase_raw
        elif isinstance(phase_raw, str):
            try:
                phase = GameplayPhase(phase_raw)
            except ValueError:
                phase = None
        return cls(
            frame=int(data["frame"]),
            room_id=int(data.get("room_id", 0)),
            area_index=int(data.get("area_index", 0)),
            game_state=int(data.get("game_state", _ORDINARY_GAME_STATE)),
            door_transition=int(data.get("door_transition", 0)),
            transition_direction=int(data.get("transition_direction", 0)),
            collected_items=int(data.get("collected_items", 0)),
            collected_beams=int(data.get("collected_beams", 0)),
            phase=phase,
        )


def is_settled_ordinary(snap: TimingSnapshot) -> bool:
    """True when the game is fully settled in controllable room gameplay."""
    phase = snap.resolved_phase()
    return (
        phase is GameplayPhase.ORDINARY_GAMEPLAY
        and snap.game_state == _ORDINARY_GAME_STATE
        and snap.door_transition == 0
        and snap.room_id != 0
    )


@dataclass(frozen=True)
class RoomVisit:
    """One completed room hop with project-native frame timing."""

    source_room_id: int
    room_id: int
    dest_room_id: int
    area_index: int
    dest_area_index: int
    entry_frame: int
    leave_frame: int
    exit_frame: int
    room_frames: int
    dwell_frames: int
    transition_frames: int
    transition_direction: int
    door_transition_at_leave: int
    collected_items: int
    collected_beams: int
    sequence_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "source_room_id": self.source_room_id,
            "source_room_id_hex": f"0x{self.source_room_id:04X}",
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "dest_room_id": self.dest_room_id,
            "dest_room_id_hex": f"0x{self.dest_room_id:04X}",
            "area_index": self.area_index,
            "dest_area_index": self.dest_area_index,
            "entry_frame": self.entry_frame,
            "leave_frame": self.leave_frame,
            "exit_frame": self.exit_frame,
            "room_frames": self.room_frames,
            "dwell_frames": self.dwell_frames,
            "transition_frames": self.transition_frames,
            "transition_direction": self.transition_direction,
            "door_transition_at_leave": self.door_transition_at_leave,
            "collected_items": self.collected_items,
            "collected_items_hex": f"0x{self.collected_items:04X}",
            "collected_beams": self.collected_beams,
            "collected_beams_hex": f"0x{self.collected_beams:04X}",
            "timing_unit": "emulator_frames",
            "timing_note": (
                "room_frames = exit_frame - entry_frame (includes door load); "
                "dwell_frames = leave_frame - entry_frame; "
                "transition_frames = exit_frame - leave_frame. "
                "Not practice-hack IGT/lag."
            ),
        }


@dataclass(frozen=True)
class DiscontinuityEvent:
    """An abandoned in-progress visit or tracking reset."""

    frame: int
    reason: DiscontinuityReason
    room_id: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "reason": self.reason.value,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "detail": self.detail,
        }


_LEAVE_KEYS = frozenset({"door_transition_at_leave", "transition_direction"})


def _make_visit(
    open_hop: OpenHop[int],
    dest: int,
    leave_frame: int,
    exit_frame: int,
    sequence_index: int,
    meta: Mapping[str, Any],
    dest_context: Mapping[str, Any],
) -> RoomVisit:
    return RoomVisit(
        source_room_id=int(open_hop.source or 0),
        room_id=open_hop.location,
        dest_room_id=dest,
        area_index=int(meta.get("area_index", 0)),
        dest_area_index=int(dest_context.get("area_index", 0)),
        entry_frame=open_hop.entry_frame,
        leave_frame=leave_frame,
        exit_frame=exit_frame,
        room_frames=exit_frame - open_hop.entry_frame,
        dwell_frames=leave_frame - open_hop.entry_frame,
        transition_frames=exit_frame - leave_frame,
        transition_direction=int(meta.get("transition_direction", 0)),
        door_transition_at_leave=int(meta.get("door_transition_at_leave", 0)),
        collected_items=int(meta.get("collected_items", 0)),
        collected_beams=int(meta.get("collected_beams", 0)),
        sequence_index=sequence_index,
    )


def _make_disc(frame: int, reason: str, room_id: int, detail: str) -> DiscontinuityEvent:
    try:
        reason_enum = DiscontinuityReason(reason)
    except ValueError:
        reason_enum = DiscontinuityReason.RESET
    return DiscontinuityEvent(
        frame=frame, reason=reason_enum, room_id=room_id, detail=detail
    )


def _snap_to_hop(snap: TimingSnapshot) -> HopFrame[int]:
    phase = snap.resolved_phase()
    ctx = {
        "area_index": snap.area_index,
        "collected_items": snap.collected_items,
        "collected_beams": snap.collected_beams,
    }
    leave_meta = {
        "door_transition_at_leave": snap.door_transition,
        "transition_direction": snap.transition_direction,
    }

    if phase is GameplayPhase.BOOT_OR_MENU:
        return HopFrame(
            frame=snap.frame,
            location=snap.room_id,
            status="abandon",
            abandon_reason=DiscontinuityReason.BOOT_OR_MENU.value,
            abandon_detail=f"game_state={snap.game_state}",
        )
    if phase is GameplayPhase.DEATH_OR_GAME_OVER:
        return HopFrame(
            frame=snap.frame,
            location=snap.room_id,
            status="abandon",
            abandon_reason=DiscontinuityReason.DEATH_OR_GAME_OVER.value,
            abandon_detail=f"game_state={snap.game_state}",
        )
    if phase is GameplayPhase.ENDING_OR_CREDITS:
        return HopFrame(
            frame=snap.frame,
            location=snap.room_id,
            status="abandon",
            abandon_reason=DiscontinuityReason.ENDING_OR_CREDITS.value,
            abandon_detail=f"game_state={snap.game_state}",
        )
    if is_settled_ordinary(snap):
        return HopFrame(
            frame=snap.frame,
            location=snap.room_id,
            status="settled",
            context=ctx,
        )
    return HopFrame(
        frame=snap.frame,
        location=snap.room_id,
        status="transition",
        leave_meta=leave_meta,
    )


@dataclass
class RoomTimer:
    """Incremental room-transition detector and hop timer.

    Feed one :class:`TimingSnapshot` (or :class:`SuperMetroidState`) per
    emulator frame via :meth:`observe`. Completed hops accumulate in
    :attr:`visits`.
    """

    _engine: HopTimer[int, RoomVisit, DiscontinuityEvent] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = HopTimer(
            make_visit=_make_visit,
            make_discontinuity=_make_disc,
            jump_reason=DiscontinuityReason.ROOM_JUMP.value,
            null_location=0,
            leave_context_keys=_LEAVE_KEYS,
            reset_ever_settled_reasons=frozenset(
                {
                    DiscontinuityReason.BOOT_OR_MENU.value,
                    DiscontinuityReason.FRAME_REGRESSION.value,
                    DiscontinuityReason.RESET.value,
                    DiscontinuityReason.DEATH_OR_GAME_OVER.value,
                }
            ),
        )

    @property
    def visits(self) -> list[RoomVisit]:
        return self._engine.visits

    @property
    def discontinuities(self) -> list[DiscontinuityEvent]:
        return self._engine.discontinuities

    def observe(self, sample: TimingSnapshot | SuperMetroidState) -> RoomVisit | None:
        """Ingest one frame sample. Return a completed visit if one just closed."""
        snap = (
            sample
            if isinstance(sample, TimingSnapshot)
            else TimingSnapshot.from_state(sample)
        )
        return self._engine.observe_frame(_snap_to_hop(snap))

    def observe_many(
        self, samples: Iterable[TimingSnapshot | SuperMetroidState]
    ) -> list[RoomVisit]:
        """Ingest a sequence; return visits completed during that sequence."""
        newly: list[RoomVisit] = []
        for sample in samples:
            visit = self.observe(sample)
            if visit is not None:
                newly.append(visit)
        return newly

    def finalize(self, *, frame: int | None = None) -> None:
        """End the session without inventing a synthetic exit hop."""
        self._engine.finalize(frame=frame)

    def report(
        self,
        *,
        source: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """JSON-serializable timing session artifact."""
        open_payload = None
        if self._engine._open is not None:
            o = self._engine._open
            open_payload = {
                "room_id": o.location,
                "room_id_hex": f"0x{o.location:04X}",
                "entry_frame": o.entry_frame,
                "in_transition": o.in_transition,
                "leave_frame": o.leave_frame,
            }
        return self._engine.report_base(
            kind="super_metroid_room_timing",
            timing_semantics={
                "frame_basis": (
                    "stable-retro env.step frames (nominal 60 Hz NTSC); "
                    "not wall-clock and not practice-hack IGT/lag counters"
                ),
                "settle_rule": (
                    "ordinary_gameplay + game_state==8 + door_transition==0 "
                    "+ room_id!=0"
                ),
                "room_frames": "exit_frame - entry_frame (dwell + door load)",
                "dwell_frames": "leave_frame - entry_frame (controllable room time)",
                "transition_frames": "exit_frame - leave_frame (door/load)",
                "practice_hack_igt_lag": False,
            },
            source=source,
            extra=extra,
            open_visit_payload=open_payload,
            visit_to_dict=lambda v: v.to_dict(),
            disc_to_dict=lambda d: d.to_dict(),
            totals={
                "total_room_frames": sum(v.room_frames for v in self.visits),
                "total_dwell_frames": sum(v.dwell_frames for v in self.visits),
                "total_transition_frames": sum(
                    v.transition_frames for v in self.visits
                ),
            },
        )


def snapshots_from_json(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[TimingSnapshot]:
    """Parse an offline fixture (list of samples or ``{"samples": [...]}``)."""
    return snapshots_from_json_mapping(data, from_mapping=TimingSnapshot.from_mapping)


def run_offline(
    samples: Sequence[TimingSnapshot | SuperMetroidState | Mapping[str, Any]],
    *,
    source: str = "offline",
) -> dict[str, Any]:
    """Process a synthetic or recorded snapshot sequence into a report."""
    timer = RoomTimer()
    normalized: list[TimingSnapshot] = []
    for sample in samples:
        if isinstance(sample, TimingSnapshot):
            normalized.append(sample)
        elif isinstance(sample, SuperMetroidState):
            normalized.append(TimingSnapshot.from_state(sample))
        else:
            normalized.append(TimingSnapshot.from_mapping(sample))
    timer.observe_many(normalized)
    if normalized:
        timer.finalize(frame=normalized[-1].frame)
    else:
        timer.finalize()
    return timer.report(source=source)


def rank_visits(
    visits: Sequence[RoomVisit | Mapping[str, Any]],
    *,
    key: str = "room_frames",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return visits sorted by a timing field (default total room_frames)."""
    return rank_by_field(
        visits,
        key=key,
        allowed=frozenset({"room_frames", "dwell_frames", "transition_frames"}),
        to_dict=lambda v: v.to_dict() if isinstance(v, RoomVisit) else dict(v),
        limit=limit,
    )


@dataclass(frozen=True)
class SplitDwell:
    """Frame gap between consecutive continuous-report splits."""

    split_id: str
    frame: int
    room_id: int
    dwell_frames: int
    previous_split_id: str | None = None
    previous_frame: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "frame": self.frame,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "dwell_frames": self.dwell_frames,
            "previous_split_id": self.previous_split_id,
            "previous_frame": self.previous_frame,
        }


def split_dwells_from_report(
    report: Mapping[str, Any],
    *,
    start_frame: int = 0,
) -> list[SplitDwell]:
    """Derive per-split dwells from a continuous JSON report's ``splits`` list.

    Does not require a live ``RoomTimer`` run — useful for offline ranking of
    high-dwell hops after a continuous tip is integrity-green.
    """
    raw_splits = report.get("splits")
    if not isinstance(raw_splits, list) or not raw_splits:
        return []
    rows: list[SplitDwell] = []
    prev_id: str | None = None
    prev_frame = start_frame
    for item in raw_splits:
        if not isinstance(item, Mapping):
            continue
        split_id = str(item.get("split_id") or item.get("id") or "")
        try:
            frame = int(item["frame"])
            room_id = int(item.get("room_id") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if not split_id:
            continue
        rows.append(
            SplitDwell(
                split_id=split_id,
                frame=frame,
                room_id=room_id,
                dwell_frames=max(0, frame - prev_frame),
                previous_split_id=prev_id,
                previous_frame=prev_frame,
            )
        )
        prev_id = split_id
        prev_frame = frame
    return rows


def rank_split_dwells(
    report: Mapping[str, Any] | Sequence[SplitDwell | Mapping[str, Any]],
    *,
    limit: int | None = None,
    min_dwell: int = 0,
) -> list[dict[str, Any]]:
    """Rank split dwells descending (tightening targets after continuous green)."""
    if isinstance(report, Mapping) and "splits" in report:
        dwells = split_dwells_from_report(report)
    elif isinstance(report, Mapping):
        dwells = []
    else:
        dwells = []
        for item in report:
            if isinstance(item, SplitDwell):
                dwells.append(item)
            else:
                dwells.append(
                    SplitDwell(
                        split_id=str(item.get("split_id", "")),
                        frame=int(item.get("frame", 0)),
                        room_id=int(item.get("room_id", 0)),
                        dwell_frames=int(item.get("dwell_frames", 0)),
                        previous_split_id=item.get("previous_split_id"),  # type: ignore[arg-type]
                        previous_frame=int(item.get("previous_frame", 0)),
                    )
                )
    filtered = [d for d in dwells if d.dwell_frames >= min_dwell]
    ranked = sorted(filtered, key=lambda d: (-d.dwell_frames, d.frame, d.split_id))
    if limit is not None:
        ranked = ranked[:limit]
    return [d.to_dict() for d in ranked]


def action_reason_hotspots(
    report: Mapping[str, Any],
    *,
    limit: int | None = 25,
    min_frames: int = 50,
) -> list[dict[str, Any]]:
    """Rank ``action_reasons`` counters from a continuous report (high dwell labels)."""
    raw = report.get("action_reasons")
    if not isinstance(raw, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for reason, count in raw.items():
        try:
            frames = int(count)
        except (TypeError, ValueError):
            continue
        if frames < min_frames:
            continue
        rows.append({"reason": str(reason), "frames": frames})
    rows.sort(key=lambda r: (-r["frames"], r["reason"]))
    if limit is not None:
        rows = rows[:limit]
    return rows
