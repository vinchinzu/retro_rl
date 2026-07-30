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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

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
        """Build from a JSON-friendly dict (offline replay)."""
        phase_raw = data.get("phase")
        phase: GameplayPhase | None
        if phase_raw is None:
            phase = None
        elif isinstance(phase_raw, GameplayPhase):
            phase = phase_raw
        else:
            phase = GameplayPhase(str(phase_raw))
        return cls(
            frame=int(data["frame"]),
            room_id=int(data["room_id"]),
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
    """One completed room hop with project-native frame timing.

    Frame semantics (all integers are emulator frames, 60 Hz NTSC nominal):

    * ``entry_frame`` — first settled ordinary frame in ``room_id``.
    * ``leave_frame`` — first non-ordinary (or transition) frame after
      dwelling; ``None`` only for incomplete visits (not stored as completed).
    * ``exit_frame`` — first settled ordinary frame in the destination room
      (same as the next visit's ``entry_frame``).
    * ``room_frames`` — ``exit_frame - entry_frame`` (dwell + door load).
    * ``dwell_frames`` — ``leave_frame - entry_frame`` (controllable time).
    * ``transition_frames`` — ``exit_frame - leave_frame`` (door/load).
    """

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


@dataclass
class _OpenVisit:
    source_room_id: int
    room_id: int
    area_index: int
    entry_frame: int
    collected_items: int
    collected_beams: int
    leave_frame: int | None = None
    door_transition_at_leave: int = 0
    transition_direction: int = 0
    in_transition: bool = False


@dataclass
class RoomTimer:
    """Incremental room-transition detector and hop timer.

    Feed one :class:`TimingSnapshot` (or :class:`SuperMetroidState`) per
    emulator frame via :meth:`observe`. Completed hops accumulate in
    :attr:`visits`.
    """

    visits: list[RoomVisit] = field(default_factory=list)
    discontinuities: list[DiscontinuityEvent] = field(default_factory=list)
    _open: _OpenVisit | None = field(default=None, repr=False)
    _last_frame: int | None = field(default=None, repr=False)
    _last_room: int = field(default=0, repr=False)
    _ever_settled: bool = field(default=False, repr=False)

    def observe(self, sample: TimingSnapshot | SuperMetroidState) -> RoomVisit | None:
        """Ingest one frame sample. Return a completed visit if one just closed."""
        snap = (
            sample
            if isinstance(sample, TimingSnapshot)
            else TimingSnapshot.from_state(sample)
        )
        completed = self._observe_snapshot(snap)
        self._last_frame = snap.frame
        return completed

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
        if self._open is None:
            return
        end_frame = frame if frame is not None else (self._last_frame or 0)
        self._abandon(
            end_frame,
            DiscontinuityReason.SESSION_END,
            self._open.room_id,
            "session finalized with open visit",
        )

    def report(
        self,
        *,
        source: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """JSON-serializable timing session artifact."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "super_metroid_room_timing",
            "timing_unit": "emulator_frames",
            "timing_semantics": {
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
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "visit_count": len(self.visits),
            "discontinuity_count": len(self.discontinuities),
            "visits": [visit.to_dict() for visit in self.visits],
            "discontinuities": [event.to_dict() for event in self.discontinuities],
            "open_visit": None
            if self._open is None
            else {
                "room_id": self._open.room_id,
                "room_id_hex": f"0x{self._open.room_id:04X}",
                "entry_frame": self._open.entry_frame,
                "in_transition": self._open.in_transition,
                "leave_frame": self._open.leave_frame,
            },
            "total_room_frames": sum(v.room_frames for v in self.visits),
            "total_dwell_frames": sum(v.dwell_frames for v in self.visits),
            "total_transition_frames": sum(v.transition_frames for v in self.visits),
        }
        if extra:
            payload["extra"] = dict(extra)
        return payload

    # --- internals ---------------------------------------------------------

    def _observe_snapshot(self, snap: TimingSnapshot) -> RoomVisit | None:
        if self._last_frame is not None and snap.frame < self._last_frame:
            self._abandon(
                snap.frame,
                DiscontinuityReason.FRAME_REGRESSION,
                snap.room_id,
                f"frame {snap.frame} < previous {self._last_frame}",
            )
            # Fall through: may re-anchor if settled after load.

        phase = snap.resolved_phase()

        if phase is GameplayPhase.BOOT_OR_MENU:
            if self._open is not None or self._ever_settled:
                self._abandon(
                    snap.frame,
                    DiscontinuityReason.BOOT_OR_MENU,
                    snap.room_id,
                    f"game_state={snap.game_state}",
                )
            return None

        if phase is GameplayPhase.DEATH_OR_GAME_OVER:
            if self._open is not None:
                self._abandon(
                    snap.frame,
                    DiscontinuityReason.DEATH_OR_GAME_OVER,
                    snap.room_id,
                    f"game_state={snap.game_state}",
                )
            return None

        if phase is GameplayPhase.ENDING_OR_CREDITS:
            if self._open is not None:
                self._abandon(
                    snap.frame,
                    DiscontinuityReason.ENDING_OR_CREDITS,
                    snap.room_id,
                    f"game_state={snap.game_state}",
                )
            return None

        if is_settled_ordinary(snap):
            return self._on_settled(snap)

        # Non-settled play (door, pause, scripted, unknown).
        if self._open is not None and not self._open.in_transition:
            self._mark_leave(snap)
        elif self._open is not None and self._open.in_transition:
            # Keep door_transition_at_leave frozen at the leave frame; only
            # fill direction if it was zero at leave and becomes known later.
            if not self._open.transition_direction and snap.transition_direction:
                self._open.transition_direction = snap.transition_direction
        return None

    def _on_settled(self, snap: TimingSnapshot) -> RoomVisit | None:
        self._ever_settled = True

        if self._open is None:
            self._open = _OpenVisit(
                source_room_id=0,
                room_id=snap.room_id,
                area_index=snap.area_index,
                entry_frame=snap.frame,
                collected_items=snap.collected_items,
                collected_beams=snap.collected_beams,
            )
            self._last_room = snap.room_id
            return None

        if not self._open.in_transition:
            if snap.room_id == self._open.room_id:
                # Still dwelling; optional inventory context refresh.
                self._open.collected_items = snap.collected_items
                self._open.collected_beams = snap.collected_beams
                self._last_room = snap.room_id
                return None
            # Settled ordinary with a different room without a transition phase:
            # save-state load, door-warp, or other discontinuity.
            self._abandon(
                snap.frame,
                DiscontinuityReason.ROOM_JUMP,
                snap.room_id,
                (
                    f"room 0x{self._open.room_id:04X} -> 0x{snap.room_id:04X} "
                    "while ordinary (no transition phase)"
                ),
            )
            self._open = _OpenVisit(
                source_room_id=0,
                room_id=snap.room_id,
                area_index=snap.area_index,
                entry_frame=snap.frame,
                collected_items=snap.collected_items,
                collected_beams=snap.collected_beams,
            )
            self._last_room = snap.room_id
            return None

        # Completing a transition.
        if snap.room_id == self._open.room_id:
            # Returned to same room (failed door / bounce) — cancel leave.
            self._open.in_transition = False
            self._open.leave_frame = None
            self._open.door_transition_at_leave = 0
            self._last_room = snap.room_id
            return None

        leave_frame = self._open.leave_frame
        if leave_frame is None:
            # Room id changed only after settle without a recorded leave frame
            # (should be rare); treat leave as the frame just before settle.
            leave_frame = max(self._open.entry_frame, snap.frame - 1)

        visit = RoomVisit(
            source_room_id=self._open.source_room_id,
            room_id=self._open.room_id,
            dest_room_id=snap.room_id,
            area_index=self._open.area_index,
            dest_area_index=snap.area_index,
            entry_frame=self._open.entry_frame,
            leave_frame=leave_frame,
            exit_frame=snap.frame,
            room_frames=snap.frame - self._open.entry_frame,
            dwell_frames=leave_frame - self._open.entry_frame,
            transition_frames=snap.frame - leave_frame,
            transition_direction=self._open.transition_direction,
            door_transition_at_leave=self._open.door_transition_at_leave,
            collected_items=self._open.collected_items,
            collected_beams=self._open.collected_beams,
            sequence_index=len(self.visits),
        )
        self.visits.append(visit)
        self._open = _OpenVisit(
            source_room_id=visit.room_id,
            room_id=snap.room_id,
            area_index=snap.area_index,
            entry_frame=snap.frame,
            collected_items=snap.collected_items,
            collected_beams=snap.collected_beams,
        )
        self._last_room = snap.room_id
        return visit

    def _mark_leave(self, snap: TimingSnapshot) -> None:
        assert self._open is not None
        self._open.in_transition = True
        self._open.leave_frame = snap.frame
        self._open.door_transition_at_leave = snap.door_transition
        self._open.transition_direction = snap.transition_direction

    def _abandon(
        self,
        frame: int,
        reason: DiscontinuityReason,
        room_id: int,
        detail: str,
    ) -> None:
        if self._open is not None:
            self.discontinuities.append(
                DiscontinuityEvent(
                    frame=frame,
                    reason=reason,
                    room_id=self._open.room_id if self._open.room_id else room_id,
                    detail=detail,
                )
            )
        elif reason is not DiscontinuityReason.SESSION_END:
            self.discontinuities.append(
                DiscontinuityEvent(
                    frame=frame,
                    reason=reason,
                    room_id=room_id,
                    detail=detail,
                )
            )
        self._open = None
        self._last_room = 0
        if reason in {
            DiscontinuityReason.BOOT_OR_MENU,
            DiscontinuityReason.FRAME_REGRESSION,
            DiscontinuityReason.RESET,
            DiscontinuityReason.DEATH_OR_GAME_OVER,
        }:
            self._ever_settled = False


def snapshots_from_json(data: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[TimingSnapshot]:
    """Parse an offline fixture (list of samples or ``{"samples": [...]}``)."""
    if isinstance(data, Mapping):
        samples = data.get("samples", data.get("frames", []))
        if not isinstance(samples, Sequence):
            raise TypeError("expected samples list in mapping fixture")
    else:
        samples = data
    return [TimingSnapshot.from_mapping(item) for item in samples]


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
    """Return visits sorted by a timing field (default total room_frames).

    Useful for spotting the slowest hops on a continuous timing artifact.
    Accepts :class:`RoomVisit` instances or their ``to_dict()`` payloads.
    """
    if key not in {"room_frames", "dwell_frames", "transition_frames"}:
        raise ValueError(
            "key must be one of room_frames, dwell_frames, transition_frames"
        )

    rows: list[dict[str, Any]] = []
    for visit in visits:
        if isinstance(visit, RoomVisit):
            row = visit.to_dict()
        else:
            row = dict(visit)
        rows.append(row)
    rows.sort(key=lambda row: int(row.get(key, 0)), reverse=True)
    if limit is not None:
        return rows[:limit]
    return rows
