"""Overworld-screen and dungeon-room transition timing for Zelda I (NES).

Detects *confirmed* location hops from project-native RAM signals
(:class:`~zelda_i.ram.ZeldaSnapshot`) and records per-hop timing in
**emulator frames** (one ``env.step`` = one frame).

This is **not** an official IGT or lag timer. Zelda I stock RAM has no
practice-hack gametime/lag counters; this module only measures settled
play between ``ADDR_SCREEN`` / ``ADDR_LEVEL`` locations using mode-based
settle rules (see ``docs/ram_map.md``).

Settle rule:
  Controllable overworld or dungeon play: ``mode == PLAY_MODE`` (5).
  Location identity is ``(level, screen)`` so overworld screens and
  dungeon rooms never collide.

A completed hop is emitted only when settled play appears in a *new*
location after a non-settled phase (scroll modes 6/7, cave enter 16,
fanfare 18, etc.). Boot/menu, death, frame rewinds, and location jumps
without a non-settled phase are discontinuities and do not invent hops.

Cave play (mode 11) and transition noise are not timed destinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from zelda_i.ram import (
    CAVE_MODE,
    PLAY_MODE,
    ZeldaSnapshot,
)

# Engine modes (see docs/ram_map.md).
_MODE_TITLE = 0
_MODE_FILE = 1
_MODE_HIT_FREEZE = 8  # brief freeze after damage; still in-room
_MODE_DEATH = 17
_MODE_TRIFORCE = 18
_TRANSITION_MODES = frozenset({6, 7, 16})  # scroll prep/scroll, cave enter


class DiscontinuityReason(str, Enum):
    """Why an in-progress visit was abandoned without a timing record."""

    FRAME_REGRESSION = "frame_regression"
    BOOT_OR_MENU = "boot_or_menu"
    LOCATION_JUMP = "location_jump"
    DEATH = "death"
    SESSION_END = "session_end"
    RESET = "reset"


class GameContext(str, Enum):
    """Settled-play context for a timed location."""

    OVERWORLD = "overworld"
    DUNGEON = "dungeon"


def context_for_level(level: int) -> GameContext:
    return GameContext.OVERWORLD if level == 0 else GameContext.DUNGEON


@dataclass(frozen=True)
class TimingSnapshot:
    """Minimal frame sample for screen/room timing (synthetic or live)."""

    frame: int
    mode: int
    level: int
    screen: int
    next_screen: int = 0
    sword: int = 0
    keys: int = 0
    health: int = 0
    triforce: int = 0

    @property
    def location_key(self) -> tuple[int, int]:
        return (int(self.level), int(self.screen))

    @property
    def context(self) -> GameContext:
        return context_for_level(self.level)

    @classmethod
    def from_snapshot(cls, snap: ZeldaSnapshot, *, frame: int) -> TimingSnapshot:
        return cls(
            frame=frame,
            mode=int(snap.mode),
            level=int(snap.level),
            screen=int(snap.screen),
            next_screen=int(snap.next_screen),
            sword=int(snap.sword),
            keys=int(snap.keys),
            health=int(snap.health),
            triforce=int(snap.triforce),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TimingSnapshot:
        """Build from a JSON-friendly dict (offline replay)."""
        return cls(
            frame=int(data["frame"]),
            mode=int(data.get("mode", PLAY_MODE)),
            level=int(data.get("level", 0)),
            screen=int(data["screen"]),
            next_screen=int(data.get("next_screen", 0)),
            sword=int(data.get("sword", 0)),
            keys=int(data.get("keys", 0)),
            health=int(data.get("health", 0)),
            triforce=int(data.get("triforce", 0)),
        )


def is_settled_play(snap: TimingSnapshot) -> bool:
    """True when Link is in controllable overworld or dungeon play."""
    return snap.mode == PLAY_MODE


def is_boot_or_menu(snap: TimingSnapshot) -> bool:
    return snap.mode in (_MODE_TITLE, _MODE_FILE)


def is_death(snap: TimingSnapshot) -> bool:
    return snap.mode == _MODE_DEATH


def is_hit_freeze(snap: TimingSnapshot) -> bool:
    """Mode 8 is a brief post-hit freeze; keep the open visit dwelling."""
    return snap.mode == _MODE_HIT_FREEZE


def is_transition_noise(snap: TimingSnapshot) -> bool:
    """Scroll / cave-enter modes that are not settled destinations."""
    return snap.mode in _TRANSITION_MODES or snap.mode == CAVE_MODE


@dataclass(frozen=True)
class LocationVisit:
    """One completed screen/room hop with emulator-frame timing.

    Frame semantics (all integers are emulator frames, 60 Hz NTSC nominal):

    * ``entry_frame`` — first settled play frame at ``(level, screen)``.
    * ``leave_frame`` — first non-settled frame after dwelling (scroll, etc.).
    * ``exit_frame`` — first settled play frame at the destination.
    * ``location_frames`` — ``exit_frame - entry_frame`` (dwell + load).
    * ``dwell_frames`` — ``leave_frame - entry_frame`` (settled time).
    * ``transition_frames`` — ``exit_frame - leave_frame`` (scroll/load).
    """

    source_level: int
    source_screen: int
    level: int
    screen: int
    dest_level: int
    dest_screen: int
    context: GameContext
    dest_context: GameContext
    entry_frame: int
    leave_frame: int
    exit_frame: int
    location_frames: int
    dwell_frames: int
    transition_frames: int
    mode_at_leave: int
    next_screen_at_leave: int
    sword: int
    keys: int
    triforce: int
    sequence_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "source_level": self.source_level,
            "source_screen": self.source_screen,
            "source_screen_hex": f"0x{self.source_screen:02X}",
            "level": self.level,
            "screen": self.screen,
            "screen_hex": f"0x{self.screen:02X}",
            "dest_level": self.dest_level,
            "dest_screen": self.dest_screen,
            "dest_screen_hex": f"0x{self.dest_screen:02X}",
            "context": self.context.value,
            "dest_context": self.dest_context.value,
            "entry_frame": self.entry_frame,
            "leave_frame": self.leave_frame,
            "exit_frame": self.exit_frame,
            "location_frames": self.location_frames,
            "dwell_frames": self.dwell_frames,
            "transition_frames": self.transition_frames,
            "mode_at_leave": self.mode_at_leave,
            "next_screen_at_leave": self.next_screen_at_leave,
            "next_screen_at_leave_hex": f"0x{self.next_screen_at_leave:02X}",
            "sword": self.sword,
            "keys": self.keys,
            "triforce": self.triforce,
            "timing_unit": "emulator_frames",
            "timing_note": (
                "location_frames = exit_frame - entry_frame (includes scroll/load); "
                "dwell_frames = leave_frame - entry_frame; "
                "transition_frames = exit_frame - leave_frame. "
                "Not official IGT/lag."
            ),
        }


@dataclass(frozen=True)
class DiscontinuityEvent:
    """An abandoned in-progress visit or tracking reset."""

    frame: int
    reason: DiscontinuityReason
    level: int
    screen: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "reason": self.reason.value,
            "level": self.level,
            "screen": self.screen,
            "screen_hex": f"0x{self.screen:02X}",
            "detail": self.detail,
        }


@dataclass
class _OpenVisit:
    source_level: int
    source_screen: int
    level: int
    screen: int
    context: GameContext
    entry_frame: int
    sword: int
    keys: int
    triforce: int
    leave_frame: int | None = None
    mode_at_leave: int = 0
    next_screen_at_leave: int = 0
    in_transition: bool = False


@dataclass
class RoomTimer:
    """Incremental screen/room-transition detector and hop timer.

    Feed one :class:`TimingSnapshot` (or :class:`ZeldaSnapshot` + frame) per
    emulator frame via :meth:`observe`. Completed hops accumulate in
    :attr:`visits`.
    """

    visits: list[LocationVisit] = field(default_factory=list)
    discontinuities: list[DiscontinuityEvent] = field(default_factory=list)
    _open: _OpenVisit | None = field(default=None, repr=False)
    _last_frame: int | None = field(default=None, repr=False)
    _ever_settled: bool = field(default=False, repr=False)

    def observe(
        self,
        sample: TimingSnapshot | ZeldaSnapshot,
        *,
        frame: int | None = None,
    ) -> LocationVisit | None:
        """Ingest one frame sample. Return a completed visit if one just closed."""
        if isinstance(sample, TimingSnapshot):
            snap = sample
        else:
            if frame is None:
                raise ValueError("frame= is required when observing ZeldaSnapshot")
            snap = TimingSnapshot.from_snapshot(sample, frame=frame)
        completed = self._observe_snapshot(snap)
        self._last_frame = snap.frame
        return completed

    def observe_many(
        self,
        samples: Iterable[TimingSnapshot | ZeldaSnapshot | tuple[ZeldaSnapshot, int]],
    ) -> list[LocationVisit]:
        """Ingest a sequence; return visits completed during that sequence."""
        newly: list[LocationVisit] = []
        for sample in samples:
            if isinstance(sample, tuple):
                visit = self.observe(sample[0], frame=sample[1])
            else:
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
            self._open.level,
            self._open.screen,
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
            "kind": "zelda_i_screen_room_timing",
            "timing_unit": "emulator_frames",
            "timing_semantics": {
                "frame_basis": (
                    "stable-retro env.step frames (nominal 60 Hz NTSC); "
                    "not wall-clock and not official IGT/lag counters"
                ),
                "settle_rule": (
                    "mode==PLAY_MODE(5) on overworld (level==0) or dungeon "
                    "(level>=1); location=(level, screen) from ADDR_LEVEL/ADDR_SCREEN"
                ),
                "location_frames": "exit_frame - entry_frame (dwell + scroll/load)",
                "dwell_frames": "leave_frame - entry_frame (settled play time)",
                "transition_frames": "exit_frame - leave_frame (scroll/load/cave noise)",
                "official_igt_lag": False,
                "ignored": (
                    "boot/title (mode 0/1), cave play (11), scroll/cave-enter "
                    "(6/7/16) as destinations, hit freeze (8) as leave, death (17)"
                ),
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
                "level": self._open.level,
                "screen": self._open.screen,
                "screen_hex": f"0x{self._open.screen:02X}",
                "context": self._open.context.value,
                "entry_frame": self._open.entry_frame,
                "in_transition": self._open.in_transition,
                "leave_frame": self._open.leave_frame,
            },
            "total_location_frames": sum(v.location_frames for v in self.visits),
            "total_dwell_frames": sum(v.dwell_frames for v in self.visits),
            "total_transition_frames": sum(v.transition_frames for v in self.visits),
        }
        if extra:
            payload["extra"] = dict(extra)
        return payload

    # --- internals ---------------------------------------------------------

    def _observe_snapshot(self, snap: TimingSnapshot) -> LocationVisit | None:
        if self._last_frame is not None and snap.frame < self._last_frame:
            self._abandon(
                snap.frame,
                DiscontinuityReason.FRAME_REGRESSION,
                snap.level,
                snap.screen,
                f"frame {snap.frame} < previous {self._last_frame}",
            )
            # Fall through: may re-anchor if settled after load.

        if is_boot_or_menu(snap):
            if self._open is not None or self._ever_settled:
                self._abandon(
                    snap.frame,
                    DiscontinuityReason.BOOT_OR_MENU,
                    snap.level,
                    snap.screen,
                    f"mode={snap.mode}",
                )
            return None

        if is_death(snap):
            if self._open is not None:
                self._abandon(
                    snap.frame,
                    DiscontinuityReason.DEATH,
                    snap.level,
                    snap.screen,
                    f"mode={snap.mode}",
                )
            return None

        if is_settled_play(snap):
            return self._on_settled(snap)

        # Hit freeze: still dwelling in the same room; do not mark leave.
        if is_hit_freeze(snap):
            return None

        # Non-settled (scroll, cave enter/play, triforce fanfare, other modes).
        if self._open is not None and not self._open.in_transition:
            self._mark_leave(snap)
        elif self._open is not None and self._open.in_transition:
            # Prefer non-zero next_screen once known during the transition.
            if not self._open.next_screen_at_leave and snap.next_screen:
                self._open.next_screen_at_leave = snap.next_screen
        return None

    def _on_settled(self, snap: TimingSnapshot) -> LocationVisit | None:
        self._ever_settled = True
        key = snap.location_key

        if self._open is None:
            self._open = _OpenVisit(
                source_level=0,
                source_screen=0,
                level=snap.level,
                screen=snap.screen,
                context=snap.context,
                entry_frame=snap.frame,
                sword=snap.sword,
                keys=snap.keys,
                triforce=snap.triforce,
            )
            return None

        open_key = (self._open.level, self._open.screen)

        if not self._open.in_transition:
            if key == open_key:
                self._open.sword = snap.sword
                self._open.keys = snap.keys
                self._open.triforce = snap.triforce
                return None
            # Settled in a different location without a non-settled phase:
            # save-state load, warp, or other discontinuity.
            self._abandon(
                snap.frame,
                DiscontinuityReason.LOCATION_JUMP,
                snap.level,
                snap.screen,
                (
                    f"({self._open.level},0x{self._open.screen:02X}) -> "
                    f"({snap.level},0x{snap.screen:02X}) while settled "
                    "(no transition phase)"
                ),
            )
            self._open = _OpenVisit(
                source_level=0,
                source_screen=0,
                level=snap.level,
                screen=snap.screen,
                context=snap.context,
                entry_frame=snap.frame,
                sword=snap.sword,
                keys=snap.keys,
                triforce=snap.triforce,
            )
            return None

        # Completing a transition.
        if key == open_key:
            # Returned to same location (failed door / cave bounce) — cancel leave.
            # Refresh inventory (e.g. sword cave exit onto the same overworld screen).
            self._open.in_transition = False
            self._open.leave_frame = None
            self._open.mode_at_leave = 0
            self._open.next_screen_at_leave = 0
            self._open.sword = snap.sword
            self._open.keys = snap.keys
            self._open.triforce = snap.triforce
            return None

        leave_frame = self._open.leave_frame
        if leave_frame is None:
            leave_frame = max(self._open.entry_frame, snap.frame - 1)

        visit = LocationVisit(
            source_level=self._open.source_level,
            source_screen=self._open.source_screen,
            level=self._open.level,
            screen=self._open.screen,
            dest_level=snap.level,
            dest_screen=snap.screen,
            context=self._open.context,
            dest_context=snap.context,
            entry_frame=self._open.entry_frame,
            leave_frame=leave_frame,
            exit_frame=snap.frame,
            location_frames=snap.frame - self._open.entry_frame,
            dwell_frames=leave_frame - self._open.entry_frame,
            transition_frames=snap.frame - leave_frame,
            mode_at_leave=self._open.mode_at_leave,
            next_screen_at_leave=self._open.next_screen_at_leave,
            sword=self._open.sword,
            keys=self._open.keys,
            triforce=self._open.triforce,
            sequence_index=len(self.visits),
        )
        self.visits.append(visit)
        self._open = _OpenVisit(
            source_level=visit.level,
            source_screen=visit.screen,
            level=snap.level,
            screen=snap.screen,
            context=snap.context,
            entry_frame=snap.frame,
            sword=snap.sword,
            keys=snap.keys,
            triforce=snap.triforce,
        )
        return visit

    def _mark_leave(self, snap: TimingSnapshot) -> None:
        assert self._open is not None
        self._open.in_transition = True
        self._open.leave_frame = snap.frame
        self._open.mode_at_leave = snap.mode
        self._open.next_screen_at_leave = snap.next_screen

    def _abandon(
        self,
        frame: int,
        reason: DiscontinuityReason,
        level: int,
        screen: int,
        detail: str,
    ) -> None:
        if self._open is not None:
            self.discontinuities.append(
                DiscontinuityEvent(
                    frame=frame,
                    reason=reason,
                    level=self._open.level,
                    screen=self._open.screen,
                    detail=detail,
                )
            )
        elif reason is not DiscontinuityReason.SESSION_END:
            self.discontinuities.append(
                DiscontinuityEvent(
                    frame=frame,
                    reason=reason,
                    level=level,
                    screen=screen,
                    detail=detail,
                )
            )
        self._open = None
        if reason in {
            DiscontinuityReason.BOOT_OR_MENU,
            DiscontinuityReason.FRAME_REGRESSION,
            DiscontinuityReason.RESET,
            DiscontinuityReason.DEATH,
        }:
            self._ever_settled = False


def snapshots_from_json(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[TimingSnapshot]:
    """Parse an offline fixture (list of samples or ``{"samples": [...]}``)."""
    if isinstance(data, Mapping):
        samples = data.get("samples", data.get("frames", []))
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("expected samples list in mapping fixture")
    else:
        samples = data
    return [TimingSnapshot.from_mapping(item) for item in samples]


def run_offline(
    samples: Sequence[TimingSnapshot | Mapping[str, Any]],
    *,
    source: str = "offline",
) -> dict[str, Any]:
    """Process a synthetic or recorded snapshot sequence into a report."""
    timer = RoomTimer()
    normalized: list[TimingSnapshot] = []
    for sample in samples:
        if isinstance(sample, TimingSnapshot):
            normalized.append(sample)
        else:
            normalized.append(TimingSnapshot.from_mapping(sample))
    timer.observe_many(normalized)
    if normalized:
        timer.finalize(frame=normalized[-1].frame)
    else:
        timer.finalize()
    return timer.report(source=source)


def bottleneck_visits(
    visits: Sequence[LocationVisit] | Sequence[Mapping[str, Any]],
    *,
    top_n: int = 5,
    key: str = "location_frames",
) -> list[dict[str, Any]]:
    """Rank completed hops by dwell+load cost for route debugging.

    Accepts live :class:`LocationVisit` objects or dicts from ``report()``.
    Default sort key is ``location_frames`` (dwell + transition).
    """
    rows: list[dict[str, Any]] = []
    for visit in visits:
        if isinstance(visit, LocationVisit):
            row = visit.to_dict()
        else:
            row = dict(visit)
        rows.append(row)
    rows.sort(key=lambda r: int(r.get(key, 0)), reverse=True)
    return rows[: max(0, top_n)]
