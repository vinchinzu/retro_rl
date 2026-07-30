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

State machine: :class:`retro_harness.hop_timer.HopTimer`.

Cave play (mode 11) and transition noise are not timed destinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from retro_harness.hop_timer import (
    HopFrame,
    HopTimer,
    OpenHop,
    snapshots_from_json_mapping,
)

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


LocationKey = tuple[int, int]
_LEAVE_KEYS = frozenset({"mode_at_leave", "next_screen_at_leave"})


def _make_visit(
    open_hop: OpenHop[LocationKey],
    dest: LocationKey,
    leave_frame: int,
    exit_frame: int,
    sequence_index: int,
    meta: Mapping[str, Any],
    dest_context: Mapping[str, Any],
) -> LocationVisit:
    level, screen = open_hop.location
    src = open_hop.source or (0, 0)
    dlevel, dscreen = dest
    ctx = meta.get("context", GameContext.OVERWORLD)
    if isinstance(ctx, str):
        ctx = GameContext(ctx)
    dctx = dest_context.get("context", context_for_level(dlevel))
    if isinstance(dctx, str):
        dctx = GameContext(dctx)
    return LocationVisit(
        source_level=src[0],
        source_screen=src[1],
        level=level,
        screen=screen,
        dest_level=dlevel,
        dest_screen=dscreen,
        context=ctx,
        dest_context=dctx,
        entry_frame=open_hop.entry_frame,
        leave_frame=leave_frame,
        exit_frame=exit_frame,
        location_frames=exit_frame - open_hop.entry_frame,
        dwell_frames=leave_frame - open_hop.entry_frame,
        transition_frames=exit_frame - leave_frame,
        mode_at_leave=int(meta.get("mode_at_leave", 0)),
        next_screen_at_leave=int(meta.get("next_screen_at_leave", 0)),
        sword=int(meta.get("sword", 0)),
        keys=int(meta.get("keys", 0)),
        triforce=int(meta.get("triforce", 0)),
        sequence_index=sequence_index,
    )


def _make_disc(
    frame: int, reason: str, location: LocationKey, detail: str
) -> DiscontinuityEvent:
    try:
        reason_enum = DiscontinuityReason(reason)
    except ValueError:
        reason_enum = DiscontinuityReason.RESET
    level, screen = location if location else (0, 0)
    return DiscontinuityEvent(
        frame=frame, reason=reason_enum, level=level, screen=screen, detail=detail
    )


def _snap_to_hop(snap: TimingSnapshot) -> HopFrame[LocationKey]:
    loc: LocationKey = (snap.level, snap.screen)
    ctx = {
        "context": snap.context,
        "sword": snap.sword,
        "keys": snap.keys,
        "triforce": snap.triforce,
    }
    leave_meta = {
        "mode_at_leave": snap.mode,
        "next_screen_at_leave": snap.next_screen,
    }

    if is_boot_or_menu(snap):
        return HopFrame(
            frame=snap.frame,
            location=loc,
            status="abandon",
            abandon_reason=DiscontinuityReason.BOOT_OR_MENU.value,
            abandon_detail=f"mode={snap.mode}",
        )
    if is_death(snap):
        return HopFrame(
            frame=snap.frame,
            location=loc,
            status="abandon",
            abandon_reason=DiscontinuityReason.DEATH.value,
            abandon_detail=f"mode={snap.mode}",
        )
    if is_settled_play(snap):
        return HopFrame(
            frame=snap.frame,
            location=loc,
            status="settled",
            context=ctx,
        )
    if is_hit_freeze(snap):
        return HopFrame(frame=snap.frame, location=loc, status="ignore")
    return HopFrame(
        frame=snap.frame,
        location=loc,
        status="transition",
        leave_meta=leave_meta,
    )


@dataclass
class RoomTimer:
    """Incremental screen/room-transition detector and hop timer.

    Feed one :class:`TimingSnapshot` (or :class:`ZeldaSnapshot` + frame) per
    emulator frame via :meth:`observe`. Completed hops accumulate in
    :attr:`visits`.
    """

    _engine: HopTimer[LocationKey, LocationVisit, DiscontinuityEvent] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._engine = HopTimer(
            make_visit=_make_visit,
            make_discontinuity=_make_disc,
            jump_reason=DiscontinuityReason.LOCATION_JUMP.value,
            null_location=(0, 0),
            leave_context_keys=_LEAVE_KEYS,
            reset_ever_settled_reasons=frozenset(
                {
                    DiscontinuityReason.BOOT_OR_MENU.value,
                    DiscontinuityReason.FRAME_REGRESSION.value,
                    DiscontinuityReason.RESET.value,
                    DiscontinuityReason.DEATH.value,
                }
            ),
        )

    @property
    def visits(self) -> list[LocationVisit]:
        return self._engine.visits

    @property
    def discontinuities(self) -> list[DiscontinuityEvent]:
        return self._engine.discontinuities

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
        return self._engine.observe_frame(_snap_to_hop(snap))

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
            level, screen = o.location
            ctx = o.context.get("context", context_for_level(level))
            if isinstance(ctx, GameContext):
                ctx_val = ctx.value
            else:
                ctx_val = str(ctx)
            open_payload = {
                "level": level,
                "screen": screen,
                "screen_hex": f"0x{screen:02X}",
                "context": ctx_val,
                "entry_frame": o.entry_frame,
                "in_transition": o.in_transition,
                "leave_frame": o.leave_frame,
            }
        return self._engine.report_base(
            kind="zelda_i_screen_room_timing",
            timing_semantics={
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
            source=source,
            extra=extra,
            open_visit_payload=open_payload,
            visit_to_dict=lambda v: v.to_dict(),
            disc_to_dict=lambda d: d.to_dict(),
            totals={
                "total_location_frames": sum(v.location_frames for v in self.visits),
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
