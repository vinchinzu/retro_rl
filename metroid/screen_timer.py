"""NES Metroid screen/map-cell transition timing (stock ROM / stable-retro).

Detects *confirmed* map-screen transitions from project-native RAM signals
(:class:`~metroid.ram.MetroidSnapshot`) and records per-hop timing in
**emulator frames** (one ``env.step`` = one frame).

NES Metroid has no practice-hack IGT / lag counters. This module never claims
wall-clock or in-game timer accuracy — only emulator-frame deltas between
settled map cells.

Settle rule (confirmed play on a map screen):
  ``engine_mode == game``, ``game_mode == playing`` (3), ``paused == 0``,
  ``in_door == 0``, map coordinates in range (``< 0xF0``), and health bytes
  initialized (not both zero).

A completed hop is emitted when settled play lands on a *new* map cell after:

* an unsettled stretch (typically ``in_door != 0`` door load), or
* a **seamless** adjacent cell change while still settled (common on
  multi-screen corridors where ``in_door`` stays 0; adjacency =
  Manhattan distance 1 on ``(map_x, map_y)``).

Boot/title, death (zero energy after play), frame rewinds/loads, and
non-adjacent map jumps while settled are discontinuities and do not produce
timing records.

State machine: :class:`retro_harness.hop_timer.HopTimer`.

Map identity is the Brinstar-style cell ``(map_x, map_y)`` from system RAM
``$50`` / ``$4F`` — the same coordinates used by ``brinstar.py`` and route
stop predicates.
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

from metroid.ram import (
    ENGINE_GAME,
    ENGINE_TITLE,
    GAME_MODE_PLAYING,
    MetroidSnapshot,
)

# Map coords above this are treated as unset / garbage (matches controllable).
_MAP_COORD_MAX = 0xF0


class DiscontinuityReason(str, Enum):
    """Why an in-progress screen visit was abandoned without a timing record."""

    FRAME_REGRESSION = "frame_regression"
    BOOT_OR_MENU = "boot_or_menu"
    MAP_JUMP = "map_jump"
    DEATH_OR_RESET = "death_or_reset"
    SESSION_END = "session_end"
    LOAD = "load"


@dataclass(frozen=True)
class TimingSnapshot:
    """Minimal frame sample for screen timing (synthetic or from live RAM)."""

    frame: int
    map_x: int
    map_y: int
    engine_mode: int = ENGINE_GAME
    game_mode: int = GAME_MODE_PLAYING
    paused: int = 0
    in_door: int = 0
    area: int = 0
    health_lo: int = 0x00
    health_hi: int = 0x03  # start energy tens/tanks nibble pattern
    equipment: int = 0
    missiles: int = 0
    missile_capacity: int = 0
    energy_tanks: int = 0
    samus_x: int = 0
    samus_y: int = 0

    @property
    def map_cell(self) -> tuple[int, int]:
        return (self.map_x, self.map_y)

    @property
    def map_id(self) -> int:
        """Packed cell id matching ``GameState.room``: ``(map_y << 8) | map_x``."""
        return (self.map_y << 8) | self.map_x

    @classmethod
    def from_snapshot(
        cls, snap: MetroidSnapshot, *, frame: int
    ) -> TimingSnapshot:
        return cls(
            frame=frame,
            map_x=snap.map_x,
            map_y=snap.map_y,
            engine_mode=snap.engine_mode,
            game_mode=snap.game_mode,
            paused=snap.paused,
            in_door=snap.in_door,
            area=snap.area,
            health_lo=snap.health_lo,
            health_hi=snap.health_hi,
            equipment=snap.equipment,
            missiles=snap.missiles,
            missile_capacity=snap.missile_capacity,
            energy_tanks=snap.energy_tanks,
            samus_x=snap.samus_x,
            samus_y=snap.samus_y,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TimingSnapshot:
        """Build from a JSON-friendly dict (offline replay)."""
        map_cell = data.get("map_cell")
        if map_cell is not None and len(map_cell) == 2:
            map_x, map_y = int(map_cell[0]), int(map_cell[1])
        else:
            map_x = int(data.get("map_x", 0))
            map_y = int(data.get("map_y", 0))
        return cls(
            frame=int(data["frame"]),
            map_x=map_x,
            map_y=map_y,
            engine_mode=int(data.get("engine_mode", ENGINE_GAME)),
            game_mode=int(data.get("game_mode", GAME_MODE_PLAYING)),
            paused=int(data.get("paused", 0)),
            in_door=int(data.get("in_door", 0)),
            area=int(data.get("area", 0)),
            health_lo=int(data.get("health_lo", 0x00)),
            health_hi=int(data.get("health_hi", 0x03)),
            equipment=int(data.get("equipment", 0)),
            missiles=int(data.get("missiles", 0)),
            missile_capacity=int(data.get("missile_capacity", 0)),
            energy_tanks=int(data.get("energy_tanks", 0)),
            samus_x=int(data.get("samus_x", 0)),
            samus_y=int(data.get("samus_y", 0)),
        )


def is_settled_play(snap: TimingSnapshot) -> bool:
    """True when the game is settled in controllable map-screen gameplay."""
    if snap.engine_mode != ENGINE_GAME:
        return False
    if snap.game_mode != GAME_MODE_PLAYING:
        return False
    if snap.paused != 0:
        return False
    if snap.in_door != 0:
        return False
    if snap.map_x >= _MAP_COORD_MAX or snap.map_y >= _MAP_COORD_MAX:
        return False
    # Reject boot/death energy (both health bytes zero) — matches readiness.
    if snap.health_lo == 0 and snap.health_hi == 0:
        return False
    return True


def is_boot_or_menu(snap: TimingSnapshot) -> bool:
    return snap.engine_mode == ENGINE_TITLE


def is_dead_energy(snap: TimingSnapshot) -> bool:
    """True when energy is fully drained under the game engine (death path)."""
    return (
        snap.engine_mode == ENGINE_GAME
        and snap.health_lo == 0
        and snap.health_hi == 0
    )


def map_cells_adjacent(ax: int, ay: int, bx: int, by: int) -> bool:
    """True when two map cells share an edge (Manhattan distance 1)."""
    return abs(ax - bx) + abs(ay - by) == 1


@dataclass(frozen=True)
class ScreenVisit:
    """One completed map-screen hop with project-native frame timing.

    Frame semantics (emulator frames, 60 Hz NTSC nominal):

    * ``entry_frame`` — first settled frame on the source map cell.
    * ``leave_frame`` — first non-settled frame after dwelling (door, mode
      change, etc.).
    * ``exit_frame`` — first settled frame on the destination map cell.
    * ``screen_frames`` — ``exit_frame - entry_frame`` (dwell + load).
    * ``dwell_frames`` — ``leave_frame - entry_frame`` (time until leave).
    * ``transition_frames`` — ``exit_frame - leave_frame`` (door/load).
    """

    source_map_x: int
    source_map_y: int
    map_x: int
    map_y: int
    dest_map_x: int
    dest_map_y: int
    area: int
    dest_area: int
    entry_frame: int
    leave_frame: int
    exit_frame: int
    screen_frames: int
    dwell_frames: int
    transition_frames: int
    in_door_at_leave: int
    equipment: int
    missiles: int
    missile_capacity: int
    energy_tanks: int
    sequence_index: int

    @property
    def map_cell(self) -> tuple[int, int]:
        return (self.map_x, self.map_y)

    @property
    def dest_map_cell(self) -> tuple[int, int]:
        return (self.dest_map_x, self.dest_map_y)

    @property
    def source_map_cell(self) -> tuple[int, int]:
        return (self.source_map_x, self.source_map_y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "source_map_x": self.source_map_x,
            "source_map_y": self.source_map_y,
            "source_map_cell": [self.source_map_x, self.source_map_y],
            "map_x": self.map_x,
            "map_y": self.map_y,
            "map_cell": [self.map_x, self.map_y],
            "map_id": (self.map_y << 8) | self.map_x,
            "dest_map_x": self.dest_map_x,
            "dest_map_y": self.dest_map_y,
            "dest_map_cell": [self.dest_map_x, self.dest_map_y],
            "dest_map_id": (self.dest_map_y << 8) | self.dest_map_x,
            "area": self.area,
            "dest_area": self.dest_area,
            "entry_frame": self.entry_frame,
            "leave_frame": self.leave_frame,
            "exit_frame": self.exit_frame,
            "screen_frames": self.screen_frames,
            "dwell_frames": self.dwell_frames,
            "transition_frames": self.transition_frames,
            "in_door_at_leave": self.in_door_at_leave,
            "equipment": self.equipment,
            "equipment_hex": f"0x{self.equipment:02X}",
            "missiles": self.missiles,
            "missile_capacity": self.missile_capacity,
            "energy_tanks": self.energy_tanks,
            "timing_unit": "emulator_frames",
            "timing_note": (
                "screen_frames = exit_frame - entry_frame (includes door load); "
                "dwell_frames = leave_frame - entry_frame; "
                "transition_frames = exit_frame - leave_frame. "
                "Not IGT/lag; NES Metroid has no practice-hack timers in this stack."
            ),
        }


@dataclass(frozen=True)
class DiscontinuityEvent:
    """An abandoned in-progress visit or tracking reset."""

    frame: int
    reason: DiscontinuityReason
    map_x: int
    map_y: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "reason": self.reason.value,
            "map_x": self.map_x,
            "map_y": self.map_y,
            "map_cell": [self.map_x, self.map_y],
            "detail": self.detail,
        }


MapCell = tuple[int, int]
_LEAVE_KEYS = frozenset({"in_door_at_leave"})


def _make_visit(
    open_hop: OpenHop[MapCell],
    dest: MapCell,
    leave_frame: int,
    exit_frame: int,
    sequence_index: int,
    meta: Mapping[str, Any],
    dest_context: Mapping[str, Any],
) -> ScreenVisit:
    mx, my = open_hop.location
    src = open_hop.source or (0, 0)
    dx, dy = dest
    return ScreenVisit(
        source_map_x=src[0],
        source_map_y=src[1],
        map_x=mx,
        map_y=my,
        dest_map_x=dx,
        dest_map_y=dy,
        area=int(meta.get("area", 0)),
        dest_area=int(dest_context.get("area", 0)),
        entry_frame=open_hop.entry_frame,
        leave_frame=leave_frame,
        exit_frame=exit_frame,
        screen_frames=exit_frame - open_hop.entry_frame,
        dwell_frames=leave_frame - open_hop.entry_frame,
        transition_frames=exit_frame - leave_frame,
        in_door_at_leave=int(meta.get("in_door_at_leave", 0)),
        equipment=int(meta.get("equipment", 0)),
        missiles=int(meta.get("missiles", 0)),
        missile_capacity=int(meta.get("missile_capacity", 0)),
        energy_tanks=int(meta.get("energy_tanks", 0)),
        sequence_index=sequence_index,
    )


def _make_disc(
    frame: int, reason: str, location: MapCell, detail: str
) -> DiscontinuityEvent:
    try:
        reason_enum = DiscontinuityReason(reason)
    except ValueError:
        reason_enum = DiscontinuityReason.LOAD
    mx, my = location if location else (0, 0)
    return DiscontinuityEvent(
        frame=frame, reason=reason_enum, map_x=mx, map_y=my, detail=detail
    )


def _snap_to_hop(snap: TimingSnapshot) -> HopFrame[MapCell]:
    loc: MapCell = (snap.map_x, snap.map_y)
    ctx = {
        "area": snap.area,
        "equipment": snap.equipment,
        "missiles": snap.missiles,
        "missile_capacity": snap.missile_capacity,
        "energy_tanks": snap.energy_tanks,
    }
    leave_meta = {"in_door_at_leave": snap.in_door}

    if is_boot_or_menu(snap):
        return HopFrame(
            frame=snap.frame,
            location=loc,
            status="abandon",
            abandon_reason=DiscontinuityReason.BOOT_OR_MENU.value,
            abandon_detail=f"engine_mode={snap.engine_mode}",
        )
    if is_dead_energy(snap):
        return HopFrame(
            frame=snap.frame,
            location=loc,
            status="abandon",
            abandon_reason=DiscontinuityReason.DEATH_OR_RESET.value,
            # Metroid abandons when open OR ever_settled
            abandon_detail="ever:health_lo=0 health_hi=0",
        )
    if is_settled_play(snap):
        return HopFrame(
            frame=snap.frame,
            location=loc,
            status="settled",
            context=ctx,
        )
    return HopFrame(
        frame=snap.frame,
        location=loc,
        status="transition",
        leave_meta=leave_meta,
    )


@dataclass
class ScreenTimer:
    """Incremental map-screen transition detector and hop timer.

    Feed one :class:`TimingSnapshot` (or :class:`MetroidSnapshot` with
    ``frame=``) per emulator frame via :meth:`observe`. Completed hops
    accumulate in :attr:`visits`.
    """

    _engine: HopTimer[MapCell, ScreenVisit, DiscontinuityEvent] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._engine = HopTimer(
            make_visit=_make_visit,
            make_discontinuity=_make_disc,
            seamless_allowed=lambda a, b: map_cells_adjacent(a[0], a[1], b[0], b[1]),
            jump_reason=DiscontinuityReason.MAP_JUMP.value,
            null_location=(0, 0),
            leave_context_keys=_LEAVE_KEYS,
            reset_ever_settled_reasons=frozenset(
                {
                    DiscontinuityReason.BOOT_OR_MENU.value,
                    DiscontinuityReason.FRAME_REGRESSION.value,
                    DiscontinuityReason.DEATH_OR_RESET.value,
                    DiscontinuityReason.LOAD.value,
                }
            ),
        )

    @property
    def visits(self) -> list[ScreenVisit]:
        return self._engine.visits

    @property
    def discontinuities(self) -> list[DiscontinuityEvent]:
        return self._engine.discontinuities

    @property
    def _open(self):
        """Compatibility shim for session helpers that inspect open visits."""
        eng = self._engine._open
        if eng is None:
            return None
        # Duck-type fields used by screen_timing_session.
        mx, my = eng.location

        class _Compat:
            map_x = mx
            map_y = my
            entry_frame = eng.entry_frame
            leave_frame = eng.leave_frame
            in_transition = eng.in_transition

        return _Compat()

    def observe(
        self,
        sample: TimingSnapshot | MetroidSnapshot,
        *,
        frame: int | None = None,
    ) -> ScreenVisit | None:
        """Ingest one frame sample. Return a completed visit if one just closed.

        When ``sample`` is a :class:`MetroidSnapshot`, pass ``frame=`` (emulator
        step index). :class:`TimingSnapshot` already carries ``frame``.
        """
        if isinstance(sample, TimingSnapshot):
            snap = sample
        else:
            if frame is None:
                raise TypeError(
                    "observe(MetroidSnapshot) requires frame= emulator index"
                )
            snap = TimingSnapshot.from_snapshot(sample, frame=frame)
        return self._engine.observe_frame(_snap_to_hop(snap))

    def observe_many(
        self,
        samples: Iterable[TimingSnapshot | Mapping[str, Any]],
    ) -> list[ScreenVisit]:
        """Ingest a sequence; return visits completed during that sequence."""
        newly: list[ScreenVisit] = []
        for sample in samples:
            if isinstance(sample, TimingSnapshot):
                visit = self.observe(sample)
            else:
                visit = self.observe(TimingSnapshot.from_mapping(sample))
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
            mx, my = o.location
            open_payload = {
                "map_x": mx,
                "map_y": my,
                "map_cell": [mx, my],
                "entry_frame": o.entry_frame,
                "in_transition": o.in_transition,
                "leave_frame": o.leave_frame,
            }
        return self._engine.report_base(
            kind="metroid_screen_timing",
            timing_semantics={
                "frame_basis": (
                    "stable-retro env.step frames (nominal 60 Hz NTSC); "
                    "not wall-clock and not IGT/lag counters"
                ),
                "settle_rule": (
                    "engine==game + game_mode==playing(3) + paused==0 + "
                    "in_door==0 + map_x/y < 0xF0 + health bytes not both zero"
                ),
                "map_identity": (
                    "system RAM map_x ($50) / map_y ($4F); same cells as "
                    "brinstar.py and MetroidSnapshot.map_cell"
                ),
                "hop_rules": (
                    "door leave (in_door!=0 then settle new cell) OR seamless "
                    "adjacent settled cell change (Manhattan distance 1)"
                ),
                "screen_frames": "exit_frame - entry_frame (dwell + door load)",
                "dwell_frames": "leave_frame - entry_frame (until leave)",
                "transition_frames": "exit_frame - leave_frame (door/load; 0 if seamless)",
                "igt_or_lag": False,
            },
            source=source,
            extra=extra,
            open_visit_payload=open_payload,
            visit_to_dict=lambda v: v.to_dict(),
            disc_to_dict=lambda d: d.to_dict(),
            totals={
                "total_screen_frames": sum(v.screen_frames for v in self.visits),
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
    timer = ScreenTimer()
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
