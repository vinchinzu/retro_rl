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

Map identity is the Brinstar-style cell ``(map_x, map_y)`` from system RAM
``$50`` / ``$4F`` — the same coordinates used by ``brinstar.py`` and route
stop predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

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


@dataclass
class _OpenVisit:
    source_map_x: int
    source_map_y: int
    map_x: int
    map_y: int
    area: int
    entry_frame: int
    equipment: int
    missiles: int
    missile_capacity: int
    energy_tanks: int
    leave_frame: int | None = None
    in_door_at_leave: int = 0
    in_transition: bool = False


@dataclass
class ScreenTimer:
    """Incremental map-screen transition detector and hop timer.

    Feed one :class:`TimingSnapshot` (or :class:`MetroidSnapshot` with
    ``frame=``) per emulator frame via :meth:`observe`. Completed hops
    accumulate in :attr:`visits`.
    """

    visits: list[ScreenVisit] = field(default_factory=list)
    discontinuities: list[DiscontinuityEvent] = field(default_factory=list)
    _open: _OpenVisit | None = field(default=None, repr=False)
    _last_frame: int | None = field(default=None, repr=False)
    _ever_settled: bool = field(default=False, repr=False)

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
        completed = self._observe_snapshot(snap)
        self._last_frame = snap.frame
        return completed

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
        if self._open is None:
            return
        end_frame = frame if frame is not None else (self._last_frame or 0)
        self._abandon(
            end_frame,
            DiscontinuityReason.SESSION_END,
            self._open.map_x,
            self._open.map_y,
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
            "kind": "metroid_screen_timing",
            "timing_unit": "emulator_frames",
            "timing_semantics": {
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
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "visit_count": len(self.visits),
            "discontinuity_count": len(self.discontinuities),
            "visits": [visit.to_dict() for visit in self.visits],
            "discontinuities": [event.to_dict() for event in self.discontinuities],
            "open_visit": None
            if self._open is None
            else {
                "map_x": self._open.map_x,
                "map_y": self._open.map_y,
                "map_cell": [self._open.map_x, self._open.map_y],
                "entry_frame": self._open.entry_frame,
                "in_transition": self._open.in_transition,
                "leave_frame": self._open.leave_frame,
            },
            "total_screen_frames": sum(v.screen_frames for v in self.visits),
            "total_dwell_frames": sum(v.dwell_frames for v in self.visits),
            "total_transition_frames": sum(
                v.transition_frames for v in self.visits
            ),
        }
        if extra:
            payload["extra"] = dict(extra)
        return payload

    # --- internals ---------------------------------------------------------

    def _observe_snapshot(self, snap: TimingSnapshot) -> ScreenVisit | None:
        if self._last_frame is not None and snap.frame < self._last_frame:
            self._abandon(
                snap.frame,
                DiscontinuityReason.FRAME_REGRESSION,
                snap.map_x,
                snap.map_y,
                f"frame {snap.frame} < previous {self._last_frame}",
            )
            # Fall through: may re-anchor if settled after load.

        if is_boot_or_menu(snap):
            if self._open is not None or self._ever_settled:
                self._abandon(
                    snap.frame,
                    DiscontinuityReason.BOOT_OR_MENU,
                    snap.map_x,
                    snap.map_y,
                    f"engine_mode={snap.engine_mode}",
                )
            return None

        if is_dead_energy(snap):
            if self._open is not None or self._ever_settled:
                self._abandon(
                    snap.frame,
                    DiscontinuityReason.DEATH_OR_RESET,
                    snap.map_x,
                    snap.map_y,
                    "health_lo=0 health_hi=0",
                )
            return None

        if is_settled_play(snap):
            return self._on_settled(snap)

        # Non-settled play (door, pause, fanfare, intro, unknown).
        if self._open is not None and not self._open.in_transition:
            self._mark_leave(snap)
        return None

    def _on_settled(self, snap: TimingSnapshot) -> ScreenVisit | None:
        self._ever_settled = True

        if self._open is None:
            self._open = _OpenVisit(
                source_map_x=0,
                source_map_y=0,
                map_x=snap.map_x,
                map_y=snap.map_y,
                area=snap.area,
                entry_frame=snap.frame,
                equipment=snap.equipment,
                missiles=snap.missiles,
                missile_capacity=snap.missile_capacity,
                energy_tanks=snap.energy_tanks,
            )
            return None

        same_cell = (
            snap.map_x == self._open.map_x and snap.map_y == self._open.map_y
        )

        if not self._open.in_transition:
            if same_cell:
                self._open.equipment = snap.equipment
                self._open.missiles = snap.missiles
                self._open.missile_capacity = snap.missile_capacity
                self._open.energy_tanks = snap.energy_tanks
                return None
            # Adjacent settled cells: seamless multi-screen scroll (in_door may
            # stay 0 for corridor screens). Non-adjacent = load/warp jump.
            if map_cells_adjacent(
                self._open.map_x, self._open.map_y, snap.map_x, snap.map_y
            ):
                return self._complete_visit(
                    snap,
                    leave_frame=snap.frame,
                    in_door_at_leave=0,
                )
            self._abandon(
                snap.frame,
                DiscontinuityReason.MAP_JUMP,
                snap.map_x,
                snap.map_y,
                (
                    f"map ({self._open.map_x},{self._open.map_y}) -> "
                    f"({snap.map_x},{snap.map_y}) while settled "
                    "(non-adjacent; no door/leave phase)"
                ),
            )
            self._open = _OpenVisit(
                source_map_x=0,
                source_map_y=0,
                map_x=snap.map_x,
                map_y=snap.map_y,
                area=snap.area,
                entry_frame=snap.frame,
                equipment=snap.equipment,
                missiles=snap.missiles,
                missile_capacity=snap.missile_capacity,
                energy_tanks=snap.energy_tanks,
            )
            return None

        # Completing a transition after a leave phase.
        if same_cell:
            # Bounce / pause return — cancel leave.
            self._open.in_transition = False
            self._open.leave_frame = None
            self._open.in_door_at_leave = 0
            return None

        leave_frame = self._open.leave_frame
        if leave_frame is None:
            leave_frame = max(self._open.entry_frame, snap.frame - 1)
        return self._complete_visit(
            snap,
            leave_frame=leave_frame,
            in_door_at_leave=self._open.in_door_at_leave,
        )

    def _complete_visit(
        self,
        snap: TimingSnapshot,
        *,
        leave_frame: int,
        in_door_at_leave: int,
    ) -> ScreenVisit:
        assert self._open is not None
        visit = ScreenVisit(
            source_map_x=self._open.source_map_x,
            source_map_y=self._open.source_map_y,
            map_x=self._open.map_x,
            map_y=self._open.map_y,
            dest_map_x=snap.map_x,
            dest_map_y=snap.map_y,
            area=self._open.area,
            dest_area=snap.area,
            entry_frame=self._open.entry_frame,
            leave_frame=leave_frame,
            exit_frame=snap.frame,
            screen_frames=snap.frame - self._open.entry_frame,
            dwell_frames=leave_frame - self._open.entry_frame,
            transition_frames=snap.frame - leave_frame,
            in_door_at_leave=in_door_at_leave,
            equipment=self._open.equipment,
            missiles=self._open.missiles,
            missile_capacity=self._open.missile_capacity,
            energy_tanks=self._open.energy_tanks,
            sequence_index=len(self.visits),
        )
        self.visits.append(visit)
        self._open = _OpenVisit(
            source_map_x=visit.map_x,
            source_map_y=visit.map_y,
            map_x=snap.map_x,
            map_y=snap.map_y,
            area=snap.area,
            entry_frame=snap.frame,
            equipment=snap.equipment,
            missiles=snap.missiles,
            missile_capacity=snap.missile_capacity,
            energy_tanks=snap.energy_tanks,
        )
        return visit

    def _mark_leave(self, snap: TimingSnapshot) -> None:
        assert self._open is not None
        self._open.in_transition = True
        self._open.leave_frame = snap.frame
        self._open.in_door_at_leave = snap.in_door

    def _abandon(
        self,
        frame: int,
        reason: DiscontinuityReason,
        map_x: int,
        map_y: int,
        detail: str,
    ) -> None:
        if self._open is not None:
            self.discontinuities.append(
                DiscontinuityEvent(
                    frame=frame,
                    reason=reason,
                    map_x=self._open.map_x if self._open.map_x else map_x,
                    map_y=self._open.map_y if self._open.map_y else map_y,
                    detail=detail,
                )
            )
        elif reason is not DiscontinuityReason.SESSION_END:
            self.discontinuities.append(
                DiscontinuityEvent(
                    frame=frame,
                    reason=reason,
                    map_x=map_x,
                    map_y=map_y,
                    detail=detail,
                )
            )
        self._open = None
        if reason in {
            DiscontinuityReason.BOOT_OR_MENU,
            DiscontinuityReason.FRAME_REGRESSION,
            DiscontinuityReason.DEATH_OR_RESET,
            DiscontinuityReason.LOAD,
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
