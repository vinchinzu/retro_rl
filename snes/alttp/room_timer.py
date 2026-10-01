"""Indoor-room and overworld-screen timing for ALTTP.

One ``env.step`` is one emulator frame. This is not IGT.

Location key is ``("in", room)`` indoors (full ``$A0``, so Link's house
stays ``0x104``) or ``("ow", dark_world, screen)`` outdoors.

Settle rule (module census, 2026-09-29): module is play (``$10`` in
``{0x07, 0x09}``) and submodule is 0. Scroll, stairs, dialog, and loads
are transitions. A hit does not leave play, so i-frames stay inside the
open visit. Intro ``0x00`` and file select ``0x01`` abandon.

``transition_kind`` labels a step ``door``, ``stairs``, ``layer_swap``,
``fall``, or ``warp``. The room timer stores that on the completed visit.

State machine: :class:`retro_harness.hop_timer.HopTimer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from retro_harness.hop_timer import HopFrame, HopTimer, OpenHop

from alttp.ram import AlttpSnapshot

# Play handlers. Submodule 0 is the only controllable state.
PLAY_MODULES = frozenset({0x07, 0x09})
BOOT_MODULES = frozenset({0x00, 0x01})


class DiscontinuityReason(str, Enum):
    FRAME_REGRESSION = "frame_regression"
    BOOT_OR_MENU = "boot_or_menu"
    LOCATION_JUMP = "location_jump"
    SESSION_END = "session_end"
    RESET = "reset"


# ("in", room) or ("ow", dark_world, screen). dark_world is a bool.
LocationKey = tuple[Any, ...]


def location_key(
    *,
    indoors: bool,
    room: int,
    dark_world: bool,
    screen: int,
) -> LocationKey:
    if indoors:
        return ("in", int(room))
    return ("ow", bool(dark_world), int(screen))


@dataclass(frozen=True)
class TimingSnapshot:
    """One frame for the hop timer. Synthetic sequences use this directly."""

    frame: int
    module: int
    submodule: int
    indoors: bool
    room: int
    dark_world: bool = False
    screen: int = 0
    floor: int = 0
    layer: int = 0

    @property
    def location_key(self) -> LocationKey:
        return location_key(
            indoors=self.indoors,
            room=self.room,
            dark_world=self.dark_world,
            screen=self.screen,
        )

    @classmethod
    def from_snapshot(cls, snap: AlttpSnapshot, *, frame: int) -> TimingSnapshot:
        return cls(
            frame=frame,
            module=int(snap.game_mode),
            submodule=int(snap.submodule),
            indoors=bool(snap.indoors),
            room=int(snap.room_id),
            dark_world=bool(snap.dark_world),
            screen=int(snap.screen_id),
            floor=int(snap.floor),
            layer=int(snap.layer),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TimingSnapshot:
        return cls(
            frame=int(data["frame"]),
            module=int(data.get("module", 0x07)),
            submodule=int(data.get("submodule", 0)),
            indoors=bool(data.get("indoors", False)),
            room=int(data.get("room", 0)),
            dark_world=bool(data.get("dark_world", False)),
            screen=int(data.get("screen", 0)),
            floor=int(data.get("floor", 0)),
            layer=int(data.get("layer", 0)),
        )


def is_settled_play(snap: TimingSnapshot) -> bool:
    return int(snap.module) in PLAY_MODULES and int(snap.submodule) == 0


# snesrev kDungeonSubmodules. 0x07 is FallingTransition. The castle well
# is not that: tile 0x5F takes spiral 0x0E, and $A0 becomes 0x72 while
# $A4 is still 0x00 (CastleRoom01, 2026-09-29). z3 calls the edge a Drop.
_STAIR_SUBMODULES = frozenset({0x06, 0x08, 0x0E, 0x10, 0x11, 0x12, 0x13})
_FALL_SUBMODULES = frozenset({0x07, 0x14})
_WARP_SUBMODULES = frozenset({0x15, 0x19})
_DOOR_SUBMODULES = frozenset({0x01, 0x02})
_OW_SCROLL_SUBMODULES = frozenset(range(0x01, 0x09)) | frozenset({0x0A})
_FALLING_ENTRANCE = 0x11
_WELL_FROM = 0x01
_WELL_TO = 0x72
_TRANSITION_BUSY = _STAIR_SUBMODULES | _FALL_SUBMODULES | _WARP_SUBMODULES | _DOOR_SUBMODULES


def _room_base(room: int) -> int:
    return int(room) & 0xFF


def _signed_floor(value: int) -> int:
    byte = int(value) & 0xFF
    return byte - 256 if byte >= 0x80 else byte


def transition_kind(prev: TimingSnapshot, snap: TimingSnapshot) -> str:
    """Label the step from ``prev`` to ``snap``.

    One of ``door``, ``stairs``, ``layer_swap``, ``fall``, ``warp``.
    Room ``0x01`` → ``0x72`` is ``fall``. The climb back up is ``stairs``.
    """
    sub = int(snap.submodule)
    prev_sub = int(prev.submodule)
    mod = int(snap.module)
    same_room = (
        bool(prev.indoors)
        and bool(snap.indoors)
        and _room_base(prev.room) == _room_base(snap.room)
    )
    if (
        same_room
        and int(prev.layer) != int(snap.layer)
        and sub not in _TRANSITION_BUSY
        and prev_sub not in _STAIR_SUBMODULES | _FALL_SUBMODULES
    ):
        return "layer_swap"
    if mod == _FALLING_ENTRANCE:
        return "fall"
    # Dungeon submodules share numbers with overworld scroll. Gate on 0x07.
    if mod == 0x07 and (sub in _FALL_SUBMODULES or prev_sub in _FALL_SUBMODULES):
        return "fall"
    # Well drop. Room id flips on the first 0x0E frame, before $A4.
    if (
        bool(prev.indoors)
        and bool(snap.indoors)
        and _room_base(prev.room) == _WELL_FROM
        and _room_base(snap.room) == _WELL_TO
    ):
        return "fall"
    if (
        mod == 0x07
        and sub == 0x0E
        and bool(prev.indoors)
        and bool(snap.indoors)
        and not same_room
        and _signed_floor(snap.floor) < _signed_floor(prev.floor)
    ):
        return "fall"
    if mod == 0x07 and (sub in _WARP_SUBMODULES or prev_sub in _WARP_SUBMODULES):
        return "warp"
    if mod == 0x07 and (sub in _STAIR_SUBMODULES or prev_sub in _STAIR_SUBMODULES):
        return "stairs"
    if mod == 0x07 and sub in _DOOR_SUBMODULES:
        return "door"
    if mod == 0x09 and sub in _OW_SCROLL_SUBMODULES:
        return "door"
    if mod in (0x08, 0x0A) or (bool(prev.indoors) and not bool(snap.indoors)):
        return "door"
    if (not bool(prev.indoors)) and bool(snap.indoors):
        return "door"
    if bool(prev.dark_world) != bool(snap.dark_world):
        return "warp"
    return "door"


@dataclass(frozen=True)
class LocationVisit:
    """One completed hop. ``location`` is where the frames were spent."""

    location: LocationKey
    dest: LocationKey
    entry_frame: int
    leave_frame: int
    exit_frame: int
    location_frames: int
    dwell_frames: int
    transition_frames: int
    sequence_index: int
    module_at_leave: int = 0
    submodule_at_leave: int = 0
    transition_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "location": list(self.location),
            "dest": list(self.dest),
            "entry_frame": self.entry_frame,
            "leave_frame": self.leave_frame,
            "exit_frame": self.exit_frame,
            "location_frames": self.location_frames,
            "dwell_frames": self.dwell_frames,
            "transition_frames": self.transition_frames,
            "module_at_leave": self.module_at_leave,
            "submodule_at_leave": self.submodule_at_leave,
            "transition_kind": self.transition_kind,
            "timing_unit": "emulator_frames",
        }


@dataclass(frozen=True)
class DiscontinuityEvent:
    frame: int
    reason: DiscontinuityReason
    location: LocationKey
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "reason": self.reason.value,
            "location": list(self.location),
            "detail": self.detail,
        }


_LEAVE_KEYS = frozenset({"module_at_leave", "submodule_at_leave", "transition_kind"})


def _make_visit(
    open_hop: OpenHop[LocationKey],
    dest: LocationKey,
    leave_frame: int,
    exit_frame: int,
    sequence_index: int,
    meta: Mapping[str, Any],
    dest_context: Mapping[str, Any],
) -> LocationVisit:
    del dest_context
    return LocationVisit(
        location=open_hop.location,
        dest=dest,
        entry_frame=open_hop.entry_frame,
        leave_frame=leave_frame,
        exit_frame=exit_frame,
        location_frames=exit_frame - open_hop.entry_frame,
        dwell_frames=leave_frame - open_hop.entry_frame,
        transition_frames=exit_frame - leave_frame,
        sequence_index=sequence_index,
        module_at_leave=int(meta.get("module_at_leave", 0)),
        submodule_at_leave=int(meta.get("submodule_at_leave", 0)),
        transition_kind=str(meta.get("transition_kind", "")),
    )


def _make_disc(
    frame: int, reason: str, location: LocationKey, detail: str
) -> DiscontinuityEvent:
    try:
        reason_enum = DiscontinuityReason(reason)
    except ValueError:
        reason_enum = DiscontinuityReason.RESET
    return DiscontinuityEvent(
        frame=frame, reason=reason_enum, location=location, detail=detail
    )


def _worth_kind(prev: TimingSnapshot, snap: TimingSnapshot) -> bool:
    if snap.location_key != prev.location_key:
        return True
    if bool(prev.indoors) and bool(snap.indoors) and int(snap.layer) != int(prev.layer):
        return True
    mod = int(snap.module)
    if mod in PLAY_MODULES and int(snap.submodule) != 0:
        return True
    return mod in (0x08, 0x0A, _FALLING_ENTRANCE)


def _snap_to_hop(
    snap: TimingSnapshot, *, transition_kind: str = ""
) -> HopFrame[LocationKey]:
    loc = snap.location_key
    if int(snap.module) in BOOT_MODULES:
        return HopFrame(
            frame=snap.frame,
            location=loc,
            status="abandon",
            abandon_reason=DiscontinuityReason.BOOT_OR_MENU.value,
            abandon_detail=f"module={int(snap.module):#x}",
        )
    if is_settled_play(snap):
        return HopFrame(frame=snap.frame, location=loc, status="settled")
    leave_meta: dict[str, Any] = {
        "module_at_leave": int(snap.module),
        "submodule_at_leave": int(snap.submodule),
    }
    if transition_kind:
        leave_meta["transition_kind"] = transition_kind
    return HopFrame(
        frame=snap.frame,
        location=loc,
        status="transition",
        leave_meta=leave_meta,
    )


@dataclass
class RoomTimer:
    """Incremental room/screen hop timer. Feed one snapshot per frame."""

    _engine: HopTimer[LocationKey, LocationVisit, DiscontinuityEvent] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._engine = HopTimer(
            make_visit=_make_visit,
            make_discontinuity=_make_disc,
            jump_reason=DiscontinuityReason.LOCATION_JUMP.value,
            null_location=("in", 0),
            leave_context_keys=_LEAVE_KEYS,
            reset_ever_settled_reasons=frozenset(
                {
                    DiscontinuityReason.BOOT_OR_MENU.value,
                    DiscontinuityReason.FRAME_REGRESSION.value,
                    DiscontinuityReason.RESET.value,
                }
            ),
        )
        self._anchor: TimingSnapshot | None = None
        self.last_kind = ""

    @property
    def visits(self) -> list[LocationVisit]:
        return self._engine.visits

    @property
    def discontinuities(self) -> list[DiscontinuityEvent]:
        return self._engine.discontinuities

    @property
    def open_location(self) -> LocationKey | None:
        open_hop = self._engine._open
        if open_hop is None:
            return None
        return open_hop.location

    def observe(
        self,
        sample: TimingSnapshot | AlttpSnapshot,
        *,
        frame: int | None = None,
    ) -> LocationVisit | None:
        if isinstance(sample, TimingSnapshot):
            snap = sample
        else:
            if frame is None:
                raise ValueError("frame= is required when observing AlttpSnapshot")
            snap = TimingSnapshot.from_snapshot(sample, frame=frame)
        kind = ""
        anchor = self._anchor
        if anchor is not None and _worth_kind(anchor, snap):
            kind = transition_kind(anchor, snap)
            if kind:
                self.last_kind = kind
        visit = self._engine.observe_frame(
            _snap_to_hop(snap, transition_kind="" if is_settled_play(snap) else kind)
        )
        if is_settled_play(snap):
            self._anchor = snap
        if visit is not None and visit.transition_kind:
            self.last_kind = visit.transition_kind
        return visit

    def observe_many(self, samples: Iterable[TimingSnapshot]) -> list[LocationVisit]:
        newly: list[LocationVisit] = []
        for sample in samples:
            visit = self.observe(sample)
            if visit is not None:
                newly.append(visit)
        return newly

    def finalize(self, *, frame: int | None = None) -> None:
        """Drop the open visit. Does not invent an exit hop."""
        self._engine.finalize(frame=frame)

    def report(self, *, source: str = "") -> dict[str, Any]:
        open_hop = self._engine._open
        open_payload = None
        if open_hop is not None:
            open_payload = {
                "location": list(open_hop.location),
                "entry_frame": open_hop.entry_frame,
                "in_transition": open_hop.in_transition,
                "leave_frame": open_hop.leave_frame,
            }
        return self._engine.report_base(
            kind="alttp_room_timing",
            timing_semantics={
                "frame_basis": "stable-retro env.step frames; not IGT",
                "settle_rule": "module in {0x07, 0x09} and submodule == 0",
                "location_key": "('in', room) indoors, ('ow', dark_world, screen) outdoors",
                "location_frames": "exit_frame - entry_frame",
                "dwell_frames": "leave_frame - entry_frame",
                "transition_frames": "exit_frame - leave_frame",
                "transition_kind": "door, stairs, layer_swap, fall, or warp",
            },
            source=source,
            open_visit_payload=open_payload,
            visit_to_dict=lambda visit: visit.to_dict(),
            disc_to_dict=lambda disc: disc.to_dict(),
            totals={
                "total_dwell_frames": sum(visit.dwell_frames for visit in self.visits),
            },
        )
