"""Shared segment result + room-visit tracking for SMZ3 routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smz3.ram import ComboSnapshot
from smz3.world import ActiveWorld


@dataclass
class RoomVisit:
    room_id: int
    enter_frame: int
    leave_frame: int | None = None
    world: str = "super_metroid"

    @property
    def dwell_frames(self) -> int | None:
        if self.leave_frame is None:
            return None
        return self.leave_frame - self.enter_frame

    def to_dict(self) -> dict[str, Any]:
        from smz3.portals import room_name

        return {
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "room_name": room_name(self.room_id),
            "enter_frame": self.enter_frame,
            "leave_frame": self.leave_frame,
            "dwell_frames": self.dwell_frames,
            "world": self.world,
        }


def track_room(
    visits: list[RoomVisit],
    snap: ComboSnapshot,
    world: ActiveWorld,
) -> None:
    """Append / close SM room visits from a combo snapshot."""
    rid = snap.sm_room_id
    if rid == 0:
        return
    if not visits or visits[-1].room_id != rid:
        if visits and visits[-1].leave_frame is None:
            visits[-1].leave_frame = snap.frame
        visits.append(
            RoomVisit(
                room_id=rid,
                enter_frame=snap.frame,
                world=world.value,
            )
        )


def close_last_visit(visits: list[RoomVisit], frame: int) -> None:
    if visits and visits[-1].leave_frame is None:
        visits[-1].leave_frame = frame


@dataclass
class SegmentResult:
    """Base outcome for any SMZ3 scripted segment."""

    ok: bool
    goal: str = ""
    frames: int = 0
    detail: str = ""
    final_snapshot: ComboSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        snap = self.final_snapshot
        return {
            "ok": self.ok,
            "goal": self.goal,
            "frames": self.frames,
            "detail": self.detail,
            "final_snapshot": snap.to_dict() if snap is not None else None,
        }


@dataclass
class SmSegmentResult(SegmentResult):
    """SM-side segment with boot frames, visits, timeout, world."""

    boot_frames: int = 0
    visits: list[RoomVisit] = field(default_factory=list)
    world: ActiveWorld = ActiveWorld.UNKNOWN
    timeout: Any | None = None  # TimeoutEvent | None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "boot_frames": self.boot_frames,
                "world": self.world.value,
                "visits": [v.to_dict() for v in self.visits],
                "timeout": self.timeout.to_dict() if self.timeout else None,
                "reached_parlor": any(v.room_id == 0x92FD for v in self.visits),
                "room_names": [v.to_dict()["room_name"] for v in self.visits],
            }
        )
        return d
