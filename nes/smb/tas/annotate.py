"""Dash-level exit marks for a 32-exit (warpless) SMB TAS replay.

``(world, dash_level)`` is the 32-exit clock — AreaNumber underground flips
are ignored. Used by ``smb.scripts.annotate_fm2``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def stage_label(world: int, dash: int) -> str:
    """1-indexed ``W-L`` from 0-indexed world + LevelNumber."""
    return f"{int(world) + 1}-{int(dash) + 1}"


def dash_key(snap: Any) -> tuple[int, int]:
    """``(world, dash_level)`` — never AreaNumber."""
    dash = int(getattr(snap, "dash_level", getattr(snap, "level", 0)))
    return int(snap.world), dash


def is_live_control(snap: Any, *, x_max: int = 80) -> bool:
    """Controllable spawn: playing, on-foot, low x, timer running."""
    timer = int(getattr(snap, "timer", 0) or 0)
    return (
        int(snap.oper_mode) == 1
        and int(snap.player_state) in (0, 7, 8)
        and timer > 0
        and 0 < int(snap.player_x) <= x_max
        and not bool(getattr(snap, "dying", False))
    )


@dataclass
class StageMark:
    """One 32-exit stage as seen in a TAS replay."""

    id: str
    world: int
    dash: int
    first_control: int | None = None
    leave_frame: int | None = None
    leave_to: str | None = None
    max_x: int = 0
    control_x: int | None = None
    control_y: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotateState:
    """Incremental scanner over snapshots."""

    marks: dict[str, StageMark] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    current: str | None = None
    death_frame: int | None = None
    ending_frame: int | None = None
    start_lives: int | None = None

    def observe(self, snap: Any, frame: int) -> None:
        if self.start_lives is None:
            lives = int(getattr(snap, "lives", -1))
            if int(snap.oper_mode) == 1 and 0 <= lives <= 8:
                self.start_lives = lives
        if self.start_lives is None:
            return
        if int(getattr(snap, "lives", self.start_lives)) < self.start_lives:
            if self.death_frame is None:
                self.death_frame = frame
            return
        if bool(getattr(snap, "dying", False)) and self.death_frame is None:
            self.death_frame = frame
            return

        key = dash_key(snap)
        if key[0] < 0 or key[0] > 7 or key[1] < 0 or key[1] > 3:
            return
        sid = stage_label(*key)
        mark = self.marks.get(sid)
        if mark is None:
            if self.current is not None and self.current in self.marks:
                prev = self.marks[self.current]
                if prev.leave_frame is None:
                    prev.leave_frame = frame
                    prev.leave_to = sid
            mark = StageMark(id=sid, world=key[0], dash=key[1])
            self.marks[sid] = mark
            self.order.append(sid)
            self.current = sid

        px = int(snap.player_x)
        if 0 < px < 20000:
            mark.max_x = max(mark.max_x, px)
        if mark.first_control is None and is_live_control(snap):
            mark.first_control = frame
            mark.control_x = px
            mark.control_y = int(getattr(snap, "player_y", 0) or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [self.marks[s].to_dict() for s in self.order if s in self.marks],
            "order": list(self.order),
            "n_stages": len(self.order),
            "death_frame": self.death_frame,
            "ending_frame": self.ending_frame,
            "start_lives": self.start_lives,
        }
