"""Shared hop lifecycle: timeout, death, scroll wait, then policy.

Dungeon dest hops subclass ``HopController`` and implement ``policy``.
Do not copy the frames/success/death preamble into each room file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

WAIT_SCROLL = (2, 3, 4, 6, 7)
WAIT_SCROLL_B = (2, 3, 4, 6, 7, 10, 16)
DEATH_MODE = 17
CELLAR_MODE = 9


def axis_dir(
    xy: tuple[int, int], dest: tuple[int, int], *, y_first: bool, tol: int = 4
) -> str | None:
    """One cardinal toward dest. None when both axes are inside ``tol``."""
    x, y = xy
    tx, ty = dest
    dy, dx = ty - y, tx - x
    axes = (("DOWN", "UP", dy), ("RIGHT", "LEFT", dx))
    if not y_first:
        axes = tuple(reversed(axes))
    for pos, neg, delta in axes:
        if abs(delta) > tol:
            return pos if delta > 0 else neg
    return None


def dungeon_align_then_push(
    snap: ZeldaSnapshot,
    *,
    push_dir: str,
    target_x: int | None = None,
    target_y: int | None = None,
    x_tol: int = 2,
    y_tol: int = 2,
    door_plane: int | None = None,
    reason: str = "door",
) -> FrameAction:
    """Align to a door band, then hold the cardinal. No sword."""
    if target_y is not None and abs(snap.link_y - target_y) > y_tol:
        btn = "UP" if snap.link_y > target_y else "DOWN"
        return FrameAction(nes_action(btn), f"{reason}_align_y")
    if door_plane is not None and push_dir in ("LEFT", "RIGHT"):
        if push_dir == "LEFT" and snap.link_x > door_plane:
            return FrameAction(nes_action("LEFT"), f"{reason}_approach")
        if push_dir == "RIGHT" and snap.link_x < door_plane:
            return FrameAction(nes_action("RIGHT"), f"{reason}_approach")
        return FrameAction(nes_action(push_dir), f"{reason}_push")
    if target_x is not None and abs(snap.link_x - target_x) > x_tol:
        btn = "LEFT" if snap.link_x > target_x else "RIGHT"
        return FrameAction(nes_action(btn), f"{reason}_align_x")
    return FrameAction(nes_action(push_dir), f"{reason}_push")


@dataclass(frozen=True)
class CellarCross:
    """Two-ladder cellar: drop to floor, cross, climb."""

    west_x: int
    east_x: int
    floor_y: int
    mouth_y: int
    tol: int = 4


def cellar_cross_dir(xy: tuple[int, int], spec: CellarCross, *, on_floor: bool) -> str:
    """DOWN to floor, then to east ladder, then UP. Caller tracks on_floor."""
    x, y = xy
    if not on_floor and y < spec.floor_y - spec.tol:
        return "DOWN"
    if abs(x - spec.east_x) > spec.tol:
        return "LEFT" if x > spec.east_x else "RIGHT"
    return "UP"


@dataclass(kw_only=True)
class HopController:
    """Timeout / death / wait-scroll guard. Subclass ``policy`` and ``arrived``."""

    max_frames: int = 4000
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    wait_modes: tuple[int, ...] = WAIT_SCROLL
    spec_id: str = ""
    require_level: int | None = None
    done_reason: str = "done"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return False

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_idle_action(), "idle")

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_{snap.screen:02x}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_idle_action(), "wait_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        del snap, force
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
        }

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def mark_fail(self, note: str, reason: str | None = None) -> FrameAction:
        self.failed = True
        self._note(note)
        return FrameAction(nes_idle_action(), reason or note)

    def mark_done(self, snap: ZeldaSnapshot, note: str | None = None) -> FrameAction:
        self.success = True
        self._note(note or self.on_arrive(snap))
        return FrameAction(nes_idle_action(), self.done_reason)

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            self._note(self.timeout_note(snap))
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == DEATH_MODE:
            return self.mark_fail("link_death")
        if snap.transitioning or snap.mode in self.wait_modes:
            return self.scroll_action(snap)
        if self.require_level is not None and snap.level != self.require_level:
            if snap.mode == PLAY_MODE and not snap.transitioning:
                return self.mark_fail(f"left_level_{snap.level}")
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        blocked = self.guard(snap)
        if blocked is not None:
            return self.emit(snap, blocked, force=self.success or self.failed)
        if self.arrived(snap):
            return self.emit(snap, self.mark_done(snap), force=True)
        return self.emit(snap, self.policy(snap))
