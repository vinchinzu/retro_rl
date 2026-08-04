"""Overworld navigation: post-sword start → Level 1 entrance (screen 0x37).

Probe-verified path (2026-07-28)::

    0x77 east@y≈140 → 0x78 north@x≈48 → 0x68 north@x≈48 → 0x58
    → north@x≈112 → 0x48 north (center) → 0x38 west (center) → 0x37
    → approach tree door ~(120,100) and push UP into Level 1 (level==1).

Natural entry begins after the wooden sword is obtained on the start screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.nav_common import (
    swing_action,
    track_stuck,
    wake_or_wait_mode,
)
from zelda_i.ram import (
    SCREEN_LEVEL1_ENTRANCE,
    SCREEN_START,
    ZeldaSnapshot,
    read_snapshot,
)

# --- Geometry (probe-stable) ---
START_EAST_Y = 140
COL_NORTH_X = 48  # north lanes on 0x78 / 0x68
BUSH_NORTH_X = 112  # north lane on 0x58 through bush grid
LEVEL1_DOOR_X = 112  # probe: enter while walking UP through x=112, y≈125–133
LEVEL1_DOOR_Y = 140  # approach from open sand south of the tree mouth
SEGMENT_MAX_FRAMES = 12000
SWORD_SWING_PERIOD = 14
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 45


class NavPhase(Enum):
    EAST_77 = auto()
    NORTH_78 = auto()
    NORTH_68 = auto()
    ALIGN_58 = auto()
    NORTH_58 = auto()
    CENTER_48 = auto()
    NORTH_48 = auto()
    CENTER_38 = auto()
    WEST_38 = auto()
    APPROACH_DOOR = auto()
    ENTER_DOOR = auto()
    DONE = auto()
    FAILED = auto()


# Waypoints on 0x58 (bush grid) toward the north lane at x=112.
# Horizontal travel needs y≈150–160; north exit opens once x≈112.
_ALIGN_58_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (48, 160),
    (80, 157),
    (112, 157),
)

_SCROLL_HOLD: dict[NavPhase, str] = {
    NavPhase.EAST_77: "RIGHT",
    NavPhase.NORTH_78: "UP",
    NavPhase.NORTH_68: "UP",
    NavPhase.NORTH_58: "UP",
    NavPhase.NORTH_48: "UP",
    NavPhase.WEST_38: "LEFT",
    NavPhase.ENTER_DOOR: "UP",
}


@dataclass
class OverworldToLevel1Controller:
    """Frame policy from post-sword start overworld to Level 1 entrance."""

    phase: NavPhase = NavPhase.EAST_77
    frames: int = 0
    phase_frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    waypoint_index: int = 0
    notes: list[str] = field(default_factory=list)
    success: bool = False
    require_dungeon: bool = True
    """If True, success requires level==1; else arriving on screen 0x37 is enough."""

    def reset(self) -> None:
        self.phase = NavPhase.EAST_77
        self.frames = 0
        self.phase_frames = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.waypoint_index = 0
        self.notes.clear()
        self.success = False

    def _set_phase(self, phase: NavPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.stuck = 0
            if note:
                self.notes.append(note)

    def _swing(self, direction: str, reason: str) -> FrameAction:
        return swing_action(
            self.phase_frames,
            direction,
            reason,
            period=SWORD_SWING_PERIOD,
            hold=SWORD_SWING_FRAMES,
        )

    def _align_and_push(
        self,
        snap: ZeldaSnapshot,
        *,
        direction: str,
        align_x: int | None = None,
        align_y: int | None = None,
        reason: str = "nav",
    ) -> FrameAction:
        # Level-1 path uses a reverse/side unstick (not the generic card-cycle).
        if self.stuck > STUCK_THRESHOLD:
            rev = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[
                direction
            ]
            side = "LEFT" if direction in ("UP", "DOWN") else "UP"
            wiggle = [rev, side, direction][self.stuck % 3]
            self.stuck = 0
            return FrameAction(nes_action(wiggle, "A"), f"{reason}_unstick")

        if (
            align_x is not None
            and abs(snap.link_x - align_x) > 3
            and 90 < snap.link_y < 200
        ):
            btn = "LEFT" if snap.link_x > align_x else "RIGHT"
            return self._swing(btn, f"{reason}_ax")
        if (
            align_y is not None
            and abs(snap.link_y - align_y) > 3
            and 30 < snap.link_x < 220
        ):
            btn = "UP" if snap.link_y > align_y else "DOWN"
            return self._swing(btn, f"{reason}_ay")
        return self._swing(direction, reason)

    def _goto_xy(
        self,
        snap: ZeldaSnapshot,
        tx: int,
        ty: int,
        reason: str = "goto",
    ) -> FrameAction:
        dx = snap.link_x - tx
        dy = snap.link_y - ty
        if abs(dx) <= 3 and abs(dy) <= 3:
            return FrameAction(nes_idle_action(), f"{reason}_at")
        if self.stuck > STUCK_THRESHOLD:
            btn = ["LEFT", "RIGHT", "UP", "DOWN"][self.stuck % 4]
            self.stuck = 0
            return FrameAction(nes_action(btn, "A"), f"{reason}_unstick")
        if abs(dx) >= abs(dy) and abs(dx) > 3:
            btn = "LEFT" if dx > 0 else "RIGHT"
        else:
            btn = "UP" if dy > 0 else "DOWN"
        return self._swing(btn, reason)

    def _follow_waypoints(
        self,
        snap: ZeldaSnapshot,
        waypoints: tuple[tuple[int, int], ...],
        reason: str,
    ) -> FrameAction:
        if self.waypoint_index >= len(waypoints):
            return FrameAction(nes_idle_action(), f"{reason}_done")
        tx, ty = waypoints[self.waypoint_index]
        dx = snap.link_x - tx
        dy = snap.link_y - ty
        if abs(dx) <= 6 and abs(dy) <= 6:
            self.waypoint_index += 1
            self.stuck = 0
            if self.waypoint_index >= len(waypoints):
                return FrameAction(nes_idle_action(), f"{reason}_done")
            tx, ty = waypoints[self.waypoint_index]
            dx = snap.link_x - tx
            dy = snap.link_y - ty
        if self.stuck > STUCK_THRESHOLD:
            opts = ("UP", "DOWN", "LEFT", "RIGHT")
            btn = opts[self.phase_frames % 4]
            self.stuck = 0
            return FrameAction(nes_action(btn, "A"), f"{reason}_unstick")
        # On entry column, finish vertical first so rightward corridors open
        if snap.link_x < 70 and abs(dy) > 4:
            btn = "UP" if dy > 0 else "DOWN"
            return self._swing(btn, f"{reason}_yfirst")
        # Prefer horizontal once y is near target (bush corridors are mid-height)
        if abs(dy) <= 10 and abs(dx) > 3:
            btn = "LEFT" if dx > 0 else "RIGHT"
            return self._swing(btn, f"{reason}_x")
        if abs(dx) >= abs(dy) and abs(dx) > 3:
            btn = "LEFT" if dx > 0 else "RIGHT"
        elif abs(dy) > 3:
            btn = "UP" if dy > 0 else "DOWN"
        else:
            btn = "LEFT" if dx > 0 else "RIGHT"
        return self._swing(btn, reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )

        if self.frames >= SEGMENT_MAX_FRAMES:
            self._set_phase(NavPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.level == 1:
            self.success = True
            self._set_phase(NavPhase.DONE, "in_level1")
            return FrameAction(nes_idle_action(), "done")

        if (
            not self.require_dungeon
            and snap.has_sword
            and snap.overworld
            and snap.screen == SCREEN_LEVEL1_ENTRANCE
        ):
            self.success = True
            self._set_phase(NavPhase.DONE, "on_level1_screen")
            return FrameAction(nes_idle_action(), "done")

        if snap.transitioning:
            hold = _SCROLL_HOLD.get(self.phase)
            if hold:
                return FrameAction(nes_action(hold), "scroll_hold")
            return FrameAction(nes_idle_action(), "scroll_idle")

        # Mode 8 = brief hit freeze; keep holding travel dir. Other modes wait.
        if snap.mode not in (5, 8, 11) and not snap.transitioning:
            return wake_or_wait_mode(self.phase_frames, snap.mode)

        self._advance_phase_for_screen(snap)

        if self.phase is NavPhase.EAST_77:
            return self._align_and_push(
                snap, direction="RIGHT", align_y=START_EAST_Y, reason="e77"
            )
        if self.phase is NavPhase.NORTH_78:
            return self._align_and_push(
                snap, direction="UP", align_x=COL_NORTH_X, reason="n78"
            )
        if self.phase is NavPhase.NORTH_68:
            return self._align_and_push(
                snap, direction="UP", align_x=COL_NORTH_X, reason="n68"
            )
        if self.phase is NavPhase.ALIGN_58:
            return self._follow_waypoints(snap, _ALIGN_58_WAYPOINTS, "a58")
        if self.phase is NavPhase.NORTH_58:
            return self._align_and_push(
                snap, direction="UP", align_x=BUSH_NORTH_X, reason="n58"
            )
        if self.phase is NavPhase.CENTER_48:
            if snap.link_y > 190:
                return self._swing("UP", "c48_off_edge")
            return self._goto_xy(snap, 120, 140, "c48")
        if self.phase is NavPhase.NORTH_48:
            if abs(snap.link_x - 120) > 12 and snap.link_y > 100:
                btn = "LEFT" if snap.link_x > 120 else "RIGHT"
                return self._swing(btn, "n48_ax")
            return self._align_and_push(snap, direction="UP", reason="n48")
        if self.phase is NavPhase.CENTER_38:
            if snap.link_y > 180:
                return self._swing("UP", "c38_off_edge")
            return self._goto_xy(snap, 120, 140, "c38")
        if self.phase is NavPhase.WEST_38:
            if snap.link_y > 170:
                return self._swing("UP", "w38_up")
            if snap.link_y < 110:
                return self._swing("DOWN", "w38_down")
            return self._align_and_push(
                snap, direction="LEFT", align_y=140, reason="w38"
            )
        if self.phase is NavPhase.APPROACH_DOOR:
            action = self._goto_xy(snap, LEVEL1_DOOR_X, LEVEL1_DOOR_Y, "door")
            if (
                abs(snap.link_x - LEVEL1_DOOR_X) <= 4
                and abs(snap.link_y - LEVEL1_DOOR_Y) <= 8
            ):
                self._set_phase(NavPhase.ENTER_DOOR, "at_door")
                return FrameAction(nes_action("UP"), "enter_push")
            return action
        if self.phase is NavPhase.ENTER_DOOR:
            if abs(snap.link_x - LEVEL1_DOOR_X) > 5:
                btn = "LEFT" if snap.link_x > LEVEL1_DOOR_X else "RIGHT"
                return self._swing(btn, "enter_ax")
            if self.phase_frames > 200:
                self._set_phase(NavPhase.APPROACH_DOOR, "retry_door")
            return FrameAction(nes_action("UP"), "enter")
        if self.phase is NavPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "failed")

    def _advance_phase_for_screen(self, snap: ZeldaSnapshot) -> None:
        sc = snap.screen
        if self.phase is NavPhase.EAST_77 and sc == 0x78:
            self._set_phase(NavPhase.NORTH_78, "on_78")
        elif self.phase is NavPhase.NORTH_78 and sc == 0x68:
            self._set_phase(NavPhase.NORTH_68, "on_68")
        elif self.phase is NavPhase.NORTH_68 and sc == 0x58:
            self.waypoint_index = 0
            self._set_phase(NavPhase.ALIGN_58, "on_58")
        elif self.phase is NavPhase.ALIGN_58:
            if abs(snap.link_x - BUSH_NORTH_X) <= 8 and snap.link_y <= 170:
                self._set_phase(NavPhase.NORTH_58, "aligned_58")
            elif self.waypoint_index >= len(_ALIGN_58_WAYPOINTS):
                self._set_phase(NavPhase.NORTH_58, "aligned_58")
            elif sc == 0x48:
                self._set_phase(NavPhase.CENTER_48, "skip_to_48")
        elif self.phase is NavPhase.NORTH_58 and sc == 0x48:
            self._set_phase(NavPhase.CENTER_48, "on_48")
        elif self.phase is NavPhase.CENTER_48:
            if abs(snap.link_x - 120) <= 16 and snap.link_y <= 170:
                self._set_phase(NavPhase.NORTH_48, "centered_48")
            elif self.phase_frames > 120:
                self._set_phase(NavPhase.NORTH_48, "force_n48")
            elif sc == 0x38:
                self._set_phase(NavPhase.CENTER_38, "skip_to_38")
        elif self.phase is NavPhase.NORTH_48 and sc == 0x38:
            self._set_phase(NavPhase.CENTER_38, "on_38")
        elif self.phase is NavPhase.CENTER_38:
            if abs(snap.link_x - 120) <= 20 and 110 <= snap.link_y <= 165:
                self._set_phase(NavPhase.WEST_38, "centered_38")
            elif self.phase_frames > 200 and snap.link_y <= 170:
                self._set_phase(NavPhase.WEST_38, "force_w38")
            elif sc == 0x37:
                self._set_phase(NavPhase.APPROACH_DOOR, "skip_to_37")
        elif self.phase is NavPhase.WEST_38 and sc == 0x37:
            self._set_phase(NavPhase.APPROACH_DOOR, "on_37")

        if sc == SCREEN_START and self.phase not in (
            NavPhase.EAST_77,
            NavPhase.DONE,
            NavPhase.FAILED,
        ):
            self._set_phase(NavPhase.EAST_77, "recover_start")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "stuck": self.stuck,
        }


def level1_entrance_success(ram: np.ndarray, *, require_dungeon: bool = True) -> bool:
    """Stop predicate: inside Level 1, or on 0x37 overworld if not requiring dungeon."""
    snap = read_snapshot(ram)
    if require_dungeon:
        return snap.level == 1
    return (
        snap.has_sword
        and snap.overworld
        and snap.screen == SCREEN_LEVEL1_ENTRANCE
    )


def level1_screen_reached(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return snap.has_sword and snap.screen == SCREEN_LEVEL1_ENTRANCE and snap.level == 0
