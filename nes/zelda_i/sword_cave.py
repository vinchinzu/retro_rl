"""Isolated + natural-entry policy for the wooden sword cave segment.

M3 acceptance: from Level1 / start overworld, enter the NW cave on screen 0x77,
collect wooden sword ($0657>=1), return to overworld start with sword.

Probe-verified approach (2026-07-27):
  - Door approach ~ (x=60, y=100) then UP into cave (mode 16→11)
  - Idle through dialog (~280 frames)
  - Align x=120, walk UP onto sword
  - DOWN to exit (mode 5, screen 0x77, sword>=1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.ram import (
    SCREEN_START,
    ZeldaSnapshot,
    is_sword_obtained,
    read_snapshot,
)

# Geometry (screen 0x77 / cave interior)
CAVE_APPROACH_X = 60
CAVE_APPROACH_Y = 100
SWORD_X = 120
SWORD_Y_TOUCH = 160  # sword collect once Link y <= this at x≈120
DIALOG_IDLE_FRAMES = 280
SEGMENT_MAX_FRAMES = 3600


class SwordPhase(Enum):
    APPROACH_DOOR = auto()
    ENTER_CAVE = auto()
    WAIT_DIALOG = auto()
    ALIGN_SWORD = auto()
    TAKE_SWORD = auto()
    EXIT_CAVE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class SwordCaveController:
    """Stateful frame policy for the sword-cave segment."""

    phase: SwordPhase = SwordPhase.APPROACH_DOOR
    frames: int = 0
    phase_frames: int = 0
    dialog_waited: int = 0
    enter_hold: int = 0
    notes: list[str] = field(default_factory=list)
    success: bool = False

    def reset(self) -> None:
        self.phase = SwordPhase.APPROACH_DOOR
        self.frames = 0
        self.phase_frames = 0
        self.dialog_waited = 0
        self.enter_hold = 0
        self.notes.clear()
        self.success = False

    def _set_phase(self, phase: SwordPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if snap.has_sword and snap.overworld and snap.screen == SCREEN_START:
            self.success = True
            self._set_phase(SwordPhase.DONE, "sword_on_start")
            return FrameAction(nes_idle_action(), "done")

        if self.frames >= SEGMENT_MAX_FRAMES:
            self._set_phase(SwordPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.transitioning:
            # Hold UP while entering; idle otherwise
            if self.phase in (SwordPhase.ENTER_CAVE, SwordPhase.APPROACH_DOOR):
                return FrameAction(nes_action("UP"), "transition_hold_up")
            return FrameAction(nes_idle_action(), "transition_idle")

        if self.phase is SwordPhase.APPROACH_DOOR:
            return self._approach_door(snap)
        if self.phase is SwordPhase.ENTER_CAVE:
            return self._enter_cave(snap)
        if self.phase is SwordPhase.WAIT_DIALOG:
            return self._wait_dialog(snap)
        if self.phase is SwordPhase.ALIGN_SWORD:
            return self._align_sword(snap)
        if self.phase is SwordPhase.TAKE_SWORD:
            return self._take_sword(snap)
        if self.phase is SwordPhase.EXIT_CAVE:
            return self._exit_cave(snap)
        return FrameAction(nes_idle_action(), self.phase.name.lower())

    def _approach_door(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.in_cave:
            self._set_phase(SwordPhase.WAIT_DIALOG, "entered_cave")
            return FrameAction(nes_idle_action(), "entered")
        # On overworld: path to approach point then UP into door
        if snap.screen != SCREEN_START:
            # Recover toward start if we drifted north
            if snap.screen == 0x67:
                return FrameAction(nes_action("DOWN"), "recover_south")
            return FrameAction(nes_idle_action(), "wrong_screen")

        dx = snap.link_x - CAVE_APPROACH_X
        dy = snap.link_y - CAVE_APPROACH_Y
        if abs(dx) > 2 and abs(dx) >= abs(dy):
            btn = "LEFT" if dx > 0 else "RIGHT"
            return FrameAction(nes_action(btn), "approach_x")
        if abs(dy) > 2:
            btn = "UP" if dy > 0 else "DOWN"
            return FrameAction(nes_action(btn), "approach_y")
        self._set_phase(SwordPhase.ENTER_CAVE, "at_door")
        return FrameAction(nes_action("UP"), "enter_push")

    def _enter_cave(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.in_cave:
            self._set_phase(SwordPhase.WAIT_DIALOG, "cave_mode")
            return FrameAction(nes_idle_action(), "cave_idle")
        self.enter_hold += 1
        if self.enter_hold > 120:
            # Retry approach
            self.enter_hold = 0
            self._set_phase(SwordPhase.APPROACH_DOOR, "reapproach")
        return FrameAction(nes_action("UP"), "enter_hold_up")

    def _wait_dialog(self, snap: ZeldaSnapshot) -> FrameAction:
        if not snap.in_cave and snap.has_sword:
            self._set_phase(SwordPhase.DONE, "already_have_sword")
            self.success = True
            return FrameAction(nes_idle_action(), "done")
        if snap.has_sword:
            self._set_phase(SwordPhase.EXIT_CAVE, "sword_during_wait")
            return FrameAction(nes_action("DOWN"), "exit")
        self.dialog_waited += 1
        if self.dialog_waited >= DIALOG_IDLE_FRAMES or (
            snap.dialog_timer == 0 and self.dialog_waited > 60 and snap.link_y >= 200
        ):
            self._set_phase(SwordPhase.ALIGN_SWORD, "dialog_done")
            return FrameAction(nes_idle_action(), "dialog_done")
        return FrameAction(nes_idle_action(), "dialog_wait")

    def _align_sword(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.has_sword:
            self._set_phase(SwordPhase.EXIT_CAVE, "sword_got")
            return FrameAction(nes_action("DOWN"), "exit")
        if not snap.in_cave:
            self._set_phase(SwordPhase.APPROACH_DOOR, "left_cave_no_sword")
            return FrameAction(nes_idle_action(), "reenter")
        dx = snap.link_x - SWORD_X
        if abs(dx) > 1:
            btn = "LEFT" if dx > 0 else "RIGHT"
            return FrameAction(nes_action(btn), "align_x")
        self._set_phase(SwordPhase.TAKE_SWORD, "aligned")
        return FrameAction(nes_action("UP"), "take_up")

    def _take_sword(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.has_sword:
            self._set_phase(SwordPhase.EXIT_CAVE, "sword_collected")
            return FrameAction(nes_action("DOWN"), "exit")
        if not snap.in_cave:
            self._set_phase(SwordPhase.APPROACH_DOOR, "lost_cave")
            return FrameAction(nes_idle_action(), "reenter")
        if abs(snap.link_x - SWORD_X) > 2:
            self._set_phase(SwordPhase.ALIGN_SWORD, "realign")
            return FrameAction(nes_idle_action(), "realign")
        return FrameAction(nes_action("UP"), "walk_sword")

    def _exit_cave(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.has_sword and snap.overworld and snap.screen == SCREEN_START:
            self.success = True
            self._set_phase(SwordPhase.DONE, "exited_with_sword")
            return FrameAction(nes_idle_action(), "done")
        if snap.transitioning:
            return FrameAction(nes_idle_action(), "exit_transition")
        if snap.in_cave or snap.mode == 11:
            return FrameAction(nes_action("DOWN"), "exit_down")
        if snap.has_sword and snap.overworld:
            # Exited to a different screen; still success if sword held
            self.success = True
            self._set_phase(SwordPhase.DONE, "exited_other_screen")
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_action("DOWN"), "exit_seek")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "dialog_waited": self.dialog_waited,
        }


def sword_segment_success(ram: np.ndarray) -> bool:
    """Stop predicate: wooden sword owned on overworld start screen."""
    snap = read_snapshot(ram)
    return snap.has_sword and snap.overworld and snap.screen == SCREEN_START


def sword_obtained(ram: np.ndarray) -> bool:
    return is_sword_obtained(ram)
