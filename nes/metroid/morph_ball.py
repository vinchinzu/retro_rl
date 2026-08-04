"""Isolated + natural-entry policy for Maru Mari (Morph Ball).

M3–M5: from Brinstar start (3,14), run left through (2,14) into morph room
(1,14), walk into the pedestal item. Success = WRAM equipment bit 4
($6878 & 0x10).

Probe-verified approach (2026-07-27):
  - LEFT / LEFT+A spans climb the multi-screen west corridor into (1,14)
  - In morph room, settle near pedestal height (y≈130–150) then RIGHT into
    the orb at ~(104, 152)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction

from metroid.ram import (
    EQUIP_MORPH,
    MORPH_MAP_X,
    MORPH_MAP_Y,
    START_MAP_X,
    read_equipment,
    read_snapshot,
)

SEGMENT_MAX_FRAMES = 5000

# Timed action spans for the west corridor climb (from map x>=3).
_PATH_SPANS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("LEFT",), 40),
    (("LEFT", "A"), 35),
    (("LEFT",), 50),
    (("LEFT", "A"), 40),
    (("LEFT",), 60),
    (("LEFT", "A"), 35),
    (("LEFT",), 80),
)

# Pedestal / item geometry in morph room (room coords).
ITEM_X = 104
ITEM_Y = 152
BAND_Y_LO = 120  # higher on screen = smaller y
BAND_Y_HI = 155


class MorphPhase(Enum):
    ALIGN = auto()
    PATH = auto()
    APPROACH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class MorphBallController:
    """Stateful frame policy for the morph-ball segment."""

    phase: MorphPhase = MorphPhase.ALIGN
    frames: int = 0
    phase_frames: int = 0
    span_index: int = 0
    span_progress: int = 0
    notes: list[str] = field(default_factory=list)
    success: bool = False

    def reset(self) -> None:
        self.phase = MorphPhase.ALIGN
        self.frames = 0
        self.phase_frames = 0
        self.span_index = 0
        self.span_progress = 0
        self.notes.clear()
        self.success = False

    def _set_phase(self, phase: MorphPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def step(self, env: Any) -> FrameAction:
        """Return the next action; does not step the emulator."""
        self.frames += 1
        self.phase_frames += 1
        ram = env.get_ram()
        equip = read_equipment(env)
        snap = read_snapshot(ram, env=env)

        if equip & EQUIP_MORPH:
            self.success = True
            self._set_phase(MorphPhase.DONE, "morph_collected")
            return FrameAction(nes_idle_action(), "done")

        if self.frames >= SEGMENT_MAX_FRAMES:
            self._set_phase(MorphPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.game_mode not in (3, 5, 8):
            return FrameAction(nes_idle_action(), "mode_wait")

        if (
            snap.map_x == MORPH_MAP_X
            and snap.map_y == MORPH_MAP_Y
            and self.phase not in (MorphPhase.APPROACH, MorphPhase.DONE)
        ):
            self._set_phase(MorphPhase.APPROACH, "in_morph_room")

        if self.phase is MorphPhase.ALIGN:
            return self._align(snap)
        if self.phase is MorphPhase.PATH:
            return self._path(snap)
        if self.phase is MorphPhase.APPROACH:
            return self._approach(snap)
        return FrameAction(nes_idle_action(), self.phase.name.lower())

    def _align(self, snap) -> FrameAction:
        if snap.map_x <= START_MAP_X and snap.map_y == 14:
            self._set_phase(MorphPhase.PATH, "aligned_start")
            return self._path(snap)
        if snap.map_x > START_MAP_X:
            return FrameAction(nes_action("LEFT"), "align_left")
        if snap.map_y > 14:
            return FrameAction(nes_action("UP"), "align_up")
        if snap.map_y < 14:
            return FrameAction(nes_action("DOWN"), "align_down")
        self._set_phase(MorphPhase.PATH, "align_fallback")
        return FrameAction(nes_action("LEFT"), "align_fallback_left")

    def _path(self, snap) -> FrameAction:
        if snap.map_x == MORPH_MAP_X and snap.map_y == MORPH_MAP_Y:
            self._set_phase(MorphPhase.APPROACH, "reached_morph_room")
            return self._approach(snap)
        if self.span_index >= len(_PATH_SPANS):
            return FrameAction(nes_action("LEFT", "A"), "path_extra_climb")
        buttons, hold = _PATH_SPANS[self.span_index]
        self.span_progress += 1
        if self.span_progress >= hold:
            self.span_index += 1
            self.span_progress = 0
        if buttons:
            return FrameAction(nes_action(*buttons), f"path_{self.span_index}")
        return FrameAction(nes_idle_action(), f"path_{self.span_index}")

    def _approach(self, snap) -> FrameAction:
        """Geometry-aware collect near pedestal (x≈104, y≈152)."""
        x, y = snap.samus_x, snap.samus_y

        # Too high (small y): let gravity drop toward pedestal band.
        if y < BAND_Y_LO:
            # Drift toward item x while falling.
            if x < ITEM_X - 8:
                return FrameAction(nes_action("RIGHT"), "fall_right")
            if x > ITEM_X + 12:
                return FrameAction(nes_action("LEFT"), "fall_left")
            return FrameAction(nes_idle_action(), "fall_idle")

        # Too low (on floor past pedestal): jump back up toward band.
        if y > BAND_Y_HI + 20:
            if x > ITEM_X + 20:
                return FrameAction(nes_action("LEFT", "A"), "reclimb_left")
            if x < ITEM_X - 20:
                return FrameAction(nes_action("RIGHT", "A"), "reclimb_right")
            return FrameAction(nes_action("A"), "reclimb_up")

        # In the vertical band near the orb — horizontal seek + touch.
        if x < ITEM_X - 4:
            return FrameAction(nes_action("RIGHT"), "seek_right")
        if x > ITEM_X + 8:
            return FrameAction(nes_action("LEFT"), "seek_left")
        # Overlapping item x: micro RIGHT/idle (proven collect pulse).
        cycle = self.phase_frames % 7
        if cycle < 4:
            return FrameAction(nes_action("RIGHT"), "touch_right")
        if cycle < 5:
            return FrameAction(nes_action("DOWN"), "touch_down")
        return FrameAction(nes_idle_action(), "touch_idle")

    def report(self) -> dict[str, object]:
        return {
            "phase": self.phase.name,
            "frames": self.frames,
            "success": self.success,
            "span_index": self.span_index,
            "notes": list(self.notes),
        }


def morph_segment_success(env: Any) -> bool:
    return bool(read_equipment(env) & EQUIP_MORPH)
