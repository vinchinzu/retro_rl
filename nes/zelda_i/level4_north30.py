"""Level 4 0x40 → 0x30 north-door controller (verified continuous)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level4_dungeon import LEVEL4, ROOM_L4_NORTH_30, ROOM_L4_ZOLS_40
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot


class North30Phase(Enum):
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4North30Controller:
    """From cleared 0x40: center x≈120, push UP into 0x30 play-ready.

    Live (rr-q8eq dense BFS): free north from north-band y≤68 @ x≈120.
    0x30 has 3× Vire ``0x12`` + 2× invuln residual ``0x2b``.
    """

    max_frames: int = 4000
    phase: North30Phase = North30Phase.ALIGN
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: North30Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(North30Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _entered_30(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_NORTH_30
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is North30Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is North30Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if self._entered_30(snap):
            self.success = True
            self._set_phase(North30Phase.DONE, "entered_0x30")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll_up")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if snap.screen == ROOM_L4_NORTH_30:
            self.success = True
            self._set_phase(North30Phase.DONE, "on_0x30")
            return FrameAction(nes_idle_action(), "done")

        if snap.screen != ROOM_L4_ZOLS_40:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if abs(snap.link_x - 120) > 6:
            self._set_phase(North30Phase.ALIGN, "align_x")
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                "align_x",
            )
        self._set_phase(North30Phase.PUSH, "push_up")
        return FrameAction(nes_action("UP"), "push_up_north")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_north_0x30",
            "target_room": f"0x{ROOM_L4_NORTH_30:02x}",
        }


def make_north_30_controller() -> Level4North30Controller:
    return Level4North30Controller()
