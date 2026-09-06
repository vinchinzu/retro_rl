"""Pause-select an already-owned B-item. Never poke ``$0656``.

START / idle 20 / RIGHT / idle 8 / START close / idle 24, matching the
live L5 ``select_b_item_menu`` timings — but RIGHT only counts as a
cursor move when ``ADDR_SELECTED_ITEM`` actually changes, and the
controller never emits B (the parent blows/places after ``success``).

Snapshot has no ``selected_item``; read ``ADDR_SELECTED_ITEM`` after
``bind_env``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.ram import ADDR_SELECTED_ITEM, ZeldaSnapshot, read_u8

B_SLOT_BOMBS = 1
B_SLOT_ARROWS = 2
B_SLOT_CANDLE = 4
B_SLOT_RECORDER = 5
B_SLOT_BAIT = 6

SLOT_NAMES = {
    B_SLOT_BOMBS: "bombs",
    B_SLOT_ARROWS: "arrows",
    B_SLOT_CANDLE: "candle",
    B_SLOT_RECORDER: "recorder",
    B_SLOT_BAIT: "bait",
}

DEATH_MODE = 17
SELECT_MAX_FRAMES = 240
OPEN_SETTLE_FRAMES = 20
CURSOR_SETTLE_FRAMES = 8
CLOSE_SETTLE_FRAMES = 64  # NES Zelda pause scroll-up takes 59 frames to accept input
MAX_CURSOR_MOVES = 8


class PauseSelectPhase(Enum):
    CHECK = auto()
    OPEN_SETTLE = auto()
    CYCLE = auto()
    CURSOR_SETTLE = auto()
    CLOSE = auto()
    CLOSE_SETTLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PauseSelectController:
    """Cycle the pause menu until ``want`` is on B, then close back to play."""

    want: int
    name: str = ""
    max_frames: int = SELECT_MAX_FRAMES
    open_settle: int = OPEN_SETTLE_FRAMES
    cursor_settle: int = CURSOR_SETTLE_FRAMES
    close_settle: int = CLOSE_SETTLE_FRAMES
    max_cursor_moves: int = MAX_CURSOR_MOVES
    phase: PauseSelectPhase = PauseSelectPhase.CHECK
    frames: int = 0
    phase_frames: int = 0
    cursor_moves: int = 0
    success: bool = False
    failed: bool = False
    skipped: bool = False
    fail_reason: str = ""
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)
    _last_selected: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = SLOT_NAMES.get(int(self.want), f"slot_{self.want}")

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _set_phase(self, phase: PauseSelectPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self._note(note)

    def _selected(self) -> int | None:
        if self._env is None:
            return None
        return int(read_u8(self._env.get_ram(), ADDR_SELECTED_ITEM))

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self.fail_reason = reason
        self._set_phase(PauseSelectPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _finish(self, reason: str) -> FrameAction:
        self.success = True
        self._set_phase(PauseSelectPhase.DONE, reason)
        return FrameAction(nes_idle_action(), reason)

    def drive(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """``None`` means the slot is selected and the parent may act now."""
        if self.success:
            return None
        action = self.step(snap)
        if self.failed:
            return action
        if self.success:
            return None
        return action

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if self._env is None:
            return self._fail("pause_select_env_not_bound")
        selected = self._selected()
        if selected is None:
            return self._fail("pause_select_env_not_bound")
        if snap.mode == DEATH_MODE:
            return self._fail("death")
        if self.frames > self.max_frames:
            return self._fail(f"select_{self.name}_timeout")

        if self.phase is PauseSelectPhase.CHECK:
            if selected == self.want:
                self.skipped = True
                return self._finish(f"{self.name}_already_selected")
            self._set_phase(PauseSelectPhase.OPEN_SETTLE, "pause_open")
            return FrameAction(nes_action("START"), "pause_open")

        if self.phase is PauseSelectPhase.OPEN_SETTLE:
            if self.phase_frames >= self.open_settle:
                self._set_phase(PauseSelectPhase.CYCLE)
            return FrameAction(nes_idle_action(), "pause_settle")

        if self.phase is PauseSelectPhase.CYCLE:
            if selected == self.want:
                self._set_phase(
                    PauseSelectPhase.CLOSE, f"{self.name}_cursor_selected"
                )
                return FrameAction(nes_idle_action(), "cursor_ready")
            if self.cursor_moves >= self.max_cursor_moves:
                return self._fail(f"{self.name}_cursor_not_found")
            self._last_selected = selected
            self._set_phase(PauseSelectPhase.CURSOR_SETTLE)
            return FrameAction(nes_action("RIGHT"), "pause_next_item")

        if self.phase is PauseSelectPhase.CURSOR_SETTLE:
            if (
                self._last_selected is not None
                and selected != self._last_selected
            ):
                self.cursor_moves += 1
                self._last_selected = selected
            if self.phase_frames >= self.cursor_settle:
                self._set_phase(PauseSelectPhase.CYCLE)
            return FrameAction(nes_idle_action(), "pause_cursor_settle")

        if self.phase is PauseSelectPhase.CLOSE:
            self._set_phase(PauseSelectPhase.CLOSE_SETTLE, "pause_close")
            return FrameAction(nes_action("START"), "pause_close")

        if self.phase is PauseSelectPhase.CLOSE_SETTLE:
            if self.phase_frames < self.close_settle:
                return FrameAction(nes_idle_action(), "pause_resume")
            if selected == self.want:
                return self._finish(f"{self.name}_selected_naturally")
            return self._fail("pause_close_contract_mismatch")

        return FrameAction(nes_idle_action(), "select")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "want": self.want,
            "name": self.name,
            "frames": self.frames,
            "phase": self.phase.name,
            "cursor_moves": self.cursor_moves,
            "fail_reason": self.fail_reason,
            "writes": 0,
            "normal_pause_input": True,
            "notes": list(self.notes),
        }


__all__ = [
    "B_SLOT_ARROWS",
    "B_SLOT_BAIT",
    "B_SLOT_BOMBS",
    "B_SLOT_CANDLE",
    "B_SLOT_RECORDER",
    "CLOSE_SETTLE_FRAMES",
    "CURSOR_SETTLE_FRAMES",
    "MAX_CURSOR_MOVES",
    "OPEN_SETTLE_FRAMES",
    "PauseSelectController",
    "PauseSelectPhase",
    "SELECT_MAX_FRAMES",
    "SLOT_NAMES",
]
