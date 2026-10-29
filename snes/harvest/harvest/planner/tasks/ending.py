"""Game ending: recognise the ROM's final sequence and wait it out.

ROM facts (see ``docs/ram_map.md`` "Ending sequence"): waking on Year 3 Summer
D30 (``year=2 season=1 day=30``) queues the ending event.  ``$7F1F47`` is its
stepper; it climbs 0x14..0x25 while the ROM shows the farm, the evaluation
screens and the Ranch-Master line.  0x25 is terminal (``CODE_83F4D7`` is a bare
RTS): the last dialogue clears and the game idles on the night-mountain map
forever.  It needs NO input (probed: no-input and A-pulse runs are frame
identical), so "credits finished" means the terminal index plus a static screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.ram_catalog import read_ram_value
from harvest.core.ranch_master import EndingIndexCapture
from harvest.tasks.nav import make_action

# Calendar: the date on which the ROM starts the ending (0-based year).
ENDING_YEAR = 2
ENDING_SEASON = 1
ENDING_DAY = 30
# ``$7F1F47`` stepper value at which the ROM has nothing left to do.
ENDING_TERMINAL_INDEX = 0x25
# Longest gap between dialogue changes seen in the ROM sequence is ~650f
# (scene fades); the terminal screen is static for 900f before we call it done.
ENDING_IDLE_FRAMES = 900
# Never observed needed; a stall this long below the terminal index falls back
# to pulsing A in case a variant waits for a button.
ENDING_STALL_PULSE_FRAMES = 4000
ENDING_MAX_FRAMES = 40_000
# The ROM starts the sequence ~700f after waking on the ending date.
ENDING_START_TIMEOUT_FRAMES = 6000


@dataclass(frozen=True)
class EndingSignals:
    """What the ROM says about the ending right now."""

    scene_index: int
    aux_index: int
    tilemap: int
    dialog_text_id: int
    input_lock: int
    ranch_report: Optional[dict] = None

    @property
    def started(self) -> bool:
        return bool(self.scene_index or self.aux_index)

    @property
    def terminal(self) -> bool:
        return self.scene_index >= ENDING_TERMINAL_INDEX

    def screen_key(self) -> Tuple[int, int, int, int, int]:
        return (
            self.scene_index,
            self.aux_index,
            self.tilemap,
            self.dialog_text_id,
            self.input_lock,
        )

    def summary(self) -> str:
        return (
            f"index=0x{self.scene_index:02X} tilemap=0x{self.tilemap:02X} "
            f"text={self.dialog_text_id} lock={self.input_lock}"
        )

    def to_dict(self) -> dict:
        row = {
            "scene_index": self.scene_index,
            "aux_index": self.aux_index,
            "tilemap": self.tilemap,
            "dialog_text_id": self.dialog_text_id,
            "input_lock": self.input_lock,
            "started": self.started,
            "terminal": self.terminal,
        }
        ranch = self.ranch_report
        if ranch:
            row["oracle"] = ranch.get("oracle")
            row["ranch_mastery_0x20"] = ranch.get("ranch_mastery_0x20")
            row["ranch_mastery_0x25"] = ranch.get("ranch_mastery_0x25")
            row["rating_text_id"] = ranch.get("rating_text_id")
            row["rating_text_id_0x20"] = ranch.get("rating_text_id_0x20")
            row["rating_text_id_0x25"] = ranch.get("rating_text_id_0x25")
            row["ranch"] = ranch
        return row


def read_ending_signals(ram: np.ndarray) -> EndingSignals:
    return EndingSignals(
        scene_index=int(read_ram_value(ram, "ending_scene_index", raw=True)),
        aux_index=int(read_ram_value(ram, "ending_aux_scene_index", raw=True)),
        tilemap=int(read_ram_value(ram, "tilemap", raw=True)),
        dialog_text_id=int(read_ram_value(ram, "dialog_text_id", raw=True)),
        input_lock=int(read_ram_value(ram, "input_lock", raw=True)),
    )


def is_ending_date(year: int, season: int, day: int) -> bool:
    """True on the ROM's ending date (0-based year), Year 3 Summer D30."""
    return (year, season, day) == (ENDING_YEAR, ENDING_SEASON, ENDING_DAY)


@dataclass
class EndingTask(Task):
    """Neutral-frame wait until the ROM's ending sequence has finished."""

    name: str = "ending"
    idle_frames: int = ENDING_IDLE_FRAMES
    stall_pulse_frames: int = ENDING_STALL_PULSE_FRAMES
    max_frames: int = ENDING_MAX_FRAMES
    start_timeout_frames: int = ENDING_START_TIMEOUT_FRAMES

    _frames: int = field(default=0, init=False)
    _last_key: Optional[Tuple[int, ...]] = field(default=None, init=False)
    _last_change: int = field(default=0, init=False)
    _peak_index: int = field(default=0, init=False)
    _signals: Optional[EndingSignals] = field(default=None, init=False)
    _score_capture: EndingIndexCapture = field(default_factory=EndingIndexCapture, init=False)
    _ranch_report: Optional[dict] = field(default=None, init=False)

    def reset(self, world: WorldState) -> None:
        self._frames = 0
        self._last_key = None
        self._last_change = 0
        self._peak_index = 0
        self._signals = None
        self._score_capture = EndingIndexCapture()
        self._ranch_report = self._score_capture.report()

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def peak_index(self) -> int:
        return self._peak_index

    @property
    def signals(self) -> Optional[EndingSignals]:
        return self._signals

    @property
    def phase_text(self) -> str:
        return "ENDING"

    @property
    def progress_text(self) -> str:
        return self._signals.summary() if self._signals is not None else ""

    def step(self, world: WorldState) -> TaskResult:
        self._frames += 1
        sig = read_ending_signals(world.ram)
        if self._score_capture.observe(world.ram, sig.scene_index) or self._ranch_report is None:
            self._ranch_report = self._score_capture.report()
        sig = replace(sig, ranch_report=self._ranch_report)
        self._signals = sig
        self._peak_index = max(self._peak_index, sig.scene_index)
        key = sig.screen_key()
        if key != self._last_key:
            self._last_key = key
            self._last_change = self._frames

        static_for = self._frames - self._last_change
        if sig.terminal and sig.input_lock == 1 and static_for >= self.idle_frames:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"ending credits complete ({sig.summary()})",
            )
        if self._peak_index == 0 and not sig.started and self._frames >= self.start_timeout_frames:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"ending never started in {self.start_timeout_frames}f",
            )
        if self._frames >= self.max_frames:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"ending did not finish in {self.max_frames}f ({sig.summary()})",
            )
        # Pulse A only when the sequence has stalled below the terminal index.
        if not sig.terminal and static_for >= self.stall_pulse_frames:
            press = (static_for % 3) != 0
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(a=press)),
                reason="ending stalled; pulsing A",
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"ending {sig.summary()}",
        )
