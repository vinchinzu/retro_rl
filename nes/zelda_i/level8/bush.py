"""Isolated 0x6D bush-geometry recon. Not the cumulative L8 entry chapter.

The spine burn controller stays fail-closed on an unverified target.  This
module may step on existing ``Level8BushOW`` / ``OW_6D`` fixtures without a
PostLevel7Handoff.  Evidence is fixture-live and ``route_eligible=false``.
Never write ``ADDR_SELECTED_ITEM``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import SCREEN_LEVEL8_BUSH
from zelda_i.level8.entry import (
    ADDR_CANDLE_USED,
    B_ITEM_CANDLE,
    BURN_MAX_FRAMES,
    CANDLE_RED,
    LEVEL8,
    POST_L7_TRIFORCE,
)
from zelda_i.ram import ADDR_CANDLE, ADDR_SELECTED_ITEM, PLAY_MODE, ZeldaSnapshot, read_u8

# Live walkable (assisted, no candle): left corridor x≈32–56 plus mid sand
# y≈88–96 east to x≈144.  Only open exit without candle: UP @ x≈48 → 0x5D.
WALKABLE_LEFT_X = (32, 56)
WALKABLE_SAND_Y = (88, 96)
WALKABLE_SAND_X_MAX = 144
OPEN_EXIT_UP_X = 48

# Dead belief: east-channel aim (136, 93) face RIGHT, push RIGHT.  Dense
# walkable burns never opened a mode-16 mouth.
DEAD_BUSH_AIM = (136, 93)

# One new hypothesis from the walkable raster: the lone bush is past the
# sampled east limit, so stand at x≈144, y≈93, fire RIGHT, then push UP
# (dungeon mouths are mode-16 UP, not a RIGHT screen exit).
HYPOTHESIS_BUSH_X = 144
HYPOTHESIS_BUSH_Y = 93
HYPOTHESIS_FACING = "RIGHT"
HYPOTHESIS_PUSH = "UP"


class ReconBurnPhase(Enum):
    AIM = auto()
    FIRE = auto()
    ENTER = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class IsolatedBushReconController:
    """Fixture-live 0x6D burn trial. Budget exhaust on 0x6D is failure."""

    link_x: int = HYPOTHESIS_BUSH_X
    link_y: int = HYPOTHESIS_BUSH_Y
    facing: str = HYPOTHESIS_FACING
    push_direction: str = HYPOTHESIS_PUSH
    tolerance: int = 4
    max_frames: int = BURN_MAX_FRAMES
    burn_budget: int = 800
    phase: ReconBurnPhase = ReconBurnPhase.AIM
    frames: int = 0
    burn_frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    candle_use_observed: bool = False
    observed_entry_room: int | None = None
    evidence: str = "fixture-live"
    route_eligible: bool = False
    _env: Any = field(default=None, init=False, repr=False)
    _validated: bool = field(default=False, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _set_phase(self, phase: ReconBurnPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._set_phase(ReconBurnPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            return self._fail("level8_entry_timeout")
        if self._env is None:
            return self._fail("bush_recon_env_not_bound")
        ram = self._env.get_ram()
        candle = read_u8(ram, ADDR_CANDLE)
        selected = read_u8(ram, ADDR_SELECTED_ITEM)
        candle_used = read_u8(ram, ADDR_CANDLE_USED)
        self.candle_use_observed = self.candle_use_observed or candle_used != 0

        if not self._validated:
            if (
                snap.level != 0
                or snap.mode != PLAY_MODE
                or snap.screen != SCREEN_LEVEL8_BUSH
            ):
                return self._fail("bush_recon_not_on_0x6d")
            if candle == 0:
                return self._fail("bush_recon_candle_unowned")
            if selected != B_ITEM_CANDLE:
                return self._fail("bush_recon_candle_not_selected")
            self._validated = True
            self.notes.append("fixture_live_bush_hypothesis_accepted")
            self.notes.append("dead_belief_136_93_right_push")

        if snap.mode == 17:
            return self._fail("link_death")
        if snap.level == LEVEL8 and snap.mode == PLAY_MODE:
            if not self.candle_use_observed:
                return self._fail("level8_entered_without_observed_candle_use")
            self.observed_entry_room = snap.screen
            self.success = True
            self._set_phase(ReconBurnPhase.DONE, "level8_live_entry")
            return FrameAction(nes_idle_action(), "done")
        if self.burn_frames >= self.burn_budget:
            return self._fail("burn_budget_exhausted_without_level8_entry")
        self.burn_frames += 1

        if snap.mode == 16:
            if not self.candle_use_observed:
                return self._fail("mouth_transition_without_candle_use")
            self._set_phase(ReconBurnPhase.ENTER, "mouth_transition_observed")
            return FrameAction(nes_action("UP"), "enter_level8")
        if self.phase is ReconBurnPhase.ENTER and snap.transitioning:
            return FrameAction(nes_action("UP"), "enter_level8_transition")
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.screen != SCREEN_LEVEL8_BUSH:
            return self._fail("left_bush_screen_without_level8_entry")
        if self.phase is ReconBurnPhase.ENTER:
            return FrameAction(nes_action("UP"), "enter_level8")

        if abs(snap.link_x - self.link_x) > self.tolerance:
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < self.link_x else "LEFT"),
                "bush_burn_align_x",
            )
        if abs(snap.link_y - self.link_y) > self.tolerance:
            return FrameAction(
                nes_action("DOWN" if snap.link_y < self.link_y else "UP"),
                "bush_burn_align_y",
            )
        self._set_phase(ReconBurnPhase.FIRE)
        cycle = self.phase_frames % 36
        if cycle < 4:
            return FrameAction(nes_action(self.facing), "bush_face")
        if cycle < 12:
            return FrameAction(nes_action("B"), "red_candle_fire")
        return FrameAction(nes_action(self.push_direction), "push_revealed_mouth")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase.name,
            "frames": self.frames,
            "burn": [self.burn_frames, self.burn_budget],
            "aim": [self.link_x, self.link_y, self.facing, self.push_direction],
            "dead_belief": DEAD_BUSH_AIM,
            "candle_use_observed": self.candle_use_observed,
            "observed_entry_room": self.observed_entry_room,
            "evidence": self.evidence,
            "route_eligible": self.route_eligible,
            "writes": 0,
            "triforce": POST_L7_TRIFORCE,
            "candle_red": CANDLE_RED,
            "notes": list(self.notes),
        }


def make_isolated_bush_recon_controller() -> IsolatedBushReconController:
    return IsolatedBushReconController()


__all__ = [
    "DEAD_BUSH_AIM",
    "HYPOTHESIS_BUSH_X",
    "HYPOTHESIS_BUSH_Y",
    "HYPOTHESIS_FACING",
    "HYPOTHESIS_PUSH",
    "IsolatedBushReconController",
    "OPEN_EXIT_UP_X",
    "WALKABLE_LEFT_X",
    "WALKABLE_SAND_X_MAX",
    "WALKABLE_SAND_Y",
    "make_isolated_bush_recon_controller",
]
