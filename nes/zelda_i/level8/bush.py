"""Isolated 0x6D bush-geometry recon. Not the cumulative L8 entry chapter.

The spine burn controller stays fail-closed on an unverified target.  This
module may step on existing ``Level8BushOW`` / ``OW_6D`` fixtures without a
PostLevel7Handoff.  Evidence is fixture-live and ``route_eligible=false``.
Never write ``ADDR_SELECTED_ITEM``.

Burn recipe (rr-u9js).  ``nes/zelda_i/logs/level8_bush_burn_sweep.json`` swept
5856 live trials over 732 standable 0x6D tiles x 4 facings x 2 pushes.  A
mode-16 mouth opened at exactly seven stands, all of them firing and pushing
the *same* direction Link faces (see ``MOUTH_STANDS``).  The default here is
(136, 93) face RIGHT / push RIGHT, which
``nes/zelda_i/custom_integrations/LegendOfZelda-Nes/Level8EntranceReconFixture.provenance.json``
reproduced into live L8 play -- screen 0x7E at (120, 205),
``reached_frame_after_push`` 111.  The older (144, 93) face RIGHT / push UP
hypothesis is refuted: all eight of its sweep trials burned the candle and saw
no mouth (``outcome: no_effect``).  Entry also does NOT complete on UP after
mode 16 -- continuing the push direction used to fire is what carries Link
through (rr-i6hq).
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

# Every stand that opened the mode-16 mouth in the 5856-trial sweep
# (logs/level8_bush_burn_sweep.json "near_misses"): one secret tile, several
# approach angles, all converging on L8 entry room 0x7E.  Facing == push on
# every one of them.
MOUTH_STANDS = (
    (120, 93, "RIGHT", "RIGHT"),
    (128, 93, "RIGHT", "RIGHT"),
    (136, 93, "RIGHT", "RIGHT"),
    (160, 77, "DOWN", "DOWN"),
    (184, 93, "LEFT", "LEFT"),
    (192, 93, "LEFT", "LEFT"),
    (200, 93, "LEFT", "LEFT"),
)

# Swept-verified default: the one stand the entrance fixture actually replayed
# into live L8 play (Level8EntranceReconFixture.provenance.json, entry room
# 0x7E at (120, 205)).
VERIFIED_BUSH_X = 136
VERIFIED_BUSH_Y = 93
VERIFIED_FACING = "RIGHT"
VERIFIED_PUSH = "RIGHT"
VERIFIED_BUSH_AIM = (VERIFIED_BUSH_X, VERIFIED_BUSH_Y)

# Refuted belief (rr-u9js): stand past the sampled east limit at (144, 93),
# fire RIGHT, then push UP because "dungeon mouths are mode-16 UP".  The sweep
# burned the candle at (144, 93) on all four facings and both pushes and never
# saw a mouth; (144, 93) is not a mouth stand at all.
REFUTED_BUSH_AIM = (144, 93)
REFUTED_FACING = "RIGHT"
REFUTED_PUSH = "UP"


class ReconBurnPhase(Enum):
    AIM = auto()
    FIRE = auto()
    ENTER = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class IsolatedBushReconController:
    """Fixture-live 0x6D burn trial. Budget exhaust on 0x6D is failure."""

    link_x: int = VERIFIED_BUSH_X
    link_y: int = VERIFIED_BUSH_Y
    facing: str = VERIFIED_FACING
    push_direction: str = VERIFIED_PUSH
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
            self.notes.append("fixture_live_bush_recipe_accepted")
            self.notes.append("refuted_aim_144_93_right_face_up_push")

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

        # rr-i6hq: UP after mode 16 does not complete the transition here.  The
        # sweep opened the mouth at seven stands and recorded entry_room=null on
        # every one; the entrance fixture only reached live L8 by continuing the
        # push direction it fired with.
        enter = nes_action(self.push_direction)
        if snap.mode == 16:
            if not self.candle_use_observed:
                return self._fail("mouth_transition_without_candle_use")
            self._set_phase(ReconBurnPhase.ENTER, "mouth_transition_observed")
            return FrameAction(enter, "enter_level8")
        if self.phase is ReconBurnPhase.ENTER and snap.transitioning:
            return FrameAction(enter, "enter_level8_transition")
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.screen != SCREEN_LEVEL8_BUSH:
            return self._fail("left_bush_screen_without_level8_entry")
        if self.phase is ReconBurnPhase.ENTER:
            return FrameAction(enter, "enter_level8")

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
            "refuted_aim": [*REFUTED_BUSH_AIM, REFUTED_FACING, REFUTED_PUSH],
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
    "MOUTH_STANDS",
    "OPEN_EXIT_UP_X",
    "REFUTED_BUSH_AIM",
    "REFUTED_FACING",
    "REFUTED_PUSH",
    "VERIFIED_BUSH_AIM",
    "VERIFIED_BUSH_X",
    "VERIFIED_BUSH_Y",
    "VERIFIED_FACING",
    "VERIFIED_PUSH",
    "IsolatedBushReconController",
    "WALKABLE_LEFT_X",
    "WALKABLE_SAND_X_MAX",
    "WALKABLE_SAND_Y",
    "make_isolated_bush_recon_controller",
]
