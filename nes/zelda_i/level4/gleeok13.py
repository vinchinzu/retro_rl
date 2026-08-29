"""Level 4 leftover 0x12 → push 0x68 → PATH_12_TO_GLEEOK enter 0x13.

clear12 v1 leftover play 0x12 (128,117). Walk to PUSH_12_STAND (112,144),
hold LEFT, then isolated hold4 token path. Isolated BFS is not this tape;
if the token dumps, leftover PNG is the next occupancy seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level4.dungeon import (
    LEVEL4,
    PUSH_12_DIR,
    PUSH_12_STAND,
    ROOM_L4_GLEEOK_13,
    ROOM_L4_VIRES_12,
)
from zelda_i.level4.maze_path import PATH_12_TO_GLEEOK, PUSH_12_HOLD, RIGHT_12_HOLD
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "Gleeok13Phase",
    "Level4Gleeok13Controller",
    "level4_gleeok13_stages",
    "level4_gleeok13_success",
    "make_gleeok13_controller",
    "attach_level4_tf_suffix",
    "run_level4_tf_suffix",
]


class Gleeok13Phase(Enum):
    STAND = auto()
    PUSH = auto()
    TOKEN = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Gleeok13Controller:
    """0x12 leftover → push stand → hold LEFT → hold4 token → 0x13."""

    max_frames: int = 8000
    phase: Gleeok13Phase = Gleeok13Phase.STAND
    frames: int = 0
    phase_frames: int = 0
    token_index: int = 0
    token_hold: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Gleeok13Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Gleeok13Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _entered_13(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_GLEEOK_13
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.phase is Gleeok13Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Gleeok13Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail(
                f"timeout_{int(snap.link_x)}_{int(snap.link_y)}"
            )
        if self._entered_13(snap):
            self.success = True
            self._set_phase(Gleeok13Phase.DONE, "entered_0x13")
            return FrameAction(nes_idle_action(), "done")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll_right")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_VIRES_12:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.phase is Gleeok13Phase.STAND:
            tx, ty = PUSH_12_STAND
            dx, dy = tx - snap.link_x, ty - snap.link_y
            if abs(dx) <= 2 and abs(dy) <= 2:
                self._set_phase(Gleeok13Phase.PUSH, "at_push_stand")
                return FrameAction(nes_action(PUSH_12_DIR), "push_block")
            # v1 leftover (128,141): y-first DOWN on the door row is solid.
            if abs(dx) > 2:
                return FrameAction(
                    nes_action("RIGHT" if dx > 0 else "LEFT"), "stand_x"
                )
            return FrameAction(
                nes_action("DOWN" if dy > 0 else "UP"), "stand_y"
            )

        if self.phase is Gleeok13Phase.PUSH:
            if self.phase_frames < PUSH_12_HOLD:
                return FrameAction(nes_action(PUSH_12_DIR), "push_block")
            self.token_index = 0
            self.token_hold = 0
            self._set_phase(Gleeok13Phase.TOKEN, "token_path")
            return FrameAction(nes_action(PATH_12_TO_GLEEOK[0]), "token_0")

        if self.phase is Gleeok13Phase.TOKEN:
            if self.token_index >= len(PATH_12_TO_GLEEOK):
                return FrameAction(nes_action("RIGHT"), "push_east")
            direction = PATH_12_TO_GLEEOK[self.token_index]
            self.token_hold += 1
            if self.token_hold >= RIGHT_12_HOLD:
                self.token_index += 1
                self.token_hold = 0
            return FrameAction(nes_action(direction), f"token_{self.token_index}")

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "token_index": self.token_index,
            "notes": list(self.notes),
            "segment": "level4_gleeok_enter_0x13",
            "target_room": f"0x{ROOM_L4_GLEEOK_13:02x}",
        }


def make_gleeok13_controller() -> Level4Gleeok13Controller:
    return Level4Gleeok13Controller()


def level4_gleeok13_stages():
    ctl = make_gleeok13_controller()
    return (("level4_gleeok_enter_0x13", ctl, ctl.max_frames),)


def level4_gleeok13_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x13; Gleeok may stay live."""
    return (
        snap.level == LEVEL4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_GLEEOK_13
        and not snap.transitioning
    )


def run_level4_tf_suffix(env, *, assist, frame_base: int):
    """South-stand Gleeok → HC → TF 0x08 from play-ready 0x13. Env-stepping."""
    from zelda_i.level4.boss_combat import make_gleeok_fight_controller

    ctl = make_gleeok_fight_controller(tag="survival_spine_l4")
    total = [int(frame_base)]
    result = ctl.run(env, assist, total)
    ok = bool(result.get("ok") and result.get("tf08"))
    return ok, int(total[0]), ctl


def attach_level4_tf_suffix(env, run, *, assist) -> bool:
    """Append the south-stand TF suffix onto a spine run. False if TF missing."""
    from zelda_i.route.chain import ControllerStageResult
    from zelda_i.level4.dungeon import LEVEL4_TRIFORCE_BIT
    from zelda_i.ram import read_snapshot

    ok, end_frame, fight = run_level4_tf_suffix(
        env, assist=assist, frame_base=run.end_frame
    )
    run.stages.append(
        ControllerStageResult(
            name="level4_gleeok_tf",
            controller=fight,
            max_frames=20000,
            frames=end_frame - run.end_frame,
            success=ok,
            frame_base=run.end_frame,
            end_frame=end_frame,
        )
    )
    run.end_frame = end_frame
    run.obs = getattr(env, "last_observation", run.obs)
    snap = read_snapshot(env.get_ram())
    run.success = ok and bool(snap.triforce & LEVEL4_TRIFORCE_BIT)
    if not run.success:
        run.failed_stage = "level4_triforce_0x08"
    return run.success
