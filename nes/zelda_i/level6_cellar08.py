"""Level 6 cellar 0x08 after the 0x3A position-write warp.

Warp leftover: mode 9 room 0x08 (208,93) rod=1 keys=4 bombs=8 TF=0x1F
bow=0 arrows=0 tile 113. Arrival is the east mouth (0x09 analog).
Do not UP (that returns). DOWN to the floor, LEFT to the west mouth, UP.
v1 west-mouth UP returned to play 0x3A (96,157) — two-mouth same-room
cellar, not Gohma. Dest is RAM play ≠ 0x08 (0x3A return is legal).
Do not invent Gohma. Do not poke bow/arrows/doors/keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_gleeok18 import PASSAGE_MODE
from zelda_i.level6_overworld import LEVEL6
from zelda_i.level6_stairs3a_warp import (
    STAIRS_3A_WARP_MAX_FRAMES,
    make_stairs_3a_warp_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "CELLAR_08_MAX_FRAMES",
    "CELLAR_08_ROOM",
    "Level6Cellar08Controller",
    "level6_cellar08_stages",
    "level6_cellar08_success",
    "make_cellar08_controller",
]

CELLAR_08_ROOM = 0x08
CELLAR_08_MAX_FRAMES = 4000
CELLAR_08_SAMPLE_PERIOD = 8
EAST_MOUTH = (208, 93)
WEST_X = 48
FLOOR_Y = 141
MOUTH_Y = 93
ALIGN_TOL = 4
MOUTH_IDLE = 240
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
CELLAR_PLAY_MODES = (9, 11)


@dataclass
class Level6Cellar08Controller:
    """East-mouth leftover → floor LEFT → west UP. Dest is RAM."""

    spec_id: str = "level6_cellar_0x08"
    room: int = CELLAR_08_ROOM
    max_frames: int = CELLAR_08_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    on_floor: bool = False
    mouth_idle: int = 0

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "rod": int(getattr(snap, "rod", 0)),
            "bow": int(getattr(snap, "bow", 0)),
            "arrows": int(getattr(snap, "arrows", 0)),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
        }
        if force or self.frames <= 2 or self.frames % CELLAR_08_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "on_floor": self.on_floor,
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _done(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), "emerged"), force=True)

    def _emerged(self, snap: ZeldaSnapshot) -> bool:
        if snap.level != LEVEL6 or snap.triforce != 0x1F:
            return False
        if int(getattr(snap, "rod", 0)) == 0:
            return False
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen == self.room:
            return False
        return True

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}"
                )
            return self._emit(snap, FrameAction(nes_idle_action(), "timeout"), force=True)
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if self._emerged(snap):
            return self._done(
                snap, f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.transitioning or snap.mode in WAIT_MODES:
            return self._emit(snap, FrameAction(nes_idle_action(), "wait_scroll"))
        if snap.mode not in CELLAR_PLAY_MODES and snap.mode != PLAY_MODE:
            return self._emit(
                snap, FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            )
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}")

        xy = (int(snap.link_x), int(snap.link_y))
        if xy[1] >= FLOOR_Y - ALIGN_TOL:
            self.on_floor = True
        if not self.on_floor:
            if xy[1] < FLOOR_Y - ALIGN_TOL:
                if xy[0] > EAST_MOUTH[0] - 24:
                    return self._emit(
                        snap, FrameAction(nes_action("LEFT", "DOWN"), "drop_clip")
                    )
                return self._emit(snap, FrameAction(nes_action("DOWN"), "drop_y"))
        if abs(xy[0] - WEST_X) > ALIGN_TOL:
            btn = "LEFT" if xy[0] > WEST_X else "RIGHT"
            return self._emit(snap, FrameAction(nes_action(btn), "cross_x"))
        if xy[1] > MOUTH_Y:
            self.mouth_idle = 0
            return self._emit(snap, FrameAction(nes_action("UP"), "climb_y"))
        self.mouth_idle += 1
        if self.mouth_idle < MOUTH_IDLE:
            return self._emit(snap, FrameAction(nes_idle_action(), "mouth_idle"))
        return self._emit(snap, FrameAction(nes_action("UP"), "mouth_up"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "mode-9 0x08 east mouth: LEFT+DOWN off (208,93), LEFT to x=48, "
                "UP west mouth. Dest RAM play != 0x08 (0x3A return dated)"
            ),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
            "on_floor": self.on_floor,
        }


def make_cellar08_controller() -> Level6Cellar08Controller:
    """Cross cellar 0x08 from the 0x3A arrival mouth. Do not return."""
    return Level6Cellar08Controller()


def level6_cellar08_stages():
    """Warp 0x3A (dedicated predecessor) then cellar 0x08 west mouth."""
    warp = make_stairs_3a_warp_controller()
    cellar = make_cellar08_controller()
    return (
        ("level6_stairs_0x3a_warp", warp, STAIRS_3A_WARP_MAX_FRAMES),
        ("level6_cellar_0x08", cellar, CELLAR_08_MAX_FRAMES),
    )


def level6_cellar08_success(snap: ZeldaSnapshot) -> bool:
    """Play dest ≠ cellar 0x08. 0x3A return is live. Rod and TF 0x1F stay."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if int(getattr(snap, "rod", 0)) == 0:
        return False
    if snap.mode == PASSAGE_MODE:
        return False
    return (
        snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != CELLAR_08_ROOM
    )
