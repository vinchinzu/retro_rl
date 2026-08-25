"""Level 6 0x3A center-hole stairs from the cellar08 spit.

cellar08 leftover play 0x3A (96,157) south of the revealed center hole.
NE 0x71 cellar 0x08 is a two-mouth same-room U-turn. Walk onto the
walkthrough hole; dest is RAM. Do not poke bow/arrows. Do not walk east.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_cellar08 import (
    CELLAR_08_MAX_FRAMES,
    CELLAR_08_ROOM,
    make_cellar08_controller,
)
from zelda_i.level6_gleeok18 import PASSAGE_MODE
from zelda_i.level6_overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.level6_stairs3a_warp import (
    STAIRS_3A_WARP_MAX_FRAMES,
    make_stairs_3a_warp_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "CENTER_HOLE",
    "CENTER_3A_MAX_FRAMES",
    "Level6Center3AController",
    "level6_center3a_stages",
    "level6_center3a_success",
    "make_center3a_controller",
]

CENTER_3A_MAX_FRAMES = 4000
CENTER_3A_SAMPLE_PERIOD = 8
CENTER_HOLE = (112, 144)
HOLE_TOL = 4
EAST_DOOR_XMIN = 200
DATED_SPIT = (96, 157)
IDLE_ON_HOLE = 16
UP_ON_HOLE = 240


@dataclass
class Level6Center3AController:
    """From cellar spit (96,157), walk onto the center hole. Dest is RAM."""

    spec_id: str = "level6_center_0x3a"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = CENTER_3A_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    on_hole: bool = False

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
            "triforce": int(snap.triforce),
        }
        if force or self.frames <= 2 or self.frames % CENTER_3A_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "on_hole": self.on_hole,
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _warped(self, snap: ZeldaSnapshot) -> bool:
        if snap.level != LEVEL6:
            return False
        if snap.mode == PASSAGE_MODE and snap.screen != CELLAR_08_ROOM:
            return True
        return (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.room
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_tile={snap.colliding_tile}"
                )
            return self._emit(snap, FrameAction(nes_idle_action(), "timeout"), force=True)
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if self._warped(snap):
            self.success = True
            self.notes.append(
                f"warped_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
            return self._emit(
                snap, FrameAction(nes_idle_action(), f"warped_{snap.mode}"), force=True
            )
        if snap.mode == PASSAGE_MODE and snap.screen == CELLAR_08_ROOM:
            return self._fail(snap, f"loop_cellar08_{snap.link_x}_{snap.link_y}")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 10):
            return self._emit(snap, FrameAction(nes_idle_action(), "wait_scroll"))
        if snap.mode != PLAY_MODE:
            return self._emit(snap, FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}"))
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self._fail(snap, f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
        if int(snap.link_x) >= EAST_DOOR_XMIN:
            return self._fail(snap, f"east_door_{snap.link_x}_{snap.link_y}")

        xy = (int(snap.link_x), int(snap.link_y))
        hx, hy = CENTER_HOLE
        if abs(xy[0] - hx) <= HOLE_TOL and abs(xy[1] - hy) <= HOLE_TOL:
            if not self.on_hole:
                self.on_hole = True
                self.phase_frames = 0
                self.notes.append(f"on_hole_{xy[0]}_{xy[1]}_tile_{snap.colliding_tile}")
            if self.phase_frames <= IDLE_ON_HOLE:
                return self._emit(snap, FrameAction(nes_idle_action(), "hole_idle"))
            if self.phase_frames > IDLE_ON_HOLE + UP_ON_HOLE:
                return self._fail(
                    snap, f"hole_no_warp_{xy[0]}_{xy[1]}_tile_{snap.colliding_tile}"
                )
            return self._emit(snap, FrameAction(nes_action("UP"), "hole_up"))
        self.on_hole = False
        if abs(xy[0] - hx) > HOLE_TOL:
            btn = "LEFT" if xy[0] > hx else "RIGHT"
            return self._emit(snap, FrameAction(nes_action(btn), "hole_x"))
        btn = "UP" if xy[1] > hy else "DOWN"
        return self._emit(snap, FrameAction(nes_action(btn), "hole_y"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                f"cellar08 spit {DATED_SPIT} → center hole {CENTER_HOLE}; "
                "dest RAM not cellar 0x08"
            ),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
            "on_hole": self.on_hole,
        }


def make_center3a_controller() -> Level6Center3AController:
    """Walk the revealed center hole from the cellar08 spit."""
    return Level6Center3AController()


def level6_center3a_stages():
    """Warp + cellar08 return, then center-hole walk. Dest is RAM."""
    return (
        ("level6_stairs_0x3a_warp", make_stairs_3a_warp_controller(), STAIRS_3A_WARP_MAX_FRAMES),
        ("level6_cellar_0x08", make_cellar08_controller(), CELLAR_08_MAX_FRAMES),
        ("level6_center_0x3a", make_center3a_controller(), CENTER_3A_MAX_FRAMES),
    )


def level6_center3a_success(snap: ZeldaSnapshot) -> bool:
    """Mode 9 dest ≠ 0x08, or play ≠ 0x3A. Rod and TF 0x1F stay."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if int(getattr(snap, "rod", 0)) == 0:
        return False
    if snap.mode == PASSAGE_MODE:
        return snap.screen != CELLAR_08_ROOM
    return (
        snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != LEVEL6_BLOCK_3A_ROOM
    )
