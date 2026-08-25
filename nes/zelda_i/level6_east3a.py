"""Level 6 0x3A east door after the 0x08 cellar return.

cellar08 leftover play 0x3A (96,157). Center hole is not CheckWarp
(center3a v1 timeout (112,141) tile 118). East door is PNG-open. y=141
RIGHT to the mouth. Dest is RAM. Do not poke bow/arrows/doors/keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_cellar08 import CELLAR_08_MAX_FRAMES, make_cellar08_controller
from zelda_i.level6_gleeok18 import PASSAGE_MODE
from zelda_i.level6_overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.level6_stairs3a_warp import (
    STAIRS_3A_WARP_MAX_FRAMES,
    make_stairs_3a_warp_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "EAST3A_MAX_FRAMES",
    "EAST_DOOR",
    "Level6East3AController",
    "level6_east3a_stages",
    "level6_east3a_success",
    "make_east3a_controller",
]

EAST3A_MAX_FRAMES = 4000
EAST3A_SAMPLE_PERIOD = 8
EAST_DOOR = (208, 141)
# v2 (96,143) tile 119 counted as y-band (tol=4) then RIGHT 0px.
DOOR_TOL = 1
DATED_SPIT = (96, 157)


@dataclass
class Level6East3AController:
    """From cellar spit, y=141 then RIGHT. Occupancy halt. Dest is RAM."""

    spec_id: str = "level6_east_0x3a"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = EAST3A_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

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
            "triforce": int(snap.triforce),
        }
        if force or self.frames <= 2 or self.frames % EAST3A_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
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
        if snap.mode == PASSAGE_MODE:
            return True
        return (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.room
        )

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
        if self._warped(snap):
            self.success = True
            self.notes.append(
                f"dest_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), f"dest_{snap.mode}"), force=True
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 10):
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_idle_action(), "wait_scroll"))
        if snap.mode != PLAY_MODE:
            return self._emit(snap, FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}"))
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self._fail(snap, f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")

        xy = (int(snap.link_x), int(snap.link_y))
        gx, gy = EAST_DOOR
        # v1 leftover (96,157) UP moved 2px; occupancy 1px-grade halted.
        if abs(xy[1] - gy) > DOOR_TOL:
            self.walker.last_dir = None
            btn = "UP" if xy[1] > gy else "DOWN"
            return self._emit(snap, FrameAction(nes_action(btn), "door_y"))
        prev = self.walker.last_dir
        before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > before:
            return self._fail(
                snap, f"occupancy_halt_{prev}_{xy[0]}_{xy[1]}_tile_{snap.colliding_tile}"
            )
        self.walker.last_dir = "RIGHT"
        if xy[0] < gx - DOOR_TOL:
            return self._emit(snap, FrameAction(nes_action("RIGHT"), "door_x"))
        return self._emit(snap, FrameAction(nes_action("RIGHT"), "door_push"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                f"cellar08 spit {DATED_SPIT} → y=141 RIGHT to {EAST_DOOR}; "
                "occupancy halt; dest RAM"
            ),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
            "misses": self.walker.misses,
        }


def make_east3a_controller() -> Level6East3AController:
    """Walk 0x3A east door from the cellar08 spit. Dest is RAM."""
    return Level6East3AController()


def level6_east3a_stages():
    """Warp + cellar08 return, then east door. Dest is RAM."""
    return (
        ("level6_stairs_0x3a_warp", make_stairs_3a_warp_controller(), STAIRS_3A_WARP_MAX_FRAMES),
        ("level6_cellar_0x08", make_cellar08_controller(), CELLAR_08_MAX_FRAMES),
        ("level6_east_0x3a", make_east3a_controller(), EAST3A_MAX_FRAMES),
    )


def level6_east3a_success(snap: ZeldaSnapshot) -> bool:
    """Mode 9 or play ≠ 0x3A. Rod and TF 0x1F stay."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if int(getattr(snap, "rod", 0)) == 0:
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    return (
        snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != LEVEL6_BLOCK_3A_ROOM
    )
