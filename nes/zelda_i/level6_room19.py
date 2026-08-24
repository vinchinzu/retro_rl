"""Level 6 0x18 east hop after Gleeok residual.

North hole idle is not mode 9 (stairs v1–v5; tile 0x77 at y=95–101).
Occupancy to the east door (208,141) then RIGHT. Do not walk RIGHT at
y=133 into the shutter face. Dest hypothesized 0x19 Map; enter-stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_overworld import LEVEL6, LEVEL6_GLEEOK_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "EAST_DOOR_X",
    "EAST_DOOR_Y",
    "EAST_DOOR_Y_TOL",
    "ROOM19_MAX_FRAMES",
    "Level6Room19Controller",
    "make_room19_controller",
]

EAST_DOOR_X = 208
EAST_DOOR_Y = 141
EAST_DOOR_Y_TOL = 4
EAST_DOOR_X_TOL = 4
ROOM19_MAX_FRAMES = 4000


@dataclass
class Level6Room19Controller:
    """Y-align 141, occupancy to (208,141), RIGHT. No stairs walk."""

    spec_id: str = "level6_room_0x19"
    room: int = LEVEL6_GLEEOK_ROOM
    goal: tuple[int, int] = (EAST_DOOR_X, EAST_DOOR_Y)
    max_frames: int = ROOM19_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
        }
        if force or self.frames <= 2 or self.frames % 250 == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "cur_opened_doors": int(snap.cur_opened_doors),
                    "open_doorway_mask": int(snap.open_doorway_mask),
                    "misses": self.walker.misses,
                    "tile": int(snap.colliding_tile),
                }
            )
        return action

    def _mark_success(self, snap: ZeldaSnapshot, reason: str, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), reason), force=True
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
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "link_death"), force=True
            )
        if (
            snap.level == LEVEL6
            and snap.screen != self.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            return self._mark_success(
                snap,
                f"arrived_{snap.screen:02x}",
                f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            self.walker.last_dir = None
            return FrameAction(nes_action("RIGHT"), "east_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            self.failed = True
            self.notes.append(f"left_level_{snap.level}")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "left_level"), force=True
            )
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("RIGHT"), "east_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        gx, gy = self.goal
        if (
            snap.link_x >= gx - EAST_DOOR_X_TOL
            and abs(snap.link_y - gy) <= EAST_DOOR_Y_TOL
        ):
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("RIGHT"), "east_push"))

        # Do not RIGHT at leftover y=133 into the shutter face.
        if abs(snap.link_y - gy) > EAST_DOOR_Y_TOL:
            dest = (int(snap.link_x), gy)
        else:
            dest = self.goal
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "east_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "east_path")
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "y=141 first, occupancy to (208,141), RIGHT; no y=133 RIGHT",
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
        }


def make_room19_controller() -> Level6Room19Controller:
    """Occupancy east of 0x18. Map pickup residual. Do not grant Rod."""
    return Level6Room19Controller()
