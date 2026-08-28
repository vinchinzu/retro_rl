"""Level 6 0x18 north-stairs hop after Gleeok residual.

Heads gone; east shutter still closed (mask 0, PNG black). Occupancy to the
north-center hole then UP. Do not grant Rod. Do not walk RIGHT into the
closed east shutter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_occupancy import l6_leftover
from zelda_i.level6_overworld import LEVEL6, LEVEL6_GLEEOK_ROOM
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "STAIRS_18_GOAL",
    "STAIRS_18_MAX_FRAMES",
    "STAIRS_18_X_TOL",
    "Level6Stairs18Controller",
    "make_stairs_18_controller",
]

# v3 leftover (120,109) tile 0x76 diamond, south of hole.
# v4 leftover (120,101) tile 0x77 still south of hole.
# v2 leftover (120,93) visually on hole; hold-UP no mode 9.
# v5 leftover (120,95) tile 0x77, still south; hole is decorative.
STAIRS_18_GOAL = (120, 96)
STAIRS_18_X_TOL = 4
STAIRS_18_MAX_FRAMES = 4000


@dataclass
class Level6Stairs18Controller:
    """Occupancy to (120,96) then idle on the 0x18 stairs hole."""

    spec_id: str = "level6_stairs_0x18"
    room: int = LEVEL6_GLEEOK_ROOM
    goal: tuple[int, int] = STAIRS_18_GOAL
    max_frames: int = STAIRS_18_MAX_FRAMES
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
            **l6_leftover(snap),
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
        if snap.mode == PASSAGE_MODE:
            return self._mark_success(
                snap,
                "stairs",
                f"stairs_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 10):
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "wait_scroll")
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
            return self._mark_success(
                snap,
                f"arrived_{snap.screen:02x}",
                f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        gx, gy = self.goal
        # v2 leftover (120,93): hold-UP walked through the hole to the north
        # wall and never entered mode 9. Idle on the hole band instead.
        if abs(snap.link_x - gx) <= STAIRS_18_X_TOL and snap.link_y <= gy:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "stairs_idle")
            )

        # v1 leftover (160,117): occupancy to (120,109) is length-tied;
        # BFS UP-first slid into the east diamond pocket. Column-first.
        if abs(snap.link_x - gx) > STAIRS_18_X_TOL:
            dest = (gx, int(snap.link_y))
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
                snap, FrameAction(nes_idle_action(), "stairs_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "stairs_path")
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "column x=120 first, occupancy to (120,96), idle on hole",
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
        }


def make_stairs_18_controller() -> Level6Stairs18Controller:
    """Occupancy onto 0x18 north stairs. Do not grant Rod."""
    return Level6Stairs18Controller()
