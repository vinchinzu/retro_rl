"""Level 6 Magical Rod pickup in cellar 0x75.

Stairs leftover: mode 9 room 0x75 (208,93) tile 0x71 rod=0. Stairs spit
west to (48,74) (v2 idle watched it). Rod is the west-column statue.
Wait spawn until xy moves, then UP the west aisle. Do not grant ADDR_ROD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_overworld import LEVEL6
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "ROD_75_MAX_FRAMES",
    "ROD_75_GOAL",
    "ROD_75_ROOM",
    "Level6RodController",
    "make_rod_75_controller",
]

ROD_75_ROOM = 0x75
# v5 leftover (48,73) rod_idle on the west statue; Rod still on the pedestal.
ROD_75_GOAL = (32, 73)
ROD_75_ALIGN_TOL = 4
ROD_75_WEST_X = 80
ROD_75_SETTLE_Y = 88
ROD_75_STABLE = 16
ROD_75_MAX_FRAMES = 4000
ROD_75_SAMPLE_PERIOD = 8
# Mode 9/11 are playable cellar; 10/16 are enter/scroll.
CELLAR_PLAY_MODES = (9, 11)
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


@dataclass
class Level6RodController:
    """Wait stairs spit, then UP west aisle to the Rod statue."""

    spec_id: str = "level6_rod_0x75"
    room: int = ROD_75_ROOM
    goal: tuple[int, int] = ROD_75_GOAL
    max_frames: int = ROD_75_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(
        default_factory=lambda: OccupancyWalker(
            grid=OccupancyGrid(xmin=32, xmax=216, ymin=40, ymax=205)
        )
    )
    spawn_xy: tuple[int, int] | None = None
    mobile: bool = False
    settled: bool = False
    settle_xy: tuple[int, int] | None = None
    stable_frames: int = 0

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "submode": int(snap.submode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "rod": self._rod(snap),
            "keys": int(snap.keys),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
        }
        if force or self.frames <= 2 or self.frames % ROD_75_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "submode": int(snap.submode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "rod": self._rod(snap),
                    "misses": self.walker.misses,
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _got_rod(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), "rod_got"), force=True
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
                    f"_mode={snap.mode}_rod={self._rod(snap)}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if self._rod(snap):
            return self._got_rod(
                snap,
                f"rod_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "wait_scroll")
        # Warp frame may still look like 0x09. Wait; do not re-push.
        if snap.screen == 0x09:
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_idle_action(), "wait_warp"))
        if snap.mode not in CELLAR_PLAY_MODES and snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}")
        if snap.mode == PLAY_MODE and snap.screen != self.room:
            return self._fail(
                snap, f"left_cellar_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )

        xy = (int(snap.link_x), int(snap.link_y))
        if self.spawn_xy is None:
            self.spawn_xy = xy
        if xy != self.spawn_xy:
            self.mobile = True
        # Wait stairs spit AND y-stable (v4 UP while spit still slid south).
        if not self.settled:
            if xy[0] > ROD_75_WEST_X or xy[1] < ROD_75_SETTLE_Y:
                self.settle_xy = None
                self.stable_frames = 0
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_spawn")
                )
            if self.settle_xy != xy:
                self.settle_xy = xy
                self.stable_frames = 0
            self.stable_frames += 1
            if self.stable_frames < ROD_75_STABLE:
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_spawn")
                )
            self.settled = True
        gx, gy = self.goal
        if abs(xy[1] - gy) > ROD_75_ALIGN_TOL:
            dest = (xy[0], gy)
        else:
            dest = self.goal
        at_dest = (
            abs(xy[0] - gx) <= ROD_75_ALIGN_TOL
            and abs(xy[1] - gy) <= ROD_75_ALIGN_TOL
        )
        if at_dest:
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_idle_action(), "rod_idle"))

        if not self.mobile:
            # Axis without occupancy until a pixel moves (spawn lock).
            self.walker.last_dir = None
            if abs(xy[1] - gy) > ROD_75_ALIGN_TOL:
                btn = "DOWN" if xy[1] < gy else "UP"
                return self._emit(snap, FrameAction(nes_action(btn), "rod_y"))
            btn = "LEFT" if xy[0] > gx else "RIGHT"
            return self._emit(snap, FrameAction(nes_action(btn), "rod_x"))

        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            if self.walker.misses > 0:
                return self._fail(snap, f"no_path_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_idle_action(), "rod_stand"))
        reason = "rod_y" if dest[0] == xy[0] else "rod_x"
        return self._emit(snap, FrameAction(nes_action(direction), reason))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "wait spit settle, UP west aisle then LEFT (32,73) ADDR_ROD",
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": list(self.goal),
        }


def make_rod_75_controller() -> Level6RodController:
    """Walk cellar 0x75 until ADDR_ROD. Do not grant the rod."""
    return Level6RodController()
