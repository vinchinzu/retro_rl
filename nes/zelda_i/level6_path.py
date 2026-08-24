"""Level 6 interior path controllers.

OccupancyWalker first. Coordinate clips only after a live miss. Isolated
emulator-state BFS is banned. Ignore object types 0x2b / 0x68. Do not poke
Rod / doors / keys. Do not grant Whistle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_KEESE_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WEST_WIZZROBE_ROOM,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "NORTH_68_MAX_FRAMES",
    "NORTH_DOOR_X",
    "NORTH_DOOR_Y",
    "Level6North68Controller",
    "make_north_48_controller",
    "make_north_58_controller",
]

NORTH_DOOR_X = 120
NORTH_DOOR_Y = 93
NORTH_BAND_Y = 109
NORTH_DOOR_X_TOL = 4
NORTH_68_MAX_FRAMES = 4000


@dataclass
class Level6North68Controller:
    """Occupancy BFS to a north door, then UP. Defaults are 0x78 → 0x68.

    Goal is play-ready dest. No combat. No path → stand.
    Door push on the north band is not occupancy-graded.
    """

    source_room: int = LEVEL6_WEST_WIZZROBE_ROOM
    dest_room: int = LEVEL6_COMPASS_ROOM
    spec_id: str = "level6_north_0x68"
    max_frames: int = NORTH_68_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def _goal(self) -> tuple[int, int]:
        return (NORTH_DOOR_X, NORTH_DOOR_Y)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if (
            snap.level == LEVEL6
            and snap.screen == self.dest_room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            note = f"arrived_{self.dest_room:02x}"
            self.notes.append(note)
            return FrameAction(nes_idle_action(), note)

        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return FrameAction(nes_idle_action(), f"wait_level_{snap.level}")
        if snap.screen == self.dest_room:
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north_settle")
        if snap.screen != self.source_room:
            self.failed = True
            self.notes.append(f"left_0x{self.source_room:02x}_to_0x{snap.screen:02x}")
            return FrameAction(
                nes_idle_action(), f"left_0x{self.source_room:02x}"
            )

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        if snap.link_y <= NORTH_BAND_Y:
            self.walker.last_dir = None
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "north_align")
            return FrameAction(nes_action("UP"), "north_push")

        direction = self.walker.next_dir(xy, self._goal())
        if direction is None:
            if abs(snap.link_x - NORTH_DOOR_X) <= 8 and snap.link_y <= 117:
                self.walker.last_dir = None
                return FrameAction(nes_action("UP"), "north_door_residual")
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "north_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "north_path")
        )

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or self.frames <= 2 or self.frames % 250 == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "room": int(snap.screen),
                    "reason": action.reason,
                }
            )
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "occupancy BFS + UP @ x≈120 on north band",
            "spec_id": self.spec_id,
            "source_room": self.source_room,
            "dest_room": self.dest_room,
        }


def make_north_58_controller() -> Level6North68Controller:
    """0x68 leftover → occupancy UP into Keese room 0x58. No fight."""
    return Level6North68Controller(
        source_room=LEVEL6_COMPASS_ROOM,
        dest_room=LEVEL6_KEESE_ROOM,
        spec_id="level6_north_0x58",
    )


def make_north_48_controller() -> Level6North68Controller:
    """0x58 leftover → occupancy UP into 0x48. Long push; do not poke doors."""
    return Level6North68Controller(
        source_room=LEVEL6_KEESE_ROOM,
        dest_room=LEVEL6_TRAPS_ROOM,
        spec_id="level6_north_0x48",
        max_frames=6000,
    )
