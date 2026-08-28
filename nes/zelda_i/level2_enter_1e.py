"""0x2e leftover (96,141) → play 0x1e. Cardinal LEFT/DOWN/UP are solid.

Bow-splice ``l6_gohma_bow_v{5,6,7}``: 1-tile gap between diamond stacks.
OccupancyWalker to the north door (120,93). First BFS dir is RIGHT (never
tried on the blocked checkbox). Dated LEFT/UP/DOWN at the gutter fire
LEFT+UP instead (diagonal is not in OccupancyWalker). Do not extend 4000f.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "DOOR_X",
    "ENTER_1E_MAX_FRAMES",
    "GUTTER_X",
    "GUTTER_Y",
    "Level2Enter1eController",
    "NORTH_DOOR",
    "ROOM_2E",
    "Enter1ePhase",
    "in_diamond_gutter",
]

ROOM_2E = 0x2E
DEST_ROOM = 0x1E
ENTER_1E_MAX_FRAMES = 4000
DOOR_X = 120
NORTH_DOOR = (120, 93)
NORTH_BAND_Y = 117
# Live leftover (96, 141). ±8 px is half a diamond tile.
GUTTER_X = (88, 104)
GUTTER_Y = (133, 149)


class Enter1ePhase(Enum):
    WALK = auto()
    DONE = auto()
    FAILED = auto()


def in_diamond_gutter(x: int, y: int) -> bool:
    """1-tile gap between 0x2e diamond stacks at door-Y."""
    return GUTTER_X[0] <= x <= GUTTER_X[1] and GUTTER_Y[0] <= y <= GUTTER_Y[1]


@dataclass
class Level2Enter1eController:
    """Occupancy to (120,93); LEFT+UP clip when the 0x2e gutter has no path."""

    dest_room: int = DEST_ROOM
    from_room: int = ROOM_2E
    goal: tuple[int, int] = NORTH_DOOR
    max_frames: int = ENTER_1E_MAX_FRAMES
    phase: Enter1ePhase = Enter1ePhase.WALK
    frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    _last_dir: str = "UP"

    def _fail(self, note: str) -> FrameAction:
        self.phase = Enter1ePhase.FAILED
        self.notes.append(note)
        self.walker.last_dir = None
        return FrameAction(nes_idle_action(), note)

    def _clip(self) -> FrameAction:
        self.walker.last_dir = None
        self._last_dir = "UP"
        return FrameAction(nes_action("LEFT", "UP"), "gutter_clip")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.screen == self.dest_room and snap.mode == PLAY_MODE:
            self.success = True
            self.phase = Enter1ePhase.DONE
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "done")
        if snap.mode == 8:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.transitioning:
            return FrameAction(nes_action(self._last_dir), "room_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.from_room:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")

        xy = (int(snap.link_x), int(snap.link_y))
        x, y = xy
        self.walker.observe(xy)
        if abs(x - DOOR_X) <= 2 and y <= NORTH_BAND_Y:
            self.walker.last_dir = None
            self._last_dir = "UP"
            return FrameAction(nes_action("UP"), "push_up")

        if in_diamond_gutter(x, y):
            # North-door BFS from here is dated UP. Door-column occupancy
            # at leftover y is RIGHT (never tried on v5–v7). Else clip.
            col = (DOOR_X, y)
            if col != self.walker.goal:
                self.walker.path = None
                self.walker.goal = col
            direction = self.walker.next_dir(xy, col)
            if direction == "RIGHT":
                self._last_dir = "RIGHT"
                return FrameAction(nes_action("RIGHT"), "gutter_right")
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"clip_f{self.frames}_{x}_{y}_{direction}")
            return self._clip()

        dest = self.goal
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)

        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{x}_{y}")
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "occupancy_stand")
        self._last_dir = direction
        return FrameAction(nes_action(direction), "north_path")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "dest_room": self.dest_room,
            "notes": list(self.notes),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "policy": (
                "occupancy to (120,93); RIGHT out of (96,141) gutter; "
                "LEFT+UP clip when BFS is not RIGHT"
            ),
        }
