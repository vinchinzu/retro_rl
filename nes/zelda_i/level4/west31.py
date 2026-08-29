"""Level 4 post-ladder 0x32 → 0x31 (waypoints around pushed block, no BFS).

v2 leftover (192,189) SE stairs. Pushed 0x68 sits on y=141; south corridor
LEFT then west-aisle UP to the west door. Isolated WEST_31_SAMPLE_PATH is
hypothesis only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level4.dungeon import LEVEL4, ROOM_L4_EAST_31, ROOM_L4_EAST_32
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "WEST_32_CLIP_BUDGET",
    "WEST_32_WAYPOINTS",
    "West31Phase",
    "Level4West31Controller",
    "level4_west31_stages",
    "level4_west31_success",
    "make_west31_controller",
]

WEST_32_CLIP_BUDGET = 96
WEST_32_SOUTH_XY = (48, 189)
WEST_32_WEST_XY = (48, 141)
WEST_32_DOOR_XY = (16, 141)
WEST_32_WAYPOINTS: tuple[tuple[int, int], ...] = (
    WEST_32_SOUTH_XY,
    WEST_32_WEST_XY,
    WEST_32_DOOR_XY,
)
WEST_32_PUSH = 280


class West31Phase(Enum):
    PATH = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4West31Controller:
    """0x32 SE leftover → south-U around 0x68 → LEFT into 0x31. No BFS."""

    max_frames: int = 4000
    phase: West31Phase = West31Phase.PATH
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0

    def _set_phase(self, phase: West31Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self._stall = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(West31Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _sample(self, snap: ZeldaSnapshot, reason: str) -> None:
        sample = {
            "frame": self.frames,
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "phase": self.phase.name,
            "path_index": self.path_index,
            "reason": reason,
            "stall": self._stall,
        }
        if (
            not self.samples
            or self.samples[-1]["reason"] != reason
            or self.frames - self.samples[-1]["frame"] >= 250
        ):
            self.samples.append(sample)

    def _entered_31(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_EAST_31
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def _path_dir(self, xy: tuple[int, int]) -> str | None:
        wps = WEST_32_WAYPOINTS
        i = self.path_index
        while i < len(wps):
            wx, wy = wps[i]
            if abs(xy[0] - wx) <= 4 and abs(xy[1] - wy) <= 4:
                i += 1
                self.path_index = i
                continue
            break
        if i >= len(wps):
            return None
        gx, gy = wps[i]
        # Stay on the south band before LEFT (y=141 is the pushed 0x68).
        if i == 0 and abs(xy[1] - WEST_32_SOUTH_XY[1]) > 2:
            return "DOWN" if xy[1] < WEST_32_SOUTH_XY[1] else "UP"
        dx, dy = gx - xy[0], gy - xy[1]
        if abs(dy) > 1 and (abs(dx) <= 8 or abs(dy) >= abs(dx)):
            return "DOWN" if dy > 0 else "UP"
        if dx != 0:
            return "RIGHT" if dx > 0 else "LEFT"
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if self._last_xy == xy:
            self._stall += 1
        else:
            self._stall = 0
            self._last_xy = xy

        if self.phase is West31Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is West31Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            self._sample(snap, "timeout")
            return self._fail(f"timeout_{xy[0]}_{xy[1]}")
        if self._entered_31(snap):
            if snap.ladder <= 0:
                return self._fail("west_no_ladder")
            self._sample(snap, "entered_0x31")
            self.success = True
            self._set_phase(West31Phase.DONE, "entered_0x31")
            return FrameAction(nes_idle_action(), "done")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("LEFT"), "scroll_left")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_EAST_32:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.phase is West31Phase.PATH:
            if self._stall >= WEST_32_CLIP_BUDGET:
                self._sample(snap, "west_solid")
                return self._fail(f"west_solid_{xy[0]}_{xy[1]}")
            direction = self._path_dir(xy)
            if direction is None:
                self._set_phase(West31Phase.PUSH, "at_west_door")
            else:
                return FrameAction(nes_action(direction), "join_west")

        if self.phase is West31Phase.PUSH:
            if abs(xy[1] - WEST_32_DOOR_XY[1]) > 8:
                return FrameAction(
                    nes_action("DOWN" if xy[1] < WEST_32_DOOR_XY[1] else "UP"),
                    "west_align_y",
                )
            if self.phase_frames >= WEST_32_PUSH:
                self._sample(snap, "push_left_timeout")
                return self._fail(f"push_left_timeout_{xy[0]}_{xy[1]}")
            if self._stall >= WEST_32_CLIP_BUDGET:
                self._sample(snap, "push_solid")
                return self._fail(f"push_solid_{xy[0]}_{xy[1]}")
            return FrameAction(nes_action("LEFT"), "west_push_left")

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "path_index": self.path_index,
            "notes": list(self.notes),
            "segment": "level4_west_0x31",
            "waypoints": [list(p) for p in WEST_32_WAYPOINTS],
            "samples": list(self.samples),
        }


def make_west31_controller() -> Level4West31Controller:
    """0x32 SE stairs leftover → 0x31 play. Ladder already set."""
    return Level4West31Controller()


def level4_west31_stages():
    ctl = make_west31_controller()
    return (
        ("level4_west_0x31", ctl, ctl.max_frames),
    )


def level4_west31_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x31 with ADDR_LADDER. Do not require KEY-UP."""
    return (
        snap.level == LEVEL4
        and snap.ladder > 0
        and snap.screen == ROOM_L4_EAST_31
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )
