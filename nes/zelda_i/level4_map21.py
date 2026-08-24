"""Level 4 0x20 → 0x21 east (waypoints, no live BFS).

v1 KEY-UP leftover (120,205) south mouth. v1/v2 RIGHT at x=120 y=141/133
is water. v3: RIGHT along south then UP the east column. Isolated
state-BFS is not this tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level4_dungeon import (
    LEVEL4,
    RIGHT_20_STAND,
    ROOM_L4_MAP_21,
    ROOM_L4_WATER_NORTH_20,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "Map21Phase",
    "Level4Map21Controller",
    "level4_map21_stages",
    "level4_map21_success",
    "make_map21_controller",
]

CLIP_BUDGET = 96
# v1 (120,141) RIGHT solid. v2 (120,133) RIGHT solid. South-around the H.
MAP_21_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (192, 205),
    (192, 141),
    RIGHT_20_STAND,
)
MAP_21_PUSH = 280


class Map21Phase(Enum):
    PATH = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Map21Controller:
    """0x20 south leftover → y=141 ladder band → RIGHT 0x21. No BFS."""

    max_frames: int = 6000
    phase: Map21Phase = Map21Phase.PATH
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0

    def _set_phase(self, phase: Map21Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self._stall = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Map21Phase.FAILED, note)
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

    def _entered_21(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_MAP_21
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if self._last_xy == xy:
            self._stall += 1
        else:
            self._stall = 0
            self._last_xy = xy

        if self.phase is Map21Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Map21Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            self._sample(snap, "timeout")
            return self._fail(f"timeout_{xy[0]}_{xy[1]}")
        if self._entered_21(snap):
            if snap.ladder <= 0:
                return self._fail("map_no_ladder")
            self.success = True
            self._set_phase(Map21Phase.DONE, "entered_0x21")
            return FrameAction(nes_idle_action(), "done")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll_right")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_WATER_NORTH_20:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.phase is Map21Phase.PATH:
            if self._stall >= CLIP_BUDGET:
                self._sample(snap, "map_solid")
                return self._fail(f"map_solid_{xy[0]}_{xy[1]}")
            wps = MAP_21_WAYPOINTS
            i = self.path_index
            while i < len(wps):
                wx, wy = wps[i]
                if abs(xy[0] - wx) <= 4 and abs(xy[1] - wy) <= 4:
                    i += 1
                    self.path_index = i
                    continue
                break
            if i >= len(wps):
                self._set_phase(Map21Phase.PUSH, "at_east_door")
            else:
                gx, gy = wps[i]
                # y-first off the south mouth onto the ladder band.
                if i == 0 and abs(xy[1] - gy) > 2:
                    return FrameAction(
                        nes_action("UP" if xy[1] > gy else "DOWN"), "join_map_y"
                    )
                dx, dy = gx - xy[0], gy - xy[1]
                if abs(dy) > 1 and (abs(dx) <= 8 or abs(dy) >= abs(dx)):
                    return FrameAction(
                        nes_action("DOWN" if dy > 0 else "UP"), "join_map_y"
                    )
                if dx != 0:
                    return FrameAction(
                        nes_action("RIGHT" if dx > 0 else "LEFT"), "join_map_x"
                    )
                return FrameAction(nes_idle_action(), "map_idle")

        if self.phase is Map21Phase.PUSH:
            if abs(xy[1] - RIGHT_20_STAND[1]) > 8:
                return FrameAction(
                    nes_action("DOWN" if xy[1] < RIGHT_20_STAND[1] else "UP"),
                    "map_align_y",
                )
            if self.phase_frames >= MAP_21_PUSH:
                self._sample(snap, "push_timeout")
                return self._fail(f"push_timeout_{xy[0]}_{xy[1]}")
            if self._stall >= CLIP_BUDGET:
                self._sample(snap, "push_solid")
                return self._fail(f"push_solid_{xy[0]}_{xy[1]}")
            return FrameAction(nes_action("RIGHT"), "map_push_right")
        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "path_index": self.path_index,
            "notes": list(self.notes),
            "segment": "level4_map_0x21",
            "waypoints": [list(p) for p in MAP_21_WAYPOINTS],
            "samples": list(self.samples),
        }


def make_map21_controller() -> Level4Map21Controller:
    return Level4Map21Controller()


def level4_map21_stages():
    ctl = make_map21_controller()
    return (
        ("level4_map_0x21", ctl, ctl.max_frames),
    )


def level4_map21_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x21 with ADDR_LADDER. Do not require map pickup."""
    return (
        snap.level == LEVEL4
        and snap.ladder > 0
        and snap.screen == ROOM_L4_MAP_21
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )
