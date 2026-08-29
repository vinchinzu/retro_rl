"""Level 4 post-ladder 0x31 maze west → 0x30 KEY-UP → 0x20 (no live BFS).

v1 west leftover (208,141) east door. Reverse of the verified east U
(192,141)→(192,173)→(160,173) then UP the east column and LEFT+UP clip
onto the inland north strip, then west alcove to the 0x30 door.
KEY-UP aligns x=120 and holds UP (ladder water + key door).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level4.dungeon import (
    KEY_30_NORTH_X,
    LEVEL4,
    ROOM_L4_EAST_31,
    ROOM_L4_NORTH_30,
    ROOM_L4_WATER_NORTH_20,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "KeyUp20Phase",
    "Level4KeyUp20Controller",
    "Level4Maze31WestController",
    "MAZE_31_WEST_WAYPOINTS",
    "Maze31WestPhase",
    "level4_keyup20_stages",
    "level4_keyup20_success",
    "make_keyup20_controller",
    "make_maze_31_west_controller",
]

CLIP_BUDGET = 96
MAZE_31_WEST_EAST_U: tuple[tuple[int, int], ...] = (
    (192, 141),
    (192, 173),
    (160, 173),
    (160, 125),
)
MAZE_31_WEST_INLAND: tuple[tuple[int, int], ...] = (
    (80, 109),
    (48, 109),
    (48, 141),
    (16, 141),
)
MAZE_31_WEST_WAYPOINTS: tuple[tuple[int, int], ...] = (
    MAZE_31_WEST_EAST_U + MAZE_31_WEST_INLAND
)
MAZE_31_NORTH_STRIP_Y = 113
MAZE_31_WEST_PUSH = 280
KEY_UP_PUSH = 450


class Maze31WestPhase(Enum):
    EAST_U = auto()
    CLIP = auto()
    INLAND = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


def _advance(wps: tuple[tuple[int, int], ...], i: int, xy: tuple[int, int]) -> int:
    while i < len(wps):
        wx, wy = wps[i]
        if abs(xy[0] - wx) <= 4 and abs(xy[1] - wy) <= 4:
            i += 1
            continue
        break
    return i


def _dir_to(xy: tuple[int, int], goal: tuple[int, int]) -> str | None:
    gx, gy = goal
    dx, dy = gx - xy[0], gy - xy[1]
    if abs(dy) > 1 and (abs(dx) <= 8 or abs(dy) >= abs(dx)):
        return "DOWN" if dy > 0 else "UP"
    if dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    return None


@dataclass
class Level4Maze31WestController:
    """0x31 east-door leftover → reverse east U + inland → LEFT 0x30."""

    max_frames: int = 6000
    phase: Maze31WestPhase = Maze31WestPhase.EAST_U
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0

    def _set_phase(self, phase: Maze31WestPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self._stall = 0
            self.path_index = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Maze31WestPhase.FAILED, note)
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

    def _entered_30(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_NORTH_30
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def _thread(
        self,
        xy: tuple[int, int],
        snap: ZeldaSnapshot,
        wps: tuple[tuple[int, int], ...],
        next_phase: Maze31WestPhase,
        next_note: str,
    ) -> FrameAction:
        if self._stall >= CLIP_BUDGET:
            self._sample(snap, "west_solid")
            return self._fail(f"west_solid_{xy[0]}_{xy[1]}")
        self.path_index = _advance(wps, self.path_index, xy)
        if self.path_index >= len(wps):
            self._set_phase(next_phase, next_note)
            return FrameAction(nes_idle_action(), next_note)
        direction = _dir_to(xy, wps[self.path_index])
        if direction is None:
            self.path_index += 1
            return FrameAction(nes_idle_action(), "wp_idle")
        return FrameAction(nes_action(direction), "join_maze_west")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if self._last_xy == xy:
            self._stall += 1
        else:
            self._stall = 0
            self._last_xy = xy

        if self.phase is Maze31WestPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Maze31WestPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            self._sample(snap, "timeout")
            return self._fail(f"timeout_{xy[0]}_{xy[1]}")
        if self._entered_30(snap):
            if snap.ladder <= 0:
                return self._fail("west_no_ladder")
            self._sample(snap, "entered_0x30")
            self.success = True
            self._set_phase(Maze31WestPhase.DONE, "entered_0x30")
            return FrameAction(nes_idle_action(), "done")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("LEFT"), "scroll_left")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_EAST_31:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.phase is Maze31WestPhase.EAST_U:
            return self._thread(
                xy, snap, MAZE_31_WEST_EAST_U, Maze31WestPhase.CLIP, "at_east_col"
            )
        if self.phase is Maze31WestPhase.CLIP:
            if xy[1] <= MAZE_31_NORTH_STRIP_Y:
                self._sample(snap, "north_strip")
                self._set_phase(Maze31WestPhase.INLAND, "north_strip")
            elif self._stall >= CLIP_BUDGET:
                self._sample(snap, "clip_solid")
                return self._fail(f"clip_solid_{xy[0]}_{xy[1]}")
            else:
                return FrameAction(nes_action("LEFT", "UP"), "maze31_west_clip")
        if self.phase is Maze31WestPhase.INLAND:
            return self._thread(
                xy, snap, MAZE_31_WEST_INLAND, Maze31WestPhase.PUSH, "at_west_door"
            )
        if self.phase is Maze31WestPhase.PUSH:
            if abs(xy[1] - 141) > 8:
                return FrameAction(
                    nes_action("DOWN" if xy[1] < 141 else "UP"), "west_align_y"
                )
            if self.phase_frames >= MAZE_31_WEST_PUSH:
                return self._fail(f"push_left_timeout_{xy[0]}_{xy[1]}")
            if self._stall >= CLIP_BUDGET:
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
            "segment": "level4_maze_west_0x30",
            "waypoints": [list(p) for p in MAZE_31_WEST_WAYPOINTS],
            "samples": list(self.samples),
        }


class KeyUp20Phase(Enum):
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4KeyUp20Controller:
    """0x30 leftover → x=120 KEY-UP into 0x20 (ladder water)."""

    max_frames: int = 4000
    phase: KeyUp20Phase = KeyUp20Phase.ALIGN
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    keys_before: int | None = None
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: KeyUp20Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(KeyUp20Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _entered_20(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_WATER_NORTH_20
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.phase is KeyUp20Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is KeyUp20Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail(f"timeout_{snap.link_x}_{snap.link_y}")
        if self._entered_20(snap):
            if snap.ladder <= 0:
                return self._fail("keyup_no_ladder")
            self.success = True
            self._set_phase(KeyUp20Phase.DONE, "entered_0x20")
            return FrameAction(nes_idle_action(), "done")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll_up")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_NORTH_30:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")
        if self.keys_before is None:
            self.keys_before = snap.keys
        if snap.keys < 1 and self.keys_before < 1:
            return self._fail("no_keys")
        if abs(snap.link_x - KEY_30_NORTH_X) > 4:
            self._set_phase(KeyUp20Phase.ALIGN, "align_x")
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < KEY_30_NORTH_X else "LEFT"),
                "align_x",
            )
        self._set_phase(KeyUp20Phase.PUSH, "push_key_up")
        return FrameAction(nes_action("UP"), "push_key_up")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_key_up_0x20",
            "key_x": KEY_30_NORTH_X,
            "keys_before": self.keys_before,
        }


def make_maze_31_west_controller() -> Level4Maze31WestController:
    return Level4Maze31WestController()


def make_keyup20_controller() -> Level4KeyUp20Controller:
    return Level4KeyUp20Controller()


def level4_keyup20_stages():
    west = make_maze_31_west_controller()
    keyup = make_keyup20_controller()
    return (
        ("level4_maze_west_0x30", west, west.max_frames),
        ("level4_key_up_0x20", keyup, keyup.max_frames),
    )


def level4_keyup20_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x20 with ADDR_LADDER. Vires may still be live."""
    return (
        snap.level == LEVEL4
        and snap.ladder > 0
        and snap.screen == ROOM_L4_WATER_NORTH_20
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )
