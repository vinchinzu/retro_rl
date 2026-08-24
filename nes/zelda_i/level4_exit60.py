"""Level 4 mode-9 0x60 → 0x32 play (coordinate waypoints, no live BFS).

v34 leftover (136,141) on the pedestal. Reverse of the inbound east dock:
RIGHT at y=141 to x=175, DOWN the dock, LEFT along y=189 (never x>=176),
UP the west aisle onto spawn stairs. Item freeze then around Keese.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon_ids import KEESE_OBJECT_TYPE
from zelda_i.level4_dungeon import LEVEL4, ROOM_L4_EAST_32, ROOM_L4_STEPLADDER
from zelda_i.level4_occupancy import (
    ROOM_60_CAUSWAY_XY,
    ROOM_60_CLIP_BUDGET,
    ROOM_60_DOCK_MOUTH_X_MIN,
    ROOM_60_EXIT_WAYPOINTS,
    ROOM_60_EXIT_X,
    ROOM_60_SPAWN_XY,
)
from zelda_i.level4_stepladder import POST_LADDER_ITEM_SETTLE, STAIRS_32_PUSH_FRAMES
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "Exit60Phase",
    "Level4Exit60Controller",
    "level4_exit60_stages",
    "level4_exit60_success",
    "make_exit60_controller",
]


class Exit60Phase(Enum):
    SETTLE = auto()
    PATH = auto()
    ENTER_STAIRS = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Exit60Controller:
    """Pedestal leftover → reverse dock waypoints → 0x32 play. No BFS."""

    max_frames: int = 12000
    phase: Exit60Phase = Exit60Phase.SETTLE
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0

    def _set_phase(self, phase: Exit60Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self._stall = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Exit60Phase.FAILED, note)
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

    def _play_32(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_EAST_32
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def _mark_done(self, note: str) -> FrameAction:
        self.success = True
        self._set_phase(Exit60Phase.DONE, note)
        return FrameAction(nes_idle_action(), "done")

    def _with_keese(self, direction: str, reason: str, snap: ZeldaSnapshot) -> FrameAction:
        keese = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id == KEESE_OBJECT_TYPE
        ]
        if keese:
            nearest = min(
                keese,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            if abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y) <= 24:
                return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

    def _path_dir(self, xy: tuple[int, int]) -> str | None:
        wps = ROOM_60_EXIT_WAYPOINTS
        i = self.path_index
        while i < len(wps):
            wx, wy = wps[i]
            if i == 0:
                arrived = xy[0] >= ROOM_60_DOCK_MOUTH_X_MIN and abs(xy[1] - wy) <= 2
            else:
                arrived = abs(xy[0] - wx) <= 4 and abs(xy[1] - wy) <= 4
            if not arrived:
                break
            i += 1
            self.path_index = i
        if i >= len(wps):
            return None
        gx, gy = wps[i]
        # v3/v5: x>=176 y>=189 dumps the SE stairs. Stay on the dock column.
        if xy[0] >= ROOM_60_EXIT_X and xy[1] >= 189:
            return "LEFT"
        if i == 0 and abs(xy[1] - ROOM_60_CAUSWAY_XY[1]) > 1:
            return "UP" if xy[1] > ROOM_60_CAUSWAY_XY[1] else "DOWN"
        # v1 leftover (176,173): LEFT back to x=175 mid-dock is solid.
        # Inbound UP at x=175/176; keep DOWN until y=189.
        if i == 1 and xy[0] >= 160 and abs(xy[1] - gy) > 2:
            return "DOWN" if xy[1] < gy else "UP"
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

        if self.phase is Exit60Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Exit60Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            self._sample(snap, "timeout")
            return self._fail(f"timeout_{xy[0]}_{xy[1]}")

        if self._play_32(snap):
            if snap.ladder <= 0:
                return self._fail("exit_no_ladder")
            self._sample(snap, "entered_0x32")
            return self._mark_done("entered_0x32")

        if snap.mode in (4, 6, 7, 10) or snap.transitioning:
            return FrameAction(nes_action("UP"), "scroll")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.screen != ROOM_L4_STEPLADDER and snap.mode != 9:
            return self._fail(f"wrong_room_0x{snap.screen:02x}_m{snap.mode}")

        if self.phase is Exit60Phase.SETTLE:
            if self.phase_frames <= POST_LADDER_ITEM_SETTLE:
                return FrameAction(nes_idle_action(), "item_freeze")
            self._set_phase(Exit60Phase.PATH, "item_thaw")
            self.path_index = 0

        if self.phase is Exit60Phase.PATH:
            if (
                self._stall >= ROOM_60_CLIP_BUDGET
            ):
                self._sample(snap, "exit_solid")
                return self._fail(f"exit_solid_{xy[0]}_{xy[1]}")
            direction = self._path_dir(xy)
            if direction is None:
                self._set_phase(Exit60Phase.ENTER_STAIRS, "at_spawn_stairs")
            else:
                return self._with_keese(direction, "join_exit", snap)

        if self.phase is Exit60Phase.ENTER_STAIRS:
            if abs(xy[0] - ROOM_60_SPAWN_XY[0]) > 8:
                return self._with_keese(
                    "LEFT" if xy[0] > ROOM_60_SPAWN_XY[0] else "RIGHT",
                    "join_exit",
                    snap,
                )
            if self.phase_frames >= STAIRS_32_PUSH_FRAMES:
                self._sample(snap, "stairs_timeout")
                return self._fail(f"stairs_timeout_{xy[0]}_{xy[1]}")
            if self._stall >= ROOM_60_CLIP_BUDGET:
                self._sample(snap, "stairs_solid")
                return self._fail(f"stairs_solid_{xy[0]}_{xy[1]}")
            return self._with_keese("UP", "enter_stairs_up", snap)

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "path_index": self.path_index,
            "notes": list(self.notes),
            "segment": "level4_exit_0x60",
            "waypoints": [list(p) for p in ROOM_60_EXIT_WAYPOINTS],
            "settle": POST_LADDER_ITEM_SETTLE,
            "samples": list(self.samples),
        }


def make_exit60_controller() -> Level4Exit60Controller:
    """Mode-9 0x60 pedestal → 0x32 play. Ladder already set."""
    return Level4Exit60Controller()


def level4_exit60_stages():
    """ADDR_LADDER leftover (136,141) → reverse dock → 0x32 play."""
    ctl = make_exit60_controller()
    return (
        ("level4_exit_0x60", ctl, ctl.max_frames),
    )


def level4_exit60_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x32 with ADDR_LADDER. Do not require west door."""
    return (
        snap.level == LEVEL4
        and snap.ladder > 0
        and snap.screen == ROOM_L4_EAST_32
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )
