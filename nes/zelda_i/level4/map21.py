"""Level 4 0x20 → 0x21 east (waypoints, no live BFS).

v1 KEY-UP leftover (120,205) south mouth. Occupancy from leftover PNGs:
H-bar y=144–159, water spine x=192–199, gold x>=200, water ends y=191.
v13 north-around reached (200,96); RIGHT there is the east wall (door is
y=141). v18 cardinal RIGHT at x=120 repeated v10 (north-door leftover
(120,95)). v19 clip stopped at v11 leftover (136,94) (DOWN/RIGHT yo-yo).
v20 restored v13 leftover (200,96); cardinal DOWN solid at (200,109).
v21 RIGHT+DOWN clip reached (208,133); PUSH y-slop 8 RIGHT into the wall.
v22: door is only y=141 — align y before RIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import should_swing_at
from zelda_i.level4.dungeon import (
    LEVEL4,
    RIGHT_20_STAND,
    ROOM_20_SPEC,
    ROOM_L4_MAP_21,
    ROOM_L4_WATER_NORTH_20,
)
from zelda_i.level4.occupancy import (
    ROOM_20_CLIP_BUDGET,
    ROOM_20_NORTH_XY,
    ROOM_20_SOUTH_XY,
    ROOM_20_SOUTH_Y_MAX,
    ROOM_20_WAYPOINTS,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "MAP_21_WAYPOINTS",
    "Clear20Phase",
    "Level4Clear20Controller",
    "Map21Phase",
    "Level4Map21Controller",
    "level4_map21_stages",
    "level4_map21_success",
    "make_map21_controller",
    "make_room_20_clear_controller",
]

CLIP_BUDGET = ROOM_20_CLIP_BUDGET
MAP_21_WAYPOINTS = ROOM_20_WAYPOINTS
MAP_21_PUSH = 280
_CLEAR20_PATROL_X: tuple[int, ...] = (48, 88, 120, 160, 200)


class Clear20Phase(Enum):
    FIGHT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Clear20Controller:
    """South-gold Vire clear (ignore 0x2b). v4–v6 knockback off the band."""

    max_frames: int = 20000
    phase: Clear20Phase = Clear20Phase.FIGHT
    frames: int = 0
    phase_frames: int = 0
    combat_frames: int = 0
    patrol_index: int = 0
    max_live_enemies: int = 0
    last_live_enemies: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Clear20Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Clear20Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _live(self, snap: ZeldaSnapshot) -> tuple:
        return ROOM_20_SPEC.live_enemies(snap)

    def _swing(self, direction: str, reason: str) -> FrameAction:
        if (self.combat_frames % 6) < 3:
            return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

    def _fight_step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.combat_frames += 1
        live = self._live(snap)
        self.last_live_enemies = len(live)
        self.max_live_enemies = max(self.max_live_enemies, len(live))
        if (
            not live
            and self.max_live_enemies >= ROOM_20_SPEC.expected_enemy_count
        ):
            self.success = True
            self._set_phase(Clear20Phase.DONE, "room_cleared")
            return FrameAction(nes_idle_action(), "done")
        y = int(snap.link_y)
        if y > ROOM_20_SOUTH_Y_MAX:
            return FrameAction(nes_action("UP"), "join_south_band")
        if y < ROOM_20_SOUTH_XY[1]:
            return FrameAction(nes_action("DOWN"), "join_south_band")
        if not live:
            return FrameAction(nes_idle_action(), "wait_spawn")
        nearest = min(
            live,
            key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
        )
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y
        above = nearest.y < ROOM_20_SOUTH_XY[1]
        if above and abs(dx) <= 28:
            return self._swing("UP", "slash_up_flyer")
        if abs(dx) > 8:
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "align_x_vire"
            )
        direction = "UP" if above or dy <= 0 else "DOWN"
        if should_swing_at(
            snap.link_x, snap.link_y, direction, (nearest,)
        ) or abs(dx) <= 16:
            return self._swing(direction, "engage")
        tx = _CLEAR20_PATROL_X[self.patrol_index % len(_CLEAR20_PATROL_X)]
        if abs(snap.link_x - tx) <= 6:
            self.patrol_index += 1
            tx = _CLEAR20_PATROL_X[self.patrol_index % len(_CLEAR20_PATROL_X)]
        return FrameAction(
            nes_action("RIGHT" if snap.link_x < tx else "LEFT"),
            "patrol_south",
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.phase is Clear20Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Clear20Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail(f"timeout_{snap.link_x}_{snap.link_y}")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_WATER_NORTH_20:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")
        if self.phase is Clear20Phase.FIGHT:
            return self._fight_step(snap)
        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "combat_frames": self.combat_frames,
            "max_live_enemies": self.max_live_enemies,
            "last_live_enemies": self.last_live_enemies,
            "notes": list(self.notes),
            "segment": "level4_clear_0x20",
            "patrol_x": list(_CLEAR20_PATROL_X),
        }


def make_room_20_clear_controller() -> Level4Clear20Controller:
    return Level4Clear20Controller()


class Map21Phase(Enum):
    PATH = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Map21Controller:
    """0x20 south leftover → y=96 north-around → DOWN x=200 → RIGHT 0x21."""

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
            if (
                abs(xy[0] - RIGHT_20_STAND[0]) <= 4
                and abs(xy[1] - RIGHT_20_STAND[1]) <= 4
            ):
                self._set_phase(Map21Phase.PUSH, "at_east_door")
            wps = MAP_21_WAYPOINTS
            i = self.path_index
            while self.phase is Map21Phase.PATH and i < len(wps):
                wx, wy = wps[i]
                y_tol = 1 if wy in (ROOM_20_NORTH_XY[1], RIGHT_20_STAND[1]) else 4
                if abs(xy[0] - wx) <= 4 and abs(xy[1] - wy) <= y_tol:
                    i += 1
                    self.path_index = i
                    self._sample(snap, f"waypoint_{i}")
                    continue
                break
            if self.phase is Map21Phase.PATH and i >= len(wps):
                self._set_phase(Map21Phase.PUSH, "at_east_door")
            elif self.phase is Map21Phase.PATH:
                gx, gy = wps[i]
                dx, dy = gx - xy[0], gy - xy[1]
                north_y = ROOM_20_NORTH_XY[1]
                # North strip eastbound: keep an east component. v19 DOWN
                # at (136,94) yo-yoed (v11). v10/v18 cardinal RIGHT at
                # x=120 sucked into the north door.
                if gy == north_y and xy[0] < gx:
                    if xy[1] < north_y or xy[0] < 136:
                        return FrameAction(
                            nes_action("RIGHT", "DOWN"), "join_map_clip"
                        )
                    if xy[1] > north_y:
                        return FrameAction(
                            nes_action("RIGHT", "UP"), "join_map_clip"
                        )
                    return FrameAction(nes_action("RIGHT"), "join_map_x")
                # v20 DOWN at (200,96) solid at y=109. Clip the wall/water
                # corner into the x=208 corridor (y≈128–175).
                if i >= 2 and xy[0] < 208 and xy[1] < 128:
                    return FrameAction(
                        nes_action("RIGHT", "DOWN"), "join_map_clip"
                    )
                hold_north = gy == north_y and dy != 0
                if hold_north or (
                    abs(dy) > 1 and (abs(dx) <= 8 or abs(dy) >= abs(dx))
                ):
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
    clear = make_room_20_clear_controller()
    path = make_map21_controller()
    return (
        ("level4_clear_0x20", clear, clear.max_frames),
        ("level4_map_0x21", path, path.max_frames),
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
