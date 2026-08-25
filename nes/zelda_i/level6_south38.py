"""Level 6 play 0x38 south after clear28-south enter-stop.

Leftover (120,93) north mouth. Forward 0x48→0x38 was occupancy run-UP
through traps (no clear). Going DOWN, skip traps the same way.
OccupancyWalker to (120,189) then DOWN. Clip LEFT+DOWN one frame after
a live miss on the north face (y=93..100), then occupancy replan. v1
held LEFT+DOWN at y=93 and slid west to x=32 (DOWN no-op, tile 119).
Do not reclear 0x38 unless a kill-door dates (south is visually open).
Do not KEY-UP 0x09. Do not CheckWarp. Do not poke bow/arrows/doors/keys.
Isolated BFS banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
)
from zelda_i.level6_path import NORTH_BAND_Y
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "SOUTH38_MAX_FRAMES",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6South38Controller",
    "level6_south38_stages",
    "level6_south38_success",
    "make_south38_controller",
]

SOUTH_DOOR_X = 120
SOUTH_DOOR_Y = 189
SOUTH_BAND_Y = 181
SOUTH_DOOR_TOL = 4
# Leftover is already (120,93). Occupancy DOWN first; clip only after miss.
NORTH_FACE_Y = 93
CLIP_PAST_Y = 101
SOUTH38_MAX_FRAMES = 6000
SOUTH38_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


@dataclass
class Level6South38Controller:
    """Occupancy to (120,189), DOWN. Skip traps. Never reclear. Never KEY-UP 0x09."""

    spec_id: str = "level6_south_0x38"
    room: int = LEVEL6_WIZZROBE_38_ROOM
    dest: int = LEVEL6_TRAPS_ROOM
    goal: tuple[int, int] = (SOUTH_DOOR_X, SOUTH_DOOR_Y)
    max_frames: int = SOUTH38_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    clip_after_miss: bool = False

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
            "rod": self._rod(snap),
            "bow": self._bow(snap),
            "arrows": self._arrows(snap),
            "tile": int(snap.colliding_tile),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
        }
        if force or self.frames <= 2 or self.frames % SOUTH38_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "rod": self._rod(snap),
                    "keys": int(snap.keys),
                    "map": int(snap.map),
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _mark_success(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.keys >= 0 and int(snap.keys) < self.keys:
            self.notes.append(
                f"key_spent_38_to_{snap.screen:02x}_{self.keys}->{int(snap.keys)}"
            )
        self.keys = int(snap.keys)
        note = (
            f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_rod={self._rod(snap)}_tf={snap.triforce:02x}"
            f"_keys={int(snap.keys)}"
        )
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), f"arrived_{snap.screen:02x}"), force=True
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.keys < 0:
            self.keys = int(snap.keys)
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_rod={self._rod(snap)}_keys={int(snap.keys)}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if snap.mode == CELLAR_MODE:
            return self._fail(
                snap,
                f"warped_cellar_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.level == 0:
            return self._fail(
                snap, f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}_{snap.screen:02x}")
        if (
            snap.screen != self.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and self._rod(snap) != 0
        ):
            if snap.screen == LEVEL6_ROD_WIZZ_ROOM:
                return self._fail(
                    snap,
                    f"key_up_09_{snap.link_x}_{snap.link_y}_keys={int(snap.keys)}",
                )
            if snap.screen == LEVEL6_WIZZROBE_28_ROOM:
                return self._fail(
                    snap,
                    f"backtrack_28_{snap.link_x}_{snap.link_y}",
                )
            if snap.screen != self.dest:
                return self._fail(
                    snap,
                    f"wrong_room_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
                )
            return self._mark_success(snap)
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            return FrameAction(nes_action("DOWN"), "south_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("DOWN"), "south_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        just_missed = self.walker.misses > misses_before
        if just_missed and (self.walker.misses <= 8 or self.frames % 60 == 0):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        gx = self.goal[0]
        if xy[1] >= SOUTH_BAND_Y:
            self.walker.last_dir = None
            self.clip_after_miss = False
            if abs(xy[0] - gx) > SOUTH_DOOR_TOL:
                btn = "LEFT" if xy[0] > gx else "RIGHT"
                return self._emit(
                    snap, FrameAction(nes_action(btn), "south_align")
                )
            return self._emit(snap, FrameAction(nes_action("DOWN"), "south_push"))

        # Occupancy DOWN first at leftover (120,93). One clip frame after a
        # miss, then occupancy replan. v1 held LEFT+DOWN at y=93 (DOWN no-op)
        # and slid west to x=32 occupancy_stand.
        if just_missed and NORTH_FACE_Y <= xy[1] < CLIP_PAST_Y:
            self.clip_after_miss = True
        if self.clip_after_miss and NORTH_FACE_Y <= xy[1] < CLIP_PAST_Y:
            self.clip_after_miss = False
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "DOWN"), "diamond_clip")
            )
        if xy[1] >= CLIP_PAST_Y:
            self.clip_after_miss = False

        dest = self.goal
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction == "UP" and xy[1] <= NORTH_BAND_Y:
            # v2: occupancy UP at leftover mouth halted; DOWN boxed y=96.
            # Peel off the door column, then occupancy south. Do not UP to 0x28.
            self.walker.last_dir = None
            btn = "RIGHT" if xy[0] <= gx else "LEFT"
            return self._emit(
                snap, FrameAction(nes_action(btn), "north_mouth_peel")
            )
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "occupancy_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "south_path")
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "occupancy DOWN from leftover (120,93) through traps (no "
                "reclear); one-shot LEFT+DOWN after a north-face miss "
                "(v1 held clip at y=93, slid west x=32); RIGHT peel when "
                "occupancy UP at north mouth (v2 halt); skip 0x38 left-0x68 "
                "push; no KEY-UP 0x09; no CheckWarp"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "dest": self.dest,
            "goal": self.goal,
            "keys": self.keys,
        }


def make_south38_controller() -> Level6South38Controller:
    """Occupancy south of 0x38. Do not reclear. Do not poke bow/arrows/doors/keys."""
    return Level6South38Controller()


def level6_south38_stages():
    """Play 0x38 leftover (120,93) → occupancy DOWN (120,189) → play 0x48."""
    ctl = make_south38_controller()
    return (
        ("level6_south_0x38", ctl, ctl.max_frames),
    )


def level6_south38_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x48 with ADDR_ROD. Enter-stop; traps may be live."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_TRAPS_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
