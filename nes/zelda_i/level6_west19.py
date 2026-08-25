"""Level 6 play 0x19 west after inland29 enter-stop.

Leftover (120,205) south mouth. Forward 0x18→0x19 was occupancy y=141
RIGHT (PNG-black shutter, no key). Return LEFT at y=141. OccupancyWalker
first; halt at first miss (no path → stand). Do not KEY-UP 0x09. Skip Map.
Do not poke bow/arrows/doors/keys. Isolated BFS banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_GLEEOK_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
)
from zelda_i.level6_path import NORTH_BAND_Y
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "WEST19_MAX_FRAMES",
    "WEST_DOOR_X",
    "WEST_DOOR_Y",
    "WEST_SPAWN_XMIN",
    "Level6West19Controller",
    "level6_west19_stages",
    "level6_west19_success",
    "make_west19_controller",
]

WEST_DOOR_X = 32
WEST_DOOR_Y = 141
WEST_DOOR_X_TOL = 4
WEST_DOOR_Y_TOL = 4
WEST_SPAWN_XMIN = 16
WEST19_MAX_FRAMES = 4000
WEST19_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


def _new_walker() -> OccupancyWalker:
    # ymin=NORTH_BAND_Y keeps BFS off the KEY-UP north door (y≈93).
    return OccupancyWalker(
        grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN, ymin=NORTH_BAND_Y)
    )


@dataclass
class Level6West19Controller:
    """Y-align 141, occupancy to (32,141), LEFT. Never KEY-UP 0x09."""

    spec_id: str = "level6_west_0x19"
    room: int = LEVEL6_MAP_ROOM
    dest: int = LEVEL6_GLEEOK_ROOM
    goal: tuple[int, int] = (WEST_DOOR_X, WEST_DOOR_Y)
    max_frames: int = WEST19_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=_new_walker)

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
        if force or self.frames <= 2 or self.frames % WEST19_SAMPLE_PERIOD == 0:
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
                f"key_spent_19_to_{snap.screen:02x}_{self.keys}->{int(snap.keys)}"
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
            if snap.screen == LEVEL6_DARK_29_ROOM:
                return self._fail(
                    snap,
                    f"backtrack_29_{snap.link_x}_{snap.link_y}",
                )
            if snap.screen != self.dest:
                return self._fail(
                    snap,
                    f"wrong_room_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
                )
            return self._mark_success(snap)
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            return FrameAction(nes_action("LEFT"), "west_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("LEFT"), "west_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        # Do not KEY-UP north to 0x09 (that door is KEY).
        if xy[1] <= NORTH_BAND_Y:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "north_key_halt")
            )

        gx, gy = self.goal
        if (
            snap.link_x <= gx + WEST_DOOR_X_TOL
            and abs(snap.link_y - gy) <= WEST_DOOR_Y_TOL
        ):
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("LEFT"), "west_push"))

        # Do not LEFT at leftover y=205 into the south mouth wall.
        if abs(snap.link_y - gy) > WEST_DOOR_Y_TOL:
            dest = (int(snap.link_x), gy)
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
                snap, FrameAction(nes_idle_action(), "occupancy_stand")
            )
        if direction == "UP" and xy[1] <= gy:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "north_key_halt")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "west_path")
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "y=141 first, occupancy to (32,141), LEFT; halt y<=109; "
                "no KEY-UP 0x09; skip Map"
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


def make_west19_controller() -> Level6West19Controller:
    """Occupancy west of 0x19. Do not poke bow/arrows/doors/keys. Skip Map."""
    return Level6West19Controller()


def level6_west19_stages():
    """Play 0x19 leftover (120,205) → occupancy LEFT y=141 → play 0x18."""
    ctl = make_west19_controller()
    return (
        ("level6_west_0x19", ctl, ctl.max_frames),
    )


def level6_west19_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x18 with ADDR_ROD. Enter-stop; Gleeok already dead."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_GLEEOK_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
