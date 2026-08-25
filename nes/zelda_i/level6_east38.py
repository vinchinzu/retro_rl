"""Level 6 play 0x38 east after clear28-south leftover.

Leftover (120,93) north mouth. bomb38-south v3 occupancy to east
(208,141) boxed (128,141) tile 118. OccupancyWalker toward east mouth
y=141; RIGHT+DOWN clip on the dated x=128 column (v1 RIGHT+UP
boxed (128,133); v2 RIGHT+DOWN y-box ±4 boxed (128,149) SE statue
face). Halt at the next occupancy miss. Isolated BFS banned. Ignore 0x2B.
Do not bomb. Do not push 0x68. Do not hold DOWN on tile 170.
Dest is RAM (do not invent). Do not KEY-UP 0x09. Do not CheckWarp.
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
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "CLIP_BOX",
    "EAST38_MAX_FRAMES",
    "EAST_DOOR",
    "Level6East38Controller",
    "level6_east38_stages",
    "level6_east38_success",
    "make_east38_controller",
]

EAST_DOOR = (208, 141)
CLIP_BOX = (128, 141)
EAST_DOOR_TOL = 4
CLIP_TOL = 4
SOUTH_DOOR_Y = 189
WEST_SPAWN_XMIN = 16
EAST38_MAX_FRAMES = 4000
EAST38_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


def _new_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))


@dataclass
class Level6East38Controller:
    """Occupancy to (208,141); RIGHT+DOWN on dated x=128 column. Dest is RAM."""

    spec_id: str = "level6_east_0x38"
    room: int = LEVEL6_WIZZROBE_38_ROOM
    goal: tuple[int, int] = EAST_DOOR
    max_frames: int = EAST38_MAX_FRAMES
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

    def _at_clip(self, xy: tuple[int, int]) -> bool:
        # v2 y-box ±4 left the column at the SE statue face (128,149).
        cx, _cy = CLIP_BOX
        return abs(xy[0] - cx) <= CLIP_TOL

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
        if force or self.frames <= 2 or self.frames % EAST38_SAMPLE_PERIOD == 0:
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
                    "bombs": int(snap.bombs),
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
            f"_keys={int(snap.keys)}_bombs={int(snap.bombs)}"
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
                    f"_bombs={int(snap.bombs)}"
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
            if snap.screen == LEVEL6_TRAPS_ROOM:
                return self._fail(
                    snap,
                    f"south_dated_48_{snap.link_x}_{snap.link_y}",
                )
            return self._mark_success(snap)
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            return FrameAction(nes_action("RIGHT"), "east_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("RIGHT"), "east_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        if xy[1] >= SOUTH_DOOR_Y:
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("UP"), "mouth_back"))

        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        just_missed = self.walker.misses > misses_before
        if just_missed and (self.walker.misses <= 8 or self.frames % 60 == 0):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        gx, gy = self.goal
        if snap.link_x >= gx - EAST_DOOR_TOL and abs(snap.link_y - gy) <= EAST_DOOR_TOL:
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("RIGHT"), "east_push"))

        # bomb38-south v3 occupancy halt at (128,141) tile 118.
        if self._at_clip(xy):
            self.walker.last_dir = None
            self.walker.path = None
            note = "east_clip"
            if note not in self.notes:
                self.notes.append(
                    f"clip_{xy[0]}_{xy[1]}_tile={int(snap.colliding_tile)}"
                )
            return self._emit(
                snap, FrameAction(nes_action("RIGHT", "DOWN"), "east_clip")
            )

        dest = self.goal
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            return self._fail(
                snap, f"occupancy_halt_{xy[0]}_{xy[1]}"
            )
        return self._emit(snap, FrameAction(nes_action(direction), "to_east"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "occupancy to east (208,141) y=141; RIGHT+DOWN clip on dated "
                "x=128 column (v1 UP boxed (128,133); v2 y-box boxed "
                "(128,149)); halt next occupancy miss; no bomb; "
                "no hold DOWN at y=189 tile 170; ignore 0x2B; do not push "
                "0x68; dest is RAM; no KEY-UP 0x09; no CheckWarp"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
            "keys": self.keys,
        }


def make_east38_controller() -> Level6East38Controller:
    """Occupancy east of 0x38 leftover. Do not poke bow/arrows/doors/bombs."""
    return Level6East38Controller()


def level6_east38_stages():
    """Play 0x38 leftover (120,93) → occupancy east door. Dest is RAM."""
    ctl = make_east38_controller()
    return (
        ("level6_east_0x38", ctl, ctl.max_frames),
    )


def level6_east38_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x38 with ADDR_ROD. Dest is RAM."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != LEVEL6_WIZZROBE_38_ROOM
        and snap.screen != LEVEL6_ROD_WIZZ_ROOM
        and snap.screen != LEVEL6_WIZZROBE_28_ROOM
        and snap.screen != LEVEL6_TRAPS_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
