"""Level 6 play 0x38 east lane after east38 v3 leftover.

Reuse the live east38 v3 prefix: occupancy toward (208,141) plus
RIGHT+DOWN on the dated x=128 column, reaching (136,149) tile 177.
Then one-frame RIGHT+UP onto y=141 east of x=136 (PNG: statue is west;
east floor open; LEFT+UP would re-enter the statue). OccupancyWalker
RIGHT on y=141. Halt at the first new occupancy miss. Isolated BFS
banned. Ignore 0x2B. Do not bomb. Do not push 0x68. Do not hold DOWN
on tile 170. Dest is RAM (do not invent). Do not KEY-UP 0x09.
Do not CheckWarp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_east38 import (
    CLIP_BOX,
    CLIP_TOL,
    EAST_DOOR,
    EAST_DOOR_TOL,
    EAST38_MAX_FRAMES,
    EAST38_SAMPLE_PERIOD,
    SOUTH_DOOR_Y,
    WEST_SPAWN_XMIN,
)
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
    "EAST38_LANE_MAX_FRAMES",
    "LANE_X",
    "LANE_Y",
    "Level6East38LaneController",
    "level6_east38_lane_stages",
    "level6_east38_lane_success",
    "make_east38_lane_controller",
]

LANE_X = 136
LANE_Y = 141
EAST38_LANE_MAX_FRAMES = EAST38_MAX_FRAMES
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


def _new_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))


@dataclass
class Level6East38LaneController:
    """east38 v3 prefix to (136,149); RIGHT+UP onto y=141; occupancy RIGHT."""

    spec_id: str = "level6_east_lane_0x38"
    room: int = LEVEL6_WIZZROBE_38_ROOM
    goal: tuple[int, int] = EAST_DOOR
    max_frames: int = EAST38_LANE_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=_new_walker)
    on_lane: bool = False
    past_lane: bool = False

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

    def _at_column(self, xy: tuple[int, int]) -> bool:
        cx, _cy = CLIP_BOX
        return abs(xy[0] - cx) <= CLIP_TOL

    def _on_east_lane(self, xy: tuple[int, int]) -> bool:
        return xy[0] >= LANE_X and abs(xy[1] - LANE_Y) <= EAST_DOOR_TOL

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

    def _reset_lane_walker(self) -> None:
        if self.on_lane:
            return
        self.on_lane = True
        self.walker = _new_walker()
        self.notes.append("lane_reset")

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

        # Live east38 v3 prefix: RIGHT+DOWN on the dated x=128 column.
        # v1: after (136,149) bounce-back re-entered this clip and dumped
        # to y=189 tile 170. Once the dated leftover is reached, stay off it.
        if self._at_column(xy) and not self.past_lane:
            self.walker.last_dir = None
            self.walker.path = None
            if "east_clip" not in self.notes:
                self.notes.append("east_clip")
                self.notes.append(
                    f"clip_{xy[0]}_{xy[1]}_tile={int(snap.colliding_tile)}"
                )
            return self._emit(
                snap, FrameAction(nes_action("RIGHT", "DOWN"), "east_clip")
            )

        # Dated occupancy miss (136,149) tile 177: clip onto y=141, x>136.
        # v1 RIGHT+UP did not change y at (136,149). v2 no column re-entry
        # climbed onto y=141 at x=128 (statue column), occupancy boxed
        # tile 177. v3 LEFT+UP (PNG other clip) after that miss.
        if (xy[0] >= LANE_X and xy[1] > LANE_Y) or (
            self.past_lane and xy[1] > LANE_Y
        ):
            self.past_lane = True
            self.walker.last_dir = None
            self.walker.path = None
            if "lane_clip" not in self.notes:
                self.notes.append("lane_clip")
                self.notes.append(
                    f"lane_{xy[0]}_{xy[1]}_tile={int(snap.colliding_tile)}"
                )
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "UP"), "lane_clip")
            )

        dest = self.goal
        if self._on_east_lane(xy):
            self._reset_lane_walker()
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
                "east38 v3 prefix occupancy + RIGHT+DOWN x=128 column to "
                "(136,149); LEFT+UP onto y=141 (v1 RIGHT+UP y-dead at "
                "y=149 dumped south; v2 no column re-entry halted "
                "(128,141) tile 177); occupancy RIGHT on y=141; halt "
                "first new miss; "
                "no bomb; no hold DOWN at y=189 tile 170; ignore 0x2B; "
                "do not push 0x68; dest is RAM; no KEY-UP 0x09; no CheckWarp"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
            "keys": self.keys,
            "on_lane": self.on_lane,
        }


def make_east38_lane_controller() -> Level6East38LaneController:
    """Occupancy east lane of 0x38 leftover. Do not poke bow/arrows/doors/bombs."""
    return Level6East38LaneController()


def level6_east38_lane_stages():
    """Play 0x38 leftover (120,93) → clip onto y=141 x>136. Dest is RAM."""
    ctl = make_east38_lane_controller()
    return (
        ("level6_east_lane_0x38", ctl, ctl.max_frames),
    )


def level6_east38_lane_success(snap: ZeldaSnapshot) -> bool:
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
