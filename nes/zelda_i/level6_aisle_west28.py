"""Level 6 play 0x28 west aisle then west, after south18 leftover.

Leftover (120,77) north mouth. west28 BLOCKED 3/3: occupancy LEFT along
the north diamond boxed x=96 (y=93/101/109); west mouth (32,141) never
reached. aisle-west28 v1: occupancy DOWN leftover miss f2 (120,79) tile
118 (live +2y vs OccupancyWalker 1px). v2: 1-frame DOWN then occupancy
from (120,79) miss f4 (120,82) tile 118 (1px then +2y). Off the north
mouth with a 1-frame DOWN (not occupancy-DOWN), then OccupancyWalker
from (120,79); do not halt on 2px overshoot (replan live). Halt at
first geometry miss. Do not LEFT along y=93 (dated). Clip only after a
new miss (west mouth LEFT+UP). Isolated BFS banned. Ignore 0x2B. Do not
DOWN at x=120 onto 0x38 south. Do not RIGHT east. Do not reclear
(kill-door undated). Dest is RAM. Do not KEY-UP 0x09. Do not CheckWarp.
Do not poke bow/arrows/doors/keys/bombs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_GLEEOK_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
)
from zelda_i.level6_path import NORTH_WEST_X
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker, WALK_DELTA

__all__ = [
    "AISLE_X",
    "AISLE_Y",
    "EAST_DOOR",
    "NORTH_MOUTH",
    "NORTH_MOUTH_Y",
    "SOUTH_DOOR_Y",
    "WEST28_AISLE_MAX_FRAMES",
    "WEST_CLIP_NOOP",
    "WEST_DOOR",
    "WEST_SPAWN_XMIN",
    "WEST_XMAX",
    "Level6AisleWest28Controller",
    "level6_aisle_west28_stages",
    "level6_aisle_west28_success",
    "make_aisle_west28_controller",
]

AISLE_X = NORTH_WEST_X
AISLE_Y = 141
AISLE_TOL = 4
WEST_DOOR = (32, 141)
EAST_DOOR = (208, 141)
WEST_DOOR_TOL = 4
SOUTH_DOOR_Y = 189
CENTER_MOUTH_X = 120
NORTH_MOUTH_Y = 77
NORTH_MOUTH = (CENTER_MOUTH_X, NORTH_MOUTH_Y)
WEST_SPAWN_XMIN = 16
WEST_XMAX = 120
# Keep occupancy off the south mouth (clear28-south trap). Dest is 141.
SOUTH_BAND_Y = 181
WEST28_AISLE_MAX_FRAMES = 4000
WEST28_AISLE_SAMPLE_PERIOD = 12
WEST_CLIP_NOOP = 192
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


def _new_walker() -> OccupancyWalker:
    # xmin=16 reaches the west mouth (default xmin=40 is east of x=32).
    # xmax=120 keeps occupancy off the sealed east mouth.
    # ymax=181 keeps BFS off the south mouth (0x38 trap).
    return OccupancyWalker(
        grid=OccupancyGrid(
            xmin=WEST_SPAWN_XMIN, xmax=WEST_XMAX, ymax=SOUTH_BAND_Y
        )
    )


@dataclass
class Level6AisleWest28Controller:
    """Occupancy to (64,141) then (32,141) LEFT. No east. No south 0x38."""

    spec_id: str = "level6_aisle_west_0x28"
    room: int = LEVEL6_WIZZROBE_28_ROOM
    aisle: tuple[int, int] = (AISLE_X, AISLE_Y)
    goal: tuple[int, int] = WEST_DOOR
    max_frames: int = WEST28_AISLE_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=_new_walker)
    clip_hold: int = 0
    clip_xy: tuple[int, int] | None = None
    mouth_step_done: bool = False

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
        if force or self.frames <= 2 or self.frames % WEST28_AISLE_SAMPLE_PERIOD == 0:
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
                f"key_spent_28_to_{snap.screen:02x}_{self.keys}->{int(snap.keys)}"
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

    def _on_aisle(self, xy: tuple[int, int]) -> bool:
        return abs(xy[0] - AISLE_X) <= AISLE_TOL and abs(xy[1] - AISLE_Y) <= AISLE_TOL

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
            if snap.screen == LEVEL6_GLEEOK_ROOM:
                return self._fail(
                    snap,
                    f"backtrack_18_{snap.link_x}_{snap.link_y}",
                )
            if snap.screen == LEVEL6_WIZZROBE_38_ROOM:
                return self._fail(
                    snap,
                    f"south_trap_38_{snap.link_x}_{snap.link_y}",
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
        if xy[1] >= SOUTH_DOOR_Y:
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("UP"), "mouth_back"))

        # v1 occupancy-DOWN leftover miss f2 (120,79) tile 118 (2px vs 1px).
        # 1-frame DOWN off the north mouth; occupancy starts from that pose.
        if not self.mouth_step_done and xy == NORTH_MOUTH:
            self.mouth_step_done = True
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action("DOWN"), "mouth_step")
            )

        gx, gy = self.goal
        if snap.link_x <= gx + WEST_DOOR_TOL and abs(snap.link_y - gy) <= WEST_DOOR_TOL:
            # Cardinal LEFT first. Clip LEFT+UP only after that live miss.
            self.walker.last_dir = None
            if xy == self.clip_xy:
                self.clip_hold += 1
            else:
                self.clip_xy = xy
                self.clip_hold = 1
            if self.clip_hold == 1:
                return self._emit(
                    snap, FrameAction(nes_action("LEFT"), "west_push")
                )
            if "west_clip" not in self.notes:
                self.notes.append("west_clip")
                self.notes.append(
                    f"clip_{xy[0]}_{xy[1]}_tile={int(snap.colliding_tile)}"
                    f"_doors={int(snap.cur_opened_doors)}"
                    f"_mask={int(snap.open_doorway_mask)}"
                )
            if self.clip_hold - 1 >= WEST_CLIP_NOOP:
                return self._fail(
                    snap,
                    f"west_clip_noop_{xy[0]}_{xy[1]}"
                    f"_tile={int(snap.colliding_tile)}"
                    f"_doors={int(snap.cur_opened_doors)}"
                    f"_mask={int(snap.open_doorway_mask)}",
                )
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "UP"), "west_clip")
            )
        self.clip_hold = 0
        self.clip_xy = None

        prev_dir = self.walker.last_dir
        prev_xy = self.walker.last_xy
        misses_before = self.walker.misses
        self.walker.observe(xy)
        just_missed = self.walker.misses > misses_before
        if just_missed:
            # v2 occupancy DOWN from (120,79) miss f4 (120,82): 1px then +2y.
            # 2px overshoot is leftover startup, not a wall. Replan live.
            overshoot = False
            if prev_dir in WALK_DELTA and prev_xy is not None:
                pdx, pdy = WALK_DELTA[prev_dir]
                if xy[0] - prev_xy[0] == 2 * pdx and xy[1] - prev_xy[1] == 2 * pdy:
                    overshoot = True
            if overshoot:
                self.notes.append(
                    f"overshoot_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}"
                )
                self.walker.last_dir = None
            else:
                self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
                return self._fail(
                    snap, f"occupancy_halt_{xy[0]}_{xy[1]}"
                )

        # Occupancy first: aisle south of the diamond, then west mouth.
        # Do not LEFT along y=93. Do not reclear (kill-door undated).
        dest = self.goal if self._on_aisle(xy) else self.aisle
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            return self._fail(
                snap, f"occupancy_halt_{xy[0]}_{xy[1]}"
            )
        if direction == "RIGHT":
            return self._fail(
                snap, f"occupancy_halt_{xy[0]}_{xy[1]}"
            )
        if (
            direction == "DOWN"
            and abs(xy[0] - CENTER_MOUTH_X) <= AISLE_TOL
            and xy[1] >= AISLE_Y
        ):
            return self._fail(
                snap, f"south_center_halt_{xy[0]}_{xy[1]}"
            )
        reason = "to_west" if self._on_aisle(xy) else "to_aisle"
        return self._emit(snap, FrameAction(nes_action(direction), reason))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "1-frame DOWN off north mouth (120,77) then occupancy from "
                "(120,79) to west aisle (64,141) south of the diamond then "
                "(32,141) LEFT; leftover mouth_step not occupancy-DOWN; 2px "
                "overshoot replans live (not halt); halt occupancy first "
                "geometry miss; no LEFT y=93; no LEFT+DOWN clip; xmax=120 no "
                "east; west-mouth LEFT+UP only after live LEFT miss; halt "
                "clip no-op 192f; no reclear; no south 0x38; no bomb; ignore "
                "0x2B; dest is RAM; no KEY-UP 0x09; no CheckWarp"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "aisle": self.aisle,
            "goal": self.goal,
            "keys": self.keys,
        }


def make_aisle_west28_controller() -> Level6AisleWest28Controller:
    """Occupancy west-aisle then west of 0x28 leftover. Do not poke doors/bombs."""
    return Level6AisleWest28Controller()


def level6_aisle_west28_stages():
    """Play 0x28 leftover (120,77) → occupancy aisle then west. Dest is RAM."""
    ctl = make_aisle_west28_controller()
    return (
        ("level6_aisle_west_0x28", ctl, ctl.max_frames),
    )


def level6_aisle_west28_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x28/0x38 with ADDR_ROD. Dest is RAM."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != LEVEL6_WIZZROBE_28_ROOM
        and snap.screen != LEVEL6_WIZZROBE_38_ROOM
        and snap.screen != LEVEL6_ROD_WIZZ_ROOM
        and snap.screen != LEVEL6_GLEEOK_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
