"""Level 6 reclear 0x39 then LEFT+UP at dated (136,141).

Reuse west39 v3 enter: occupancy 0x3A LEFT, replan leftover miss
(tile 119), west_align DOWN, west_push into 0x39. Occupancy-patrol
remaining Vires (ignore 0x2B). LEFT+DOWN clips onto y=141 at dated
(144,109) / (142,141) / (139,141). Then LEFT+UP at dated (136,141)
(east39 reverse of RIGHT+UP, not LEFT+DOWN). OccupancyWalker LEFT on
the band that clip produces. Halt at the first new occupancy miss.
Isolated BFS banned. Do not KEY-UP 0x09 / 0x29. Do not CheckWarp
0x3A stairs. Do not bomb. Dest is RAM. Do not invent Gohma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_dungeon import ROOM_39_SPEC, make_clear_39_controller
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
)
from zelda_i.level6_path import NORTH_BAND_Y, NORTH_DOOR_X, WEST_CLIP_X
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "DATED_DOWN",
    "DATED_LEFT",
    "DATED_LEFT2",
    "DATED_LEFT3",
    "LANE_Y",
    "WEST39_UPCLIP_MAX_FRAMES",
    "WEST_DOOR",
    "WEST_SPAWN_XMIN",
    "Level6West39UpclipController",
    "level6_west39_upclip_stages",
    "level6_west39_upclip_success",
    "make_west39_upclip_controller",
]

WEST_DOOR = (32, 141)
WEST_DOOR_TOL = 4
WEST_SPAWN_XMIN = 16
LANE_Y = 141
DATED_DOWN = (144, 109)
# v1 occupancy y=141 LEFT 0px tile 119. PNG south floor open.
DATED_LEFT = (142, 141)
# v2 occupancy LEFT 0px tile 119 after the (142,141) clip.
DATED_LEFT2 = (139, 141)
# v3 occupancy LEFT 0px tile 117 after the (139,141) LEFT+DOWN clip.
DATED_LEFT3 = (136, 141)
WEST39_UPCLIP_MAX_FRAMES = 20000
WEST39_UPCLIP_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
START_ROOM = LEVEL6_BLOCK_3A_ROOM
VIA_ROOM = LEVEL6_DARK_39_ROOM


def _enter_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))


def _lane_walker(y: int = LANE_Y) -> OccupancyWalker:
    return OccupancyWalker(
        grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN, ymin=y, ymax=y)
    )


@dataclass
class Level6West39UpclipController:
    """v3 enter 0x39, reclear, LEFT+UP at (136,141). Dest is RAM."""

    spec_id: str = "level6_west39_upclip_0x39"
    room: int = START_ROOM
    max_frames: int = WEST39_UPCLIP_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    upclipped: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=_enter_walker)
    fighter: Any = None

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

    def _goal(self) -> tuple[int, int]:
        if self.room == START_ROOM:
            return (WEST_CLIP_X, LANE_Y)
        band = int(self.walker.grid.ymin)
        if self.upclipped and band == int(self.walker.grid.ymax) and band != LANE_Y:
            return (WEST_DOOR[0], band)
        return WEST_DOOR

    def _north_of_lane(self, xy: tuple[int, int]) -> bool:
        return xy[1] < LANE_Y

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
            "room": int(self.room),
        }
        if force or self.frames <= 2 or self.frames % WEST39_UPCLIP_SAMPLE_PERIOD == 0:
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
                    "room": int(self.room),
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
                f"key_spent_{self.room:02x}_to_{snap.screen:02x}"
                f"_{self.keys}->{int(snap.keys)}"
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
            snap,
            FrameAction(nes_idle_action(), f"arrived_{snap.screen:02x}"),
            force=True,
        )

    def _arrive_39(self, snap: ZeldaSnapshot) -> None:
        self.notes.append(
            f"arrived_39_{snap.link_x}_{snap.link_y}_keys={int(snap.keys)}"
        )
        self.room = VIA_ROOM
        self.keys = int(snap.keys)
        self.walker = _lane_walker()
        self.fighter = None

    def _west_clip(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        self.walker.last_dir = None
        self.walker.path = None
        clip_note = f"clip_{xy[0]}_{xy[1]}_tile={int(snap.colliding_tile)}"
        if "west_clip" not in self.notes:
            self.notes.append("west_clip")
        if clip_note not in self.notes:
            self.notes.append(clip_note)
        return self._emit(
            snap, FrameAction(nes_action("LEFT", "DOWN"), "west_clip")
        )

    def _west_upclip(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        self.walker.last_dir = None
        self.walker.path = None
        self.upclipped = True
        clip_note = f"upclip_{xy[0]}_{xy[1]}_tile={int(snap.colliding_tile)}"
        if "west_upclip" not in self.notes:
            self.notes.append("west_upclip")
        if clip_note not in self.notes:
            self.notes.append(clip_note)
        return self._emit(
            snap, FrameAction(nes_action("LEFT", "UP"), "west_upclip")
        )

    def _bind_upclip_band(self, xy: tuple[int, int]) -> None:
        y = int(xy[1])
        if int(self.walker.grid.ymin) == y and int(self.walker.grid.ymax) == y:
            return
        self.walker = _lane_walker(y)
        note = f"upclip_band_{y}"
        if note not in self.notes:
            self.notes.append(note)

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
                    f"north_29_{snap.link_x}_{snap.link_y}_keys={int(snap.keys)}",
                )
            if snap.screen == START_ROOM:
                return self._fail(
                    snap,
                    f"backtrack_3a_{snap.link_x}_{snap.link_y}",
                )
            if snap.screen == VIA_ROOM:
                self._arrive_39(snap)
                return self._emit(
                    snap, FrameAction(nes_action("LEFT"), "room_settle"), force=True
                )
            if not level6_west39_upclip_success(snap):
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
        if snap.screen == VIA_ROOM and ROOM_39_SPEC.live_enemies(snap):
            if self.fighter is None:
                self.fighter = make_clear_39_controller()
                self.notes.append(f"reclear_39_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(snap, self.fighter.step(snap))
        self.fighter = None

        if self.room == VIA_ROOM and xy == DATED_LEFT3:
            return self._west_upclip(snap, xy)
        if self.upclipped and xy != DATED_LEFT3:
            self._bind_upclip_band(xy)

        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        dated_down_miss = (
            self.room == VIA_ROOM
            and prev_dir == "DOWN"
            and xy == DATED_DOWN
        )
        dated_left_miss = (
            self.room == VIA_ROOM
            and prev_dir == "LEFT"
            and xy in (DATED_LEFT, DATED_LEFT2, DATED_LEFT3)
        )
        if self.walker.misses > misses_before:
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
            # v1 leftover LEFT (144,141) 0px tile 119. v3 enter replans.
            # Dated occupancy DOWN (144,109) tile 118: clip, do not halt.
            # v1 occupancy LEFT (142,141) tile 119 0px: clip, do not halt.
            # v2 occupancy LEFT (139,141) tile 119 0px: clip, do not halt.
            # v3 occupancy LEFT (136,141) tile 117 0px: LEFT+UP, do not halt.
            if (
                self.room != START_ROOM
                and not dated_down_miss
                and not dated_left_miss
            ):
                return self._fail(snap, f"occupancy_halt_{xy[0]}_{xy[1]}")

        gx = self._goal()[0]
        if xy[0] <= gx:
            self.walker.last_dir = None
            if abs(xy[1] - LANE_Y) > WEST_DOOR_TOL:
                btn = "UP" if xy[1] > LANE_Y else "DOWN"
                return self._emit(snap, FrameAction(nes_action(btn), "west_align"))
            return self._emit(snap, FrameAction(nes_action("LEFT"), "west_push"))

        if self.room == VIA_ROOM and not self.upclipped and (
            self._north_of_lane(xy) or xy in (DATED_LEFT, DATED_LEFT2)
        ):
            return self._west_clip(snap, xy)

        dest = self._goal()
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            if self.room == START_ROOM:
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "occupancy_stand")
                )
            return self._fail(snap, f"occupancy_halt_{xy[0]}_{xy[1]}")
        if (
            direction == "UP"
            and snap.screen == VIA_ROOM
            and xy[1] <= NORTH_BAND_Y
            and abs(xy[0] - NORTH_DOOR_X) <= WEST_DOOR_TOL
        ):
            return self._fail(snap, f"north_key_halt_{xy[0]}_{xy[1]}")
        if direction == "DOWN" and self.room == VIA_ROOM:
            # Dated west39 v3 occupancy DOWN (144,109). Stay LEFT on the band.
            if dest[1] == int(self.walker.grid.ymin):
                direction = "LEFT"
        reason = "left_path" if self.room == START_ROOM else "west_lane"
        return self._emit(snap, FrameAction(nes_action(direction), reason))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "west39 v3 enter: occupancy 0x3A LEFT replan leftover miss, "
                "west_align DOWN then west_push; reclear 0x39 Vires (ignore "
                "0x2B); LEFT+DOWN clip dated occupancy DOWN (144,109) tile "
                "118 and LEFT (142,141)/(139,141) tile 119; LEFT+UP clip "
                "dated occupancy LEFT (136,141) tile 117 (east39 reverse, "
                "not LEFT+DOWN); OccupancyWalker LEFT on the clip band; "
                "halt first new miss; no north 0x29; no KEY-UP 0x09; no "
                "stairs3a CheckWarp; dest is RAM; no bomb"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self._goal(),
            "keys": self.keys,
        }


def make_west39_upclip_controller() -> Level6West39UpclipController:
    """Reclear 0x39 then LEFT+UP at (136,141). Do not poke doors/bombs."""
    return Level6West39UpclipController()


def level6_west39_upclip_stages():
    """Play 0x3A leftover (144,141) → 0x39 reclear → LEFT+UP. Dest is RAM."""
    ctl = make_west39_upclip_controller()
    return (
        ("level6_west39_upclip_0x39", ctl, ctl.max_frames),
    )


def level6_west39_upclip_success(snap: ZeldaSnapshot) -> bool:
    """Play dest ≠ 0x3A and ≠ 0x29 and ≠ 0x39. Dest is RAM."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != START_ROOM
        and snap.screen != LEVEL6_DARK_29_ROOM
        and snap.screen != VIA_ROOM
        and snap.screen != LEVEL6_ROD_WIZZ_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
