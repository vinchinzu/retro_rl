"""Level 6 0x18 east hop after Gleeok residual.

North hole idle is not mode 9 (stairs v1–v5; tile 0x77 at y=95–101).
Occupancy to the east door (208,141) then RIGHT. Do not walk RIGHT at
y=133 into the shutter face. Dest hypothesized 0x19 Map; enter-stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon_ids import INVULN_MOVER_OBJECT_TYPE
from zelda_i.level6_dungeon import LEVEL6_MAP_BIT
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_GLEEOK_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "EAST_DOOR_X",
    "EAST_DOOR_Y",
    "EAST_DOOR_Y_TOL",
    "ROOM19_MAX_FRAMES",
    "MAP_19_GOAL",
    "MAP_19_MAX_FRAMES",
    "MAP_19_STANDS",
    "NORTH_09_COLUMN_X",
    "NORTH_09_DOOR_X",
    "ROOM09_MAX_FRAMES",
    "SETTLE_19_IDLE_FRAMES",
    "SETTLE_19_MAX_FRAMES",
    "Level6Map19Controller",
    "Level6Room09Controller",
    "Level6Room19Controller",
    "Level6Settle19Controller",
    "make_map19_controller",
    "make_room09_controller",
    "make_room19_controller",
    "make_settle_09_controller",
    "make_settle_19_controller",
    "make_settle_29_controller",
    "make_settle_39_controller",
    "make_settle_3a_controller",
]

EAST_DOOR_X = 208
EAST_DOOR_Y = 141
EAST_DOOR_Y_TOL = 4
EAST_DOOR_X_TOL = 4
ROOM19_MAX_FRAMES = 4000


@dataclass
class Level6Room19Controller:
    """Y-align 141, occupancy to (208,141), RIGHT. No stairs walk."""

    spec_id: str = "level6_room_0x19"
    room: int = LEVEL6_GLEEOK_ROOM
    goal: tuple[int, int] = (EAST_DOOR_X, EAST_DOOR_Y)
    max_frames: int = ROOM19_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
        }
        if force or self.frames <= 2 or self.frames % 250 == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "cur_opened_doors": int(snap.cur_opened_doors),
                    "open_doorway_mask": int(snap.open_doorway_mask),
                    "misses": self.walker.misses,
                    "tile": int(snap.colliding_tile),
                }
            )
        return action

    def _mark_success(self, snap: ZeldaSnapshot, reason: str, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), reason), force=True
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "link_death"), force=True
            )
        if (
            snap.level == LEVEL6
            and snap.screen != self.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            return self._mark_success(
                snap,
                f"arrived_{snap.screen:02x}",
                f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            self.walker.last_dir = None
            return FrameAction(nes_action("RIGHT"), "east_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            self.failed = True
            self.notes.append(f"left_level_{snap.level}")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "left_level"), force=True
            )
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("RIGHT"), "east_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        gx, gy = self.goal
        if (
            snap.link_x >= gx - EAST_DOOR_X_TOL
            and abs(snap.link_y - gy) <= EAST_DOOR_Y_TOL
        ):
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("RIGHT"), "east_push"))

        # Do not RIGHT at leftover y=133 into the shutter face.
        if abs(snap.link_y - gy) > EAST_DOOR_Y_TOL:
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
                snap, FrameAction(nes_idle_action(), "east_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "east_path")
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "y=141 first, occupancy to (208,141), RIGHT; no y=133 RIGHT",
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
        }


def make_room19_controller() -> Level6Room19Controller:
    """Occupancy east of 0x18. Map pickup residual. Do not grant Rod."""
    return Level6Room19Controller()


SETTLE_19_IDLE_FRAMES = 160
SETTLE_19_SAMPLE_PERIOD = 12
SETTLE_19_MAX_FRAMES = 400
_CENSUS_SKIP_TYPES = frozenset({0, INVULN_MOVER_OBJECT_TYPE})


def _live_census_objects(snap: ZeldaSnapshot) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for obj in snap.objects:
        type_id = int(obj.type_id)
        if obj.slot == 0 or type_id == 0:
            continue
        rows.append(
            {
                "slot": int(obj.slot),
                "type": type_id,
                "x": int(obj.x),
                "y": int(obj.y),
                "hp": int(obj.hp),
            }
        )
    return rows


@dataclass
class Level6Settle19Controller:
    """Idle at leftover. Do not walk into beams / wizzrobes."""

    spec_id: str = "level6_settle_0x19"
    room: int = LEVEL6_MAP_ROOM
    idle_frames: int = SETTLE_19_IDLE_FRAMES
    sample_period: int = SETTLE_19_SAMPLE_PERIOD
    max_frames: int = SETTLE_19_MAX_FRAMES
    frames: int = 0
    idle_in_room: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    type_histogram: dict[str, int] = field(default_factory=dict)
    leftover: dict[str, int] = field(default_factory=dict)
    policy: str = "IDLE at 0x19 west mouth; census spawn; do not walk"

    def _record(self, snap: ZeldaSnapshot, *, force: bool = False) -> None:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "room_item_id": int(snap.room_item_id),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
        }
        live = _live_census_objects(snap)
        counts: dict[int, int] = {}
        for row in live:
            type_id = int(row["type"])
            if type_id in _CENSUS_SKIP_TYPES:
                continue
            counts[type_id] = counts.get(type_id, 0) + 1
        for type_id, n in counts.items():
            key = f"0x{type_id:02x}"
            prev = self.type_histogram.get(key, 0)
            if n > prev:
                self.type_histogram[key] = n
        if force or self.frames <= 2 or self.frames % self.sample_period == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "objects": live,
                    "cur_opened_doors": int(snap.cur_opened_doors),
                    "open_doorway_mask": int(snap.open_doorway_mask),
                    "room_item_id": int(snap.room_item_id),
                    "map": int(snap.map),
                }
            )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                )
            self._record(snap, force=True)
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            self._record(snap, force=True)
            return FrameAction(nes_idle_action(), "link_death")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            self.failed = True
            self.notes.append(f"left_level_{snap.level}")
            return FrameAction(nes_idle_action(), "left_level")
        if snap.screen != self.room:
            self.failed = True
            self.notes.append(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
            return FrameAction(nes_idle_action(), f"left_0x{self.room:02x}")

        self.idle_in_room += 1
        self._record(snap, force=self.idle_in_room >= self.idle_frames)
        if self.idle_in_room >= self.idle_frames:
            self.success = True
            hist = ",".join(
                f"{k}x{n}" for k, n in sorted(self.type_histogram.items())
            )
            self.notes.append(
                f"settled_{self.room:02x}_{snap.link_x}_{snap.link_y}_{hist}"
            )
            return FrameAction(nes_idle_action(), "settled")
        return FrameAction(nes_idle_action(), "spawn_idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "idle_in_room": self.idle_in_room,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": self.policy,
            "type_histogram": dict(self.type_histogram),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
        }


def make_settle_19_controller() -> Level6Settle19Controller:
    """Idle ~160f in play 0x19 and census objects. Do not walk into the beam."""
    return Level6Settle19Controller()


def make_settle_09_controller() -> Level6Settle19Controller:
    """Idle ~160f in play 0x09 south mouth. Do not walk into wizzrobes."""
    return Level6Settle19Controller(
        spec_id="level6_settle_0x09",
        room=LEVEL6_ROD_WIZZ_ROOM,
        policy="IDLE at 0x09 south mouth; census spawn; do not walk",
    )


def make_settle_29_controller() -> Level6Settle19Controller:
    """Idle ~160f in dark 0x29. Census types; do not grant candle."""
    return Level6Settle19Controller(
        spec_id="level6_settle_0x29",
        room=LEVEL6_DARK_29_ROOM,
        policy="IDLE at 0x29 north mouth; census spawn; no candle",
    )


def make_settle_39_controller() -> Level6Settle19Controller:
    """Idle ~160f in dark 0x39. Census types; do not invent Gohma."""
    return Level6Settle19Controller(
        spec_id="level6_settle_0x39",
        room=LEVEL6_DARK_39_ROOM,
        policy="IDLE at 0x39 north mouth; census spawn; no candle/Gohma",
    )


def make_settle_3a_controller() -> Level6Settle19Controller:
    """Idle ~160f in play 0x3A. Census types; do not push the block."""
    return Level6Settle19Controller(
        spec_id="level6_settle_0x3a",
        room=LEVEL6_BLOCK_3A_ROOM,
        policy="IDLE at 0x3A west mouth; census spawn; do not push",
    )


# v1 leftover (176,158): occupancy boxed 4-cardinal then (176,93).
# v2 column x=120 south-band: leftover (120,181) ON the sprite, map still 0x0A.
# v3 idle (120,141) 120f then column through the sprite; leftover (120,179) 0x0A.
# v4 occupancy y-first to (136,141) freeze-miss boxed leftover then (176,93).
# v5 occupancy x-first wandered 244 misses; never idled (136,141); leftover (112,189).
# v6 axis LEFT to x=136 then idle (136,141): leftover (136,137) map still 0x0A.
# No persistent non-enemy object slot (0x2b + dead 0x14 only). Skip Map.
MAP_19_GOAL = (136, 141)
MAP_19_STANDS: tuple[tuple[int, int], ...] = (MAP_19_GOAL,)
MAP_19_STAND_TOL = 4
MAP_19_MAX_FRAMES = 6000
MAP_19_SAMPLE_PERIOD = 12


@dataclass
class Level6Map19Controller:
    """Axis onto (136,141) Map cell. Success is ADDR_MAP bit 0x20."""

    spec_id: str = "level6_map_0x19"
    room: int = LEVEL6_MAP_ROOM
    goal: tuple[int, int] = MAP_19_GOAL
    max_frames: int = MAP_19_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        item_id = int(snap.room_item_id)
        map_bits = int(snap.map)
        objects = _live_census_objects(snap)
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "map": map_bits,
            "room_item_id": item_id,
            "tile": int(snap.colliding_tile),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "triforce": int(snap.triforce),
        }
        changed = bool(self.samples) and (
            self.samples[-1]["map"] != map_bits
            or self.samples[-1]["room_item_id"] != item_id
        )
        if (
            force
            or changed
            or self.frames <= 2
            or self.frames % MAP_19_SAMPLE_PERIOD == 0
        ):
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "reason": action.reason,
                    "map": map_bits,
                    "room_item_id": item_id,
                    "tile": int(snap.colliding_tile),
                    "goal": self.goal,
                    "misses": self.walker.misses,
                    "objects": objects,
                }
            )
        return action

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_map={snap.map:02x}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "link_death"), force=True
            )
        if (int(snap.map) & LEVEL6_MAP_BIT) != 0:
            self.success = True
            self.notes.append(f"map_{snap.link_x}_{snap.link_y}")
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "map_got"), force=True
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            self.failed = True
            self.notes.append(f"left_level_{snap.level}")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "left_level"), force=True
            )
        if snap.screen != self.room:
            self.failed = True
            self.notes.append(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")
            return self._emit(
                snap, FrameAction(nes_idle_action(), f"left_0x{self.room:02x}"),
                force=True,
            )

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        gx, gy = self.goal
        # Axis only: occupancy from leftover boxed (v1/v4) then wandered (v5).
        if abs(xy[0] - gx) > MAP_19_STAND_TOL:
            self.walker.last_dir = None
            btn = "LEFT" if xy[0] > gx else "RIGHT"
            return self._emit(snap, FrameAction(nes_action(btn), "map_column"))
        if abs(xy[1] - gy) > MAP_19_STAND_TOL:
            self.walker.last_dir = None
            btn = "DOWN" if xy[1] < gy else "UP"
            return self._emit(snap, FrameAction(nes_action(btn), "map_row"))
        self.walker.last_dir = None
        return self._emit(snap, FrameAction(nes_idle_action(), "map_idle"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "axis LEFT to x=136 then idle (136,141); ADDR_MAP|0x20",
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
        }


def make_map19_controller() -> Level6Map19Controller:
    """Axis onto the 0x19 Map cell. Do not grant ADDR_MAP."""
    return Level6Map19Controller()


# Map is optional. PNG north door is locked; walkthrough KEY-UP then wizzrobes.
# Occupancy from leftover (176,158) boxed (v1/v4); axis LEFT is free (v6).
# v1 KEY-UP: LEFT to x=120 then occupancy UP freeze-missed y=157→138, wandered
# south, spent the key at (120,189) (cur_opened_doors DOWN=4). North still locked.
# v2: axis LEFT to v6-free x=136, occupancy to the north door, stand at y>=181.
NORTH_09_COLUMN_X = 136
NORTH_09_DOOR_X = 120
NORTH_09_DOOR_Y = 93
NORTH_09_BAND_Y = 109
NORTH_09_SOUTH_Y = 181
NORTH_09_X_TOL = 4
ROOM09_MAX_FRAMES = 4000
ROOM09_SAMPLE_PERIOD = 12


@dataclass
class Level6Room09Controller:
    """Axis LEFT out of the east pocket, occupancy KEY-UP. Skip Map."""

    spec_id: str = "level6_room_0x09"
    room: int = LEVEL6_MAP_ROOM
    dest: int = LEVEL6_ROD_WIZZ_ROOM
    goal: tuple[int, int] = (NORTH_09_DOOR_X, NORTH_09_DOOR_Y)
    max_frames: int = ROOM09_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    keys_at_start: int | None = None
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        objects = _live_census_objects(snap)
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "tile": int(snap.colliding_tile),
        }
        if force or self.frames <= 2 or self.frames % ROOM09_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "reason": action.reason,
                    "screen": int(snap.screen),
                    "keys": int(snap.keys),
                    "map": int(snap.map),
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
                    "objects": objects,
                }
            )
        return action

    def _mark_success(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), f"arrived_{snap.screen:02x}"),
            force=True,
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.keys_at_start is None:
            self.keys_at_start = int(snap.keys)
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_keys={snap.keys}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "link_death"), force=True
            )
        if (
            snap.level == LEVEL6
            and snap.screen != self.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            return self._mark_success(
                snap,
                f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                f"_keys={self.keys_at_start}->{snap.keys}",
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            self.failed = True
            self.notes.append(f"left_level_{snap.level}")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "left_level"), force=True
            )
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        # East pocket occupancy is unrecoverable. Axis LEFT to v6-free x=136.
        # v1 LEFT to x=120 then UP freeze-missed and occupancy spent the south key.
        if xy[0] > NORTH_09_COLUMN_X + NORTH_09_X_TOL:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action("LEFT"), "north_column")
            )
        if xy[1] >= NORTH_09_SOUTH_Y:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"south_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "north_south_halt")
            )
        if snap.link_y <= NORTH_09_BAND_Y:
            self.walker.last_dir = None
            if abs(snap.link_x - NORTH_09_DOOR_X) > NORTH_09_X_TOL:
                btn = "LEFT" if snap.link_x > NORTH_09_DOOR_X else "RIGHT"
                return self._emit(
                    snap, FrameAction(nes_action(btn), "north_align")
                )
            return self._emit(snap, FrameAction(nes_action("UP"), "north_push"))

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
                snap, FrameAction(nes_idle_action(), "north_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "north_path")
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "axis LEFT to x=136, occupancy KEY-UP y<=109; halt y>=181"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "keys_at_start": self.keys_at_start,
            "spec_id": self.spec_id,
            "room": self.room,
            "dest": self.dest,
            "goal": self.goal,
        }


def make_room09_controller() -> Level6Room09Controller:
    """Occupancy KEY-UP out of 0x19. Skip Map. Do not poke the door."""
    return Level6Room09Controller()
