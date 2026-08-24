"""Level 6 Gleeok 0x18 south-stand fight (type 0x44).

Live settle (`l6_settle18_continuous_v1`): body is **0x44**, not L4 0x43.
Fireball residual 0x56. Head 0x46 not seen during idle. Reuse L4 south-stand
policy; do not copy the L4 TF suffix. Do not require Map. Diamond south mouth
uses the live 0x28 LEFT+UP clip (cardinal UP at y=181 is solid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.door_graph.core import DoorDir
from zelda_i.dungeon_ids import (
    GLEEOK_3HEAD_OBJECT_TYPE,
    GLEEOK_HEAD_OBJECT_TYPE,
    INVULN_MOVER_OBJECT_TYPE,
)
from zelda_i.level4_boss_combat import (
    FIREBALL_DODGE_DIST,
    STAND_DY,
    _fireball_dodge_dir,
    _south_stand_action,
    gleeok_fireballs,
    gleeok_heads_live,
)
from zelda_i.level6_overworld import LEVEL6, LEVEL6_GLEEOK_ROOM
from zelda_i.level6_path import CLIP_CLEAR_Y
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker, WALK_DELTA

__all__ = [
    "EAST_DOOR_BIT",
    "GLEEOK_18_MAX_FRAMES",
    "Level6Gleeok18Controller",
    "Level6PostGleeok18Controller",
    "PASSAGE_MODE",
    "POSTGLEEOK_18_CENSUS_FRAMES",
    "POSTGLEEOK_18_MAX_FRAMES",
    "POSTGLEEOK_STAND_X",
    "POSTGLEEOK_STAND_Y",
    "STAIRS_KEEP_Y",
    "east_door_open",
    "gleeok_3head_live",
    "make_gleeok_18_controller",
    "make_postgleeok_18_controller",
]

GLEEOK_18_MAX_FRAMES = 20000


def gleeok_3head_live(snap: ZeldaSnapshot) -> list:
    """Body slots type 0x44 (HP may be 0 mid-fight — TYPE presence)."""
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) == GLEEOK_3HEAD_OBJECT_TYPE
    ]


@dataclass
class Level6Gleeok18Controller:
    """Diamond clip inland, then L4 south-stand on 0x44. Stop when body is gone."""

    spec_id: str = "level6_gleeok_0x18"
    room: int = LEVEL6_GLEEOK_ROOM
    max_frames: int = GLEEOK_18_MAX_FRAMES
    stand_dy: int = STAND_DY
    fireball_dodge_dist: int = FIREBALL_DODGE_DIST
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    saw_0x44: bool = False
    saw_0x46: bool = False

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        bodies = gleeok_3head_live(snap)
        if bodies:
            self.saw_0x44 = True
        if gleeok_heads_live(snap):
            self.saw_0x46 = True
        if force or self.frames <= 2 or self.frames % 250 == 0:
            body = bodies[0] if bodies else None
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "reason": action.reason,
                    "bx": None if body is None else int(body.x),
                    "by": None if body is None else int(body.y),
                    "bhp": None if body is None else int(body.hp),
                    "heads": len(gleeok_heads_live(snap)),
                    "n44": len(bodies),
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
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 9, 10):
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

        bodies = gleeok_3head_live(snap)
        if bodies:
            self.saw_0x44 = True
        if not bodies:
            if not self.saw_0x44:
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_body")
                )
            self.success = True
            self.notes.append(
                f"body_gone_{snap.link_x}_{snap.link_y}_0x46={int(self.saw_0x46)}"
            )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "body_gone"), force=True
            )

        if snap.link_y > CLIP_CLEAR_Y:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(
                    f"clip_f{self.frames}_{snap.link_x}_{snap.link_y}"
                )
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "UP"), "diamond_clip")
            )

        dodge = _fireball_dodge_dir(snap, thr=self.fireball_dodge_dist)
        if dodge is not None:
            return self._emit(
                snap, FrameAction(nes_action(dodge), "fb_dodge")
            )

        act = _south_stand_action(snap, bodies[0], stand_dy=self.stand_dy)
        reason = (
            "south_stand"
            if list(act) == list(nes_action("UP", "A"))
            else "south_walk"
        )
        return self._emit(snap, FrameAction(act, reason))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "LEFT+UP y>173 then L4 south-stand on 0x44",
            "saw_0x44": self.saw_0x44,
            "saw_0x46": self.saw_0x46,
            "stand_dy": self.stand_dy,
            "spec_id": self.spec_id,
            "room": self.room,
            "body_type": GLEEOK_3HEAD_OBJECT_TYPE,
            "head_type": GLEEOK_HEAD_OBJECT_TYPE,
        }


def make_gleeok_18_controller() -> Level6Gleeok18Controller:
    """South-stand 0x44 until the body is gone. Map / Rod residual."""
    return Level6Gleeok18Controller()


PASSAGE_MODE = 9
EAST_DOOR_BIT = int(DoorDir.RIGHT)
# Leftover inland (121,133) is already south of body (124,111)+STAND_DY.
POSTGLEEOK_STAND_X = 120
POSTGLEEOK_STAND_Y = 133
# North stairs hole is ~y=96–109. Dodge/stand stay south of it.
STAIRS_KEEP_Y = 125
POSTGLEEOK_18_SAMPLE_PERIOD = 12
POSTGLEEOK_18_CENSUS_FRAMES = 192
POSTGLEEOK_18_AFTER_HEADS = 96
POSTGLEEOK_18_MAX_FRAMES = 4000
_CENSUS_SKIP_TYPES = frozenset({0, INVULN_MOVER_OBJECT_TYPE})
_DODGE_PX = 32


@dataclass(frozen=True)
class _FixedStand:
    """Stationary south-stand target. Do not chase live 0x46."""

    x: int
    y: int


def east_door_open(snap: ZeldaSnapshot) -> bool:
    """Walkable east is ``open_doorway_mask`` bit0.

    v1 leftover: ``cur_opened_doors`` 0→5 while the PNG shutter stayed black
    and mask stayed 0. Do not treat ``cur_opened_doors`` RIGHT as open.
    """
    return bool(int(snap.open_doorway_mask) & EAST_DOOR_BIT)


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


def _dodge_goal(snap: ZeldaSnapshot) -> tuple[int, int] | None:
    """Occupancy target away from the nearest 0x56. Keep y south of stairs."""
    dodge = _fireball_dodge_dir(snap, thr=FIREBALL_DODGE_DIST)
    if dodge is None:
        return None
    dx, dy = WALK_DELTA[dodge]
    x = int(snap.link_x) + dx * _DODGE_PX
    y = int(snap.link_y) + dy * _DODGE_PX
    x = max(48, min(192, x))
    y = max(STAIRS_KEEP_Y, min(173, y))
    return (x, y)


@dataclass
class Level6PostGleeok18Controller:
    """Residual 0x46 south-stand + door census. Do not walk to stairs."""

    spec_id: str = "level6_postgleeok_0x18"
    room: int = LEVEL6_GLEEOK_ROOM
    max_frames: int = POSTGLEEOK_18_MAX_FRAMES
    sample_period: int = POSTGLEEOK_18_SAMPLE_PERIOD
    census_frames: int = POSTGLEEOK_18_CENSUS_FRAMES
    after_heads: int = POSTGLEEOK_18_AFTER_HEADS
    stand_x: int = POSTGLEEOK_STAND_X
    stand_y: int = POSTGLEEOK_STAND_Y
    frames: int = 0
    idle_in_room: int = 0
    idle_after_heads: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    type_histogram: dict[str, int] = field(default_factory=dict)
    saw_0x44: bool = False
    saw_0x46: bool = False
    saw_0x56: bool = False
    last_doors: dict[str, int] = field(default_factory=dict)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def _record_census(self, snap: ZeldaSnapshot, *, force: bool = False) -> None:
        self.last_doors = {
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
        }
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "room_item_id": int(snap.room_item_id),
            "triforce": int(snap.triforce),
        }
        live = _live_census_objects(snap)
        counts: dict[int, int] = {}
        for row in live:
            type_id = int(row["type"])
            if type_id == GLEEOK_3HEAD_OBJECT_TYPE:
                self.saw_0x44 = True
            if type_id == GLEEOK_HEAD_OBJECT_TYPE:
                self.saw_0x46 = True
            if type_id == 0x56:
                self.saw_0x56 = True
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
                    "n44": len(gleeok_3head_live(snap)),
                    "n46": len(gleeok_heads_live(snap)),
                    "n56": len(gleeok_fireballs(snap)),
                    "east": int(east_door_open(snap)),
                }
            )

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self._record_census(snap, force=force)
        return action

    def _mark_success(
        self, snap: ZeldaSnapshot, reason: str, note: str
    ) -> FrameAction:
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
                    f"_doors={snap.cur_opened_doors}_{snap.open_doorway_mask}"
                    f"_0x46={len(gleeok_heads_live(snap))}"
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
        if snap.mode == PASSAGE_MODE:
            return self._mark_success(
                snap,
                "stairs",
                f"stairs_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 10):
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

        self.idle_in_room += 1
        xy = (int(snap.link_x), int(snap.link_y))
        self.walker.observe(xy)

        if not gleeok_3head_live(snap) and east_door_open(snap):
            return self._mark_success(
                snap,
                "east_open",
                f"east_open_{snap.link_x}_{snap.link_y}"
                f"_doors={snap.cur_opened_doors}_{snap.open_doorway_mask}",
            )

        heads = gleeok_heads_live(snap)
        if heads:
            self.idle_after_heads = 0
        else:
            self.idle_after_heads += 1
        if (
            not gleeok_3head_live(snap)
            and not heads
            and self.idle_in_room >= self.census_frames
            and self.idle_after_heads >= self.after_heads
        ):
            return self._mark_success(
                snap,
                "heads_gone",
                f"heads_gone_{snap.link_x}_{snap.link_y}"
                f"_doors={snap.cur_opened_doors}_{snap.open_doorway_mask}",
            )

        goal = _dodge_goal(snap)
        if goal is not None:
            direction = self.walker.next_dir(xy, goal)
            if direction == "UP" and snap.link_y <= STAIRS_KEEP_Y:
                self.walker.last_dir = None
                direction = None
            if direction is not None:
                return self._emit(
                    snap, FrameAction(nes_action(direction), "fb_dodge")
                )
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "fb_stand")
            )

        self.walker.last_dir = None
        bodies = gleeok_3head_live(snap)
        if bodies:
            act = _south_stand_action(snap, bodies[0], stand_dy=STAND_DY)
            reason = (
                "south_stand"
                if list(act) == list(nes_action("UP", "A"))
                else "south_walk"
            )
            return self._emit(snap, FrameAction(act, reason))
        if heads:
            dummy = _FixedStand(
                x=self.stand_x, y=self.stand_y - STAND_DY
            )
            act = _south_stand_action(snap, dummy, stand_dy=STAND_DY)
            reason = (
                "south_stand"
                if list(act) == list(nes_action("UP", "A"))
                else "south_walk"
            )
            return self._emit(snap, FrameAction(act, reason))
        return self._emit(snap, FrameAction(nes_idle_action(), "residual_idle"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "idle_in_room": self.idle_in_room,
            "idle_after_heads": self.idle_after_heads,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "IDLE census; 0x46 south-stand at (120,133); "
                "occupancy dodge 0x56; no stairs walk"
            ),
            "saw_0x44": self.saw_0x44,
            "saw_0x46": self.saw_0x46,
            "saw_0x56": self.saw_0x56,
            "type_histogram": dict(self.type_histogram),
            "cur_opened_doors": self.last_doors.get("cur_opened_doors"),
            "open_doorway_mask": self.last_doors.get("open_doorway_mask"),
            "east_open": bool(
                (self.last_doors.get("open_doorway_mask") or 0) & EAST_DOOR_BIT
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "spec_id": self.spec_id,
            "room": self.room,
            "stand": (self.stand_x, self.stand_y),
            "stairs_keep_y": STAIRS_KEEP_Y,
        }


def make_postgleeok_18_controller() -> Level6PostGleeok18Controller:
    """Census residual + doors after 0x44 body-gone. Do not grant Rod."""
    return Level6PostGleeok18Controller()
