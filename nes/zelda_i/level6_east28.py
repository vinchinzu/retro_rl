"""Level 6 play 0x28 east census after south18 leftover.

Leftover (120,77) north mouth. v1 east_push (208,141) tile 223 no-op
doors=0. v2 reclear opened south (doors 0→4 mask 0→8) not east
(RIGHT bit 0x01 never set). Cardinal RIGHT still tile 223. Clip
RIGHT+UP after that miss (west38 LEFT+UP analog). Occupancy-patrol
live Like-Like / wizzrobe (ignore 0x2B), then occupancy to (208,141).
Halt at first occupancy miss / clip no-op. Isolated BFS banned.
Natural bombs. No count poke. Do not bomb-east this hop. Do not go
south into 0x38. Dest is RAM. Do not KEY-UP 0x09. Do not CheckWarp.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import DoorRoute, GenericDungeonRoomController
from zelda_i.dungeon_ids import LIKE_LIKE_OBJECT_TYPE, WIZZROBE_BLUE_OBJECT_TYPE
from zelda_i.level6_dungeon import ROOM_28_SPEC
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_GLEEOK_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "EAST_DOOR",
    "EAST28_MAX_FRAMES",
    "SOUTH_DOOR_Y",
    "WEST_DOOR",
    "Level6East28Controller",
    "level6_east28_stages",
    "level6_east28_success",
    "make_east28_controller",
]

EAST_DOOR = (208, 141)
WEST_DOOR = (32, 141)
EAST_DOOR_TOL = 4
SOUTH_DOOR_Y = 189
SOUTH_BAND_Y = 181
EAST28_MAX_FRAMES = 16000
EAST28_SAMPLE_PERIOD = 12
# v1 cardinal RIGHT ~192f no-op then knockback. v2 recleared so no
# knockback and held 13k. Halt clip if xy stuck that long.
EAST_CLIP_NOOP = 192
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
# Live leftover types (v1 PNG wizzrobe + Like-Like). Ignore 0x2B. Do not
# register (clobbers ROOM_28_SPEC). inland_dash=0: already in-room (UP
# would backtrack 0x18). Patrol/bounds stay north of south mouth.
EAST28_RECLEAR_SPEC = replace(
    ROOM_28_SPEC,
    spec_id="level6_east28_reclear",
    enemy_types=(
        LIKE_LIKE_OBJECT_TYPE,
        WIZZROBE_BLUE_OBJECT_TYPE,
        WIZZROBE_ORANGE_TYPE,
    ),
    expected_enemy_count=1,
    entry=DoorRoute("RIGHT", ((120, 77), EAST_DOOR)),
    combat=replace(
        ROOM_28_SPEC.combat,
        inland_dash=0,
        occupancy_bounds=(16, 216, 77, SOUTH_BAND_Y),
        patrol=tuple(
            p for p in ROOM_28_SPEC.combat.patrol if p[1] < SOUTH_BAND_Y
        ),
    ),
    max_frames=EAST28_MAX_FRAMES,
)


@dataclass
class Level6East28Controller:
    """Reclear then occupancy to (208,141) RIGHT. No south. Halt first miss."""

    spec_id: str = "level6_east_0x28"
    room: int = LEVEL6_WIZZROBE_28_ROOM
    goal: tuple[int, int] = EAST_DOOR
    max_frames: int = EAST28_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    fighter: Any = None
    clip_hold: int = 0
    clip_xy: tuple[int, int] | None = None

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
        if force or self.frames <= 2 or self.frames % EAST28_SAMPLE_PERIOD == 0:
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

        # Kill-door analog: reclear live types before east. OccupancyWalker after.
        live = EAST28_RECLEAR_SPEC.live_enemies(snap)
        if live:
            if self.fighter is None:
                self.fighter = GenericDungeonRoomController(
                    spec=EAST28_RECLEAR_SPEC
                )
                self.notes.append(
                    f"reclear_28_{xy[0]}_{xy[1]}"
                    f"_doors={int(snap.cur_opened_doors)}"
                    f"_mask={int(snap.open_doorway_mask)}"
                )
            self.walker.last_dir = None
            return self._emit(snap, self.fighter.step(snap))
        if self.fighter is not None:
            self.notes.append(
                f"recleared_28_{xy[0]}_{xy[1]}"
                f"_doors={int(snap.cur_opened_doors)}"
                f"_mask={int(snap.open_doorway_mask)}"
            )
            self.fighter = None

        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        just_missed = self.walker.misses > misses_before
        if just_missed and (self.walker.misses <= 8 or self.frames % 60 == 0):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        gx, gy = self.goal
        if snap.link_x >= gx - EAST_DOOR_TOL and abs(snap.link_y - gy) <= EAST_DOOR_TOL:
            # v2 cardinal RIGHT tile 223 ~13k no-op after reclear (south
            # doors=4, east bit 0x01 never). Clip after that miss.
            self.walker.last_dir = None
            if xy == self.clip_xy:
                self.clip_hold += 1
            else:
                self.clip_xy = xy
                self.clip_hold = 1
            if "east_clip" not in self.notes:
                self.notes.append("east_clip")
                self.notes.append(
                    f"clip_{xy[0]}_{xy[1]}_tile={int(snap.colliding_tile)}"
                    f"_doors={int(snap.cur_opened_doors)}"
                    f"_mask={int(snap.open_doorway_mask)}"
                )
            if self.clip_hold >= EAST_CLIP_NOOP:
                return self._fail(
                    snap,
                    f"east_clip_noop_{xy[0]}_{xy[1]}"
                    f"_tile={int(snap.colliding_tile)}"
                    f"_doors={int(snap.cur_opened_doors)}"
                    f"_mask={int(snap.open_doorway_mask)}",
                )
            return self._emit(
                snap, FrameAction(nes_action("RIGHT", "UP"), "east_clip")
            )
        self.clip_hold = 0
        self.clip_xy = None

        # south28 v1 BFS-DOWN @ x=120 boxed y=93. Occupancy x-align at
        # current y first, then the east mouth. Clip only after a live miss.
        if abs(xy[0] - gx) > EAST_DOOR_TOL:
            dest = (gx, xy[1])
        else:
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
                "occupancy-patrol live Like-Like/blue/orange (ignore 0x2B) "
                "then occupancy x-align (208,y) then (208,141); v2 cardinal "
                "RIGHT tile 223 no-op after reclear opened south not east; "
                "v3 RIGHT+UP clip after that miss; halt clip no-op 192f or "
                "occupancy miss; no south 0x38; no bomb-east; no count "
                "poke; no push 0x68; dest is RAM; no KEY-UP 0x09; no "
                "CheckWarp"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
            "keys": self.keys,
        }


def make_east28_controller() -> Level6East28Controller:
    """Reclear then occupancy east of 0x28 leftover. Do not poke doors/bombs."""
    return Level6East28Controller()


def level6_east28_stages():
    """Play 0x28 leftover (120,77) → reclear → occupancy east. Dest is RAM."""
    ctl = make_east28_controller()
    return (
        ("level6_east_0x28", ctl, ctl.max_frames),
    )


def level6_east28_success(snap: ZeldaSnapshot) -> bool:
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
