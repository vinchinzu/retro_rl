"""Level 6 play 0x28 west-aisle south after south18 enter-stop.

Leftover (120,77) north mouth. Center south mouth of 0x28 is sealed
(mask=0, doors=0; south28 v2 DOWN tile 170 no scroll; v3 RIGHT+DOWN
leftover (124,189) still 0x28). Forward 0x38→0x28 used left-0x68 then
west aisle x=64 because the 0x38 north shutter was sealed. Reverse that
aisle; do not the center mouth.

v1: occupancy+clip reached west aisle (64,189); cardinal DOWN tile 221
no scroll. Like-Like knock leftover (32,189); 2× Like-Like + 1 blue
wizzrobe live. Kill-door hyp: 0x28 south was the 0x38 enter mouth going
north; return may need reclear. Occupancy-patrol live types, then
OccupancyWalker west aisle DOWN. LEFT+DOWN is only the dated north-face
clip to *reach* the aisle, not a door. Halt at first occupancy miss
(no path → stand). Do not KEY-UP 0x09. Do not CheckWarp. Do not poke
bow/arrows/doors/keys. Isolated BFS banned.
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
from zelda_i.level6_path import NORTH_BAND_Y, NORTH_WEST_X
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "AISLE28_MAX_FRAMES",
    "AISLE_X",
    "SOUTH_DOOR_Y",
    "Level6Aisle28Controller",
    "level6_aisle28_stages",
    "level6_aisle28_success",
    "make_aisle28_controller",
]

AISLE_X = NORTH_WEST_X
SOUTH_DOOR_Y = 189
SOUTH_BAND_Y = 181
AISLE_TOL = 4
CENTER_MOUTH_X = 120
# v1 occupancy DOWN @ x=120 boxed y=93 tiles 118/119. Mirror of
# DIAMOND_FACE_Y=181 / CLIP_CLEAR_Y=173 (8px past the face).
NORTH_FACE_Y = 93
CLIP_PAST_Y = 101
AISLE28_MAX_FRAMES = 16000
AISLE28_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
# Live leftover types, not northbound 2× orange. Do not register (clobbers
# ROOM_28_SPEC). inland_dash=0: already in-room from north mouth (UP would
# backtrack 0x18).
AISLE28_RECLEAR_SPEC = replace(
    ROOM_28_SPEC,
    spec_id="level6_aisle28_reclear",
    enemy_types=(
        LIKE_LIKE_OBJECT_TYPE,
        WIZZROBE_BLUE_OBJECT_TYPE,
        WIZZROBE_ORANGE_TYPE,
    ),
    expected_enemy_count=1,
    entry=DoorRoute("DOWN", ((120, 77), (120, 141))),
    combat=replace(ROOM_28_SPEC.combat, inland_dash=0),
)


@dataclass
class Level6Aisle28Controller:
    """Occupancy to west aisle x=64, DOWN. Never the center south mouth."""

    spec_id: str = "level6_aisle_0x28"
    room: int = LEVEL6_WIZZROBE_28_ROOM
    dest: int = LEVEL6_WIZZROBE_38_ROOM
    goal: tuple[int, int] = (AISLE_X, SOUTH_DOOR_Y)
    max_frames: int = AISLE28_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    fighter: Any = None

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
        if force or self.frames <= 2 or self.frames % AISLE28_SAMPLE_PERIOD == 0:
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
                f"key_spent_28_to_{snap.screen:02x}_{self.keys}->{int(snap.keys)}"
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
            if snap.screen == LEVEL6_GLEEOK_ROOM:
                return self._fail(
                    snap,
                    f"backtrack_18_{snap.link_x}_{snap.link_y}",
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
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        # Kill-door: reclear live types before south. OccupancyWalker after.
        if AISLE28_RECLEAR_SPEC.live_enemies(snap):
            if self.fighter is None:
                self.fighter = GenericDungeonRoomController(spec=AISLE28_RECLEAR_SPEC)
                self.notes.append(f"reclear_28_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(snap, self.fighter.step(snap))
        self.fighter = None

        # Dated north-face miss: cardinal DOWN @ x=120 boxes y=93. LEFT+DOWN
        # only to reach the west aisle, never as a south door.
        if xy[1] < CLIP_PAST_Y and xy[0] > AISLE_X + AISLE_TOL:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "DOWN"), "diamond_clip")
            )
        if xy[1] >= SOUTH_BAND_Y:
            self.walker.last_dir = None
            # Dated no-scroll: do not stand/push center south mouth x=120 y=189.
            if abs(xy[0] - AISLE_X) > AISLE_TOL:
                btn = "LEFT" if xy[0] > AISLE_X else "RIGHT"
                reason = (
                    "center_mouth_skip"
                    if abs(xy[0] - CENTER_MOUTH_X) <= AISLE_TOL
                    else "aisle_align"
                )
                return self._emit(snap, FrameAction(nes_action(btn), reason))
            return self._emit(snap, FrameAction(nes_action("DOWN"), "south_push"))

        # v2 dest=(64, y) on the north diamond row LEFT-boxed (96,101) then
        # UP→north_door_halt. Reclear opened south (doors=4, mask=8). Walk
        # the south-aisle goal; OccupancyWalker first after that miss.
        dest = self.goal
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction == "UP" and xy[1] <= NORTH_BAND_Y:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "north_door_halt")
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
                "occupancy-patrol live Like-Like/blue/orange then occupancy "
                "to (64,189) DOWN (not dest-same-y); LEFT+DOWN clip y<101 "
                "to reach aisle (not a door); never center south mouth "
                "y=189; halt UP y<=109; skip 0x38 left-0x68 push; no "
                "KEY-UP 0x09; no CheckWarp"
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


def make_aisle28_controller() -> Level6Aisle28Controller:
    """West-aisle south of 0x28. Do not poke bow/arrows/doors/keys. No CheckWarp."""
    return Level6Aisle28Controller()


def level6_aisle28_stages():
    """Play 0x28 leftover (120,77) → west aisle x=64 DOWN → play 0x38."""
    ctl = make_aisle28_controller()
    return (
        ("level6_aisle_0x28", ctl, ctl.max_frames),
    )


def level6_aisle28_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x38 with ADDR_ROD. Enter-stop; enemies may be gone."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_WIZZROBE_38_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
