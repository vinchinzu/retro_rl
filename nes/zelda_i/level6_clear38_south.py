"""Level 6 play 0x38 reclear then center-south after clear28-south enter-stop.

Leftover (120,93) north mouth. south38 occupancy reached PNG-open south
(120,189) then cardinal DOWN tile 170 no scroll (doors=0 mask=0) — same
signature as 0x28 south before reclear. Reclear of 0x28 opened that mouth.

Occupancy-patrol live traps / wizzrobe / Like-Like (do not skip combat),
then OccupancyWalker to (120,189) DOWN. Halt at first occupancy miss.
Clip RIGHT+DOWN only after a new miss. Do not occupancy UP at the north
mouth (south38 v2 halt). Do not persistent LEFT+DOWN at y=93 (south38 v1
slid west). south38 stays dedicated-red. Do not KEY-UP 0x09. Do not
CheckWarp. Do not poke bow/arrows/doors/keys. Isolated BFS banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import DoorRoute, GenericDungeonRoomController
from zelda_i.dungeon_ids import (
    INVULN_MOVER_OBJECT_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
    WIZZROBE_BLUE_OBJECT_TYPE,
)
from zelda_i.level6_dungeon import ROOM_38_SPEC
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.level6_path import NORTH_BAND_Y
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "CLEAR38_SOUTH_MAX_FRAMES",
    "SPAWN_WAIT_FRAMES",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6Clear38SouthController",
    "level6_clear38_south_stages",
    "level6_clear38_south_success",
    "make_clear38_south_controller",
]

SOUTH_DOOR_X = 120
SOUTH_DOOR_Y = 189
SOUTH_BAND_Y = 181
SOUTH_DOOR_TOL = 4
# v2 leftover: slot1 type 0x2B hp240 sitting west of the south mouth.
TRAP_TYPE = INVULN_MOVER_OBJECT_TYPE
TRAP_DOOR_X_TOL = 20
# south38 v1 occupancy DOWN @ x=120 boxed y=96; LEFT+DOWN at y=93 slid west.
NORTH_FACE_Y = 93
CLIP_PAST_Y = 101
CLEAR38_SOUTH_MAX_FRAMES = 16000
CLEAR38_SOUTH_SAMPLE_PERIOD = 12
# v1 occupancy-walked empty leftover in 144f; orange sprite still live at
# timeout and live_enemies never fired. Wait for spawn before south.
SPAWN_WAIT_FRAMES = 180
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
# Live leftover types (v3 orange wizzrobe). inland_dash=0: already in-room
# from north mouth (UP would backtrack 0x28). type_only: HP=0 teleport/spawn.
_CLEAR38_LIVE_TYPES = (
    LIKE_LIKE_OBJECT_TYPE,
    WIZZROBE_BLUE_OBJECT_TYPE,
    WIZZROBE_ORANGE_TYPE,
)
CLEAR38_SOUTH_RECLEAR_SPEC = replace(
    ROOM_38_SPEC,
    spec_id="level6_clear38_south_reclear",
    enemy_types=_CLEAR38_LIVE_TYPES,
    type_only_enemy_types=_CLEAR38_LIVE_TYPES,
    expected_enemy_count=1,
    entry=DoorRoute("DOWN", ((120, 93), (120, 141))),
    combat=replace(ROOM_38_SPEC.combat, inland_dash=0),
    max_frames=CLEAR38_SOUTH_MAX_FRAMES,
)


@dataclass
class Level6Clear38SouthController:
    """Reclear then occupancy to center south (120,189) DOWN. Never KEY-UP 0x09."""

    spec_id: str = "level6_clear_south_0x38"
    room: int = LEVEL6_WIZZROBE_38_ROOM
    dest: int = LEVEL6_TRAPS_ROOM
    goal: tuple[int, int] = (SOUTH_DOOR_X, SOUTH_DOOR_Y)
    max_frames: int = CLEAR38_SOUTH_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    fighter: Any = None
    clip_after_miss: bool = False
    seen_live: bool = False

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

    def _door_trap(self, snap: ZeldaSnapshot):
        """Invuln 0x2B on the south-door column (v2 census, not a wizzrobe)."""
        for obj in snap.objects:
            if int(obj.slot) < 1 or int(obj.type_id) != TRAP_TYPE:
                continue
            if abs(int(obj.x) - SOUTH_DOOR_X) <= TRAP_DOOR_X_TOL and int(
                obj.y
            ) >= SOUTH_BAND_Y:
                return obj
        return None

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
            "objs": [
                {
                    "slot": int(obj.slot),
                    "type": int(obj.type_id),
                    "hp": int(obj.hp),
                    "x": int(obj.x),
                    "y": int(obj.y),
                }
                for obj in snap.objects
                if int(obj.slot) >= 1 and int(obj.type_id) != 0
            ],
        }
        if force or self.frames <= 2 or self.frames % CLEAR38_SOUTH_SAMPLE_PERIOD == 0:
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
        # Kill-door: wait for spawn, then reclear, then occupancy south.
        # v1 occupancy-walked empty leftover; live_enemies never fired.
        live = CLEAR38_SOUTH_RECLEAR_SPEC.live_enemies(snap)
        if live:
            self.seen_live = True
        waiting = (not self.seen_live) and self.frames <= SPAWN_WAIT_FRAMES
        if live or waiting:
            if self.fighter is None:
                self.fighter = GenericDungeonRoomController(
                    spec=CLEAR38_SOUTH_RECLEAR_SPEC
                )
                note = (
                    f"reclear_38_{xy[0]}_{xy[1]}"
                    if live
                    else f"spawn_wait_38_{xy[0]}_{xy[1]}"
                )
                self.notes.append(note)
            self.walker.last_dir = None
            self.clip_after_miss = False
            return self._emit(snap, self.fighter.step(snap))
        self.fighter = None

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
            trap = self._door_trap(snap)
            if trap is not None and abs(xy[0] - gx) <= SOUTH_DOOR_TOL:
                # v2 held DOWN on tile 170 with 0x2B west of the mouth.
                # Occupancy-patrol the trap off the column, then DOWN.
                btn = "RIGHT" if xy[0] <= int(trap.x) else "LEFT"
                return self._emit(
                    snap, FrameAction(nes_action(btn), "trap_peel")
                )
            # south38 dated cardinal DOWN @ (120,189) tile 170. Hold DOWN;
            # do not RIGHT+DOWN here (v1 bounced to x=126 tile 119).
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
                snap, FrameAction(nes_action("RIGHT", "DOWN"), "diamond_clip")
            )
        if xy[1] >= CLIP_PAST_Y:
            self.clip_after_miss = False

        dest = self.goal
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction == "UP" and xy[1] <= NORTH_BAND_Y:
            # v2: occupancy UP at leftover mouth halted. Peel RIGHT, not UP.
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
                "spawn-wait then occupancy-patrol live traps/wizzrobe/"
                "Like-Like; occupancy to (120,189) DOWN; peel 0x2B off the "
                "south column then DOWN (v2 held DOWN tile 170, 0x2B live); "
                "one-shot RIGHT+DOWN after a north-face miss only; RIGHT peel "
                "when occupancy UP at north mouth; skip dedicated south38; "
                "no KEY-UP 0x09; no CheckWarp"
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


def make_clear38_south_controller() -> Level6Clear38SouthController:
    """Reclear then center-south of 0x38. Do not poke bow/arrows/doors/keys."""
    return Level6Clear38SouthController()


def level6_clear38_south_stages():
    """Play 0x38 leftover (120,93) → reclear → (120,189) DOWN → play 0x48."""
    ctl = make_clear38_south_controller()
    return (
        ("level6_clear_south_0x38", ctl, ctl.max_frames),
    )


def level6_clear38_south_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x48 with ADDR_ROD. Enter-stop; traps may be live."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_TRAPS_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
