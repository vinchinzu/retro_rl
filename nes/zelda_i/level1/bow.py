"""Level 1 0x23 KEY-LEFT after clear23. Dest play 0x22 (ROM bow room).

Q1 ROM $18700: 0x23 W=key, 0x22 E=key N/S/W=wall item=0x03. Enter-stop only.
Do not claim ADDR_BOW. Do not poke bow/arrows/doors/keys. Isolated BFS banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level1.finish import ROOM_KEY_GORIYA, ROOM_KEY_STALFOS_MAZE, level1_triforce_stages
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.screen_glance import BOW22_LEAVE, GlanceLeftover, grade_controller
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "BOW22_MAX_FRAMES",
    "LEVEL1_BOW_ROOM",
    "Level1Bow22Controller",
    "NORTH_BAND_Y",
    "NORTH_JOIN_X",
    "WEST_AISLE_X",
    "WEST_DOOR_X",
    "WEST_DOOR_Y",
    "level1_bow_glance",
    "level1_bow_stages",
    "level1_bow_success",
    "level1_to_clear23_stages",
    "make_bow22_controller",
]

LEVEL1 = 1
LEVEL1_BOW_ROOM = 0x22  # ROM west of 0x23; bow cellar is the next hop
WEST_DOOR_X, WEST_DOOR_Y, DOOR_TOL = 32, 141, 4
WEST_AISLE_X = 64
UPPER_CHANNEL_Y = 117
# Westwall v1–v3: UP at x=64 and x=80 from y=117 is tile 119 solid.
# Plus stem is ROOM_23_SPEC (112, 93) / (112, 133); key stand (114, 117).
NORTH_JOIN_X = 112
NORTH_BAND_Y = 93
WEST_SPAWN_XMIN = 16
BOW22_MAX_FRAMES = 4000
SAMPLE_PERIOD = 12
DEATH_MODE, CELLAR_MODE = 17, 9
WAIT_SCROLL = (2, 3, 4, 6, 7, 10, 16)


def level1_to_clear23_stages():
    """Survival L1 suffix through Goriya key 0x23. Does not backtrack to 0x44."""
    stages = []
    for item in level1_triforce_stages(natural_entry=True, survival=True):
        stages.append(item)
        if item[0] == "clear23_key":
            break
    return tuple(stages)


def make_bow22_controller() -> "Level1Bow22Controller":
    """Occupancy plus-stem KEY-LEFT of 0x23. Do not poke bow/arrows."""
    return Level1Bow22Controller()


def level1_bow_stages():
    """Prefix through clear23, then KEY-LEFT. Dest is exact play 0x22."""
    return (
        *level1_to_clear23_stages(),
        ("level1_bow_0x22", make_bow22_controller(), BOW22_MAX_FRAMES),
    )


def level1_bow_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L1 0x22. Do not require ADDR_BOW. Reject 0x23/0x33."""
    return (
        snap.level == LEVEL1
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL1_BOW_ROOM
    )


def level1_bow_glance(controller) -> GlanceLeftover:
    """East-mouth leftover after KEY-LEFT into 0x22."""
    return grade_controller(controller, BOW22_LEAVE)


def _leftover(snap: ZeldaSnapshot) -> dict[str, int]:
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "bow": int(snap.bow),
        "arrows": int(snap.arrows),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "triforce": int(snap.triforce),
    }


@dataclass
class Level1Bow22Controller:
    """Plus-stem north-around 0x23 then KEY-LEFT. Dest play 0x22. Enter-stop."""

    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(init=False)
    spec_id: str = "level1_bow_0x22"
    room: int = ROOM_KEY_GORIYA
    dest: int = LEVEL1_BOW_ROOM
    goal: tuple[int, int] = (WEST_DOOR_X, WEST_DOOR_Y)
    max_frames: int = BOW22_MAX_FRAMES

    def __post_init__(self) -> None:
        self.walker = OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or self.frames <= 2 or self.frames % SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
                }
            )
        self.leftover = _leftover(snap)
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _mark_success(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.keys >= 0 and int(snap.keys) < self.keys:
            self.notes.append(
                f"key_spent_23_to_22_{self.keys}->{int(snap.keys)}"
            )
        self.keys = int(snap.keys)
        self.success = True
        self.notes.append(
            f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_bow={int(snap.bow)}_keys={int(snap.keys)}"
        )
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), "arrived_22"), force=True
        )

    def _stage(self, xy: tuple[int, int]) -> tuple[tuple[int, int], str]:
        """Channel x-first to 112, UP y=93, LEFT (32,93), DOWN door. Not x=80."""
        x, y = xy
        gx, gy = self.goal
        # x112 v1: climb and north LEFT live. y>97 is not "on channel" on the
        # west column — that pull sent (32,109) east and stood (32,117).
        west_column = x <= gx + DOOR_TOL
        on_channel = y > NORTH_BAND_Y + DOOR_TOL
        if west_column and y < gy - DOOR_TOL:
            return (gx, gy), "door_drop"
        if on_channel and x != NORTH_JOIN_X:
            return (NORTH_JOIN_X, UPPER_CHANNEL_Y), "west_path"
        if on_channel:
            return (NORTH_JOIN_X, NORTH_BAND_Y), "north_band"
        if x > gx + DOOR_TOL:
            return (gx, NORTH_BAND_Y), "west_wall"
        return (gx, gy), "door_drop"

    def _walk(self, snap: ZeldaSnapshot) -> FrameAction:
        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
        gx, gy = self.goal
        if abs(snap.link_y - gy) <= DOOR_TOL and snap.link_x <= gx + DOOR_TOL:
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("LEFT"), "west_push"))
        dest, reason = self._stage(xy)
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_idle_action(), f"{reason}_stand"))
        return self._emit(snap, FrameAction(nes_action(direction), reason))

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.keys < 0:
            self.keys = int(snap.keys)
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if not any(n.startswith("timeout") for n in self.notes):
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_keys={int(snap.keys)}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == DEATH_MODE:
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
        if snap.screen == ROOM_KEY_STALFOS_MAZE and snap.mode == PLAY_MODE:
            return self._fail(
                snap, f"backtrack_33_{snap.link_x}_{snap.link_y}"
            )
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.level == LEVEL1
            and snap.screen == LEVEL1_BOW_ROOM
        ):
            return self._mark_success(snap)
        if snap.screen not in (ROOM_KEY_GORIYA, LEVEL1_BOW_ROOM) and snap.mode == PLAY_MODE:
            return self._fail(
                snap, f"wrong_room_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.transitioning or snap.mode in WAIT_SCROLL:
            self.walker.last_dir = None
            return FrameAction(nes_action("LEFT"), "west_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL1:
            return self._fail(snap, f"left_level_{snap.level}",)
        if snap.screen != ROOM_KEY_GORIYA:
            self.walker.last_dir = None
            return FrameAction(nes_action("LEFT"), "west_settle")
        return self._walk(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "occupancy LEFT to (112,117) then UP y=93 LEFT (32,93) "
                "DOWN (32,141); dest play 0x22; no ADDR_BOW; no x=80 UP"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
            "dest_room": self.dest,
            "keys": self.keys,
        }
