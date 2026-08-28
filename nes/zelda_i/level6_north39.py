"""0x3A west → reclear 0x39 → enter-stop play 0x29.

clear3a leftover: play 0x3A (144,141) rod=1 keys=4 bow=0 arrows=0 TF=0x1F
map=0x0A; center 0x68 unpushed; west door open. OccupancyWalker first.
Halt at first occupancy miss (no path → stand). Coordinate clip only after
a live miss (v1: 0x39 east spawn (208,141) tile 118 boxed 4-cardinal;
LEFT+UP inland. v2: north_push (120,93) tile 118, keys=4, Vires live —
reclear 0x39 kill-door before UP. v3: arrived play 0x29 (120,205);
inland UP boxed tile 244 — north39 stops at that enter). Do not inland
0x29. Do not take the east door. Do not CheckWarp. Do not poke
bow/arrows/doors/keys. Do not invent Gohma. Isolated BFS banned.
Fail OW 0x22 as ow_early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_dungeon import ROOM_39_SPEC, make_clear_39_controller
from zelda_i.level6_occupancy import l6_leftover
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
)
from zelda_i.level6_path import NORTH_BAND_Y, NORTH_DOOR_X, NORTH_DOOR_Y, WEST_CLIP_X
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "EAST_DOOR_X",
    "EAST_DOOR_Y",
    "NORTH39_DEST",
    "NORTH39_LEGS",
    "NORTH39_MAX_FRAMES",
    "Level6North39Controller",
    "level6_north39_stages",
    "level6_north39_success",
    "make_north39_controller",
]

EAST_DOOR_X = 208
EAST_DOOR_Y = 141
DOOR_TOL = 4
# West spawn ~x=32 is outside OccupancyGrid xmin=40.
WEST_SPAWN_XMIN = 16
NORTH39_MAX_FRAMES = 20000
NORTH39_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
# (room, door button, occupancy goal). Dest 0x29 is enter-stop, not a leg.
NORTH39_LEGS: tuple[tuple[int, str, tuple[int, int]], ...] = (
    (LEVEL6_BLOCK_3A_ROOM, "LEFT", (WEST_CLIP_X, EAST_DOOR_Y)),
    (LEVEL6_DARK_39_ROOM, "UP", (NORTH_DOOR_X, NORTH_DOOR_Y)),
)
NORTH39_DEST = LEVEL6_DARK_29_ROOM
NORTH39_NEXT: dict[int, int] = {
    NORTH39_LEGS[0][0]: NORTH39_LEGS[1][0],
    NORTH39_LEGS[1][0]: NORTH39_DEST,
}


def _new_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))


@dataclass
class Level6North39Controller:
    """Occupancy 0x3A west → reclear 0x39 → enter-stop play 0x29."""

    spec_id: str = "level6_north39_0x29"
    max_frames: int = NORTH39_MAX_FRAMES
    frames: int = 0
    hop: int = 0
    keys: int = -1
    stop_room: int = NORTH39_DEST
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=_new_walker)
    fighter: Any = None

    def _in_legs(self) -> bool:
        return 0 <= self.hop < len(NORTH39_LEGS)

    def _room(self) -> int:
        return NORTH39_LEGS[self.hop][0]

    def _dir(self) -> str:
        return NORTH39_LEGS[self.hop][1]

    def _goal(self) -> tuple[int, int]:
        return NORTH39_LEGS[self.hop][2]

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL6
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.stop_room
            and snap.rod != 0
            and snap.triforce == 0x1F
        )

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            **l6_leftover(snap),
            "level": int(snap.level),
            "map": int(snap.map),
            "hop": int(self.hop),
            "room": int(self._room()) if self._in_legs() else -1,
        }
        if force or self.frames <= 2 or self.frames % NORTH39_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "level": int(snap.level),
                    "reason": action.reason,
                    "rod": int(snap.rod),
                    "keys": int(snap.keys),
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
                    "hop": int(self.hop),
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _mark_done(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), "entered_29"), force=True
        )

    def _enter_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"enter_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_rod={snap.rod}_tf={snap.triforce:02x}"
            f"_keys={int(snap.keys)}"
        )

    def _advance(self, snap: ZeldaSnapshot, dest: int) -> None:
        prev_keys = self.keys
        if prev_keys >= 0 and int(snap.keys) < prev_keys:
            self.notes.append(
                f"key_spent_{self._room():02x}_to_{dest:02x}"
                f"_{prev_keys}->{int(snap.keys)}"
            )
        self.keys = int(snap.keys)
        self.notes.append(
            f"arrived_{dest:02x}_{snap.link_x}_{snap.link_y}"
            f"_keys={int(snap.keys)}"
        )
        self.hop += 1
        self.walker = _new_walker()
        self.fighter = None

    def _door_band(self, xy: tuple[int, int]) -> tuple[str, str] | None:
        gx, gy = self._goal()
        direction = self._dir()
        if direction == "LEFT" and xy[0] <= gx:
            if abs(xy[1] - gy) > DOOR_TOL:
                return ("UP" if xy[1] > gy else "DOWN", "west_align")
            return ("LEFT", "west_push")
        if direction == "UP" and xy[1] <= NORTH_BAND_Y:
            if abs(xy[0] - gx) > DOOR_TOL:
                return ("LEFT" if xy[0] > gx else "RIGHT", "north_align")
            return ("UP", "north_push")
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.keys < 0:
            self.keys = int(snap.keys)
            for i, (room, _, _) in enumerate(NORTH39_LEGS):
                if snap.level == LEVEL6 and snap.screen == room:
                    self.hop = i
                    break
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_hop={self.hop}_rod={snap.rod}"
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
        if (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            return self._fail(
                snap,
                f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            if not self._in_legs():
                return FrameAction(nes_idle_action(), "wait_dest")
            return FrameAction(nes_action(self._dir()), f"{self._dir().lower()}_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level == 0:
            return self._fail(
                snap,
                f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}_{snap.screen:02x}")
        if not self._in_legs():
            if self._at_stop(snap):
                return self._mark_done(snap, self._enter_note(snap))
            return self._fail(
                snap,
                f"dest_{snap.screen:02x}_from_{self.stop_room:02x}"
                f"_{snap.link_x}_{snap.link_y}",
            )
        dest = NORTH39_NEXT.get(self._room())
        if snap.screen != self._room():
            if dest is not None and snap.screen == dest:
                self._advance(snap, int(snap.screen))
                self.walker.last_dir = None
                if self._at_stop(snap):
                    return self._mark_done(snap, self._enter_note(snap))
                settle = (
                    nes_action(self._dir())
                    if self._in_legs()
                    else nes_idle_action()
                )
                return self._emit(
                    snap,
                    FrameAction(settle, "room_settle"),
                    force=True,
                )
            return self._fail(
                snap,
                f"dest_{snap.screen:02x}_from_{self._room():02x}"
                f"_{snap.link_x}_{snap.link_y}",
            )
        if self._at_stop(snap):
            return self._mark_done(snap, self._enter_note(snap))

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before:
            note = f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}_r{self._room():02x}"
            if self.walker.misses <= 8 or self.frames % 60 == 0:
                self.notes.append(note)

        if (
            snap.screen == LEVEL6_DARK_39_ROOM
            and ROOM_39_SPEC.live_enemies(snap)
        ):
            if self.fighter is None:
                self.fighter = make_clear_39_controller()
                self.notes.append(f"reclear_39_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(snap, self.fighter.step(snap))
        self.fighter = None

        band = self._door_band(xy)
        if band is not None:
            btn, reason = band
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action(btn), reason))

        # v1 leftover play 0x39 (208,141) tile 118 boxed 4-cardinal.
        if (
            self._dir() == "UP"
            and self.walker.misses >= 1
            and xy[0] >= EAST_DOOR_X - DOOR_TOL
        ):
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "UP"), "east_spawn_clip")
            )

        dest_xy = self._goal()
        if dest_xy != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest_xy
        direction = self.walker.next_dir(xy, dest_xy)
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(
                    f"stand_f{self.frames}_{xy[0]}_{xy[1]}_r{self._room():02x}"
                )
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "occupancy_stand")
            )
        return self._emit(
            snap,
            FrameAction(nes_action(direction), f"{self._dir().lower()}_path"),
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "occupancy 0x3A west→0x39 reclear Vires→enter 0x29; "
                "no inland 0x29; no OW"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "hop": self.hop,
            "keys": self.keys,
        }


def make_north39_controller() -> Level6North39Controller:
    """0x3A west → reclear 0x39 → enter 0x29. Do not inland. Do not OW."""
    return Level6North39Controller()


def level6_north39_stages():
    """Play 0x3A leftover (144,141) → west → reclear 0x39 → enter 0x29."""
    ctl = make_north39_controller()
    return (
        ("level6_north39_0x29", ctl, ctl.max_frames),
    )


def level6_north39_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x29 enter-stop. Enemies may be live. No inland, no OW."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_DARK_29_ROOM
        and snap.triforce == 0x1F
        and snap.rod != 0
    )
