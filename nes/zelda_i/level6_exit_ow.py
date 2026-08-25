"""Leave L6 from cleared 0x3A. north39 enter-stop or OW mouth 0x22.

clear3a leftover: play 0x3A (144,141) rod=1 keys=4 bow=0 arrows=0 TF=0x1F
map=0x0A; center 0x68 unpushed; west door open. OccupancyWalker first.
Halt at first occupancy miss (no path → stand). Coordinate clip only after
a live miss (v1: 0x39 east spawn (208,141) tile 118 boxed 4-cardinal;
LEFT+UP inland. v2: north_push (120,93) tile 118, keys=4, Vires live —
reclear 0x39 kill-door before UP. v3: arrived play 0x29 (120,205);
inland UP boxed tile 244 — north39 stops at that enter). Do not inland
0x29. Do not take the east door. Do not CheckWarp. Do not poke
bow/arrows/doors/keys. Do not invent Gohma. Isolated BFS banned.

Source leave (inbound rooms, not invented): west 0x3A→0x39 → north 0x29
→ 0x19 → west 0x18 → south 0x28/0x38/0x48/0x58/0x68 → 0x78 → east 0x79
→ DOWN OW 0x22. A north return that spends a key is live evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import LEVEL6_ENTRY_ROOM, SCREEN_LEVEL6_ENTRANCE
from zelda_i.level6_dungeon import ROOM_39_SPEC, make_clear_39_controller
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
    "EXIT_OW_LEGS",
    "EXIT_OW_MAX_FRAMES",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6ExitOwController",
    "level6_exit_ow_stages",
    "level6_exit_ow_success",
    "level6_north39_stages",
    "level6_north39_success",
    "make_exit_ow_controller",
    "make_north39_controller",
]

EAST_DOOR_X = 208
EAST_DOOR_Y = 141
SOUTH_DOOR_X = 120
SOUTH_DOOR_Y = 189
SOUTH_BAND_Y = 181
DOOR_TOL = 4
# West spawn ~x=32 is outside OccupancyGrid xmin=40.
WEST_SPAWN_XMIN = 16
EXIT_OW_MAX_FRAMES = 20000
EXIT_OW_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
# (room, door button, occupancy goal). Next dungeon room is the following row;
# last row dest is OW SCREEN_LEVEL6_ENTRANCE.
EXIT_OW_LEGS: tuple[tuple[int, str, tuple[int, int]], ...] = (
    (LEVEL6_BLOCK_3A_ROOM, "LEFT", (WEST_CLIP_X, EAST_DOOR_Y)),
    (0x39, "UP", (NORTH_DOOR_X, NORTH_DOOR_Y)),
    (0x29, "UP", (NORTH_DOOR_X, NORTH_DOOR_Y)),
    (0x19, "LEFT", (WEST_CLIP_X, EAST_DOOR_Y)),
    (0x18, "DOWN", (SOUTH_DOOR_X, SOUTH_DOOR_Y)),
    (0x28, "DOWN", (SOUTH_DOOR_X, SOUTH_DOOR_Y)),
    (0x38, "DOWN", (SOUTH_DOOR_X, SOUTH_DOOR_Y)),
    (0x48, "DOWN", (SOUTH_DOOR_X, SOUTH_DOOR_Y)),
    (0x58, "DOWN", (SOUTH_DOOR_X, SOUTH_DOOR_Y)),
    (0x68, "DOWN", (SOUTH_DOOR_X, SOUTH_DOOR_Y)),
    (0x78, "RIGHT", (EAST_DOOR_X, EAST_DOOR_Y)),
    (LEVEL6_ENTRY_ROOM, "DOWN", (SOUTH_DOOR_X, SOUTH_DOOR_Y)),
)
EXIT_OW_NEXT: dict[int, int] = {
    EXIT_OW_LEGS[i][0]: EXIT_OW_LEGS[i + 1][0]
    for i in range(len(EXIT_OW_LEGS) - 1)
}


def _new_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))


@dataclass
class Level6ExitOwController:
    """Occupancy leave from cleared 0x3A. Stop: OW 0x22, or enter-stop room."""

    spec_id: str = "level6_exit_ow_0x22"
    max_frames: int = EXIT_OW_MAX_FRAMES
    frames: int = 0
    hop: int = 0
    keys: int = -1
    stop_room: int | None = None
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=_new_walker)
    fighter: Any = None

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

    def _room(self) -> int:
        return EXIT_OW_LEGS[self.hop][0]

    def _dir(self) -> str:
        return EXIT_OW_LEGS[self.hop][1]

    def _goal(self) -> tuple[int, int]:
        return EXIT_OW_LEGS[self.hop][2]

    def _ow(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == SCREEN_LEVEL6_ENTRANCE
            and self._rod(snap) != 0
            and snap.triforce == 0x1F
        )

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.stop_room is None:
            return False
        return (
            snap.level == LEVEL6
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.stop_room
            and self._rod(snap) != 0
            and snap.triforce == 0x1F
        )

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "level": int(snap.level),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
            "rod": self._rod(snap),
            "bow": self._bow(snap),
            "arrows": self._arrows(snap),
            "tile": int(snap.colliding_tile),
            "hop": int(self.hop),
            "room": int(self._room()) if self.hop < len(EXIT_OW_LEGS) else -1,
        }
        if force or self.frames <= 2 or self.frames % EXIT_OW_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "level": int(snap.level),
                    "reason": action.reason,
                    "rod": self._rod(snap),
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
        reason = (
            "entered_29"
            if self.stop_room == LEVEL6_DARK_29_ROOM
            else "exited_ow"
        )
        return self._emit(
            snap, FrameAction(nes_idle_action(), reason), force=True
        )

    def _enter_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"enter_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_rod={self._rod(snap)}_tf={snap.triforce:02x}"
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
        if direction == "RIGHT" and xy[0] >= gx - DOOR_TOL:
            if abs(xy[1] - gy) > DOOR_TOL:
                return ("UP" if xy[1] > gy else "DOWN", "east_align")
            return ("RIGHT", "east_push")
        if direction == "UP" and xy[1] <= NORTH_BAND_Y:
            if abs(xy[0] - gx) > DOOR_TOL:
                return ("LEFT" if xy[0] > gx else "RIGHT", "north_align")
            return ("UP", "north_push")
        if direction == "DOWN" and xy[1] >= SOUTH_BAND_Y:
            if abs(xy[0] - gx) > DOOR_TOL:
                return ("LEFT" if xy[0] > gx else "RIGHT", "south_align")
            return ("DOWN", "south_push")
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.keys < 0:
            self.keys = int(snap.keys)
            for i, (room, _, _) in enumerate(EXIT_OW_LEGS):
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
                    f"_mode={snap.mode}_hop={self.hop}_rod={self._rod(snap)}"
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
        if self._ow(snap):
            if self.stop_room is not None:
                return self._fail(
                    snap,
                    f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
                )
            if self._room() != LEVEL6_ENTRY_ROOM and self.hop < len(EXIT_OW_LEGS) - 1:
                return self._fail(
                    snap,
                    f"ow_early_{self._room():02x}_{snap.link_x}_{snap.link_y}",
                )
            return self._mark_done(
                snap,
                f"ow_22_{snap.link_x}_{snap.link_y}_rod={self._rod(snap)}"
                f"_tf={snap.triforce:02x}_keys={int(snap.keys)}",
            )
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            return FrameAction(nes_action(self._dir()), f"{self._dir().lower()}_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level == 0:
            self.walker.last_dir = None
            if self.stop_room is not None:
                return self._fail(
                    snap,
                    f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
                )
            return self._emit(snap, FrameAction(nes_action("DOWN"), "ow_settle"))
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}_{snap.screen:02x}")
        dest = EXIT_OW_NEXT.get(self._room())
        if snap.screen != self._room():
            if dest is not None and snap.screen == dest:
                self._advance(snap, int(snap.screen))
                self.walker.last_dir = None
                if self._at_stop(snap):
                    return self._mark_done(snap, self._enter_note(snap))
                return self._emit(
                    snap,
                    FrameAction(nes_action(self._dir()), "room_settle"),
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
                if self.stop_room == LEVEL6_DARK_29_ROOM
                else (
                    "occupancy 0x3A west→0x39 north→0x29/0x19 west→0x18 south "
                    "chain→0x78 east→0x79 DOWN OW 0x22; LEFT+UP after 0x39 "
                    "east-mouth miss; reclear 0x39 Vires before north; no stairs3a"
                )
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "hop": self.hop,
            "keys": self.keys,
        }


def make_exit_ow_controller() -> Level6ExitOwController:
    """Occupancy leave L6 from 0x3A. Do not poke bow/arrows/doors."""
    return Level6ExitOwController()


def make_north39_controller() -> Level6ExitOwController:
    """0x3A west → reclear 0x39 → enter 0x29. Do not inland. Do not OW."""
    return Level6ExitOwController(
        spec_id="level6_north39_0x29",
        stop_room=LEVEL6_DARK_29_ROOM,
    )


def level6_exit_ow_stages():
    """Play 0x3A leftover (144,141) → occupancy leave → OW 0x22."""
    ctl = make_exit_ow_controller()
    return (
        ("level6_exit_ow_0x22", ctl, ctl.max_frames),
    )


def level6_exit_ow_success(snap: ZeldaSnapshot) -> bool:
    """OW play 0x22 level 0, rod owned, TF 0x1F. No stairs3a dest."""
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == SCREEN_LEVEL6_ENTRANCE
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )


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
        and int(getattr(snap, "rod", 0)) != 0
    )
