"""Level 4 maze token-path controllers and timing knobs.

Room specs stay in ``level4_dungeon``. Hold-scripts use ``HoldTokenPath``;
death/timeout/scroll share ``MazeHop``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import DungeonPhase, GenericDungeonRoomController
from zelda_i.dungeon.hop_controller import axis_dir
from zelda_i.level4.dungeon import (
    COMPASS_PICKUP_XY,
    KEY_40_PICKUP_XY,
    KEY_61_EAST_Y,
    LEVEL4,
    LEVEL4_COMPASS_BIT,
    MAZE_31_EAST_X_MIN,
    MAZE_31_EAST_Y,
    MAZE_31_EAST_Y_TOL,
    ROOM_40_SPEC,
    ROOM_L4_COMPASS_62,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    ROOM_L4_ZOLS_40,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.dungeon.token_path import HoldTokenPath

# 0x12 → Gleeok: hold4 PATH_12_TO_GLEEOK (rr-rvae dual).
PUSH_12_HOLD = 70  # frames holding LEFT at stand
RIGHT_12_HOLD = 4
PATH_12_TO_GLEEOK: tuple[str, ...] = (
    ("RIGHT",) * 6 + ("DOWN",) + ("RIGHT",) * 16 + ("UP",) + ("RIGHT",) * 7
)

# Dark-maze 0x62 compass (rr-9so0). Pickup ~(136,132); return west then LEFT → 0x61.
MAZE_IN_HOLD = 6
MAZE_OUT_HOLD = 4
MAZE_62_TO_COMPASS: tuple[str, ...] = (
    ("DOWN",) * 4 + ("RIGHT",) * 4 + ("UP",) * 3 + ("RIGHT",) + ("UP",) * 3
)
MAZE_62_RETURN_WEST: tuple[str, ...] = (
    ("DOWN",) * 3 + ("LEFT",) + ("DOWN",) * 4 + ("LEFT",) + ("DOWN",)
    + ("LEFT",) * 9 + ("UP",) + ("LEFT",) * 8 + ("UP",) + ("LEFT",) * 2
    + ("UP",) + ("LEFT",) + ("UP",) * 3 + ("LEFT",)
)

# 0x50 → 0x40 (rr-xc3x). Center+UP is blocked; waypoint seek then long UP.
# MAZE_50_TO_NORTH is fallback/docs (hold MAZE_50_HOLD from ≈(160,149)).
MAZE_50_HOLD = 6
MAZE_50_LONG_UP = 280
MAZE_50_TO_NORTH: tuple[str, ...] = (
    ("DOWN",) * 4 + ("LEFT",) * 6 + ("UP",) * 8 + ("RIGHT",) * 2
    + ("UP",) * 3 + ("LEFT",) + ("UP",) * 4
)
MAZE_50_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (160, 181), (112, 181), (112, 120), (128, 100), (120, 72), (120, 56),
)

MAP_21_HOLD = 6
MAP_21_SAMPLE_PATH: tuple[str, ...] = (
    ("LEFT",) * 2 + ("UP",) + ("RIGHT",) * 22 + ("DOWN",) * 11
)

# 0x40 key (rr-q8eq / rr-zavx): ALIGN to KEY_40_PATH_ANCHOR then hold6 maze.
MAZE_40_KEY_HOLD = 6
KEY_40_PATH_ANCHOR = (136, 165)
MAZE_40_TO_KEY: tuple[str, ...] = (
    ("UP",) * 2 + ("RIGHT",) * 5 + ("UP",) * 4 + ("LEFT",) * 5
)
KEY_40_HUNT: tuple[tuple[int, int], ...] = (
    (136, 117), (120, 117), KEY_40_PICKUP_XY, (128, 117),
    (112, 117), (136, 125), (120, 109),
)

WAIT_SCROLL = (4, 6, 7)
DEATH_MODE = 17
STALL_LIMIT = 24


def _idle(reason: str) -> FrameAction:
    return FrameAction(nes_idle_action(), reason)


def _act(direction: str, reason: str) -> FrameAction:
    return FrameAction(nes_action(direction), reason)


def _path(tokens: tuple[str, ...], hold: int) -> Any:
    return field(default_factory=lambda: HoldTokenPath(tokens, hold))


def _walk_toward(snap: ZeldaSnapshot, tx: int, ty: int) -> str:
    return axis_dir(
        (int(snap.link_x), int(snap.link_y)), (tx, ty), y_first=False, tol=4
    ) or "UP"


def _key40_hunt_dir(snap: ZeldaSnapshot, phase_frames: int) -> str:
    tx, ty = KEY_40_HUNT[min(phase_frames // 120, len(KEY_40_HUNT) - 1)]
    if abs(snap.link_x - tx) > 5 or abs(snap.link_y - ty) > 5:
        return _walk_toward(snap, tx, ty)
    return ("LEFT", "UP", "RIGHT", "DOWN")[(phase_frames // 8) % 4]


@dataclass
class MazeHop:
    """Death / timeout / wait-level / scroll / wait-mode preamble."""

    max_frames: int = 4000
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    path_index: int = 0
    scroll_hold: str | None = None
    dest_screen: int | None = None
    play_room: int | None = None
    arrive_note: str = "done"
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def on_phase(self, phase: Any) -> None:
        pass

    def scroll_dir(self, snap: ZeldaSnapshot) -> str | None:
        return self.scroll_hold

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return self.dest_screen is not None and snap.screen == self.dest_screen

    def policy(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        return _idle("idle")

    def _set_phase(self, phase: Any, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self._stall = 0
            if note:
                self.notes.append(note)
            self.on_phase(phase)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(type(self.phase).FAILED, note)
        return _idle(note)

    def _mark_done(self, note: str) -> FrameAction:
        self.success = True
        self._set_phase(type(self.phase).DONE, note)
        return _idle("done")

    def _sample(self, snap: ZeldaSnapshot, reason: str) -> None:
        sample = {
            "frame": self.frames, "x": int(snap.link_x), "y": int(snap.link_y),
            "phase": self.phase.name, "path_index": self.path_index,
            "reason": reason, "stall": self._stall,
        }
        prev = self.samples[-1] if self.samples else None
        if prev is None or prev["reason"] != reason or self.frames - prev["frame"] >= 250:
            self.samples.append(sample)

    def _stall_fail(self, snap: ZeldaSnapshot, reason: str, xy: tuple[int, int]) -> FrameAction:
        self._sample(snap, reason)
        return self._fail(f"{reason}_{xy[0]}_{xy[1]}")

    def hold(self, path: HoldTokenPath, prefix: str) -> FrameAction | None:
        direction = path.advance()
        self.path_index = path.index
        if direction is None:
            return None
        return _act(direction, f"{prefix}_{direction}")

    def thread(
        self,
        xy: tuple[int, int],
        snap: ZeldaSnapshot,
        wps: tuple[tuple[int, int], ...],
        prefix: str,
        on_done: Callable[[], FrameAction],
    ) -> FrameAction:
        if self._stall >= STALL_LIMIT:
            return self._stall_fail(snap, "thread_stuck", xy)
        if self.path_index < len(wps):
            gx, gy = wps[self.path_index]
            if abs(xy[0] - gx) <= 4 and abs(xy[1] - gy) <= 4:
                self._sample(snap, f"waypoint_{self.path_index}")
                self.path_index += 1
                self._stall = 0
        if self.path_index >= len(wps):
            return on_done()
        gx, gy = wps[self.path_index]
        direction = axis_dir(xy, (gx, gy), y_first=False, tol=4) or "UP"
        return _act(direction, f"{prefix}_{direction}")

    def maze_guard(self, snap: ZeldaSnapshot, *, level: int = 4) -> FrameAction | None:
        name = self.phase.name
        if name in ("DONE", "FAILED"):
            return _idle(name.lower())
        if snap.mode == DEATH_MODE:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            self._sample(snap, "timeout")
            return self._fail("timeout")
        if snap.level != level:
            return _idle("wait_level4")
        if snap.transitioning or snap.mode in WAIT_SCROLL:
            scroll = self.scroll_dir(snap)
            if scroll:
                return _act(scroll, f"scroll_{scroll.lower()}")
            return _idle(f"wait_scroll_{snap.mode}")
        if snap.mode == 8:
            return _idle("hurt_freeze")
        if snap.mode != PLAY_MODE:
            return _idle(f"wait_mode_{snap.mode}")
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        self._stall = self._stall + 1 if self._last_xy == xy else 0
        self._last_xy = xy
        blocked = self.maze_guard(snap, level=LEVEL4)
        if blocked is not None:
            return blocked
        if self.arrived(snap):
            self._sample(snap, self.arrive_note)
            return self._mark_done(self.arrive_note)
        if self.play_room is not None and snap.screen != self.play_room:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")
        return self.policy(snap, xy)

    def report_base(self, segment: str, **extra: Any) -> dict[str, Any]:
        return {
            "success": self.success, "phase": self.phase.name,
            "frames": self.frames, "notes": list(self.notes),
            "path_index": self.path_index, "segment": segment, **extra,
        }


class Compass62Phase(Enum):
    MAZE_IN = auto()
    MAZE_OUT = auto()
    EXIT_LEFT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Compass62Controller(MazeHop):
    """Cleared 0x62: HoldTokenPath to compass, return west, LEFT to 0x61."""

    max_frames: int = 12000
    phase: Compass62Phase = Compass62Phase.MAZE_IN
    compass_at_frame: int | None = None
    arrive_note: str = "compass_and_0x61"
    maze_in: HoldTokenPath = _path(MAZE_62_TO_COMPASS, MAZE_IN_HOLD)
    maze_out: HoldTokenPath = _path(MAZE_62_RETURN_WEST, MAZE_OUT_HOLD)

    def on_phase(self, phase: Any) -> None:
        if phase is Compass62Phase.MAZE_IN:
            self.maze_in.reset()
        elif phase is Compass62Phase.MAZE_OUT:
            self.maze_out.reset()

    def scroll_dir(self, snap: ZeldaSnapshot) -> str | None:
        if self.phase is Compass62Phase.EXIT_LEFT or snap.screen in (
            ROOM_L4_COMPASS_62, ROOM_L4_VIRES_61,
        ):
            return "LEFT"
        return None

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return bool(snap.compass & LEVEL4_COMPASS_BIT) and snap.screen == ROOM_L4_VIRES_61

    def policy(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        if bool(snap.compass & LEVEL4_COMPASS_BIT) and self.compass_at_frame is None:
            self.compass_at_frame = self.frames
            self.notes.append(f"compass_bit_f{self.frames}")

        if self.phase is Compass62Phase.MAZE_IN:
            if snap.screen != ROOM_L4_COMPASS_62:
                return self._fail(f"maze_in_wrong_room_0x{snap.screen:02x}")
            if bool(snap.compass & LEVEL4_COMPASS_BIT):
                self._set_phase(Compass62Phase.MAZE_OUT, "got_compass")
            else:
                act = self.hold(self.maze_in, "maze_in")
                return act if act else self._fail("maze_in_path_exhausted_no_compass")

        if self.phase is Compass62Phase.MAZE_OUT:
            if snap.screen == ROOM_L4_VIRES_61:
                self._set_phase(Compass62Phase.EXIT_LEFT, "already_0x61")
            elif snap.screen != ROOM_L4_COMPASS_62:
                return self._fail(f"maze_out_wrong_room_0x{snap.screen:02x}")
            else:
                act = self.hold(self.maze_out, "maze_out")
                if act is None:
                    self._set_phase(Compass62Phase.EXIT_LEFT, "return_path_done")
                else:
                    return act

        if snap.screen not in (ROOM_L4_COMPASS_62, ROOM_L4_VIRES_61):
            return self._fail(f"exit_wrong_room_0x{snap.screen:02x}")
        if snap.screen == ROOM_L4_COMPASS_62 and abs(snap.link_y - KEY_61_EAST_Y) > 8:
            return _act("UP" if snap.link_y > KEY_61_EAST_Y else "DOWN", "align_exit_y")
        return _act("LEFT", "exit_left")

    def report(self) -> dict[str, Any]:
        return self.report_base(
            "level4_compass_0x62", compass_at_frame=self.compass_at_frame,
            maze_in=list(MAZE_62_TO_COMPASS), maze_out=list(MAZE_62_RETURN_WEST),
            pickup_xy=list(COMPASS_PICKUP_XY),
        )


def make_compass_62_controller() -> Level4Compass62Controller:
    return Level4Compass62Controller()


class North40Phase(Enum):
    WAYPOINTS = auto()
    PUSH_UP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4North40Controller(MazeHop):
    """Cleared 0x50: coordinate-gated walk, then UP into 0x40."""

    max_frames: int = 10000
    phase: North40Phase = North40Phase.WAYPOINTS
    dest_screen: int | None = ROOM_L4_ZOLS_40
    play_room: int | None = ROOM_L4_VIRES_50
    arrive_note: str = "entered_0x40"

    def scroll_dir(self, snap: ZeldaSnapshot) -> str | None:
        if snap.screen in (ROOM_L4_VIRES_50, ROOM_L4_ZOLS_40):
            return "UP"
        return None

    def policy(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        if snap.link_y <= 80 and abs(snap.link_x - 120) <= 16:
            self._set_phase(North40Phase.PUSH_UP, "near_north_band")
            return _act("UP", "push_up_north")

        if self.phase is North40Phase.WAYPOINTS:
            goals = MAZE_50_WAYPOINTS[:-1]
            if self.path_index < len(goals):
                gx, gy = goals[self.path_index]
                if abs(xy[0] - gx) <= 4 and abs(xy[1] - gy) <= 4:
                    self._sample(snap, f"waypoint_{self.path_index}")
                    self.path_index += 1
                    if self.path_index < len(goals):
                        gx, gy = goals[self.path_index]
            if self.path_index >= len(goals):
                self._set_phase(North40Phase.PUSH_UP, "north_band_reached")
                return _act("UP", "push_up_north")
            gx, gy = goals[self.path_index]
            if self.path_index == 4 and xy[1] > 96:
                direction = "UP" if xy[1] > gy else "DOWN"
            else:
                direction = axis_dir(xy, (gx, gy), y_first=False, tol=4) or "UP"
            if self._stall >= 18:
                self._sample(snap, f"stuck_{direction}_to_{gx}_{gy}")
                return self._fail(f"waypoint_{self.path_index}_stuck_{direction}")
            return _act(direction, f"maze50_seek_{self.path_index}_{direction}")

        if self.phase_frames >= MAZE_50_LONG_UP + 120:
            self._sample(snap, "push_up_timeout")
            return self._fail("push_up_timeout")
        if self._stall and self._stall % 250 == 0:
            self._sample(snap, "push_up_stuck")
        return _act("UP", "push_up_north")

    def report(self) -> dict[str, Any]:
        return self.report_base(
            "level4_north_0x40",
            waypoints=[list(w) for w in MAZE_50_WAYPOINTS],
            maze_path=list(MAZE_50_TO_NORTH), hold=MAZE_50_HOLD,
            long_up=MAZE_50_LONG_UP, samples=list(self.samples),
        )


def make_north_40_controller() -> Level4North40Controller:
    return Level4North40Controller()


class Key40Phase(Enum):
    FIGHT = auto()
    ALIGN = auto()
    PATH = auto()
    HUNT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Key40Controller(MazeHop):
    """Clear 0x40 then ``MAZE_40_TO_KEY`` hold6; ALIGN to KEY_40_PATH_ANCHOR first."""

    max_frames: int = 25000
    phase: Key40Phase = Key40Phase.FIGHT
    keys_before: int | None = None
    play_room: int | None = ROOM_L4_ZOLS_40
    arrive_note: str = "key_collected"
    maze: HoldTokenPath = _path(MAZE_40_TO_KEY, MAZE_40_KEY_HOLD)
    _clear: GenericDungeonRoomController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._clear = GenericDungeonRoomController(ROOM_40_SPEC)
        self._clear.phase = DungeonPhase.FIGHT

    def on_phase(self, phase: Any) -> None:
        if phase is Key40Phase.PATH:
            self.maze.reset()

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.screen == ROOM_L4_ZOLS_40
            and self.keys_before is not None
            and snap.keys > self.keys_before
            and len(ROOM_40_SPEC.live_enemies(snap)) == 0
        )

    def _hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.phase_frames >= 1200:
            return self._fail("key_hunt_timeout")
        return _act(_key40_hunt_dir(snap, self.phase_frames), "key_hunt")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.keys_before is None and snap.screen == ROOM_L4_ZOLS_40:
            self.keys_before = snap.keys
        return super().step(snap)

    def policy(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        if self.phase is Key40Phase.FIGHT:
            live = ROOM_40_SPEC.live_enemies(snap)
            if (
                not live
                and self._clear.max_live_enemies >= ROOM_40_SPEC.expected_enemy_count
            ):
                self._set_phase(Key40Phase.ALIGN, "room_cleared")
            else:
                return self._clear.step(snap)

        if self.phase is Key40Phase.ALIGN:
            ax, ay = KEY_40_PATH_ANCHOR
            if abs(snap.link_x - ax) <= 6 and abs(snap.link_y - ay) <= 6:
                self._set_phase(Key40Phase.PATH, "aligned_path_anchor")
            elif self.phase_frames >= 900:
                self._set_phase(Key40Phase.PATH, "align_timeout")
            else:
                d = _walk_toward(snap, ax, ay)
                return _act(d, f"align_{d}")

        if self.phase is Key40Phase.PATH:
            act = self.hold(self.maze, "maze40")
            if act is None:
                self._set_phase(Key40Phase.HUNT, "path_done")
            else:
                return act

        if self.phase is Key40Phase.HUNT:
            return self._hunt(snap)
        return _idle("idle")

    def report(self) -> dict[str, Any]:
        return self.report_base(
            "level4_key_0x40", keys_before=self.keys_before,
            maze_path=list(MAZE_40_TO_KEY), hold=MAZE_40_KEY_HOLD,
            path_anchor=list(KEY_40_PATH_ANCHOR),
            pickup_xy=list(KEY_40_PICKUP_XY), clear=self._clear.report(),
        )


def make_room_40_key_controller() -> Level4Key40Controller:
    return Level4Key40Controller()


# 0x31 west-door leftover. Cardinals stick at ~(32,141); RIGHT+UP clips
# into ~(48,133), then waypoints to mid-maze ~(128,133).
MAZE_31_INLAND_X = 48
MAZE_31_ALCOVE_Y = 141
MAZE_31_ALCOVE_Y_TOL = 8
MAZE_31_MID = (128, 133)
MAZE_31_MID_TOL = 8
MAZE_31_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (48, 109), (80, 109), (80, 173), MAZE_31_MID,
)


class Maze31InlandPhase(Enum):
    CLIP = auto()
    THREAD = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Maze31InlandController(MazeHop):
    """West-door leftover → RIGHT+UP clip → waypoints to mid-maze."""

    phase: Maze31InlandPhase = Maze31InlandPhase.CLIP
    scroll_hold: str | None = "RIGHT"
    play_room: int | None = ROOM_L4_EAST_31
    arrive_note: str = "mid_maze"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        gx, gy = MAZE_31_MID
        return (
            abs(int(snap.link_x) - gx) <= MAZE_31_MID_TOL
            and abs(int(snap.link_y) - gy) <= 12
        )

    def _left_alcove(self, snap: ZeldaSnapshot) -> bool:
        x, y = int(snap.link_x), int(snap.link_y)
        return x >= MAZE_31_INLAND_X or (
            x >= 40 and abs(y - MAZE_31_ALCOVE_Y) > MAZE_31_ALCOVE_Y_TOL
        )

    def _snap_waypoint(self, xy: tuple[int, int]) -> None:
        self.path_index = min(
            range(len(MAZE_31_WAYPOINTS)),
            key=lambda i: (
                abs(MAZE_31_WAYPOINTS[i][0] - xy[0])
                + abs(MAZE_31_WAYPOINTS[i][1] - xy[1]),
                -i,
            ),
        )

    def policy(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        if self.phase is Maze31InlandPhase.CLIP:
            if self._left_alcove(snap):
                self._sample(snap, "left_alcove")
                self._set_phase(Maze31InlandPhase.THREAD, "left_alcove")
                self._snap_waypoint(xy)
            elif self._stall >= STALL_LIMIT:
                return self._stall_fail(snap, "alcove_clip_stuck", xy)
            elif abs(xy[1] - MAZE_31_ALCOVE_Y) <= MAZE_31_ALCOVE_Y_TOL:
                return FrameAction(nes_action("RIGHT", "UP"), "maze31_alcove_clip")
            else:
                return _act("RIGHT", "maze31_alcove_right")
        return self.thread(
            xy, snap, MAZE_31_WAYPOINTS, "maze31_thread",
            lambda: self._mark_done("mid_maze"),
        )

    def report(self) -> dict[str, Any]:
        return self.report_base(
            "level4_inland_0x31", mid=list(MAZE_31_MID),
            waypoints=[list(w) for w in MAZE_31_WAYPOINTS],
            samples=list(self.samples),
        )


def make_maze_31_inland_controller() -> Level4Maze31InlandController:
    return Level4Maze31InlandController()


# 0x31 leftover (112,141): UP off water, SE clip, south U, then RIGHT → 0x32.
MAZE_31_NORTH_STRIP_Y = 113
MAZE_31_SE_X = 132
MAZE_31_SE_Y = 125
MAZE_31_EAST_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (160, 173), (192, 173), (192, 141), (200, 141),
)
MAZE_31_EAST_PUSH = 280


class Maze31EastPhase(Enum):
    JOIN = auto()
    CLIP = auto()
    THREAD = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Maze31EastController(MazeHop):
    """Cleared 0x31 leftover → waypoints → free RIGHT into 0x32."""

    phase: Maze31EastPhase = Maze31EastPhase.JOIN
    scroll_hold: str | None = "RIGHT"
    dest_screen: int | None = ROOM_L4_EAST_32
    play_room: int | None = ROOM_L4_EAST_31
    arrive_note: str = "entered_0x32"

    def _at_east_band(self, snap: ZeldaSnapshot) -> bool:
        return (
            int(snap.link_x) >= MAZE_31_EAST_X_MIN
            and abs(int(snap.link_y) - MAZE_31_EAST_Y) <= MAZE_31_EAST_Y_TOL
        )

    def _push_east(self, xy: tuple[int, int], snap: ZeldaSnapshot, note: str) -> FrameAction:
        if self.phase is not Maze31EastPhase.PUSH:
            self._sample(snap, "east_band")
            self._set_phase(Maze31EastPhase.PUSH, note)
        if self.phase_frames >= MAZE_31_EAST_PUSH:
            self._sample(snap, "push_right_timeout")
            return self._fail("push_right_timeout")
        if abs(xy[1] - KEY_61_EAST_Y) > MAZE_31_EAST_Y_TOL:
            return _act("DOWN" if xy[1] < KEY_61_EAST_Y else "UP", "maze31_east_align_y")
        return _act("RIGHT", "maze31_east_push")

    def policy(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        if self._at_east_band(snap) or self.phase is Maze31EastPhase.PUSH:
            return self._push_east(xy, snap, "east_band")

        if self.phase is Maze31EastPhase.JOIN:
            if xy[1] <= MAZE_31_NORTH_STRIP_Y:
                self._sample(snap, "north_strip")
                self._set_phase(Maze31EastPhase.CLIP, "north_strip")
            elif self._stall >= STALL_LIMIT:
                return self._stall_fail(snap, "join_stuck", xy)
            else:
                return _act("UP", "maze31_east_join_UP")

        if self.phase is Maze31EastPhase.CLIP:
            if xy[0] >= MAZE_31_SE_X and xy[1] >= MAZE_31_SE_Y:
                self._sample(snap, "se_corridor")
                self._set_phase(Maze31EastPhase.THREAD, "se_corridor")
                self.path_index = 0
            elif self._stall >= STALL_LIMIT:
                return self._stall_fail(snap, "se_clip_stuck", xy)
            else:
                return FrameAction(nes_action("RIGHT", "DOWN"), "maze31_east_se_clip")

        return self.thread(
            xy, snap, MAZE_31_EAST_WAYPOINTS, "maze31_east",
            lambda: self._push_east(xy, snap, "waypoints_done"),
        )

    def report(self) -> dict[str, Any]:
        return self.report_base(
            "level4_east_0x32",
            waypoints=[list(w) for w in MAZE_31_EAST_WAYPOINTS],
            samples=list(self.samples),
        )


def make_maze_31_east_controller() -> Level4Maze31EastController:
    return Level4Maze31EastController()
