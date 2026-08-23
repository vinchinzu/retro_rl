"""Level 4 maze token-path controllers and timing knobs.

Room specs and stop predicates remain in ``level4_dungeon``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.level4_dungeon import (
    COMPASS_PICKUP_XY,
    KEY_40_PICKUP_XY,
    KEY_61_EAST_Y,
    LEVEL4,
    LEVEL4_COMPASS_BIT,
    ROOM_40_SPEC,
    ROOM_L4_COMPASS_62,
    ROOM_L4_EAST_31,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    ROOM_L4_ZOLS_40,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

# 0x12 → Gleeok: hold4 PATH_12_TO_GLEEOK (rr-rvae dual).
PUSH_12_HOLD = 70  # frames holding LEFT at stand
RIGHT_12_HOLD = 4
PATH_12_TO_GLEEOK: tuple[str, ...] = (
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "DOWN",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
)

# Dark-maze 0x62 compass (rr-9so0 live BFS from Level4Room62Cleared).
# Hold each token for MAZE_IN_HOLD / MAZE_OUT_HOLD frames. Pickup ~ (136,132).
# After compass, corridor path back to west vestibule then LEFT → 0x61 play.
MAZE_IN_HOLD = 6
MAZE_OUT_HOLD = 4
MAZE_62_TO_COMPASS: tuple[str, ...] = (
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "UP",
)
MAZE_62_RETURN_WEST: tuple[str, ...] = (
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "LEFT",
)

# 0x50 → 0x40 north (rr-xc3x live). Interior blocks block center+UP.
# Prefer waypoint seek (robust to clear_50 end pose) then long UP.
# Token path kept as fallback / docs (hold MAZE_50_HOLD from ≈(160,149)).
MAZE_50_HOLD = 6
MAZE_50_LONG_UP = 280
MAZE_50_TO_NORTH: tuple[str, ...] = (
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "UP",
)
# Live intermediate cells on successful BFS (tol ±8).
MAZE_50_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (160, 181),
    (112, 181),
    (112, 120),
    (128, 100),
    (120, 72),
    (120, 56),
)

# Sample path from a common post-thrash pose (hold MAP_21_HOLD):
MAP_21_HOLD = 6
MAP_21_SAMPLE_PATH: tuple[str, ...] = (
    "LEFT",
    "LEFT",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
)

# Live scripted key path after combat clear pose ≈(136–140, 164–165).
# East-corridor route (rr-q8eq BFS): UP×2 RIGHT×5 UP×4 LEFT×5 hold6 → key ~(136,117).
# Clear pose varies (skip-compass combat often ends ~west mid); ALIGN to
# ``KEY_40_PATH_ANCHOR`` before the maze so PATH is pose-stable (rr-zavx).
MAZE_40_KEY_HOLD = 6
KEY_40_PATH_ANCHOR = (136, 165)
MAZE_40_TO_KEY: tuple[str, ...] = (
    "UP",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
)


# --- 0x62 dark maze: compass + return LEFT → 0x61 (rr-9so0) ---


class Compass62Phase(Enum):
    MAZE_IN = auto()
    MAZE_OUT = auto()
    EXIT_LEFT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Compass62Controller:
    """From cleared 0x62: maze to compass pickup, return west, LEFT to 0x61.

    Live (rr-9so0): hold scripted dirs (BFS) — open seek fails on maze walls.
    Success: ``ADDR_COMPASS & 0x08`` and play-ready on 0x61.
    """

    max_frames: int = 12000
    phase: Compass62Phase = Compass62Phase.MAZE_IN
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    hold_left: int = 0
    success: bool = False
    compass_at_frame: int | None = None
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Compass62Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.path_index = 0
            self.hold_left = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Compass62Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Compass62Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Compass62Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            bool(snap.compass & LEVEL4_COMPASS_BIT)
            and snap.level == LEVEL4
            and snap.screen == ROOM_L4_VIRES_61
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(Compass62Phase.DONE, "compass_and_0x61")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")

        # Scroll through door transitions while exiting west.
        if snap.transitioning or snap.mode in (4, 6, 7):
            if self.phase is Compass62Phase.EXIT_LEFT or (
                snap.screen in (ROOM_L4_COMPASS_62, ROOM_L4_VIRES_61)
            ):
                return FrameAction(nes_action("LEFT"), "scroll_left")
            return FrameAction(nes_idle_action(), f"wait_scroll_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if bool(snap.compass & LEVEL4_COMPASS_BIT) and self.compass_at_frame is None:
            self.compass_at_frame = self.frames
            self.notes.append(f"compass_bit_f{self.frames}")

        if self.phase is Compass62Phase.MAZE_IN:
            if snap.screen != ROOM_L4_COMPASS_62:
                return self._fail(f"maze_in_wrong_room_0x{snap.screen:02x}")
            if bool(snap.compass & LEVEL4_COMPASS_BIT):
                self._set_phase(Compass62Phase.MAZE_OUT, "got_compass")
                # fall through to MAZE_OUT this frame
            else:
                if self.path_index >= len(MAZE_62_TO_COMPASS):
                    return self._fail("maze_in_path_exhausted_no_compass")
                direction = MAZE_62_TO_COMPASS[self.path_index]
                self.hold_left += 1
                if self.hold_left >= MAZE_IN_HOLD:
                    self.path_index += 1
                    self.hold_left = 0
                return FrameAction(nes_action(direction), f"maze_in_{direction}")

        if self.phase is Compass62Phase.MAZE_OUT:
            if snap.screen == ROOM_L4_VIRES_61:
                self._set_phase(Compass62Phase.EXIT_LEFT, "already_0x61")
            elif snap.screen != ROOM_L4_COMPASS_62:
                return self._fail(f"maze_out_wrong_room_0x{snap.screen:02x}")
            elif self.path_index >= len(MAZE_62_RETURN_WEST):
                self._set_phase(Compass62Phase.EXIT_LEFT, "return_path_done")
            else:
                direction = MAZE_62_RETURN_WEST[self.path_index]
                self.hold_left += 1
                if self.hold_left >= MAZE_OUT_HOLD:
                    self.path_index += 1
                    self.hold_left = 0
                return FrameAction(nes_action(direction), f"maze_out_{direction}")

        # EXIT_LEFT: push west door / finish settle on 0x61
        if snap.screen == ROOM_L4_VIRES_61 and snap.mode == PLAY_MODE:
            if bool(snap.compass & LEVEL4_COMPASS_BIT) and not snap.transitioning:
                self.success = True
                self._set_phase(Compass62Phase.DONE, "settled_0x61")
                return FrameAction(nes_idle_action(), "done")
        if snap.screen not in (ROOM_L4_COMPASS_62, ROOM_L4_VIRES_61):
            return self._fail(f"exit_wrong_room_0x{snap.screen:02x}")
        # Align y≈141 when still in 0x62 vestibule, then LEFT.
        if snap.screen == ROOM_L4_COMPASS_62 and abs(snap.link_y - KEY_61_EAST_Y) > 8:
            return FrameAction(
                nes_action("UP" if snap.link_y > KEY_61_EAST_Y else "DOWN"),
                "align_exit_y",
            )
        return FrameAction(nes_action("LEFT"), "exit_left")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "compass_at_frame": self.compass_at_frame,
            "path_index": self.path_index,
            "segment": "level4_compass_0x62",
            "maze_in": list(MAZE_62_TO_COMPASS),
            "maze_out": list(MAZE_62_RETURN_WEST),
            "pickup_xy": list(COMPASS_PICKUP_XY),
        }


def make_compass_62_controller() -> Level4Compass62Controller:
    return Level4Compass62Controller()


# --- 0x50 cleared → north scripted → 0x40 (rr-xc3x) ---


class North40Phase(Enum):
    WAYPOINTS = auto()
    PUSH_UP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4North40Controller:
    """From cleared 0x50: coordinate-gated walk then UP into 0x40.

    Continuous v7: coordinate gates are reliable from the live clear_50 pose.
    The final corner is UP at x≈128 to y≈93, LEFT to x≈120, then long UP.
    """

    max_frames: int = 10000
    phase: North40Phase = North40Phase.WAYPOINTS  # WAYPOINTS = token path phase
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    hold_left: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def _sample(self, snap: ZeldaSnapshot, reason: str) -> None:
        """Keep transition/final and sparse stuck evidence in the report."""
        sample = {
            "frame": self.frames,
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "phase": self.phase.name,
            "path_index": self.path_index,
            "reason": reason,
            "stall": self._stall,
        }
        if not self.samples or self.samples[-1] != sample:
            self.samples.append(sample)

    def _set_phase(self, phase: North40Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.path_index = 0
            self.hold_left = 0
            self._stall = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(North40Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _entered_40(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_ZOLS_40
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if self._last_xy == xy:
            self._stall += 1
        else:
            self._stall = 0
            self._last_xy = xy

        if self.phase is North40Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is North40Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if self._entered_40(snap):
            self._sample(snap, "entered_0x40")
            self.success = True
            self._set_phase(North40Phase.DONE, "entered_0x40")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")

        if snap.transitioning or snap.mode in (4, 6, 7):
            if snap.screen in (ROOM_L4_VIRES_50, ROOM_L4_ZOLS_40):
                return FrameAction(nes_action("UP"), "scroll_up")
            return FrameAction(nes_idle_action(), f"wait_scroll_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if snap.screen == ROOM_L4_ZOLS_40:
            self.success = True
            self._set_phase(North40Phase.DONE, "on_0x40")
            return FrameAction(nes_idle_action(), "done")

        if snap.screen != ROOM_L4_VIRES_50:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        # Early north-band boost: if already near door, just push UP.
        if snap.link_y <= 80 and abs(snap.link_x - 120) <= 16:
            self._set_phase(North40Phase.PUSH_UP, "near_north_band")
            return FrameAction(nes_action("UP"), "push_up_north")

        if self.phase is North40Phase.WAYPOINTS:
            # The last documented waypoint is above ordinary room movement.
            # Reach the playable north band, then use the door residual.
            goals = MAZE_50_WAYPOINTS[:-1]
            if self.path_index >= len(goals):
                self._set_phase(North40Phase.PUSH_UP, "north_band_reached")
                return FrameAction(nes_action("UP"), "push_up_north")
            gx, gy = goals[self.path_index]
            if abs(xy[0] - gx) <= 4 and abs(xy[1] - gy) <= 4:
                self._sample(snap, f"waypoint_{self.path_index}")
                self.path_index += 1
                if self.path_index >= len(goals):
                    self._set_phase(North40Phase.PUSH_UP, "north_band_reached")
                    return FrameAction(nes_action("UP"), "push_up_north")
                gx, gy = goals[self.path_index]
            # Each leg changes one axis. This makes the live prediction
            # falsifiable and turns at the first verified band instead of
            # skipping a blocked open-loop token.
            vertical_first = self.path_index == 4 and xy[1] > 96
            if vertical_first:
                direction = "UP" if xy[1] > gy else "DOWN"
            elif abs(xy[0] - gx) > 4:
                direction = "LEFT" if xy[0] > gx else "RIGHT"
            else:
                direction = "UP" if xy[1] > gy else "DOWN"
            if self._stall >= 18:
                self._sample(snap, f"stuck_{direction}_to_{gx}_{gy}")
                return self._fail(f"waypoint_{self.path_index}_stuck_{direction}")
            return FrameAction(nes_action(direction), f"maze50_seek_{self.path_index}_{direction}")

        if self.phase_frames >= MAZE_50_LONG_UP + 120:
            self._sample(snap, "push_up_timeout")
            return self._fail("push_up_timeout")
        if self._stall and self._stall % 250 == 0:
            self._sample(snap, "push_up_stuck")
        return FrameAction(nes_action("UP"), "push_up_north")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "path_index": self.path_index,
            "segment": "level4_north_0x40",
            "waypoints": [list(w) for w in MAZE_50_WAYPOINTS],
            "maze_path": list(MAZE_50_TO_NORTH),
            "hold": MAZE_50_HOLD,
            "long_up": MAZE_50_LONG_UP,
            "samples": list(self.samples),
        }


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
class Level4Key40Controller:
    """Clear 0x40 Zols+gels then scripted path to RoomItemId 0x19 key.

    Live (rr-q8eq): wooden sword splits Zol→Gel; after clear, center-band key
    is not reachable via naive mid-room patrol (south pocket walls). Use
    ``MAZE_40_TO_KEY`` hold6 from the common clear pose.

    rr-zavx: after clear, ALIGN to ``KEY_40_PATH_ANCHOR`` first — skip-compass
    combat end pose ~(72,125) makes the maze path miss the key band.
    """

    max_frames: int = 25000
    phase: Key40Phase = Key40Phase.FIGHT
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    hold_left: int = 0
    success: bool = False
    keys_before: int | None = None
    notes: list[str] = field(default_factory=list)
    _clear: GenericDungeonRoomController = field(init=False, repr=False)
    _hunt_i: int = 0
    _hunt_targets: tuple[tuple[int, int], ...] = (
        (136, 117),
        (120, 117),
        KEY_40_PICKUP_XY,
        (128, 117),
        (112, 117),
        (136, 125),
        (120, 109),
    )

    def __post_init__(self) -> None:
        self._clear = GenericDungeonRoomController(ROOM_40_SPEC)
        self._clear.phase = DungeonPhase.FIGHT

    def _set_phase(self, phase: Key40Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Key40Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _walk_toward(self, snap: ZeldaSnapshot, tx: int, ty: int) -> str:
        if abs(snap.link_x - tx) > 4:
            return "RIGHT" if snap.link_x < tx else "LEFT"
        if abs(snap.link_y - ty) > 4:
            return "DOWN" if snap.link_y < ty else "UP"
        return "UP"

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Key40Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Key40Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if self.keys_before is None and snap.screen == ROOM_L4_ZOLS_40:
            self.keys_before = snap.keys

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_ZOLS_40
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and self.keys_before is not None
            and snap.keys > self.keys_before
            and len(ROOM_40_SPEC.live_enemies(snap)) == 0
        ):
            self.success = True
            self._set_phase(Key40Phase.DONE, "key_collected")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_idle_action(), f"wait_scroll_{snap.mode}")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_ZOLS_40:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

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
                self.path_index = 0
                self.hold_left = 0
            elif self.phase_frames >= 900:
                # Give up on anchor; still try maze from current pose + hunt.
                self._set_phase(Key40Phase.PATH, "align_timeout")
                self.path_index = 0
                self.hold_left = 0
            else:
                d = self._walk_toward(snap, ax, ay)
                return FrameAction(nes_action(d), f"align_{d}")

        if self.phase is Key40Phase.PATH:
            if self.path_index >= len(MAZE_40_TO_KEY):
                self._set_phase(Key40Phase.HUNT, "path_done")
            else:
                direction = MAZE_40_TO_KEY[self.path_index]
                self.hold_left += 1
                if self.hold_left >= MAZE_40_KEY_HOLD:
                    self.path_index += 1
                    self.hold_left = 0
                return FrameAction(nes_action(direction), f"maze40_{direction}")

        if self.phase is Key40Phase.HUNT:
            # Walk key-band waypoints then orbit (pose-stable recovery).
            if self.phase_frames >= 1200:
                return self._fail("key_hunt_timeout")
            tgt_i = min(
                self.phase_frames // 120, len(self._hunt_targets) - 1
            )
            tx, ty = self._hunt_targets[tgt_i]
            if abs(snap.link_x - tx) > 5 or abs(snap.link_y - ty) > 5:
                d = self._walk_toward(snap, tx, ty)
            else:
                orbit = ("LEFT", "UP", "RIGHT", "DOWN")
                d = orbit[(self.phase_frames // 8) % len(orbit)]
            return FrameAction(nes_action(d), "key_hunt")

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "path_index": self.path_index,
            "keys_before": self.keys_before,
            "segment": "level4_key_0x40",
            "maze_path": list(MAZE_40_TO_KEY),
            "hold": MAZE_40_KEY_HOLD,
            "path_anchor": list(KEY_40_PATH_ANCHOR),
            "pickup_xy": list(KEY_40_PICKUP_XY),
            "clear": self._clear.report(),
        }


def make_room_40_key_controller() -> Level4Key40Controller:
    """Clear 0x40 Zols+gels + collect key via scripted east-corridor path."""
    return Level4Key40Controller()


# 0x31 west-door leftover. Cardinals stick at ~(32,141) on the maze wall
# (continuous v1/v2). RIGHT+UP clips into the vertical corridor ~(48,133);
# coordinate waypoints then thread to mid-maze ~(128,133).
MAZE_31_INLAND_X = 48
MAZE_31_ALCOVE_Y = 141
MAZE_31_ALCOVE_Y_TOL = 8
MAZE_31_MID = (128, 133)
MAZE_31_MID_TOL = 8
MAZE_31_NORTH_Y = 109
# v5 (80,109) RIGHT no-op → DOWN. v6 (80,141) RIGHT no-op → keep DOWN
# to the south channel, then RIGHT/UP to mid-maze.
MAZE_31_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (48, 109),
    (80, 109),
    (80, 173),
    MAZE_31_MID,
)


class Maze31InlandPhase(Enum):
    CLIP = auto()
    THREAD = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Maze31InlandController:
    """West-door leftover → RIGHT+UP clip → coordinate waypoints to mid-maze."""

    max_frames: int = 4000
    phase: Maze31InlandPhase = Maze31InlandPhase.CLIP
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    path_index: int = 0
    _last_xy: tuple[int, int] | None = None
    _stall: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def _sample(self, snap: ZeldaSnapshot, reason: str) -> None:
        sample = {
            "frame": self.frames,
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "phase": self.phase.name,
            "reason": reason,
            "stall": self._stall,
        }
        if (
            not self.samples
            or self.samples[-1]["reason"] != reason
            or self.frames - self.samples[-1]["frame"] >= 250
        ):
            self.samples.append(sample)

    def _set_phase(self, phase: Maze31InlandPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self._stall = 0
            if phase is Maze31InlandPhase.THREAD:
                self.path_index = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Maze31InlandPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _left_alcove(self, snap: ZeldaSnapshot) -> bool:
        x, y = int(snap.link_x), int(snap.link_y)
        if x >= MAZE_31_INLAND_X:
            return True
        return x >= 40 and abs(y - MAZE_31_ALCOVE_Y) > MAZE_31_ALCOVE_Y_TOL

    def _at_mid(self, snap: ZeldaSnapshot) -> bool:
        gx, gy = MAZE_31_MID
        return (
            abs(int(snap.link_x) - gx) <= MAZE_31_MID_TOL
            and abs(int(snap.link_y) - gy) <= 12
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if self._last_xy == xy:
            self._stall += 1
        else:
            self._stall = 0
            self._last_xy = xy

        if self.phase is Maze31InlandPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Maze31InlandPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            self._sample(snap, "timeout")
            return self._fail("timeout")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll_right")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_EAST_31:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self._at_mid(snap):
            self._sample(snap, "mid_maze")
            self.success = True
            self._set_phase(Maze31InlandPhase.DONE, "mid_maze")
            return FrameAction(nes_idle_action(), "done")

        if self.phase is Maze31InlandPhase.CLIP:
            if self._left_alcove(snap):
                self._sample(snap, "left_alcove")
                self._set_phase(Maze31InlandPhase.THREAD, "left_alcove")
                self._snap_waypoint(xy)
            elif self._stall >= 24:
                self._sample(snap, "alcove_clip_stuck")
                return self._fail(f"alcove_clip_stuck_{xy[0]}_{xy[1]}")
            else:
                if abs(xy[1] - MAZE_31_ALCOVE_Y) <= MAZE_31_ALCOVE_Y_TOL:
                    return FrameAction(
                        nes_action("RIGHT", "UP"), "maze31_alcove_clip"
                    )
                return FrameAction(nes_action("RIGHT"), "maze31_alcove_right")

        return self._thread_step(xy, snap)

    def _snap_waypoint(self, xy: tuple[int, int]) -> None:
        self.path_index = min(
            range(len(MAZE_31_WAYPOINTS)),
            key=lambda i: (
                abs(MAZE_31_WAYPOINTS[i][0] - xy[0])
                + abs(MAZE_31_WAYPOINTS[i][1] - xy[1]),
                -i,
            ),
        )

    def _thread_step(
        self, xy: tuple[int, int], snap: ZeldaSnapshot
    ) -> FrameAction:
        if self._stall >= 24:
            self._sample(snap, "thread_stuck")
            return self._fail(f"thread_stuck_{xy[0]}_{xy[1]}")
        if self.path_index >= len(MAZE_31_WAYPOINTS):
            self._sample(snap, "mid_maze")
            self.success = True
            self._set_phase(Maze31InlandPhase.DONE, "mid_maze")
            return FrameAction(nes_idle_action(), "done")
        gx, gy = MAZE_31_WAYPOINTS[self.path_index]
        if abs(xy[0] - gx) <= 4 and abs(xy[1] - gy) <= 4:
            self._sample(snap, f"waypoint_{self.path_index}")
            self.path_index += 1
            self._stall = 0
            if self.path_index >= len(MAZE_31_WAYPOINTS):
                self.success = True
                self._set_phase(Maze31InlandPhase.DONE, "mid_maze")
                return FrameAction(nes_idle_action(), "done")
            gx, gy = MAZE_31_WAYPOINTS[self.path_index]
        if abs(xy[0] - gx) > 4:
            direction = "RIGHT" if xy[0] < gx else "LEFT"
        else:
            direction = "DOWN" if xy[1] < gy else "UP"
        return FrameAction(nes_action(direction), f"maze31_thread_{direction}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_inland_0x31",
            "mid": list(MAZE_31_MID),
            "path_index": self.path_index,
            "waypoints": [list(w) for w in MAZE_31_WAYPOINTS],
            "samples": list(self.samples),
        }


def make_maze_31_inland_controller() -> Level4Maze31InlandController:
    return Level4Maze31InlandController()
