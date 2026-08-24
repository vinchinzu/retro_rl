"""Level 6 interior path controllers.

OccupancyWalker first. Coordinate clips only after a live miss. Isolated
emulator-state BFS is banned. Ignore object type 0x2b. 0x68 is the left-block
push in 0x38 (sample y; do not poke). Do not poke Rod / doors / keys. Do not
grant Whistle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_KEESE_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WEST_WIZZROBE_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "BLOCK_OBJECT_TYPE",
    "NORTH_68_MAX_FRAMES",
    "NORTH_DOOR_X",
    "NORTH_DOOR_Y",
    "Level6North68Controller",
    "Level6Push38Controller",
    "left_block_0x68",
    "make_north_28_controller",
    "make_north_38_controller",
    "make_north_48_controller",
    "make_north_58_controller",
    "south_face_stand",
]

NORTH_DOOR_X = 120
NORTH_DOOR_Y = 93
NORTH_BAND_Y = 109
NORTH_DOOR_X_TOL = 4
NORTH_68_MAX_FRAMES = 4000
# West mouth x=32 boxes cardinals; clip inland before the push stand.
WEST_CLIP_X = 48
BLOCK_OBJECT_TYPE = 0x68
# Hold UP from one tile south of live 0x68 until that object's y drops 8px.
PUSH_SOUTH_OFFSET = 16
PUSH_ALIGN_TOL = 2
PUSH_MOVED_PX = 8
PUSH_38_MAX_HOLD = 600
WAIT_BLOCK_MAX = 120
PUSH_38_MAX_FRAMES = 8000
# x=120 UP from the south band hits the block pair; aisle is west of 0x68.
NORTH_WEST_X = 64


def left_block_0x68(snap: ZeldaSnapshot) -> ZeldaObject | None:
    """Westernmost live 0x68. Ignore Bubble 0x40 / invuln 0x2b."""
    blocks = [obj for obj in snap.objects if int(obj.type_id) == BLOCK_OBJECT_TYPE]
    if not blocks:
        return None
    return min(blocks, key=lambda obj: (int(obj.x), int(obj.y)))


def south_face_stand(block: ZeldaObject) -> tuple[int, int]:
    """One tile south of a 0x68. UP from here should register a push."""
    return (int(block.x), int(block.y) + PUSH_SOUTH_OFFSET)


@dataclass
class Level6North68Controller:
    """Occupancy BFS to a north door, then UP. Defaults are 0x78 → 0x68.

    Goal is play-ready dest. No combat. No path → stand.
    Door push on the north band is not occupancy-graded.
    """

    source_room: int = LEVEL6_WEST_WIZZROBE_ROOM
    dest_room: int = LEVEL6_COMPASS_ROOM
    spec_id: str = "level6_north_0x68"
    max_frames: int = NORTH_68_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def _goal(self) -> tuple[int, int]:
        return (NORTH_DOOR_X, NORTH_DOOR_Y)

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

        if (
            snap.level == LEVEL6
            and snap.screen == self.dest_room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            note = f"arrived_{self.dest_room:02x}"
            self.notes.append(note)
            return FrameAction(nes_idle_action(), note)

        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return FrameAction(nes_idle_action(), f"wait_level_{snap.level}")
        if snap.screen == self.dest_room:
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north_settle")
        if snap.screen != self.source_room:
            self.failed = True
            self.notes.append(f"left_0x{self.source_room:02x}_to_0x{snap.screen:02x}")
            return FrameAction(
                nes_idle_action(), f"left_0x{self.source_room:02x}"
            )

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        if snap.link_y <= NORTH_BAND_Y:
            self.walker.last_dir = None
            if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return FrameAction(nes_action(direction), "north_align")
            return FrameAction(nes_action("UP"), "north_push")

        direction = self.walker.next_dir(xy, self._goal())
        if direction is None:
            if abs(snap.link_x - NORTH_DOOR_X) <= 8 and snap.link_y <= 117:
                self.walker.last_dir = None
                return FrameAction(nes_action("UP"), "north_door_residual")
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "north_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "north_path")
        )

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or self.frames <= 2 or self.frames % 250 == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "room": int(snap.screen),
                    "reason": action.reason,
                }
            )
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "occupancy BFS + UP @ x≈120 on north band",
            "spec_id": self.spec_id,
            "source_room": self.source_room,
            "dest_room": self.dest_room,
        }


def make_north_58_controller() -> Level6North68Controller:
    """0x68 leftover → occupancy UP into Keese room 0x58. No fight."""
    return Level6North68Controller(
        source_room=LEVEL6_COMPASS_ROOM,
        dest_room=LEVEL6_KEESE_ROOM,
        spec_id="level6_north_0x58",
    )


def make_north_48_controller() -> Level6North68Controller:
    """0x58 leftover → occupancy UP into 0x48. Long push; do not poke doors."""
    return Level6North68Controller(
        source_room=LEVEL6_KEESE_ROOM,
        dest_room=LEVEL6_TRAPS_ROOM,
        spec_id="level6_north_0x48",
        max_frames=6000,
    )


def make_north_38_controller() -> Level6North68Controller:
    """0x48 leftover → occupancy run-UP into 0x38. Do not fight traps."""
    return Level6North68Controller(
        source_room=LEVEL6_TRAPS_ROOM,
        dest_room=LEVEL6_WIZZROBE_38_ROOM,
        spec_id="level6_north_0x38",
        max_frames=6000,
    )


class Push38Phase(Enum):
    CLIP = auto()
    TO_PUSH = auto()
    PUSH = auto()
    NORTH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Push38Controller:
    """0x38 west leftover → clip inland → live left 0x68 UP → west-aisle 0x28."""

    spec_id: str = "level6_north_0x28"
    source_room: int = LEVEL6_WIZZROBE_38_ROOM
    dest_room: int = LEVEL6_WIZZROBE_28_ROOM
    max_frames: int = PUSH_38_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: Push38Phase = Push38Phase.CLIP
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    block_slot: int | None = None
    block_x0: int | None = None
    block_y0: int | None = None

    def _set_phase(self, phase: Push38Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self.failed = True
        self._set_phase(Push38Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL6
            and snap.screen == self.dest_room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def _block(self, snap: ZeldaSnapshot) -> ZeldaObject | None:
        if self.block_slot is not None:
            found = next(
                (
                    obj
                    for obj in snap.objects
                    if obj.slot == self.block_slot
                    and int(obj.type_id) == BLOCK_OBJECT_TYPE
                ),
                None,
            )
            if found is not None:
                return found
        block = left_block_0x68(snap)
        if block is None:
            return None
        if self.block_slot is None:
            self.block_slot = int(block.slot)
            self.block_x0 = int(block.x)
            self.block_y0 = int(block.y)
            self.notes.append(f"left_block_{block.slot}_{block.x}_{block.y}")
        return block

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
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
            return self._fail("link_death")
        if self._arrived(snap):
            self.success = True
            self._set_phase(Push38Phase.DONE, f"arrived_{self.dest_room:02x}")
            return FrameAction(nes_idle_action(), f"arrived_{self.dest_room:02x}")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 9, 10):
            return FrameAction(nes_action("UP"), "north_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return FrameAction(nes_idle_action(), f"wait_level_{snap.level}")
        if snap.screen == self.dest_room:
            return FrameAction(nes_action("UP"), "north_settle")
        if snap.screen != self.source_room:
            return self._fail(f"left_0x{self.source_room:02x}_to_0x{snap.screen:02x}")

        xy = (int(snap.link_x), int(snap.link_y))
        if self.phase is Push38Phase.CLIP:
            if snap.link_x >= WEST_CLIP_X:
                self._set_phase(Push38Phase.TO_PUSH, f"inland_{xy[0]}_{xy[1]}")
            else:
                if self.frames <= 8 or self.frames % 60 == 0:
                    self.notes.append(f"west_clip_f{self.frames}_{xy[0]}_{xy[1]}")
                return self._emit(
                    snap, FrameAction(nes_action("RIGHT", "UP"), "west_clip")
                )

        if self.phase is Push38Phase.TO_PUSH:
            block = self._block(snap)
            if block is None:
                if self.phase_frames >= WAIT_BLOCK_MAX:
                    return self._fail(f"no_block_0x68_{xy[0]}_{xy[1]}")
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_block")
                )
            tx, ty = south_face_stand(block)
            if abs(xy[0] - tx) <= PUSH_ALIGN_TOL and abs(xy[1] - ty) <= PUSH_ALIGN_TOL:
                self._set_phase(
                    Push38Phase.PUSH,
                    f"at_push_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
                )
            elif xy[1] < ty - PUSH_ALIGN_TOL:
                # North of the south face: DOWN first so RIGHT misses the 0x68 west face.
                return self._emit(snap, FrameAction(nes_action("DOWN"), "stand_y"))
            elif abs(xy[0] - tx) > PUSH_ALIGN_TOL:
                direction = "LEFT" if xy[0] > tx else "RIGHT"
                return self._emit(snap, FrameAction(nes_action(direction), "stand_x"))
            else:
                direction = "UP" if xy[1] > ty else "DOWN"
                return self._emit(snap, FrameAction(nes_action(direction), "stand_y"))

        if self.phase is Push38Phase.PUSH:
            block = self._block(snap)
            if block is None:
                return self._fail(f"lost_block_{xy[0]}_{xy[1]}")
            if self.block_y0 is None:
                self.block_x0 = int(block.x)
                self.block_y0 = int(block.y)
            if int(block.y) <= int(self.block_y0) - PUSH_MOVED_PX:
                self._set_phase(
                    Push38Phase.NORTH,
                    f"pushed_{self.block_x0}_{self.block_y0}"
                    f"_to_{int(block.x)}_{int(block.y)}",
                )
            elif self.phase_frames >= PUSH_38_MAX_HOLD:
                return self._fail(
                    f"push_no_move_{xy[0]}_{xy[1]}"
                    f"_block_{int(block.x)}_{int(block.y)}"
                )
            else:
                return self._emit(
                    snap, FrameAction(nes_action("UP"), "push_left_block")
                )

        if self.phase is Push38Phase.NORTH:
            if snap.link_y <= NORTH_BAND_Y:
                if abs(snap.link_x - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
                    direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                    return FrameAction(nes_action(direction), "north_align")
                return FrameAction(nes_action("UP"), "north_push")
            if abs(xy[0] - NORTH_WEST_X) > NORTH_DOOR_X_TOL:
                direction = "LEFT" if xy[0] > NORTH_WEST_X else "RIGHT"
                return self._emit(snap, FrameAction(nes_action(direction), "north_west"))
            return self._emit(snap, FrameAction(nes_action("UP"), "north_aisle"))

        return FrameAction(nes_idle_action(), "failed")

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or self.frames <= 2 or self.frames % 250 == 0:
            block = self._block(snap) if self.block_slot is not None else left_block_0x68(snap)
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "room": int(snap.screen),
                    "phase": self.phase.name,
                    "reason": action.reason,
                    "bx": None if block is None else int(block.x),
                    "by": None if block is None else int(block.y),
                }
            )
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "phase": self.phase.name,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "west clip + live 0x68 south-face UP until y moves + west-aisle north",
            "block_slot": self.block_slot,
            "block_xy0": (
                None
                if self.block_x0 is None or self.block_y0 is None
                else [self.block_x0, self.block_y0]
            ),
            "spec_id": self.spec_id,
            "source_room": self.source_room,
            "dest_room": self.dest_room,
        }


def make_north_28_controller() -> Level6Push38Controller:
    """0x38 leftover → clip, live left 0x68 UP until y moves, west-aisle 0x28."""
    return Level6Push38Controller()
