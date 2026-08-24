"""Level 6 0x09 left-block stairs toward Magical Rod.

Clear leftover (112,173); live left 0x68 (96,144). Reuse 0x38 south-face UP
until that object's y drops ≥8px. CheckWarp needs still; do not hold-UP.
NE hole is decorative (v8 on-graphic tile 0x77). v9 vacated (96,144) yo-yos
143/145. v10: no live pair 0x68 (visual right block is a tile; rx=-1).
v11 idle NW (48,109) tile 118 still mode 5. Next SW (48,173). Halt y>=181.
Do not occupancy. Do not grant ADDR_ROD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_gleeok18 import PASSAGE_MODE
from zelda_i.level6_overworld import LEVEL6, LEVEL6_ROD_WIZZ_ROOM
from zelda_i.level6_path import (
    BLOCK_OBJECT_TYPE,
    PUSH_ALIGN_TOL,
    PUSH_38_MAX_HOLD,
    PUSH_MOVED_PX,
    WAIT_BLOCK_MAX,
    left_block_0x68,
    south_face_stand,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "STAIRS_09_MAX_FRAMES",
    "Stairs09Phase",
    "Level6Stairs09Controller",
    "make_stairs_09_controller",
]

STAIRS_09_MAX_FRAMES = 4000
STAIRS_09_SAMPLE_PERIOD = 12
# Exact idle; CheckWarp misses ALIGN_TOL. Halt south mouth (don't spend key).
STAIR_ALIGN_TOL = 0
STAIRS_09_SOUTH_HALT_Y = 181
# v11 idle (48,109) tile 118 not warp. Next SW after left y-move.
STAIRS_09_HOLE = (48, 173)


class Stairs09Phase(Enum):
    TO_PUSH = auto()
    PUSH = auto()
    TO_HOLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Stairs09Controller:
    """Left 0x68 y-move, then idle SW (48,173). Success is mode 9."""

    spec_id: str = "level6_stairs_0x09"
    room: int = LEVEL6_ROD_WIZZ_ROOM
    max_frames: int = STAIRS_09_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    idle_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: Stairs09Phase = Stairs09Phase.TO_PUSH
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    block_slot: int | None = None
    block_x0: int | None = None
    block_y0: int | None = None

    def _set_phase(self, phase: Stairs09Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        self._set_phase(Stairs09Phase.FAILED, note)
        return self._emit(
            snap, FrameAction(nes_idle_action(), note), force=True
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

    def _warped(self, snap: ZeldaSnapshot) -> bool:
        if snap.level != LEVEL6:
            return False
        if snap.mode == PASSAGE_MODE:
            return True
        return (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.room
        )

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        block = self._block(snap)
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "bx": -1 if block is None else int(block.x),
            "by": -1 if block is None else int(block.y),
            "keys": int(snap.keys),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
        }
        if force or self.frames <= 2 or self.frames % STAIRS_09_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "phase": self.phase.name,
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "bx": None if block is None else int(block.x),
                    "by": None if block is None else int(block.y),
                    "misses": self.walker.misses,
                }
            )
        return action

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
                    f"_mode={snap.mode}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if self._warped(snap):
            self.success = True
            self._set_phase(
                Stairs09Phase.DONE,
                f"warped_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
            self.walker.last_dir = None
            return self._emit(
                snap,
                FrameAction(nes_idle_action(), f"warped_{snap.mode}"),
                force=True,
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 10):
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self._fail(
                snap, f"left_0x{self.room:02x}_to_0x{snap.screen:02x}"
            )

        xy = (int(snap.link_x), int(snap.link_y))
        if xy[1] >= STAIRS_09_SOUTH_HALT_Y:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "south_halt")
            )

        if self.phase is Stairs09Phase.TO_PUSH:
            block = self._block(snap)
            if block is None:
                if self.phase_frames >= WAIT_BLOCK_MAX:
                    return self._fail(snap, f"no_block_0x68_{xy[0]}_{xy[1]}")
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_block")
                )
            tx, ty = south_face_stand(block)
            if abs(xy[0] - tx) <= PUSH_ALIGN_TOL and abs(xy[1] - ty) <= PUSH_ALIGN_TOL:
                self._set_phase(
                    Stairs09Phase.PUSH,
                    f"at_push_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
                )
            else:
                # v1 occupancy leftover (112,173) freeze-miss boxed 4-cardinal.
                self.walker.last_dir = None
                if xy[1] < ty - PUSH_ALIGN_TOL:
                    return self._emit(
                        snap, FrameAction(nes_action("DOWN"), "stand_y")
                    )
                if abs(xy[0] - tx) > PUSH_ALIGN_TOL:
                    btn = "LEFT" if xy[0] > tx else "RIGHT"
                    return self._emit(
                        snap, FrameAction(nes_action(btn), "stand_x")
                    )
                btn = "UP" if xy[1] > ty else "DOWN"
                return self._emit(
                    snap, FrameAction(nes_action(btn), "stand_y")
                )

        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        if self.phase is Stairs09Phase.PUSH:
            block = self._block(snap)
            if block is None:
                return self._fail(snap, f"lost_block_{xy[0]}_{xy[1]}")
            if self.block_y0 is None:
                self.block_x0 = int(block.x)
                self.block_y0 = int(block.y)
            if int(block.y) <= int(self.block_y0) - PUSH_MOVED_PX:
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(
                    Stairs09Phase.TO_HOLE,
                    f"pushed_{self.block_x0}_{self.block_y0}"
                    f"_to_{int(block.x)}_{int(block.y)}",
                )
            elif self.phase_frames >= PUSH_38_MAX_HOLD:
                return self._fail(
                    snap,
                    f"push_no_move_{xy[0]}_{xy[1]}"
                    f"_block_{int(block.x)}_{int(block.y)}",
                )
            else:
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_action("UP"), "push_left_block")
                )

        if self.phase is Stairs09Phase.TO_HOLE:
            gx, gy = STAIRS_09_HOLE
            self.walker.last_dir = None
            if abs(xy[0] - gx) > STAIR_ALIGN_TOL:
                btn = "LEFT" if xy[0] > gx else "RIGHT"
                return self._emit(snap, FrameAction(nes_action(btn), "hole_x"))
            if abs(xy[1] - gy) > STAIR_ALIGN_TOL:
                btn = "UP" if xy[1] > gy else "DOWN"
                if btn == "DOWN" and xy[1] >= STAIRS_09_SOUTH_HALT_Y - 1:
                    return self._emit(
                        snap, FrameAction(nes_idle_action(), "south_halt")
                    )
                return self._emit(snap, FrameAction(nes_action(btn), "hole_y"))
            return self._emit(snap, FrameAction(nes_idle_action(), "hole_idle"))

        return self._emit(snap, FrameAction(nes_idle_action(), "failed"), force=True)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "phase": self.phase.name,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "axis south-face UP until y-move, idle SW (48,173)"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "block_slot": self.block_slot,
            "block_xy0": (
                None
                if self.block_x0 is None or self.block_y0 is None
                else [self.block_x0, self.block_y0]
            ),
            "spec_id": self.spec_id,
            "room": self.room,
        }


def make_stairs_09_controller() -> Level6Stairs09Controller:
    """Push left 0x68 in 0x09 then idle SW (48,173). Do not grant ADDR_ROD."""
    return Level6Stairs09Controller()
