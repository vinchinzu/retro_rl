"""Level 6 0x09 left-block stairs toward Magical Rod.

Clear leftover (112,173); live left 0x68 (96,144). Reuse 0x38 south-face UP
until that object's y drops ≥8px. v13 SW (48,172) tile 119 idle 3887f is
floor, not a hole. Remaining 0x68 slot11 (208,96) — v10 no_right_0x68 was
early. Next: south-face that NE 0x68 until y-move, then still-stand.
CheckWarp needs still; do not hold-UP. Do not occupancy. Do not grant
ADDR_ROD. Halt y>=181.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
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
    "ne_block_0x68",
]

STAIRS_09_MAX_FRAMES = 4000
STAIRS_09_SAMPLE_PERIOD = 8
STAIRS_09_SOUTH_HALT_Y = 181
STAIRS_09_IDLE_MIN = 240
# v13: left 0x68 y-moves then slot11 jumps 96,131 → 208,96 (~16f later).
NE_BLOCK_X_MIN = 184


def ne_block_0x68(snap: ZeldaSnapshot) -> ZeldaObject | None:
    """NE 0x68 (x>=184). Do not pick the still-despawning left block."""
    blocks = [
        obj
        for obj in snap.objects
        if int(obj.type_id) == BLOCK_OBJECT_TYPE and int(obj.x) >= NE_BLOCK_X_MIN
    ]
    if not blocks:
        return None
    return max(blocks, key=lambda obj: (int(obj.x), int(obj.y)))


class Stairs09Phase(Enum):
    TO_PUSH = auto()
    PUSH = auto()
    TO_NE = auto()
    PUSH_NE = auto()
    IDLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Stairs09Controller:
    """Left 0x68 y-move, then south-face NE 0x68. Success is mode 9."""

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
    leftover: dict[str, Any] = field(default_factory=dict)
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

    def _unlock(self) -> None:
        self.block_slot = None
        self.block_x0 = None
        self.block_y0 = None

    def _lock(self, block: ZeldaObject, tag: str) -> None:
        if self.block_slot is not None:
            return
        self.block_slot = int(block.slot)
        self.block_x0 = int(block.x)
        self.block_y0 = int(block.y)
        self.notes.append(f"{tag}_{block.slot}_{block.x}_{block.y}")

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        self._set_phase(Stairs09Phase.FAILED, note)
        return self._emit(
            snap, FrameAction(nes_idle_action(), note), force=True
        )

    def _find_block(self, snap: ZeldaSnapshot, *, ne: bool) -> ZeldaObject | None:
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
        return ne_block_0x68(snap) if ne else left_block_0x68(snap)

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

    def _blocks_68(self, snap: ZeldaSnapshot) -> list[dict[str, int]]:
        return [
            {"slot": int(obj.slot), "x": int(obj.x), "y": int(obj.y)}
            for obj in snap.objects
            if int(obj.type_id) == BLOCK_OBJECT_TYPE
        ]

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        ne = self.phase in (
            Stairs09Phase.TO_NE,
            Stairs09Phase.PUSH_NE,
            Stairs09Phase.IDLE,
        )
        block = self._find_block(snap, ne=ne)
        blocks = self._blocks_68(snap)
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "submode": int(snap.submode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "rod": int(getattr(snap, "rod", 0)),
            "bx": -1 if block is None else int(block.x),
            "by": -1 if block is None else int(block.y),
            "blocks": blocks,
            "keys": int(snap.keys),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
            "idle_frames": int(self.idle_frames),
        }
        if force or self.frames <= 2 or self.frames % STAIRS_09_SAMPLE_PERIOD == 0:
            buttons = [
                name
                for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
                if idx is not None and int(action.action[idx])
            ]
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "submode": int(snap.submode),
                    "screen": int(snap.screen),
                    "phase": self.phase.name,
                    "reason": action.reason,
                    "action": "none" if not buttons else "+".join(buttons),
                    "tile": int(snap.colliding_tile),
                    "rod": int(getattr(snap, "rod", 0)),
                    "bx": None if block is None else int(block.x),
                    "by": None if block is None else int(block.y),
                    "blocks": blocks,
                    "idle_frames": int(self.idle_frames),
                    "misses": self.walker.misses,
                }
            )
        return action

    def _axis_to_south_face(
        self, snap: ZeldaSnapshot, xy: tuple[int, int], block: ZeldaObject, tag: str
    ) -> FrameAction | None:
        tx, ty = south_face_stand(block)
        if abs(xy[0] - tx) <= PUSH_ALIGN_TOL and abs(xy[1] - ty) <= PUSH_ALIGN_TOL:
            return None
        self.walker.last_dir = None
        if xy[1] < ty - PUSH_ALIGN_TOL:
            return self._emit(snap, FrameAction(nes_action("DOWN"), f"{tag}_y"))
        if abs(xy[0] - tx) > PUSH_ALIGN_TOL:
            btn = "LEFT" if xy[0] > tx else "RIGHT"
            return self._emit(snap, FrameAction(nes_action(btn), f"{tag}_x"))
        btn = "UP" if xy[1] > ty else "DOWN"
        return self._emit(snap, FrameAction(nes_action(btn), f"{tag}_y"))

    def _hold_push(
        self, snap: ZeldaSnapshot, xy: tuple[int, int], *, ne: bool, reason: str
    ) -> FrameAction | None:
        block = self._find_block(snap, ne=ne)
        if block is None:
            return self._fail(snap, f"lost_block_{xy[0]}_{xy[1]}")
        if self.block_y0 is None:
            self.block_x0 = int(block.x)
            self.block_y0 = int(block.y)
        if int(block.y) <= int(self.block_y0) - PUSH_MOVED_PX:
            self.walker.last_dir = None
            self.walker.path = None
            note = (
                f"pushed_{self.block_x0}_{self.block_y0}"
                f"_to_{int(block.x)}_{int(block.y)}"
            )
            if ne:
                self._set_phase(Stairs09Phase.IDLE, note)
            else:
                self._unlock()
                self._set_phase(Stairs09Phase.TO_NE, note)
            return None
        if self.phase_frames >= PUSH_38_MAX_HOLD:
            return self._fail(
                snap,
                f"push_no_move_{xy[0]}_{xy[1]}"
                f"_block_{int(block.x)}_{int(block.y)}",
            )
        self.walker.last_dir = None
        return self._emit(snap, FrameAction(nes_action("UP"), reason))

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
            block = self._find_block(snap, ne=False)
            if block is None:
                if self.phase_frames >= WAIT_BLOCK_MAX:
                    return self._fail(snap, f"no_block_0x68_{xy[0]}_{xy[1]}")
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_block")
                )
            self._lock(block, "left_block")
            walked = self._axis_to_south_face(snap, xy, block, "stand")
            if walked is not None:
                return walked
            self._set_phase(
                Stairs09Phase.PUSH,
                f"at_push_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
            )

        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        if self.phase is Stairs09Phase.PUSH:
            held = self._hold_push(snap, xy, ne=False, reason="push_left_block")
            if held is not None:
                return held

        if self.phase is Stairs09Phase.TO_NE:
            block = self._find_block(snap, ne=True)
            if block is None:
                if self.phase_frames >= WAIT_BLOCK_MAX:
                    return self._fail(snap, f"no_ne_0x68_{xy[0]}_{xy[1]}")
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_ne_block")
                )
            self._lock(block, "ne_block")
            walked = self._axis_to_south_face(snap, xy, block, "ne")
            if walked is not None:
                return walked
            self._set_phase(
                Stairs09Phase.PUSH_NE,
                f"at_ne_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
            )

        if self.phase is Stairs09Phase.PUSH_NE:
            held = self._hold_push(snap, xy, ne=True, reason="push_ne_block")
            if held is not None:
                return held

        if self.phase is Stairs09Phase.IDLE:
            self.idle_frames += 1
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "stairs_idle")
            )

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
                "axis south-face left 0x68 until y-move, then NE 0x68 "
                "(208,96) south-face UP, idle"
            ),
            "idle_frames": int(self.idle_frames),
            "idle_min": STAIRS_09_IDLE_MIN,
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
    """Push left 0x68 then NE (208,96) 0x68 in 0x09. Do not grant ADDR_ROD."""
    return Level6Stairs09Controller()
