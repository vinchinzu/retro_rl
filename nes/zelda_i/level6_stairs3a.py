"""Level 6 cleared 0x3A center-block stairs.

Leftover play 0x3A (144,141) rod=1 keys=4 bombs=8 TF=0x1F map=0x0A
bow=0 arrows=0; west door open; bubble residual. Live center 0x68
south-face UP until y-move / warp. OccupancyWalker to the south face; LEFT+DOWN clip after a live miss
(v1 leftover (144,141) tile 118 boxed 4-cardinal). Success is mode 9
or a new play room.
Do not invent Gohma. Do not poke ADDR_BOW / ADDR_ARROWS / doors / keys.
Do not grant Map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_gleeok18 import PASSAGE_MODE
from zelda_i.level6_overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.level6_path import (
    BLOCK_OBJECT_TYPE,
    PUSH_ALIGN_TOL,
    PUSH_38_MAX_HOLD,
    PUSH_MOVED_PX,
    WAIT_BLOCK_MAX,
    south_face_stand,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "STAIRS_3A_MAX_FRAMES",
    "Stairs3APhase",
    "Level6Stairs3AController",
    "center_block_0x68",
    "level6_stairs3a_stages",
    "level6_stairs3a_success",
    "make_stairs_3a_controller",
]

STAIRS_3A_MAX_FRAMES = 4000
STAIRS_3A_SAMPLE_PERIOD = 8
# Search key for the center 0x68 — not a walk target.
_CENTER_XY = (120, 144)


def center_block_0x68(snap: ZeldaSnapshot) -> ZeldaObject | None:
    """0x68 closest to room center. Ignore Bubble 0x40 / invuln 0x2b."""
    blocks = [
        obj for obj in snap.objects if int(obj.type_id) == BLOCK_OBJECT_TYPE
    ]
    if not blocks:
        return None
    cx, cy = _CENTER_XY
    return min(
        blocks,
        key=lambda obj: abs(int(obj.x) - cx) + abs(int(obj.y) - cy),
    )


class Stairs3APhase(Enum):
    TO_PUSH = auto()
    PUSH = auto()
    ON_HOLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Stairs3AController:
    """Occupancy south-face of live center 0x68, then UP. Dest is RAM."""

    spec_id: str = "level6_stairs_0x3a"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = STAIRS_3A_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: Stairs3APhase = Stairs3APhase.TO_PUSH
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    block_slot: int | None = None
    block_x0: int | None = None
    block_y0: int | None = None

    def _set_phase(self, phase: Stairs3APhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _lock(self, block: ZeldaObject) -> None:
        if self.block_slot is not None:
            return
        self.block_slot = int(block.slot)
        self.block_x0 = int(block.x)
        self.block_y0 = int(block.y)
        self.notes.append(f"center_block_{block.slot}_{block.x}_{block.y}")

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        self._set_phase(Stairs3APhase.FAILED, note)
        return self._emit(
            snap, FrameAction(nes_idle_action(), note), force=True
        )

    def _find_block(self, snap: ZeldaSnapshot) -> ZeldaObject | None:
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
        return center_block_0x68(snap)

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

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        block = self._find_block(snap)
        blocks = self._blocks_68(snap)
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "submode": int(snap.submode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "rod": self._rod(snap),
            "bow": self._bow(snap),
            "arrows": self._arrows(snap),
            "bx": -1 if block is None else int(block.x),
            "by": -1 if block is None else int(block.y),
            "blocks": blocks,
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
        }
        if force or self.frames <= 2 or self.frames % STAIRS_3A_SAMPLE_PERIOD == 0:
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
                    "rod": self._rod(snap),
                    "bow": self._bow(snap),
                    "arrows": self._arrows(snap),
                    "bx": None if block is None else int(block.x),
                    "by": None if block is None else int(block.y),
                    "blocks": blocks,
                    "keys": int(snap.keys),
                    "misses": self.walker.misses,
                }
            )
        return action

    def _at_south_face(
        self, xy: tuple[int, int], block: ZeldaObject
    ) -> bool:
        tx, ty = south_face_stand(block)
        return (
            abs(xy[0] - tx) <= PUSH_ALIGN_TOL
            and abs(xy[1] - ty) <= PUSH_ALIGN_TOL
        )

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
                Stairs3APhase.DONE,
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
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        if self.phase is Stairs3APhase.TO_PUSH:
            block = self._find_block(snap)
            if block is None:
                if self.phase_frames >= WAIT_BLOCK_MAX:
                    return self._fail(snap, f"no_block_0x68_{xy[0]}_{xy[1]}")
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_block")
                )
            self._lock(block)
            if self._at_south_face(xy, block):
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(
                    Stairs3APhase.PUSH,
                    f"at_push_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
                )
            else:
                dest = south_face_stand(block)
                # v1 leftover (144,141) tile 118 boxed 4-cardinal.
                if self.walker.misses > 0:
                    self.walker.last_dir = None
                    if (
                        xy[0] > dest[0] + PUSH_ALIGN_TOL
                        and xy[1] < dest[1] - PUSH_ALIGN_TOL
                    ):
                        return self._emit(
                            snap,
                            FrameAction(nes_action("LEFT", "DOWN"), "stand_clip"),
                        )
                    if xy[1] < dest[1] - PUSH_ALIGN_TOL:
                        return self._emit(
                            snap, FrameAction(nes_action("DOWN"), "stand_y")
                        )
                    if abs(xy[0] - dest[0]) > PUSH_ALIGN_TOL:
                        btn = "LEFT" if xy[0] > dest[0] else "RIGHT"
                        return self._emit(
                            snap, FrameAction(nes_action(btn), "stand_x")
                        )
                    btn = "UP" if xy[1] > dest[1] else "DOWN"
                    return self._emit(
                        snap, FrameAction(nes_action(btn), "stand_y")
                    )
                if dest != self.walker.goal:
                    self.walker.path = None
                    self.walker.goal = dest
                direction = self.walker.next_dir(xy, dest)
                if direction is None:
                    self.walker.last_dir = None
                    if self.frames <= 8 or self.frames % 60 == 0:
                        self.notes.append(
                            f"stand_f{self.frames}_{xy[0]}_{xy[1]}"
                        )
                    return self._emit(
                        snap, FrameAction(nes_idle_action(), "stand_wait")
                    )
                return self._emit(
                    snap, FrameAction(nes_action(direction), "stand_path")
                )

        if self.phase is Stairs3APhase.PUSH:
            block = self._find_block(snap)
            if block is None:
                return self._fail(snap, f"lost_block_{xy[0]}_{xy[1]}")
            if self.block_y0 is None:
                self.block_x0 = int(block.x)
                self.block_y0 = int(block.y)
            if int(block.y) <= int(self.block_y0) - PUSH_MOVED_PX:
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(
                    Stairs3APhase.ON_HOLE,
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
                    snap, FrameAction(nes_action("UP"), "push_block")
                )

        if self.phase is Stairs3APhase.ON_HOLE:
            self.walker.last_dir = None
            hx = int(self.block_x0 or xy[0])
            hy = int(self.block_y0 or xy[1])
            if abs(xy[0] - hx) <= PUSH_ALIGN_TOL and abs(xy[1] - hy) <= PUSH_ALIGN_TOL:
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "hole_idle")
                )
            if abs(xy[0] - hx) > PUSH_ALIGN_TOL:
                btn = "LEFT" if xy[0] > hx else "RIGHT"
                return self._emit(
                    snap, FrameAction(nes_action(btn), "hole_x")
                )
            btn = "UP" if xy[1] > hy else "DOWN"
            return self._emit(
                snap, FrameAction(nes_action(btn), "hole_y")
            )

        return self._emit(
            snap, FrameAction(nes_idle_action(), "failed"), force=True
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "phase": self.phase.name,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "occupancy south-face center 0x68; LEFT+DOWN clip after "
                "tile-118 miss; UP until y-move; then original-xy idle. "
                "dest is RAM (mode 9 or play != 0x3A)"
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


def make_stairs_3a_controller() -> Level6Stairs3AController:
    """South-face center 0x68 in 0x3A. Do not poke bow/arrows/doors."""
    return Level6Stairs3AController()


def level6_stairs3a_stages():
    """0x3A leftover → south-face center 0x68. Dest is RAM. No Gohma."""
    stairs = make_stairs_3a_controller()
    return (
        ("level6_stairs_0x3a", stairs, STAIRS_3A_MAX_FRAMES),
    )


def level6_stairs3a_success(snap: ZeldaSnapshot) -> bool:
    """Mode 9 cellar or a new L6 play room. Rod and TF 0x1F stay."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if int(getattr(snap, "rod", 0)) == 0:
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    return (
        snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != LEVEL6_BLOCK_3A_ROOM
    )
