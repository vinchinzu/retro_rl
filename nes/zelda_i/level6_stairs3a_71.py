"""Level 6 0x3A stairs: live push, then tile 0x71 still-stand.

Leftover play 0x3A (144,141) rod=1 keys=4 bombs=8 TF=0x1F bow=0 arrows=0.
Reuse stairs3a push: LEFT+DOWN clip after dated leftover miss (144,141)
tile 118, south-face UP, y-move 8px. Then stairs09 analog: NE 0x68
(x>=184; live leftover slot11 jump 208,96) south-face UP onto tile 0x71,
still-stand CheckWarp. Do not idle on tile 119 (v3 leftover 112,146).
Do not hold-UP past the hole (v2 leftover 112,133 tile 179). Occupancy
halt at first new miss. Isolated BFS banned. Do not walk east door.
Do not take 0x29. Do not invent Gohma. Do not poke ADDR_BOW / ADDR_ARROWS.
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
from zelda_i.level6_stairs09 import ne_block_0x68
from zelda_i.level6_stairs3a import center_block_0x68
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "DATED_LEFTOVER",
    "HOLE_TILE",
    "STAIRS_3A_71_MAX_FRAMES",
    "WARP_TILE",
    "Stairs3A71Phase",
    "Level6Stairs3A71Controller",
    "level6_stairs3a_71_stages",
    "level6_stairs3a_71_success",
    "make_stairs_3a_71_controller",
]

STAIRS_3A_71_MAX_FRAMES = 4000
STAIRS_3A_71_SAMPLE_PERIOD = 8
WARP_TILE = 0x71
HOLE_TILE = 119
EAST_DOOR_XMIN = 200
EAST_DOOR_Y = 141
EAST_ROOM = 0x3B
WEST_ROOM = 0x39
NORTH_29 = 0x29
KEY_UP_09 = 0x09
DATED_LEFTOVER = (144, 141)
HOLE_COLUMN_TOL = 4


class Stairs3A71Phase(Enum):
    TO_PUSH = auto()
    PUSH = auto()
    TO_NE = auto()
    PUSH_NE = auto()
    IDLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Stairs3A71Controller:
    """Push center 0x68, then still-stand tile 0x71. Dest is RAM."""

    spec_id: str = "level6_stairs_0x3a_71"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = STAIRS_3A_71_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    idle_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: Stairs3A71Phase = Stairs3A71Phase.TO_PUSH
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    block_slot: int | None = None
    block_x0: int | None = None
    block_y0: int | None = None
    hole_x: int | None = None

    def _set_phase(self, phase: Stairs3A71Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _lock(self, block: ZeldaObject, tag: str) -> None:
        if self.block_slot is not None:
            return
        self.block_slot = int(block.slot)
        self.block_x0 = int(block.x)
        self.block_y0 = int(block.y)
        self.notes.append(f"{tag}_{block.slot}_{block.x}_{block.y}")

    def _unlock(self) -> None:
        self.block_slot = None
        self.block_x0 = None
        self.block_y0 = None

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        self._set_phase(Stairs3A71Phase.FAILED, note)
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
        return ne_block_0x68(snap) if ne else center_block_0x68(snap)

    def _forbidden_play(self, screen: int) -> str | None:
        if screen == NORTH_29:
            return "north_29"
        if screen == KEY_UP_09:
            return "key_up_09"
        if screen == EAST_ROOM:
            return "east_door"
        if screen == WEST_ROOM:
            return "west_door"
        return None

    def _warped(self, snap: ZeldaSnapshot) -> bool:
        if snap.level != LEVEL6:
            return False
        if snap.mode == PASSAGE_MODE:
            return True
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.room
        ):
            return self._forbidden_play(int(snap.screen)) is None
        return False

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
        ne = self.phase in (
            Stairs3A71Phase.TO_NE,
            Stairs3A71Phase.PUSH_NE,
            Stairs3A71Phase.IDLE,
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
            "idle_frames": int(self.idle_frames),
        }
        if force or self.frames <= 2 or self.frames % STAIRS_3A_71_SAMPLE_PERIOD == 0:
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
                    "idle_frames": int(self.idle_frames),
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

    def _warp_idle(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction:
        self.idle_frames += 1
        self.walker.last_dir = None
        if "warp_tile_71" not in self.notes:
            self.notes.append(f"warp_tile_71_{xy[0]}_{xy[1]}")
        self._set_phase(Stairs3A71Phase.IDLE, f"at_71_{xy[0]}_{xy[1]}")
        return self._emit(snap, FrameAction(nes_idle_action(), "warp_idle"))

    def _axis_to_ne(
        self, snap: ZeldaSnapshot, xy: tuple[int, int], block: ZeldaObject
    ) -> FrameAction | None:
        tx, ty = south_face_stand(block)
        if self._at_south_face(xy, block):
            return None
        self.walker.last_dir = None
        if xy[0] >= EAST_DOOR_XMIN and abs(xy[1] - EAST_DOOR_Y) <= 8:
            return self._fail(
                snap, f"east_door_{xy[0]}_{xy[1]}"
            )
        hole_x = int(self.hole_x if self.hole_x is not None else xy[0])
        if (
            abs(xy[0] - hole_x) <= HOLE_COLUMN_TOL
            and xy[1] > ty + PUSH_ALIGN_TOL
        ):
            return self._emit(
                snap, FrameAction(nes_action("RIGHT"), "ne_sidestep")
            )
        if xy[1] > ty + PUSH_ALIGN_TOL and xy[0] < EAST_DOOR_XMIN:
            return self._emit(snap, FrameAction(nes_action("UP"), "ne_y"))
        if xy[1] < ty - PUSH_ALIGN_TOL:
            return self._emit(snap, FrameAction(nes_action("DOWN"), "ne_y"))
        if abs(xy[0] - tx) > PUSH_ALIGN_TOL:
            btn = "LEFT" if xy[0] > tx else "RIGHT"
            return self._emit(snap, FrameAction(nes_action(btn), "ne_x"))
        btn = "UP" if xy[1] > ty else "DOWN"
        return self._emit(snap, FrameAction(nes_action(btn), "ne_y"))

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
        xy = (int(snap.link_x), int(snap.link_y))
        if snap.mode in (6, 7, 16) and xy[0] >= EAST_DOOR_XMIN:
            return self._fail(snap, f"east_door_{xy[0]}_{xy[1]}")
        if self._warped(snap):
            self.success = True
            self._set_phase(
                Stairs3A71Phase.DONE,
                f"warped_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
            self.walker.last_dir = None
            return self._emit(
                snap,
                FrameAction(nes_idle_action(), f"warped_{snap.mode}"),
                force=True,
            )
        banned = self._forbidden_play(int(snap.screen))
        if banned is not None and snap.mode == PLAY_MODE:
            return self._fail(
                snap, f"{banned}_{snap.screen:02x}_{xy[0]}_{xy[1]}"
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

        tile = int(snap.colliding_tile)
        if tile == WARP_TILE:
            return self._warp_idle(snap, xy)
        if self.phase is Stairs3A71Phase.IDLE:
            self.idle_frames += 1
            self.walker.last_dir = None
            if tile == HOLE_TILE:
                self._set_phase(
                    Stairs3A71Phase.TO_NE,
                    f"hole_tile_119_{xy[0]}_{xy[1]}",
                )
            else:
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "warp_idle")
                )

        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        new_miss = self.walker.misses > misses_before
        if new_miss and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
        leftover_miss = xy == DATED_LEFTOVER and self.walker.misses == 1
        if new_miss and not leftover_miss:
            return self._fail(
                snap, f"occupancy_halt_{xy[0]}_{xy[1]}"
            )

        if self.phase is Stairs3A71Phase.TO_PUSH:
            block = self._find_block(snap, ne=False)
            if block is None:
                if self.phase_frames >= WAIT_BLOCK_MAX:
                    return self._fail(snap, f"no_block_0x68_{xy[0]}_{xy[1]}")
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_block")
                )
            self._lock(block, "center_block")
            if self._at_south_face(xy, block):
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(
                    Stairs3A71Phase.PUSH,
                    f"at_push_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
                )
            else:
                dest = south_face_stand(block)
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
                    return self._fail(
                        snap, f"occupancy_halt_{xy[0]}_{xy[1]}"
                    )
                if dest != self.walker.goal:
                    self.walker.path = None
                    self.walker.goal = dest
                direction = self.walker.next_dir(xy, dest)
                if direction is None:
                    self.walker.last_dir = None
                    if self.walker.misses > 0:
                        return self._fail(
                            snap, f"occupancy_halt_{xy[0]}_{xy[1]}"
                        )
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

        if self.phase is Stairs3A71Phase.PUSH:
            block = self._find_block(snap, ne=False)
            if block is None:
                return self._fail(snap, f"lost_block_{xy[0]}_{xy[1]}")
            if self.block_y0 is None:
                self.block_x0 = int(block.x)
                self.block_y0 = int(block.y)
            if int(block.y) <= int(self.block_y0) - PUSH_MOVED_PX:
                self.hole_x = int(self.block_x0)
                self.walker.last_dir = None
                self.walker.path = None
                note = (
                    f"pushed_{self.block_x0}_{self.block_y0}"
                    f"_to_{int(block.x)}_{int(block.y)}"
                )
                self._unlock()
                self._set_phase(Stairs3A71Phase.TO_NE, note)
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

        if self.phase is Stairs3A71Phase.TO_NE:
            if tile == HOLE_TILE:
                self.walker.last_dir = None
                if xy[0] >= EAST_DOOR_XMIN:
                    return self._fail(snap, f"east_door_{xy[0]}_{xy[1]}")
                return self._emit(
                    snap, FrameAction(nes_action("RIGHT"), "ne_sidestep")
                )
            block = self._find_block(snap, ne=True)
            if block is None:
                if self.phase_frames >= WAIT_BLOCK_MAX:
                    return self._fail(snap, f"no_ne_0x68_{xy[0]}_{xy[1]}")
                self.walker.last_dir = None
                if (
                    self.hole_x is not None
                    and abs(xy[0] - int(self.hole_x)) <= HOLE_COLUMN_TOL
                ):
                    return self._emit(
                        snap, FrameAction(nes_action("RIGHT"), "ne_sidestep")
                    )
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_ne_block")
                )
            self._lock(block, "ne_block")
            walked = self._axis_to_ne(snap, xy, block)
            if walked is not None:
                return walked
            self._set_phase(
                Stairs3A71Phase.PUSH_NE,
                f"at_ne_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
            )

        if self.phase is Stairs3A71Phase.PUSH_NE:
            block = self._find_block(snap, ne=True)
            if block is None:
                return self._fail(snap, f"lost_ne_{xy[0]}_{xy[1]}")
            if int(block.y) <= int(self.block_y0 or block.y) - PUSH_MOVED_PX:
                self.walker.last_dir = None
                self.notes.append(
                    f"ne_moved_{self.block_x0}_{self.block_y0}"
                    f"_to_{int(block.x)}_{int(block.y)}"
                )
            if self.phase_frames >= PUSH_38_MAX_HOLD:
                return self._fail(
                    snap,
                    f"push_ne_no_71_{xy[0]}_{xy[1]}_tile={tile}",
                )
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("UP"), "push_ne_block"))

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
                "LEFT+DOWN clip after leftover (144,141) tile-118 miss; "
                "south-face UP until y-move 8px; NE 0x68 south-face UP onto "
                "tile 0x71 still-stand; no tile-119 idle; no hold-UP past hole; "
                "halt first new occupancy miss; dest is RAM"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "idle_frames": int(self.idle_frames),
            "block_slot": self.block_slot,
            "block_xy0": (
                None
                if self.block_x0 is None or self.block_y0 is None
                else [self.block_x0, self.block_y0]
            ),
            "spec_id": self.spec_id,
            "room": self.room,
        }


def make_stairs_3a_71_controller() -> Level6Stairs3A71Controller:
    """Push 0x3A center 0x68 then still-stand 0x71. Do not poke bow/arrows."""
    return Level6Stairs3A71Controller()


def level6_stairs3a_71_stages():
    """0x3A leftover → push → tile 0x71 still-stand. Dest is RAM. No Gohma."""
    stairs = make_stairs_3a_71_controller()
    return (
        ("level6_stairs_0x3a_71", stairs, STAIRS_3A_71_MAX_FRAMES),
    )


def level6_stairs3a_71_success(snap: ZeldaSnapshot) -> bool:
    """Mode 9 cellar or a new L6 play room. Rod and TF 0x1F stay."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if int(getattr(snap, "rod", 0)) == 0:
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    if (
        snap.mode != PLAY_MODE
        or snap.transitioning
        or snap.screen == LEVEL6_BLOCK_3A_ROOM
    ):
        return False
    if snap.screen in (NORTH_29, KEY_UP_09, EAST_ROOM, WEST_ROOM):
        return False
    return True
