"""Level 6 0x3A stairs: live push, then NE 0x68 south-face onto 0x71.

Leftover play 0x3A (144,141) rod=1 keys=4 bombs=8 TF=0x1F bow=0 arrows=0.
Reuse stairs3a-71 live push: LEFT+DOWN clip after dated leftover miss
(144,141) tile 118, DOWN to south-face y=160, south-face UP, center 0x68
y-move 8px (112,144→136). Then to NE 0x68 (208,96) without RIGHT on tile
119 at x=184 (v3 leftover (184,147) RIGHT 0px). After live push, y=149
around corridor is dated (v1 (114,149) RIGHT knockback; v2 DOWN 0px then
(122,149) RIGHT halt). RIGHT+DOWN around the pushed block; do not
occupancy_halt on that row. Sidestep around 119: LEFT off the hole
column, UP west of 184 to south-face y, RIGHT at y=112. South-face that
object, UP onto tile 0x71. Do not expect NE 0x68 to y-move. Occupancy
halt at first new miss after leaving y=149. Isolated BFS banned. Do not
walk east door. Do not take 0x29. Do not invent Gohma. Do not poke
ADDR_BOW / ADDR_ARROWS.
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
from zelda_i.level6_stairs09 import NE_BLOCK_X_MIN, ne_block_0x68
from zelda_i.level6_stairs3a import center_block_0x68
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "AROUND_X",
    "DATED_LEFTOVER",
    "HOLE_COLUMN",
    "HOLE_TILE",
    "SOUTH_AROUND_Y",
    "STAIRS_3A_NE_MAX_FRAMES",
    "V1_TO_NE",
    "V2_TO_NE",
    "V3_HOLE",
    "WARP_TILE",
    "Stairs3ANEPhase",
    "Level6Stairs3ANEController",
    "level6_stairs3a_ne_stages",
    "level6_stairs3a_ne_success",
    "make_stairs_3a_ne_controller",
]

STAIRS_3A_NE_MAX_FRAMES = 4000
STAIRS_3A_NE_SAMPLE_PERIOD = 8
WARP_TILE = 0x71
HOLE_TILE = 119
EAST_DOOR_XMIN = 200
EAST_DOOR_Y = 141
EAST_ROOM = 0x3B
WEST_ROOM = 0x39
NORTH_29 = 0x29
KEY_UP_09 = 0x09
DATED_LEFTOVER = (144, 141)
# v1 leftover after live push: first TO_NE RIGHT knockback (tile 116).
V1_TO_NE = (114, 149)
# v2 leftover: DOWN 0px on y=149, then RIGHT occupancy_halt (tile 118).
V2_TO_NE = (122, 149)
V3_HOLE = (184, 147)
HOLE_COLUMN = NE_BLOCK_X_MIN
HOLE_COLUMN_TOL = 8
# West of tile-119 column (184), east of center stairs (112).
AROUND_X = 160
# South of pushed center 0x68, then RIGHT around toward NE.
SOUTH_AROUND_Y = 160
STAIRS_HOLE_TOL = 4


class Stairs3ANEPhase(Enum):
    TO_PUSH = auto()
    PUSH = auto()
    TO_NE = auto()
    PUSH_NE = auto()
    IDLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Stairs3ANEController:
    """Push center 0x68, around 119, south-face NE 0x68 onto 0x71."""

    spec_id: str = "level6_stairs_0x3a_ne"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = STAIRS_3A_NE_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    idle_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: Stairs3ANEPhase = Stairs3ANEPhase.TO_PUSH
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    block_slot: int | None = None
    block_x0: int | None = None
    block_y0: int | None = None
    hole_x: int | None = None

    def _set_phase(self, phase: Stairs3ANEPhase, note: str = "") -> None:
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
        self._set_phase(Stairs3ANEPhase.FAILED, note)
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
            Stairs3ANEPhase.TO_NE,
            Stairs3ANEPhase.PUSH_NE,
            Stairs3ANEPhase.IDLE,
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
        if force or self.frames <= 2 or self.frames % STAIRS_3A_NE_SAMPLE_PERIOD == 0:
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

    def _walk(self, snap: ZeldaSnapshot, btn: str, reason: str) -> FrameAction:
        self.walker.last_dir = btn
        return self._emit(snap, FrameAction(nes_action(btn), reason))

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
        self._set_phase(Stairs3ANEPhase.IDLE, f"at_71_{xy[0]}_{xy[1]}")
        return self._emit(snap, FrameAction(nes_idle_action(), "warp_idle"))

    def _on_v3_hole(self, xy: tuple[int, int], tile: int) -> bool:
        return tile == HOLE_TILE and abs(xy[0] - HOLE_COLUMN) <= HOLE_COLUMN_TOL

    def _around_corridor(self, xy: tuple[int, int]) -> bool:
        # y=149 east of the pushed block, west of AROUND_X. DOWN 0px (v2).
        return xy[1] == V1_TO_NE[1] and xy[0] < AROUND_X

    def _east_door(self, xy: tuple[int, int]) -> bool:
        return xy[0] >= EAST_DOOR_XMIN and abs(xy[1] - EAST_DOOR_Y) <= 8

    def _axis_to_ne(
        self, snap: ZeldaSnapshot, xy: tuple[int, int], block: ZeldaObject
    ) -> FrameAction | None:
        tx, ty = south_face_stand(block)
        if self._at_south_face(xy, block):
            return None
        if self._east_door(xy):
            return self._fail(snap, f"east_door_{xy[0]}_{xy[1]}")
        tile = int(snap.colliding_tile)
        # v3 leftover (184,147) tile 119: RIGHT 0px. Go around, not RIGHT.
        if self._on_v3_hole(xy, tile) or (
            xy[1] > ty + PUSH_ALIGN_TOL
            and abs(xy[0] - HOLE_COLUMN) <= HOLE_COLUMN_TOL
        ):
            return self._walk(snap, "LEFT", "ne_around")
        if xy[1] > ty + PUSH_ALIGN_TOL:
            if xy[0] < AROUND_X:
                return self._walk(snap, "RIGHT", "ne_sidestep")
            return self._walk(snap, "UP", "ne_y")
        if xy[1] < ty - PUSH_ALIGN_TOL:
            return self._walk(snap, "DOWN", "ne_y")
        if abs(xy[0] - tx) > PUSH_ALIGN_TOL:
            btn = "LEFT" if xy[0] > tx else "RIGHT"
            return self._walk(snap, btn, "ne_x")
        btn = "UP" if xy[1] > ty else "DOWN"
        return self._walk(snap, btn, "ne_y")

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
                Stairs3ANEPhase.DONE,
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
        if self.phase is Stairs3ANEPhase.IDLE:
            self.idle_frames += 1
            self.walker.last_dir = None
            if tile == HOLE_TILE:
                self._set_phase(
                    Stairs3ANEPhase.TO_NE,
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
        # y=149 around corridor: v1 (114,149) + v2 (122,149). Halt after leaving.
        dated_to_ne = (
            self.phase is Stairs3ANEPhase.TO_NE and self._around_corridor(xy)
        )
        if new_miss and not leftover_miss and not dated_to_ne:
            return self._fail(
                snap, f"occupancy_halt_{xy[0]}_{xy[1]}"
            )

        if self.phase is Stairs3ANEPhase.TO_PUSH:
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
                    Stairs3ANEPhase.PUSH,
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
                    if xy[1] < dest[1] - PUSH_ALIGN_TOL:
                        return self._emit(
                            snap, FrameAction(nes_action("DOWN"), "stand_y")
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

        if self.phase is Stairs3ANEPhase.PUSH:
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
                self._set_phase(Stairs3ANEPhase.TO_NE, note)
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

        if self.phase is Stairs3ANEPhase.TO_NE:
            if self._around_corridor(xy):
                # v2 DOWN 0px at (114,149). RIGHT+DOWN around the pushed block.
                self.walker.last_dir = None
                return self._emit(
                    snap,
                    FrameAction(nes_action("RIGHT", "DOWN"), "ne_around_clip"),
                )
            if tile == HOLE_TILE:
                self.walker.last_dir = None
                if self._east_door(xy):
                    return self._fail(snap, f"east_door_{xy[0]}_{xy[1]}")
                if abs(xy[0] - HOLE_COLUMN) <= HOLE_COLUMN_TOL:
                    return self._walk(snap, "LEFT", "ne_around")
                return self._walk(snap, "RIGHT", "ne_sidestep")
            block = self._find_block(snap, ne=True)
            if block is None:
                if self.phase_frames >= WAIT_BLOCK_MAX:
                    return self._fail(snap, f"no_ne_0x68_{xy[0]}_{xy[1]}")
                if xy[0] < AROUND_X or (
                    self.hole_x is not None
                    and abs(xy[0] - int(self.hole_x)) <= STAIRS_HOLE_TOL
                ):
                    return self._walk(snap, "RIGHT", "ne_sidestep")
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "wait_ne_block")
                )
            self._lock(block, "ne_block")
            walked = self._axis_to_ne(snap, xy, block)
            if walked is not None:
                return walked
            self._set_phase(
                Stairs3ANEPhase.PUSH_NE,
                f"at_ne_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
            )

        if self.phase is Stairs3ANEPhase.PUSH_NE:
            block = self._find_block(snap, ne=True)
            if block is None:
                return self._fail(snap, f"lost_ne_{xy[0]}_{xy[1]}")
            # NE 0x68 does not y-move (stairs09). Hold UP onto tile 0x71.
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
                "DOWN to south-face y=160 after clip x-aligns; south-face UP "
                "until y-move 8px; after push, RIGHT+DOWN around y=149 "
                "(114,149)/(122,149) (DOWN 0px; do not occupancy_halt that "
                "row); RIGHT off stairs to x=160 then UP to south-face y "
                "(no UP/RIGHT on tile 119 at x=184); LEFT around v3 hole "
                "(184,147); NE 0x68 (208,96) south-face UP onto tile 0x71 "
                "(object may not y-move); halt first new occupancy miss "
                "after leaving y=149; dest is RAM"
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


def make_stairs_3a_ne_controller() -> Level6Stairs3ANEController:
    """Push 0x3A center 0x68, around 119, south-face NE onto 0x71."""
    return Level6Stairs3ANEController()


def level6_stairs3a_ne_stages():
    """0x3A leftover → push → around 119 → NE 0x68 → tile 0x71. Dest is RAM."""
    stairs = make_stairs_3a_ne_controller()
    return (
        ("level6_stairs_0x3a_ne", stairs, STAIRS_3A_NE_MAX_FRAMES),
    )


def level6_stairs3a_ne_success(snap: ZeldaSnapshot) -> bool:
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
