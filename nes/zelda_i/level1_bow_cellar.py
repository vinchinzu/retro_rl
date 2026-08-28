"""Level 1 play 0x22: push west 0x68 UP, take center stairs, dest mode 9.

The staircase is visible at room center before the push. Enter from the east,
thread the south diamond from its stable face, push the west block UP, and
walk through the opened gap. This hop is enter-cellar only. Do not poke
ADDR_BOW / ADDR_ARROWS. Isolated BFS banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level1_bow import LEVEL1_BOW_ROOM, level1_bow_stages
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.screen_glance import BOW_CELLAR_LEAVE, GlanceLeftover, grade_controller
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "BOW_CELLAR_MAX_FRAMES",
    "EAST_INLAND_X",
    "SOUTH_FACE_Y",
    "SOUTH_GAP_X",
    "SOUTH_MOUTH_Y",
    "STAIRS_STAND_X",
    "STAIRS_STAND_Y",
    "VACATED_SLOT_Y",
    "Level1BowCellarController",
    "LEVEL1_BOW_CELLAR_ROOM",
    "level1_bow_cellar_glance",
    "level1_bow_cellar_glance_fields",
    "level1_bow_cellar_stages",
    "level1_bow_cellar_success",
    "make_bow_cellar_controller",
    "south_face_stand",
    "westmost_block_0x68",
]

LEVEL1 = 1
LEVEL1_BOW_CELLAR_ROOM = 0x7F
BLOCK_OBJECT_TYPE = 0x68
PUSH_SOUTH_OFFSET = 13
PUSH_ALIGN_TOL = 2
PUSH_FACE_TOL = 3
PUSH_MOVED_PX = 8
PUSH_MAX_HOLD = 600
WAIT_BLOCK_MAX = 120
EAST_SPAWN_XMAX = 232
# v2 leftover (208,93): UP at x=208 is the NE statue.
# v3 leftover (176,141) tile 118: LEFT at y=141 hits the east diamond.
# v4 leftover (144,109) tile 118: LEFT at y=109 is the north diamond.
EAST_INLAND_X = 176
SOUTH_MOUTH_Y = 189
SOUTH_GAP_X = 128
SOUTH_FACE_Y = 181
# Original west 0x68 y. After the UP push the slot is empty; RIGHT from
# leftover (96,149) is solid until Link walks through this y.
VACATED_SLOT_Y = 144
# The existing center stair square accepts the standard UW CheckWarps pose.
STAIRS_STAND_X = 128
STAIRS_STAND_Y = 141
BOW_CELLAR_MAX_FRAMES = 4000
SAMPLE_PERIOD = 12
DEATH_MODE = 17
WAIT_SCROLL = (2, 3, 4, 6, 7, 10, 16)


def westmost_block_0x68(snap: ZeldaSnapshot) -> ZeldaObject | None:
    """Westernmost live 0x68. Ignore empty slots."""
    blocks = [obj for obj in snap.objects if int(obj.type_id) == BLOCK_OBJECT_TYPE]
    if not blocks:
        return None
    return min(blocks, key=lambda obj: (int(obj.x), int(obj.y)))


def south_face_stand(block: ZeldaObject) -> tuple[int, int]:
    """South of a 0x68. UP from here should register a push."""
    return (int(block.x), int(block.y) + PUSH_SOUTH_OFFSET)


def make_bow_cellar_controller() -> "Level1BowCellarController":
    """Westmost 0x68 south-face UP, then center stairs. Do not poke bow."""
    return Level1BowCellarController()


def level1_bow_cellar_stages():
    """Enter-stop 0x22, then west-block stairs. Dest is mode 9."""
    return (
        *level1_bow_stages(),
        ("level1_bow_cellar", make_bow_cellar_controller(), BOW_CELLAR_MAX_FRAMES),
    )


def level1_bow_cellar_success(snap: ZeldaSnapshot) -> bool:
    """L1 mode-9 cellar. Do not require ADDR_BOW. Reject play 0x22/0x23."""
    return snap.level == LEVEL1 and snap.mode == PASSAGE_MODE


def level1_bow_cellar_glance(controller) -> GlanceLeftover:
    """Mode-9 0x7F leftover after the 0x22 stairs. ADDR_BOW may still be 0."""
    return grade_controller(controller, BOW_CELLAR_LEAVE)


def level1_bow_cellar_glance_fields(snap: ZeldaSnapshot) -> dict[str, int]:
    """Mode-9 leftover after the 0x22 stairs. ADDR_BOW may still be 0."""
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "bow": int(snap.bow),
        "arrows": int(snap.arrows),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "triforce": int(snap.triforce),
    }


class CellarPhase(Enum):
    TO_PUSH = auto()
    PUSH = auto()
    TO_STAIRS = auto()
    DONE = auto()
    FAILED = auto()


def _leftover(snap: ZeldaSnapshot, block: ZeldaObject | None) -> dict[str, Any]:
    return {
        **level1_bow_cellar_glance_fields(snap),
        "bx": -1 if block is None else int(block.x),
        "by": -1 if block is None else int(block.y),
    }


@dataclass
class Level1BowCellarController:
    """South-face westmost 0x68, hold UP, then idle on center stairs."""

    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: CellarPhase = CellarPhase.TO_PUSH
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(init=False)
    spec_id: str = "level1_bow_cellar"
    room: int = LEVEL1_BOW_ROOM
    block_slot: int | None = None
    block_x0: int | None = None
    block_y0: int | None = None
    max_frames: int = BOW_CELLAR_MAX_FRAMES

    def __post_init__(self) -> None:
        self.walker = OccupancyWalker(grid=OccupancyGrid(xmax=EAST_SPAWN_XMAX))

    def _set_phase(self, phase: CellarPhase, note: str = "") -> None:
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
        self.notes.append(f"west_block_{block.slot}_{block.x}_{block.y}")

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
        return westmost_block_0x68(snap)

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        block = self._find_block(snap)
        if force or self.frames <= 2 or self.frames % SAMPLE_PERIOD == 0:
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
                    "bow": int(snap.bow),
                    "bx": None if block is None else int(block.x),
                    "by": None if block is None else int(block.y),
                    "misses": self.walker.misses,
                }
            )
        self.leftover = _leftover(snap, block)
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        self._set_phase(CellarPhase.FAILED, note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _mark_success(self, snap: ZeldaSnapshot) -> FrameAction:
        self.success = True
        self._set_phase(
            CellarPhase.DONE,
            f"warped_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_bow={int(snap.bow)}",
        )
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), "arrived_cellar"), force=True
        )

    def _at_south_face(self, xy: tuple[int, int], block: ZeldaObject) -> bool:
        tx, ty = south_face_stand(block)
        return (
            abs(xy[0] - tx) <= PUSH_ALIGN_TOL
            and abs(xy[1] - ty) <= PUSH_FACE_TOL
        )

    def _push_stage(
        self, xy: tuple[int, int], block: ZeldaObject
    ) -> tuple[tuple[int, int], str]:
        """South mouth, stable south face, then LEFT+UP around the diamond."""
        x, y = xy
        gx, gy = south_face_stand(block)
        if x > EAST_INLAND_X:
            return (EAST_INLAND_X, 141), "west_inland"
        if y < SOUTH_MOUTH_Y - PUSH_ALIGN_TOL and x > SOUTH_GAP_X + PUSH_ALIGN_TOL:
            return (EAST_INLAND_X, SOUTH_MOUTH_Y), "south_peel"
        if x > SOUTH_GAP_X + PUSH_ALIGN_TOL:
            return (SOUTH_GAP_X, SOUTH_MOUTH_Y), "south_gap"
        if y > SOUTH_FACE_Y:
            return (SOUTH_GAP_X, SOUTH_FACE_Y), "south_face"
        if x > gx + PUSH_ALIGN_TOL and y > gy + PUSH_ALIGN_TOL:
            return (gx, gy), "southwest_clip"
        if y > gy + PUSH_FACE_TOL:
            return (gx, gy), "push_align_y"
        if abs(x - gx) > PUSH_ALIGN_TOL:
            return (gx, gy), "push_align_x"
        return (gx, gy), "stand_path"

    def _stairs_step(self, snap: ZeldaSnapshot) -> FrameAction:
        """UP through the vacated west-block slot, then RIGHT onto stairs."""
        x, y = int(snap.link_x), int(snap.link_y)
        self.walker.last_dir = None
        slot_y = self.block_y0 if self.block_y0 is not None else VACATED_SLOT_Y
        # Live leftover (96,149): RIGHT is solid south of the original 0x68.
        if x < STAIRS_STAND_X and y > slot_y:
            return self._emit(
                snap, FrameAction(nes_action("UP"), "stairs_slot")
            )
        # CheckWarps UW: X must be a multiple of $10. PUSH_ALIGN_TOL idle at
        # x=126 (tile 0x73) never warps; stand is x=128, y=141 (= $10k+$D).
        if x < STAIRS_STAND_X:
            return self._emit(
                snap, FrameAction(nes_action("RIGHT"), "stairs_east")
            )
        if y > STAIRS_STAND_Y:
            return self._emit(
                snap, FrameAction(nes_action("UP"), "stairs_north")
            )
        if y < STAIRS_STAND_Y:
            return self._emit(
                snap, FrameAction(nes_action("DOWN"), "stairs_south")
            )
        if x > STAIRS_STAND_X:
            return self._emit(
                snap, FrameAction(nes_action("LEFT"), "stairs_west")
            )
        return self._emit(snap, FrameAction(nes_idle_action(), "stairs_idle"))

    def _walk_to(self, snap: ZeldaSnapshot, dest: tuple[int, int], reason: str) -> FrameAction:
        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_idle_action(), f"{reason}_stand"))
        return self._emit(snap, FrameAction(nes_action(direction), reason))

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if not any(n.startswith("timeout") for n in self.notes):
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_bow={int(snap.bow)}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == DEATH_MODE:
            return self._fail(snap, "link_death")
        if snap.level == 0:
            return self._fail(
                snap, f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.mode == PASSAGE_MODE and snap.level == LEVEL1:
            return self._mark_success(snap)
        if snap.screen == 0x23 and snap.mode == PLAY_MODE:
            return self._fail(
                snap, f"backtrack_23_{snap.link_x}_{snap.link_y}"
            )
        if snap.transitioning or snap.mode in WAIT_SCROLL:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL1:
            return self._fail(snap, f"left_level_{snap.level}")
        if snap.screen != LEVEL1_BOW_ROOM:
            return self._fail(
                snap, f"wrong_room_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )

        xy = (int(snap.link_x), int(snap.link_y))
        if self.phase is CellarPhase.TO_PUSH:
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
                    CellarPhase.PUSH,
                    f"at_push_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
                )
                return self._emit(snap, FrameAction(nes_action("UP"), "push_block"))
            dest, reason = self._push_stage(xy, block)
            if reason in {"south_peel", "south_gap", "south_face", "push_align_y", "push_align_x"}:
                self.walker.last_dir = None
                if reason == "south_peel":
                    btn = "DOWN"
                elif reason == "south_gap":
                    btn = "LEFT"
                elif reason in {"south_face", "push_align_y"}:
                    btn = "UP"
                else:
                    btn = "LEFT" if xy[0] > dest[0] else "RIGHT"
                return self._emit(
                    snap, FrameAction(nes_action(btn), reason)
                )
            if reason == "southwest_clip":
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_action("LEFT", "UP"), "southwest_clip")
                )
            return self._walk_to(snap, dest, reason)

        if self.phase is CellarPhase.PUSH:
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
                    CellarPhase.TO_STAIRS,
                    f"pushed_{self.block_x0}_{self.block_y0}"
                    f"_to_{int(block.x)}_{int(block.y)}",
                )
            elif self.phase_frames >= PUSH_MAX_HOLD:
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

        if self.phase is CellarPhase.TO_STAIRS:
            return self._stairs_step(snap)

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
                "LEFT inland x=176; DOWN y=189; LEFT only to x=128; "
                "cardinal UP to the stable south face y=181; LEFT+UP to "
                "west-block south stand; push UP until its y moves; UP "
                "through vacated slot y=144 then RIGHT to exact x=128 "
                "(CheckWarps X multiple of $10; y=141 is $10k+$D); idle; "
                "dest mode 9; no ADDR_BOW poke"
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
