"""Level 1 play 0x22 leftover: westmost 0x68 DOWN, stairs, dest mode 9.

Leftover (160,157) SE diamond edge, keys=0 bow=0. Wiki: 4 blade traps,
push the west block down, stairs, cellar bow. This hop is enter-cellar
only. Do not claim ADDR_BOW. Do not poke bow/arrows/doors/keys.
Isolated BFS banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level1_bow import LEVEL1_BOW_ROOM, level1_bow_stages
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "BOW_CELLAR_MAX_FRAMES",
    "EAST_INLAND_X",
    "SOUTH_LANE_Y",
    "WEST_AISLE_X",
    "Level1BowCellarController",
    "level1_bow_cellar_glance_fields",
    "level1_bow_cellar_stages",
    "level1_bow_cellar_success",
    "make_bow_cellar_controller",
    "north_face_stand",
    "westmost_block_0x68",
]

LEVEL1 = 1
BLOCK_OBJECT_TYPE = 0x68
PUSH_NORTH_OFFSET = 16
PUSH_ALIGN_TOL = 2
PUSH_MOVED_PX = 8
PUSH_MAX_HOLD = 600
WAIT_BLOCK_MAX = 120
EAST_SPAWN_XMAX = 232
# v2 leftover (208,93): UP at x=208 is the NE statue.
# v3 leftover (176,141) tile 118: LEFT at y=141 hits the east diamond.
# v4 leftover (144,109) tile 118: LEFT at y=109 is the north diamond.
# northwall leftover (112,109) tile 178: UP y=93 from x=144 live; LEFT
# at y=93 reaches x=113 then tile 119 (bricked north door column).
# south189 leftover (176,189) tile 117: DOWN x=176 to y=189 live; LEFT
# at y=189 reaches x=127 then tile 119 (bricked south door column).
# south173 leftover (144,173) tile 118: LEFT y=173 176→144 live; LEFT
# at (144,173) is the south diamond (SE point; v4 mirror).
# south157 leftover (160,157) tile 118: LEFT y=157 176→160 live; LEFT
# at (160,157) is the SE diamond edge (v3–v4 diagonal).
EAST_INLAND_X = 176
SOUTH_LANE_Y = 157
WEST_AISLE_X = 64
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


def north_face_stand(block: ZeldaObject) -> tuple[int, int]:
    """One tile north of a 0x68. DOWN from here should register a push."""
    return (int(block.x), int(block.y) - PUSH_NORTH_OFFSET)


def make_bow_cellar_controller() -> "Level1BowCellarController":
    """Westmost 0x68 north-face DOWN. Dest mode 9. Do not poke bow."""
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
    ON_HOLE = auto()
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
    """North-face westmost 0x68, hold DOWN, idle original xy. Dest mode 9."""

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

    def _at_north_face(self, xy: tuple[int, int], block: ZeldaObject) -> bool:
        tx, ty = north_face_stand(block)
        return (
            abs(xy[0] - tx) <= PUSH_ALIGN_TOL
            and abs(xy[1] - ty) <= PUSH_ALIGN_TOL
        )

    def _stage(
        self, xy: tuple[int, int], block: ZeldaObject
    ) -> tuple[tuple[int, int], str]:
        """East mouth LEFT, south y=157, west aisle, north face. Not y=173."""
        x, y = xy
        gx, gy = north_face_stand(block)
        # northwall: LEFT y=93 pinches at the bricked north door.
        # south189: LEFT y=189 pinches at the bricked south door.
        # south173: LEFT y=173 is the south diamond. Do not UP at x=208.
        # Do not LEFT past x=144 at y=109 or y=173.
        if x > EAST_INLAND_X:
            return (EAST_INLAND_X, 141), "west_inland"
        if x > WEST_AISLE_X + PUSH_ALIGN_TOL and y < SOUTH_LANE_Y - PUSH_ALIGN_TOL:
            return (min(x, EAST_INLAND_X), SOUTH_LANE_Y), "south_peel"
        if x > WEST_AISLE_X + PUSH_ALIGN_TOL:
            return (WEST_AISLE_X, SOUTH_LANE_Y), "west_south"
        if y > gy + PUSH_ALIGN_TOL:
            return (WEST_AISLE_X, gy), "north_aisle"
        if x < gx - PUSH_ALIGN_TOL:
            return (gx, gy), "east_face"
        return (gx, gy), "stand_path"

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
            if self._at_north_face(xy, block):
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(
                    CellarPhase.PUSH,
                    f"at_push_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
                )
                return self._emit(snap, FrameAction(nes_action("DOWN"), "push_block"))
            dest, reason = self._stage(xy, block)
            return self._walk_to(snap, dest, reason)

        if self.phase is CellarPhase.PUSH:
            block = self._find_block(snap)
            if block is None:
                return self._fail(snap, f"lost_block_{xy[0]}_{xy[1]}")
            if self.block_y0 is None:
                self.block_x0 = int(block.x)
                self.block_y0 = int(block.y)
            if int(block.y) >= int(self.block_y0) + PUSH_MOVED_PX:
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(
                    CellarPhase.ON_HOLE,
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
                    snap, FrameAction(nes_action("DOWN"), "push_block")
                )

        if self.phase is CellarPhase.ON_HOLE:
            self.walker.last_dir = None
            hx = int(self.block_x0 or xy[0])
            hy = int(self.block_y0 or xy[1])
            if abs(xy[0] - hx) <= PUSH_ALIGN_TOL and abs(xy[1] - hy) <= PUSH_ALIGN_TOL:
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "hole_idle")
                )
            if abs(xy[0] - hx) > PUSH_ALIGN_TOL:
                btn = "LEFT" if xy[0] > hx else "RIGHT"
                return self._emit(snap, FrameAction(nes_action(btn), "hole_x"))
            btn = "UP" if xy[1] > hy else "DOWN"
            return self._emit(snap, FrameAction(nes_action(btn), "hole_y"))

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
                "LEFT inland x=176, DOWN y=157 (south_peel, not diamond "
                "y=173 / door y=189), LEFT west aisle x=64 toward x=96, "
                "UP to north-face y, RIGHT onto westmost 0x68, DOWN until "
                "y+8; idle original xy; dest mode 9; no ADDR_BOW; no "
                "LEFT past x=144 at y=109/y=173; no UP at x=208"
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
