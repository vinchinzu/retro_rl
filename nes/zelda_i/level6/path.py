"""Level 6 interior path controllers.

OccupancyWalker first. Coordinate clips only after a live miss. Isolated
emulator-state BFS is banned. Ignore object type 0x2b. 0x68 is the left-block
push in 0x38 (sample y; do not poke). Do not poke Rod / doors / keys. Do not
grant Whistle.
"""

from __future__ import annotations

from zelda_i.dungeon.hop_controller import room_step

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6.overworld import (
    LEVEL6,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_KEESE_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WEST_WIZZROBE_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
)
from zelda_i.dungeon.bomb_wall import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.dungeon.hop_controller import LatticeDoorWalker, deployed_ladder
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyWalker

__all__ = [
    "BLOCK_OBJECT_TYPE",
    "NORTH_68_MAX_FRAMES",
    "NORTH_DOOR_X",
    "NORTH_DOOR_Y",
    "Level6North68Controller",
    "Level6Push38Controller",
    "BombWall28East",
    "Level6DoorWalkController",
    "left_block_0x68",
    "make_bomb_east_28_controller",
    "make_north_09_controller",
    "make_north_19_controller",
    "make_south_39_controller",
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
# Historical west-clear leftover. Center x=120 UP is the middle statue row.
HISTORICAL_AISLE_X = 144
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
# 0x28 leftover (120,181) UP is solid (v2); LEFT to x=80 then UP still solid
# (v3 leftover 80,181). Peel y=189 then x=80 UP walks to 181 then solid (v4).
# v5: LEFT+UP along the y=181 south face (cardinals cannot thread).
SOUTH_MOUTH_Y = 189
DIAMOND_FACE_Y = 181
CLIP_CLEAR_Y = 173


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
    # False after a live leftover box (0x28 v1 freeze-missed the first UP).
    use_occupancy: bool = True
    # When set, LEFT/RIGHT onto this column then UP (0x28 diamond south face).
    aisle_x: int | None = None
    peeled: bool = False
    # LEFT+UP while y > CLIP_CLEAR_Y (0x28 v5; cardinal UP at y=181 is solid).
    clip_left_up: bool = False
    # Idle first: a press while the ROM finishes a bomb-hole entry puts the
    # stepladder down on the 0x29 moat and pins Link to its axis.
    settle_frames: int = 0
    # ROM-lattice door walk first; the hand rules below are the fallback.
    _env: Any = field(default=None, repr=False)
    _door: LatticeDoorWalker = field(default_factory=LatticeDoorWalker, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

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

        if self.frames <= self.settle_frames:
            return FrameAction(nes_idle_action(), "entry_settle")
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

        door = self._door.action(self._env, snap, "UP", "lattice_door")
        if door is not None:
            self.walker.last_dir = None
            return self._emit(snap, door)

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
                # v6 leftover (96,109): cardinal RIGHT is solid. Clip toward door.
                if self.clip_left_up and snap.link_x < NORTH_DOOR_X:
                    return self._emit(
                        snap,
                        FrameAction(nes_action("RIGHT", "UP"), "door_clip"),
                    )
                direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
                return self._emit(
                    snap, FrameAction(nes_action(direction), "north_align")
                )
            return self._emit(snap, FrameAction(nes_action("UP"), "north_push"))

        # 0x78 leftover (104,149): occupancy UP misses the west statue.
        # v1 stood (104,149). v2 (104,158) still between the west pair.
        # v3 CLIP_CLEAR_Y=173 then replan stood (104,173), 56 misses.
        # v4: DOWN to y=189, RIGHT to x=120, occupancy UP leftover
        # (120,149) — center-column UP is the middle statue row.
        # v5: from x>=120, RIGHT to historical x=144, then occupancy UP.
        # Do not re-probe boxed cells mid-peel. Do not retry x=120 UP.
        if (
            snap.screen == LEVEL6_WEST_WIZZROBE_ROOM
            and snap.link_y >= 141
        ):
            if snap.link_x < HISTORICAL_AISLE_X or snap.link_x > HISTORICAL_AISLE_X + 4:
                self.peeled = True
                self.walker.last_dir = None
                if snap.link_x < NORTH_DOOR_X and snap.link_y < SOUTH_MOUTH_Y:
                    if self.frames <= 8 or self.frames % 60 == 0:
                        self.notes.append(f"peel_south_{xy[0]}_{xy[1]}")
                    return self._emit(
                        snap, FrameAction(nes_action("DOWN"), "peel_south_statue")
                    )
                btn = "RIGHT" if snap.link_x < HISTORICAL_AISLE_X else "LEFT"
                reason = (
                    "peel_to_aisle"
                    if snap.link_x > HISTORICAL_AISLE_X
                    else (
                        "peel_east_aisle"
                        if snap.link_x >= NORTH_DOOR_X
                        else "peel_east_door"
                    )
                )
                if self.frames <= 8 or self.frames % 60 == 0:
                    self.notes.append(f"{reason}_{xy[0]}_{xy[1]}")
                return self._emit(snap, FrameAction(nes_action(btn), reason))

        if not self.use_occupancy:
            if self.clip_left_up and snap.link_y > CLIP_CLEAR_Y:
                if self.frames <= 8 or self.frames % 60 == 0:
                    self.notes.append(f"clip_f{self.frames}_{xy[0]}_{xy[1]}")
                return self._emit(
                    snap,
                    FrameAction(nes_action("LEFT", "UP"), "diamond_clip"),
                )
            # v5 leftover (96,173): RIGHT to x=120 is a diamond. Hold UP.
            return self._emit(
                snap, FrameAction(nes_action("UP"), "north_hold")
            )

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
            "policy": (
                "LEFT+UP at y=181, hold UP, RIGHT+UP at y=109"
                if self.clip_left_up
                else f"DOWN to y={SOUTH_MOUTH_Y} then aisle x={self.aisle_x} UP"
                if not self.use_occupancy and self.aisle_x is not None
                else "hold UP @ x≈120"
                if not self.use_occupancy
                else (
                    f"0x78 west-pocket DOWN to y={SOUTH_MOUTH_Y} "
                    f"RIGHT to x={HISTORICAL_AISLE_X} then occupancy UP"
                )
            ),
            "aisle_x": self.aisle_x,
            "clip_left_up": self.clip_left_up,
            "peeled": self.peeled,
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
            else:
                # ROM lattice to the south face (the 0x68 is in the tile map,
                # so the route goes around it). The axis rules flipped at the
                # face for 1034 reversals (R19).
                step = room_step(snap, (tx, ty), tol=PUSH_ALIGN_TOL)
                if step is not None:
                    return self._emit(snap, FrameAction(nes_action(step), "stand_path"))

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


# ROM door table: 0x28 E is a bomb wall onto 0x29 W, whose N key door is
# 0x19 S. 0x29 -> 0x19 -> 0x09 skips the 0x28 clear and the 0x18 Gleeok
# (4.3-5.4 hearts in C8 resumes) and spends the key the return path spent
# on 0x19 S anyway.
BOMB_28_EAST_STAND = (208, 141)
# Link's pose range on a dungeon room's walkable ring (x lo/hi, y lo/hi).
INTERIOR_BOUNDS = (32, 208, 93, 189)
ENTRY_SETTLE_FRAMES = 20


class BombWall28East:
    """Geometry stand for ``BombWallController``: 0x28 bomb-RIGHT -> 0x29."""

    room = LEVEL6_WIZZROBE_28_ROOM
    stand = BOMB_28_EAST_STAND
    face = "RIGHT"
    opens_to = LEVEL6_DARK_29_ROOM


def make_bomb_east_28_controller() -> BombWallController:
    """0x28 south-aisle arrival -> bomb the east wall -> 0x29. No fight."""
    return BombWallController(
        wall=BombWall28East(),
        level=LEVEL6,
        select_item=B_SLOT_BOMBS,
        stand_tol=2,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=6000,
    )


def make_north_19_controller() -> Level6North68Controller:
    """0x29 west bomb hole -> lattice to the north key door -> 0x19. No fight."""
    return Level6North68Controller(
        source_room=LEVEL6_DARK_29_ROOM,
        dest_room=LEVEL6_MAP_ROOM,
        spec_id="level6_north_0x19",
        settle_frames=ENTRY_SETTLE_FRAMES,
    )


def make_north_09_controller() -> Level6North68Controller:
    """0x19 south mouth -> west-bank lattice to the key door -> 0x09. No Map."""
    return Level6North68Controller(
        source_room=LEVEL6_MAP_ROOM,
        dest_room=LEVEL6_ROD_WIZZ_ROOM,
        spec_id="level6_north_0x09",
    )


@dataclass
class Level6DoorWalkController:
    """ROM-lattice walk to one door of ``source_room`` and through it.

    ``LatticeDoorWalker`` routes on the live ``$6530`` lattice (around a
    moat, not into it) and owns the stepladder release. No combat.
    """

    source_room: int
    dest_room: int
    direction: str
    spec_id: str
    max_frames: int = NORTH_68_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    inside: bool = False
    # Hold ``direction`` down this door column instead of routing: 0x29's
    # stepladder bridges both 16 px moats at x=120 and the island between.
    straight_x: int | None = None
    _env: Any = field(default=None, repr=False)
    _door: LatticeDoorWalker = field(default_factory=LatticeDoorWalker, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _inland(self, snap: ZeldaSnapshot) -> str | None:
        """Out of the entry doorway first: a sideways press there is solid.

        0x29's north mouth parked Link at (120,82) pressing LEFT for 650
        frames. Latches once Link is on the interior ring.
        """
        if self.inside:
            return None
        x, y = int(snap.link_x), int(snap.link_y)
        lo_x, hi_x, lo_y, hi_y = INTERIOR_BOUNDS
        if y < lo_y:
            return "DOWN"
        if y > hi_y:
            return "UP"
        if x < lo_x:
            return "RIGHT"
        if x > hi_x:
            return "LEFT"
        self.inside = True
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if snap.level == LEVEL6 and snap.screen == self.dest_room and snap.mode == PLAY_MODE and not snap.transitioning:
            self.success = True
            self.notes.append(f"arrived_{self.dest_room:02x}_{snap.link_x}_{snap.link_y}")
            return FrameAction(nes_idle_action(), f"arrived_{self.dest_room:02x}")
        if snap.mode == 17 or self.frames >= self.max_frames:
            self.failed = True
            self.notes.append(f"failed_{snap.screen:02x}_{snap.link_x}_{snap.link_y}_mode{snap.mode}")
            return FrameAction(nes_idle_action(), "door_walk_failed")
        if snap.transitioning or snap.mode != PLAY_MODE or snap.screen != self.source_room:
            return FrameAction(nes_action(self.direction), "door_walk_scroll")
        if self.straight_x is not None:
            x = int(snap.link_x)
            if x != self.straight_x and deployed_ladder(snap) is None:
                side = "RIGHT" if x < self.straight_x else "LEFT"
                return FrameAction(nes_action(side), "door_walk_column")
            return FrameAction(nes_action(self.direction), "door_walk_straight")
        if (inland := self._inland(snap)) is not None:
            return FrameAction(nes_action(inland), "door_walk_inland")
        door = self._door.action(self._env, snap, self.direction, "door_walk")
        return door or FrameAction(nes_action(self.direction), "door_walk_push")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
        }


def make_south_39_controller() -> Level6DoorWalkController:
    """Post-Rod 0x29 north mouth -> DOWN x=120 over both moats -> open S door.

    The old occupancy door hop (``SOUTH29_SPEC``, removed) walked into
    the moat, dropped the stepladder, then refused UP (``forbid_up``) and
    stood 800 frames among the Wizzrobes (c9_from_coast, 7.75h).
    """
    return Level6DoorWalkController(
        source_room=LEVEL6_DARK_29_ROOM,
        dest_room=LEVEL6_DARK_39_ROOM,
        direction="DOWN",
        spec_id="level6_south_0x29",
        straight_x=120,
    )
