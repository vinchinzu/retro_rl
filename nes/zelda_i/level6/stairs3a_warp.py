"""Level 6 0x3A stairs via locked south-band cardinals onto 0x71.

Walk-on at y=149 is BLOCKED (ne71 v1 LEFT 158,149; v2 UP 144,149; v3 UP
136,149). Center hole after the push is decorative tile 119 / 0x77. Real
CheckWarp is tile 0x71 at (208, 93). After the live center push, peel south
of y=149, RIGHT to x=208, UP the east column onto 0x71. Do not retry
occupancy at y=149. Do not restore stairs3a* names. Do not poke x/y.

Do not write room, door, inventory, Triforce, capacity, facing, mode, or
load state. Dest is RAM. Do not invent/fight Gohma. Do not poke bow/arrows.
Do not grant Map.

The center-0x68 push (occupancy south-face, then UP until the block moves)
used to be a standalone `stairs3a` hop; that walk-on is superseded (position
poke removed, `position_writes=0`) so it now lives here as the ``PUSH``
phase's inner helper, folded in rather than kept as a separate module.
"""

from __future__ import annotations

from zelda_i.dungeon.hop_controller import room_step

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.level6.occupancy import l6_leftover, l6_play_dest_success
from zelda_i.level6.overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.level6.path import (
    BLOCK_OBJECT_TYPE,
    PUSH_ALIGN_TOL,
    PUSH_38_MAX_HOLD,
    PUSH_MOVED_PX,
    WAIT_BLOCK_MAX,
    south_face_stand,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyWalker

__all__ = [
    "STAIRS_3A_WARP_MAX_FRAMES",
    "WARP_XY",
    "SOUTH_BAND_Y",
    "EAST_COLUMN_X",
    "Level6Stairs3AWarpController",
    "Stairs3AWarpPhase",
    "is_center_block_pushed",
    "level6_stairs3a_warp_stages",
    "level6_stairs3a_warp_success",
    "make_stairs_3a_warp_controller",
]

STAIRS_3A_WARP_MAX_FRAMES = 4000
STAIRS_3A_WARP_SAMPLE_PERIOD = 8
# Proven 0x09 CheckWarp: south-face NE 0x68 UP onto tile 0x71.
WARP_XY = (208, 93)
EAST_DOOR_XMIN = 200
EAST_ROOM = 0x3B
WEST_ROOM = 0x39
NORTH_29 = 0x29
KEY_UP_09 = 0x09
SOUTH_BAND_Y = 181
EAST_COLUMN_X = 208
# south_face_stand of NE 0x68 (208, 96).
NE_SOUTH_FACE_Y = 112
WALK_PHASE_MAX = 1200

# --- inner: live center-0x68 push (was the standalone stairs3a hop) -------

_PUSH_MAX_FRAMES = 4000
_PUSH_SAMPLE_PERIOD = 8
# Search key for the center 0x68 — not a walk target.
_CENTER_XY = (120, 144)


def center_block_0x68(snap: ZeldaSnapshot) -> ZeldaObject | None:
    """0x68 near room center (112, 144). Exclude NE warp block at (208, 96)."""
    cx, cy = _CENTER_XY
    blocks = [
        obj
        for obj in snap.objects
        if int(obj.type_id) == BLOCK_OBJECT_TYPE
        and abs(int(obj.x) - cx) + abs(int(obj.y) - cy) <= 32
    ]
    if not blocks:
        return None
    return min(
        blocks,
        key=lambda obj: abs(int(obj.x) - cx) + abs(int(obj.y) - cy),
    )


def is_center_block_pushed(snap: ZeldaSnapshot) -> bool:
    """True when 0x3A center block has moved, NE warp block exists, or mode 9."""
    if snap.mode in (9, 16):
        return True
    if snap.colliding_tile == 0x71:
        return True
    for obj in snap.objects:
        if int(obj.type_id) == BLOCK_OBJECT_TYPE and int(obj.x) >= 184:
            return True
    return False


class _PushPhase(Enum):
    TO_PUSH = auto()
    PUSH = auto()
    ON_HOLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class _PushController:
    """Occupancy south-face of live center 0x68, then UP. Dest is RAM.

    Live-center-push helper for ``Level6Stairs3AWarpController.PUSH`` phase.
    Do not invent Gohma. Do not poke ADDR_BOW / ADDR_ARROWS / doors / keys.
    Do not grant Map.
    """

    spec_id: str = "level6_stairs_0x3a"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = _PUSH_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: _PushPhase = _PushPhase.TO_PUSH
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    block_slot: int | None = None
    block_x0: int | None = None
    block_y0: int | None = None

    def _set_phase(self, phase: _PushPhase, note: str = "") -> None:
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
        self._set_phase(_PushPhase.FAILED, note)
        return self._emit(
            snap, FrameAction(nes_idle_action(), note), force=True
        )

    def _is_pushed(self, snap: ZeldaSnapshot) -> bool:
        if is_center_block_pushed(snap):
            return True
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
                if int(found.x) >= 184:
                    return True
                if self.block_x0 is not None and abs(int(found.x) - self.block_x0) >= 8:
                    return True
                if self.block_y0 is not None and abs(int(found.y) - self.block_y0) >= PUSH_MOVED_PX:
                    return True
        return False

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
                if int(found.x) >= 184:
                    return None
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
        return int(snap.rod)

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(snap.bow)

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(snap.arrows)

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        block = self._find_block(snap)
        blocks = self._blocks_68(snap)
        self.leftover = {
            **l6_leftover(snap),
            "submode": int(snap.submode),
            "bx": -1 if block is None else int(block.x),
            "by": -1 if block is None else int(block.y),
            "blocks": blocks,
            "map": int(snap.map),
        }
        if force or self.frames <= 2 or self.frames % _PUSH_SAMPLE_PERIOD == 0:
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
                _PushPhase.DONE,
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

        if self.phase is _PushPhase.TO_PUSH:
            if self._is_pushed(snap):
                self.success = True
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(_PushPhase.ON_HOLE, "center_already_pushed")
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "center_already_pushed")
                )
            block = self._find_block(snap)
            if block is None:
                if self._is_pushed(snap):
                    self.success = True
                    self.walker.last_dir = None
                    self.walker.path = None
                    self._set_phase(_PushPhase.ON_HOLE, "center_already_pushed")
                    return self._emit(
                        snap, FrameAction(nes_idle_action(), "center_already_pushed")
                    )
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
                    _PushPhase.PUSH,
                    f"at_push_{xy[0]}_{xy[1]}_block_{int(block.x)}_{int(block.y)}",
                )
            else:
                # ROM lattice to the south face; the hand detour flipped
                # DOWN/RIGHT at (64,157)<->(64,158) for 2457 frames (R17).
                step = room_step(snap, south_face_stand(block), tol=PUSH_ALIGN_TOL)
                if step is not None:
                    return self._emit(snap, FrameAction(nes_action(step), "stand_path"))
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "stand_wait")
                )

        if self.phase is _PushPhase.PUSH:
            if self._is_pushed(snap):
                self.success = True
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(_PushPhase.ON_HOLE, "center_pushed")
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "center_pushed")
                )
            block = self._find_block(snap)
            if block is None:
                if self._is_pushed(snap):
                    self.success = True
                    self.walker.last_dir = None
                    self.walker.path = None
                    self._set_phase(_PushPhase.ON_HOLE, "center_pushed")
                    return self._emit(
                        snap, FrameAction(nes_idle_action(), "center_pushed")
                    )
                return self._fail(snap, f"lost_block_{xy[0]}_{xy[1]}")
            if self.block_y0 is None:
                self.block_x0 = int(block.x)
                self.block_y0 = int(block.y)
            if (
                abs(int(block.y) - int(self.block_y0)) >= PUSH_MOVED_PX
                or abs(int(block.x) - int(self.block_x0)) >= 8
                or int(block.x) >= 184
            ):
                self.success = True
                self.walker.last_dir = None
                self.walker.path = None
                self._set_phase(
                    _PushPhase.ON_HOLE,
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

        if self.phase is _PushPhase.ON_HOLE:
            # Warp peels south; this idle is not the CheckWarp walk.
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


def _make_push_controller() -> _PushController:
    """South-face center 0x68 in 0x3A. Do not poke bow/arrows/doors."""
    return _PushController()


# --- outer: south-band walk from the pushed hole onto 0x71 ----------------


class Stairs3AWarpPhase(Enum):
    PUSH = auto()
    PEEL = auto()
    EAST = auto()
    NORTH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Stairs3AWarpController(HopController):
    """Live center push, then south-band RIGHT and east-column UP onto 0x71."""

    spec_id: str = "level6_stairs_0x3a_warp"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = STAIRS_3A_WARP_MAX_FRAMES
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    phase_frames: int = 0
    phase: Stairs3AWarpPhase = Stairs3AWarpPhase.PUSH
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    position_assist: dict[str, Any] = field(
        default_factory=lambda: {"position_writes": 0, "progression_writes": 0}
    )
    env: Any | None = None
    inner: Any = field(default_factory=_make_push_controller)

    def bind_env(self, env: Any) -> None:
        self.env = env

    def _set_phase(self, phase: Stairs3AWarpPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            **l6_leftover(snap),
            "map": int(snap.map),
            "phase": self.phase.name,
        }
        if force or self.frames <= 2 or self.frames % STAIRS_3A_WARP_SAMPLE_PERIOD == 0:
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
                    "rod": int(snap.rod),
                    "bow": int(snap.bow),
                    "arrows": int(snap.arrows),
                    "keys": int(snap.keys),
                }
            )
        return action

    def mark_fail(self, note: str, reason: str | None = None) -> FrameAction:
        self._set_phase(Stairs3AWarpPhase.FAILED, note)
        return super().mark_fail(note, reason)

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return level6_stairs3a_warp_success(snap)

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"warped_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def mark_done(self, snap: ZeldaSnapshot, note: str | None = None) -> FrameAction:
        self.done_reason = f"warped_{snap.mode}"
        self._set_phase(Stairs3AWarpPhase.DONE, note or self.on_arrive(snap))
        return super().mark_done(snap, note)

    def _walk_timeout(self, snap: ZeldaSnapshot, tag: str) -> FrameAction | None:
        if self.phase_frames <= WALK_PHASE_MAX:
            return None
        return self.mark_fail(
            f"{tag}_no_dest_{snap.link_x}_{snap.link_y}_tile_{snap.colliding_tile}"
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.screen == EAST_ROOM:
            return self.mark_fail(f"east_room_0x{EAST_ROOM:02x}")
        if snap.screen != self.room:
            return self.mark_fail(
                f"left_0x{self.room:02x}_to_0x{snap.screen:02x}"
            )
        x, y = int(snap.link_x), int(snap.link_y)

        if self.phase is Stairs3AWarpPhase.PUSH:
            if (
                is_center_block_pushed(snap)
                or self.inner.phase is _PushPhase.ON_HOLE
                or self.inner.success
            ):
                self._set_phase(Stairs3AWarpPhase.PEEL, "center_pushed")
                return FrameAction(nes_action("DOWN"), "peel_south")
            action = self.inner.step(snap)
            if self.inner.failed:
                return self.mark_fail(
                    self.inner.notes[-1] if self.inner.notes else "push_fail"
                )
            if (
                self.inner.phase is _PushPhase.ON_HOLE
                or self.inner.success
                or is_center_block_pushed(snap)
            ):
                self._set_phase(Stairs3AWarpPhase.PEEL, "center_pushed")
                return FrameAction(nes_action("DOWN"), "peel_south")
            return action

        timed = self._walk_timeout(snap, self.phase.name.lower())
        if timed is not None:
            return timed

        if self.phase is Stairs3AWarpPhase.PEEL:
            if y >= SOUTH_BAND_Y:
                self._set_phase(Stairs3AWarpPhase.EAST, f"south_band_{x}_{y}")
            else:
                return FrameAction(nes_action("DOWN"), "peel_south")

        if self.phase is Stairs3AWarpPhase.EAST:
            if y < SOUTH_BAND_Y:
                return FrameAction(nes_action("DOWN"), "peel_south")
            if x != EAST_COLUMN_X:
                btn = "RIGHT" if x < EAST_COLUMN_X else "LEFT"
                return FrameAction(nes_action(btn), "east_column")
            self._set_phase(Stairs3AWarpPhase.NORTH, f"east_column_{x}_{y}")

        if self.phase is Stairs3AWarpPhase.NORTH:
            reason = "warp_up" if y <= NE_SOUTH_FACE_Y else "column_up"
            return FrameAction(nes_action("UP"), reason)

        return FrameAction(nes_idle_action(), "failed")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.phase_frames += 1
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "phase": self.phase.name,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "live center 0x68 push, peel south of y=149, RIGHT to "
                f"x={EAST_COLUMN_X}, UP onto {WARP_XY}; dest is RAM "
                "(mode 9 or play != 0x3A, forbid 0x3B/0x39/0x29/0x09)"
            ),
            "leftover": dict(self.leftover),
            "position_assist": dict(self.position_assist),
            "spec_id": self.spec_id,
            "room": self.room,
            "warp_xy": list(WARP_XY),
        }


def make_stairs_3a_warp_controller() -> Level6Stairs3AWarpController:
    """Push 0x3A center 0x68, then south-band walk onto 0x71. No position write."""
    return Level6Stairs3AWarpController()


def level6_stairs3a_warp_stages():
    """0x3A leftover → live push → south-band walk onto 0x71. Dest is RAM."""
    stairs = make_stairs_3a_warp_controller()
    return (
        ("level6_stairs_0x3a_warp", stairs, STAIRS_3A_WARP_MAX_FRAMES),
    )


def level6_stairs3a_warp_success(snap: ZeldaSnapshot) -> bool:
    """Mode 9 cellar or a new L6 play room. Rod and TF 0x1F stay."""
    return l6_play_dest_success(
        snap,
        not_room=LEVEL6_BLOCK_3A_ROOM,
        forbid=(NORTH_29, KEY_UP_09, EAST_ROOM, WEST_ROOM),
    )
