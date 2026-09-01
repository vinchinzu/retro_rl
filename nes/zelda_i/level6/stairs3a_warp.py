"""Level 6 0x3A stairs via locked south-band cardinals onto 0x71.

Walk-on at y=149 is BLOCKED (ne71 v1 LEFT 158,149; v2 UP 144,149; v3 UP
136,149). Center hole after the push is decorative tile 119 / 0x77. Real
CheckWarp is tile 0x71 at (208, 93). After the live center push, peel south
of y=149, RIGHT to x=208, UP the east column onto 0x71. Do not retry
occupancy at y=149. Do not restore stairs3a* names. Do not poke x/y.

Do not write 10→room, door, inventory, Triforce, capacity, facing, mode, or
load state. Dest is RAM. Do not invent/fight Gohma. Do not poke bow/arrows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.level6.occupancy import l6_leftover, l6_play_dest_success
from zelda_i.level6.overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.level6.stairs3a import (
    Stairs3APhase,
    make_stairs_3a_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "STAIRS_3A_WARP_MAX_FRAMES",
    "WARP_XY",
    "SOUTH_BAND_Y",
    "EAST_COLUMN_X",
    "Level6Stairs3AWarpController",
    "Stairs3AWarpPhase",
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
    inner: Any = field(default_factory=make_stairs_3a_controller)

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
        if self.phase is not Stairs3AWarpPhase.NORTH:
            if x >= EAST_DOOR_XMIN and y in range(133, 150):
                return self.mark_fail(f"east_door_{x}_{y}")

        if self.phase is Stairs3AWarpPhase.PUSH:
            action = self.inner.step(snap)
            if self.inner.failed:
                return self.mark_fail(
                    self.inner.notes[-1] if self.inner.notes else "push_fail"
                )
            if self.inner.phase is Stairs3APhase.ON_HOLE:
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
