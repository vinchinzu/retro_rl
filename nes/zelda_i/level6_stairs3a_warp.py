"""Level 6 0x3A stairs via one disclosed Link-position write.

Walk-on from clear3a leftover is BLOCKED (stairs3a / -71 / -ne / -ne71 /
-neclip / -neunder, 3 reds each). Operator exception: one (x, y) write
onto the 0x09 analog warp ``(208, 93)`` after the live center 0x68 push.
Do not write room, door, inventory, Triforce, capacity, facing, mode, or
load state. Dest is RAM. Do not invent/fight Gohma. Do not poke bow/arrows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.assist import poke_link_position
from zelda_i.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.level6_occupancy import l6_leftover, l6_play_dest_success
from zelda_i.level6_overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.level6_stairs3a import (
    Stairs3APhase,
    make_stairs_3a_controller,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "STAIRS_3A_WARP_MAX_FRAMES",
    "WARP_XY",
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
IDLE_AFTER_POKE = 16
UP_AFTER_IDLE = 240


class Stairs3AWarpPhase(Enum):
    PUSH = auto()
    POKE = auto()
    IDLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Stairs3AWarpController(HopController):
    """Live center push, then one (x, y) write onto 0x71. Dest is RAM."""

    spec_id: str = "level6_stairs_0x3a_warp"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = STAIRS_3A_WARP_MAX_FRAMES
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    phase_frames: int = 0
    phase: Stairs3AWarpPhase = Stairs3AWarpPhase.PUSH
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    position_assist: dict[str, Any] | None = None
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
        self.leftover = {**l6_leftover(snap), "map": int(snap.map)}
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
        if snap.level != LEVEL6:
            return False
        if snap.mode == PASSAGE_MODE:
            return True
        return (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.room
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"warped_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def mark_done(self, snap: ZeldaSnapshot, note: str | None = None) -> FrameAction:
        self.done_reason = f"warped_{snap.mode}"
        self._set_phase(Stairs3AWarpPhase.DONE, note or self.on_arrive(snap))
        return super().mark_done(snap, note)

    def _poke(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.env is None:
            return self.mark_fail("no_env_for_position_write")
        wx, wy = WARP_XY
        self.position_assist = poke_link_position(
            self.env,
            wx,
            wy,
            room=self.room,
            from_xy=(int(snap.link_x), int(snap.link_y)),
        )
        n = int(self.position_assist.get("position_writes") or 0)
        if n != 1:
            return self.mark_fail("position_write_failed")
        self._set_phase(
            Stairs3AWarpPhase.IDLE,
            f"poked_{int(snap.link_x)}_{int(snap.link_y)}_to_{wx}_{wy}",
        )
        return FrameAction(nes_idle_action(), "position_write")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self.mark_fail(
                f"left_0x{self.room:02x}_to_0x{snap.screen:02x}"
            )
        if int(snap.link_x) >= EAST_DOOR_XMIN and int(snap.link_y) in range(
            133, 150
        ):
            return self.mark_fail(f"east_door_{snap.link_x}_{snap.link_y}")

        if self.phase is Stairs3AWarpPhase.PUSH:
            action = self.inner.step(snap)
            if self.inner.failed:
                return self.mark_fail(
                    self.inner.notes[-1] if self.inner.notes else "push_fail"
                )
            if self.inner.phase is Stairs3APhase.ON_HOLE:
                self._set_phase(Stairs3AWarpPhase.POKE, "center_pushed")
                return self._poke(snap)
            return action

        if self.phase is Stairs3AWarpPhase.POKE:
            return self._poke(snap)

        if self.phase is Stairs3AWarpPhase.IDLE:
            if self.phase_frames <= IDLE_AFTER_POKE:
                return FrameAction(nes_idle_action(), "warp_idle")
            if self.phase_frames > IDLE_AFTER_POKE + UP_AFTER_IDLE:
                return self.mark_fail(
                    f"warp_no_dest_{snap.link_x}_{snap.link_y}_tile_{snap.colliding_tile}"
                )
            return FrameAction(nes_action("UP"), "warp_up")

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
                "live center 0x68 push, then one disclosed Link-position "
                f"write to {WARP_XY}; dest is RAM (mode 9 or play != 0x3A)"
            ),
            "leftover": dict(self.leftover),
            "position_assist": self.position_assist,
            "spec_id": self.spec_id,
            "room": self.room,
            "warp_xy": list(WARP_XY),
        }


def make_stairs_3a_warp_controller() -> Level6Stairs3AWarpController:
    """Push 0x3A center 0x68, then one (x, y) write onto 0x71."""
    return Level6Stairs3AWarpController()


def level6_stairs3a_warp_stages():
    """0x3A leftover → live push → one position write. Dest is RAM."""
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
