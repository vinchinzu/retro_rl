"""Level 4 leftover 0x01 → bomb-RIGHT 0x12 Vire clear (no live BFS).

key01 v3 leftover play 0x01 (120,133). DOWN the bomb hole into 0x11, then
BOMB_RIGHT stand (192,141) (AGENTS trap; not free RIGHT). Ignore 0x35 on
0x11 and block 0x68 on 0x12. Isolated PATH_12_TO_GLEEOK is the next hop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.bomb_wall_path import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.dungeon import DungeonPhase
from zelda_i.level4_dungeon import (
    LEVEL4,
    ROOM_12_SPEC,
    ROOM_L4_KEY_01,
    ROOM_L4_MID_11,
    ROOM_L4_VIRES_12,
)
from zelda_i.level4_path import make_room_12_clear_controller
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "BOMB_11_EAST_FACE",
    "BOMB_11_EAST_STAND",
    "BombWall11East",
    "South11Phase",
    "Level4South11Controller",
    "level4_clear12_stages",
    "level4_clear12_success",
    "make_bomb_11_east_controller",
    "make_south_11_controller",
]

BOMB_11_EAST_STAND = (192, 141)
BOMB_11_EAST_FACE = "RIGHT"
SOUTH_11_X = 120
SOUTH_11_X_TOL = 6


class South11Phase(Enum):
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4South11Controller:
    """0x01 leftover (120,133) → x=120 hold DOWN into 0x11. Ignore Keese."""

    max_frames: int = 4000
    phase: South11Phase = South11Phase.ALIGN
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: South11Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(South11Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def _entered_11(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_MID_11
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.phase is South11Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is South11Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail(
                f"timeout_{int(snap.link_x)}_{int(snap.link_y)}"
            )
        if self._entered_11(snap):
            self.success = True
            self._set_phase(South11Phase.DONE, "entered_0x11")
            return FrameAction(nes_idle_action(), "done")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("DOWN"), "scroll_down")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen == ROOM_L4_MID_11:
            self.success = True
            self._set_phase(South11Phase.DONE, "on_0x11")
            return FrameAction(nes_idle_action(), "done")
        if snap.screen != ROOM_L4_KEY_01:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")
        if abs(snap.link_x - SOUTH_11_X) > SOUTH_11_X_TOL:
            self._set_phase(South11Phase.ALIGN, "align_x")
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < SOUTH_11_X else "LEFT"),
                "align_x",
            )
        self._set_phase(South11Phase.PUSH, "push_down")
        return FrameAction(nes_action("DOWN"), "push_down_south")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_south_0x11",
            "target_room": f"0x{ROOM_L4_MID_11:02x}",
        }


class BombWall11East:
    """Geometry stand: 0x11 bomb-RIGHT → 0x12."""

    room = ROOM_L4_MID_11
    stand = BOMB_11_EAST_STAND
    face = BOMB_11_EAST_FACE
    opens_to = ROOM_L4_VIRES_12


def make_south_11_controller() -> Level4South11Controller:
    return Level4South11Controller()


def make_bomb_11_east_controller() -> BombWallController:
    """0x11 leftover → bomb east → 0x12. No 0x35 clear."""
    return BombWallController(
        wall=BombWall11East(),
        level=LEVEL4,
        approach_waypoints=(BOMB_11_EAST_STAND,),
        approach_tol=2,
        stand_tol=2,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=8000,
    )


def level4_clear12_stages():
    """0x01 leftover → DOWN 0x11 → bomb-RIGHT 0x12 → Vire clear."""
    south = make_south_11_controller()
    bomb = make_bomb_11_east_controller()
    clear = make_room_12_clear_controller()
    clear.phase = DungeonPhase.FIGHT
    return (
        ("level4_south_0x11", south, south.max_frames),
        ("level4_bomb_east_0x12", bomb, bomb.max_frames),
        ("level4_clear_0x12", clear, ROOM_12_SPEC.max_frames),
    )


def level4_clear12_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x12 with Vires dead. Block 0x68 may stay."""
    return (
        snap.level == LEVEL4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_VIRES_12
        and not snap.transitioning
        and not ROOM_12_SPEC.live_enemies(snap)
    )
