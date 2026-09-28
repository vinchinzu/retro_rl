"""Level 4 0x20 -> 0x21 east, on the ROM lattice, Vires left alive.

0x20's east door is open (``pin_probe.py --doors``: ``E=open->0x21``), so
the room needs no clear. The old stages fought its Vires from the south
band first (7,048 frames on clean_poweron_c12, 7,636 on n3_credits, 1.26
hearts lost) and then walked a hand waypoint table with diagonal clip
presses round the water. The walk is now ``room_step`` to the door stand
under ``rollout.PolicyGuard``: the lattice knows the water and the walls,
the guard knows where a Vire will be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import room_step
from zelda_i.level4.dungeon import (
    LEVEL4,
    RIGHT_20_STAND,
    ROOM_L4_MAP_21,
    ROOM_L4_WATER_NORTH_20,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.rollout import PolicyGuard
from zelda_i.walk import live_env

__all__ = [
    "Level4Room20WalkController",
    "level4_map21_stages",
    "level4_map21_success",
    "make_map21_controller",
]

# Vires dive at ~2 px/f and split into Keese: roll only with one this near.
ROOM_20_GUARD_RADIUS = 64


@dataclass
class Level4Room20WalkController:
    """0x20 south mouth -> the (208, 141) east stand -> RIGHT into 0x21."""

    max_frames: int = 6000
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    _at_door: bool = False
    _env: Any = field(default=None, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail(self, note: str) -> FrameAction:
        self.failed = True
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail(f"timeout_{snap.link_x}_{snap.link_y}")
        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_MAP_21
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            if snap.ladder <= 0:
                return self._fail("map_no_ladder")
            self.success = True
            self.notes.append("entered_0x21")
            return FrameAction(nes_idle_action(), "done")
        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll_right")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != ROOM_L4_WATER_NORTH_20:
            return self._fail(f"wrong_room_0x{snap.screen:02x}")
        # On the stand, the push owns the frame: RIGHT walks Link off the
        # stand's tolerance, and a re-approach there walked him back
        # (208 <-> 210 for 2000 frames). A knock off the door row re-walks.
        sx, sy = RIGHT_20_STAND
        x, y = int(snap.link_x), int(snap.link_y)
        if self._at_door and (abs(y - sy) > 2 or x < sx - 4):
            self._at_door = False
        if not self._at_door:
            env = self._env if self._env is not None else live_env.current()
            step = room_step(snap, RIGHT_20_STAND, tol=1, env=env)
            if step is not None:
                return FrameAction(nes_action(step), "room20_lattice")
            self._at_door = True
        return FrameAction(nes_action("RIGHT"), "room20_door")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_map_0x21",
        }


def make_map21_controller() -> PolicyGuard:
    return PolicyGuard(Level4Room20WalkController(), trigger_radius=ROOM_20_GUARD_RADIUS)


def level4_map21_stages():
    path = make_map21_controller()
    return (("level4_map_0x21", path, path.max_frames),)


def level4_map21_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x21 with ADDR_LADDER. Do not require map pickup."""
    return (
        snap.level == LEVEL4
        and snap.screen == ROOM_L4_MAP_21
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.ladder > 0
    )
