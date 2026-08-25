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
from zelda_i.level6_gleeok18 import PASSAGE_MODE
from zelda_i.level6_overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.level6_stairs3a import (
    Stairs3APhase,
    make_stairs_3a_controller,
)
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    PLAY_MODE,
    ZeldaSnapshot,
)

__all__ = [
    "STAIRS_3A_WARP_MAX_FRAMES",
    "WARP_XY",
    "Level6Stairs3AWarpController",
    "Stairs3AWarpPhase",
    "level6_stairs3a_warp_stages",
    "level6_stairs3a_warp_success",
    "make_stairs_3a_warp_controller",
    "poke_link_position",
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


def poke_link_position(
    env: Any,
    x: int,
    y: int,
    *,
    room: int,
    from_xy: tuple[int, int],
) -> dict[str, Any]:
    """Write only ``ADDR_LINK_X`` / ``ADDR_LINK_Y``. Not Clean."""
    writes: list[dict[str, Any]] = []
    notes: list[str] = []
    assigned = 0
    try:
        mem = env.unwrapped.data.memory
        if hasattr(mem, "assign"):
            mem.assign(ADDR_LINK_X, "|u1", int(x) & 0xFF)
            mem.assign(ADDR_LINK_Y, "|u1", int(y) & 0xFF)
            assigned = 2
            notes.append("memory.assign")
        elif hasattr(mem, "set_byte"):
            mem.set_byte(ADDR_LINK_X, int(x) & 0xFF)
            mem.set_byte(ADDR_LINK_Y, int(y) & 0xFF)
            assigned = 2
            notes.append("memory.set_byte")
    except Exception as exc:
        notes.append(f"poke_fail={exc!r}")
    writes.append(
        {
            "field": "link_x",
            "address": ADDR_LINK_X,
            "from": int(from_xy[0]),
            "to": int(x),
        }
    )
    writes.append(
        {
            "field": "link_y",
            "address": ADDR_LINK_Y,
            "from": int(from_xy[1]),
            "to": int(y),
        }
    )
    return {
        "writes": writes,
        "notes": notes,
        "room": int(room),
        "room_hex": f"0x{int(room):02x}",
        "xy": [int(x), int(y)],
        "from_xy": [int(from_xy[0]), int(from_xy[1])],
        "position_writes": 1 if assigned == 2 else 0,
        "addresses": [ADDR_LINK_X, ADDR_LINK_Y],
        "progression_writes": 0,
        "capacity_writes": 0,
        "door_writes": 0,
        "inventory_writes": 0,
        "triforce_writes": 0,
        "state_load": False,
        "mid_run_state_load": False,
    }


class Stairs3AWarpPhase(Enum):
    PUSH = auto()
    POKE = auto()
    IDLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6Stairs3AWarpController:
    """Live center push, then one (x, y) write onto 0x71. Dest is RAM."""

    spec_id: str = "level6_stairs_0x3a_warp"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = STAIRS_3A_WARP_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    phase: Stairs3AWarpPhase = Stairs3AWarpPhase.PUSH
    notes: list[str] = field(default_factory=list)
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

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        self._set_phase(Stairs3AWarpPhase.FAILED, note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

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

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "rod": int(getattr(snap, "rod", 0)),
            "bow": int(getattr(snap, "bow", 0)),
            "arrows": int(getattr(snap, "arrows", 0)),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
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
                    "rod": int(getattr(snap, "rod", 0)),
                    "bow": int(getattr(snap, "bow", 0)),
                    "arrows": int(getattr(snap, "arrows", 0)),
                    "keys": int(snap.keys),
                }
            )
        return action

    def _poke(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.env is None:
            return self._fail(snap, "no_env_for_position_write")
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
            return self._fail(snap, "position_write_failed")
        self._set_phase(
            Stairs3AWarpPhase.IDLE,
            f"poked_{int(snap.link_x)}_{int(snap.link_y)}_to_{wx}_{wy}",
        )
        return self._emit(
            snap, FrameAction(nes_idle_action(), "position_write"), force=True
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
                Stairs3AWarpPhase.DONE,
                f"warped_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
            return self._emit(
                snap,
                FrameAction(nes_idle_action(), f"warped_{snap.mode}"),
                force=True,
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7, 10):
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self._fail(
                snap, f"left_0x{self.room:02x}_to_0x{snap.screen:02x}"
            )
        if int(snap.link_x) >= EAST_DOOR_XMIN and int(snap.link_y) in range(
            133, 150
        ):
            return self._fail(snap, f"east_door_{snap.link_x}_{snap.link_y}")

        if self.phase is Stairs3AWarpPhase.PUSH:
            action = self.inner.step(snap)
            if self.inner.failed:
                return self._fail(
                    snap, self.inner.notes[-1] if self.inner.notes else "push_fail"
                )
            if self.inner.phase is Stairs3APhase.ON_HOLE:
                self._set_phase(Stairs3AWarpPhase.POKE, "center_pushed")
                return self._poke(snap)
            return self._emit(snap, action)

        if self.phase is Stairs3AWarpPhase.POKE:
            return self._poke(snap)

        if self.phase is Stairs3AWarpPhase.IDLE:
            if self.phase_frames <= IDLE_AFTER_POKE:
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "warp_idle")
                )
            if self.phase_frames > IDLE_AFTER_POKE + UP_AFTER_IDLE:
                return self._fail(
                    snap,
                    f"warp_no_dest_{snap.link_x}_{snap.link_y}_tile_{snap.colliding_tile}",
                )
            return self._emit(snap, FrameAction(nes_action("UP"), "warp_up"))

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
