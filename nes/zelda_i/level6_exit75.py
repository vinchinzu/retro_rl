"""Level 6 cellar 0x75 return to play 0x09 after Magical Rod.

Rod leftover: mode 9 room 0x75 (136,141) rod=1 on the (empty) east
pedestal. v1 cardinal DOWN at (168,141) tile 250. v2 cardinal UP at
(208,141) tile 243. Inbound climbed the south face with RIGHT+UP; v3
LEFT+DOWN from the east column to drop, then floor LEFT and west UP.
Do not grant items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_occupancy import l6_leftover
from zelda_i.level6_overworld import LEVEL6, LEVEL6_ROD_WIZZ_ROOM
from zelda_i.level6_rod import (
    ROD_75_ALIGN_TOL,
    ROD_75_EAST_X,
    ROD_75_FLOOR_Y,
    ROD_75_ROOM,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "EXIT_75_MAX_FRAMES",
    "EXIT_75_WEST_X",
    "EXIT_75_MOUTH_X",
    "EXIT_75_MOUTH_Y",
    "Level6Exit75Controller",
    "level6_exit75_stages",
    "level6_exit75_success",
    "make_exit75_controller",
]

EXIT_75_WEST_X = 48
EXIT_75_MOUTH_X = 208
EXIT_75_MOUTH_Y = 93
EXIT_75_SETTLE = 16
EXIT_75_WARP_IDLE = 240
EXIT_75_MAX_FRAMES = 4000
EXIT_75_SAMPLE_PERIOD = 8
CELLAR_PLAY_MODES = (9, 11)
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


@dataclass
class Level6Exit75Controller:
    """Pedestal leftover → reverse east column → west mouth → play 0x09."""

    spec_id: str = "level6_exit_0x75"
    room: int = ROD_75_ROOM
    dest_room: int = LEVEL6_ROD_WIZZ_ROOM
    max_frames: int = EXIT_75_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    on_floor: bool = False
    mouth_idle: int = 0

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(snap.rod)

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(snap.bow)

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(snap.arrows)

    def _play_09(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == LEVEL6
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.dest_room
            and self._rod(snap) != 0
        )

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            **l6_leftover(snap),
            "submode": int(snap.submode),
            "map": int(snap.map),
        }
        if force or self.frames <= 2 or self.frames % EXIT_75_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "submode": int(snap.submode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "rod": self._rod(snap),
                    "bow": self._bow(snap),
                    "arrows": self._arrows(snap),
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _mark_done(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), "exited"), force=True)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_rod={self._rod(snap)}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if self._play_09(snap):
            return self._mark_done(
                snap,
                f"play_09_{snap.link_x}_{snap.link_y}_rod={self._rod(snap)}",
            )
        if (
            snap.level == LEVEL6
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.room
            and self._rod(snap) != 0
        ):
            return self._mark_done(
                snap,
                f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if self._rod(snap) == 0:
            return self._fail(snap, "rod_missing")
        if snap.mode == 8:
            return self._emit(snap, FrameAction(nes_idle_action(), "hurt_freeze"))
        if snap.transitioning or snap.mode in WAIT_MODES:
            btn = "UP" if snap.mode in (10, 16) else None
            if btn:
                return self._emit(snap, FrameAction(nes_action(btn), "exit_scroll"))
            return self._emit(snap, FrameAction(nes_idle_action(), "wait_scroll"))
        if snap.mode not in CELLAR_PLAY_MODES and snap.mode != PLAY_MODE:
            return self._emit(
                snap, FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            )
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}")
        if snap.mode == PLAY_MODE and snap.screen != self.room:
            return self._fail(
                snap, f"left_cellar_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )

        xy = (int(snap.link_x), int(snap.link_y))
        if xy[1] >= ROD_75_FLOOR_Y:
            self.on_floor = True
        if self.frames <= EXIT_75_SETTLE and not self.on_floor:
            return self._emit(snap, FrameAction(nes_idle_action(), "item_settle"))

        # v1 DOWN at (168,141) tile 250. v2 UP at (208,141) tile 243.
        # y=141 RIGHT is free. LEFT+DOWN reverses inbound RIGHT+UP climb.
        if not self.on_floor:
            if xy[0] < ROD_75_EAST_X - ROD_75_ALIGN_TOL:
                return self._emit(snap, FrameAction(nes_action("RIGHT"), "to_east"))
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "DOWN"), "drop_clip")
            )

        # Floor recovery (not the v2 path). LEFT to west, UP spit.
        if abs(xy[0] - EXIT_75_WEST_X) > ROD_75_ALIGN_TOL and xy[1] >= ROD_75_FLOOR_Y - 4:
            return self._emit(snap, FrameAction(nes_action("LEFT"), "cross_x"))
        if abs(xy[0] - EXIT_75_WEST_X) > ROD_75_ALIGN_TOL:
            btn = "LEFT" if xy[0] > EXIT_75_WEST_X else "RIGHT"
            return self._emit(snap, FrameAction(nes_action(btn), "climb_ax"))
        if xy[1] > EXIT_75_MOUTH_Y:
            self.mouth_idle = 0
            return self._emit(snap, FrameAction(nes_action("UP"), "climb_y"))
        self.mouth_idle += 1
        if self.mouth_idle < EXIT_75_WARP_IDLE:
            return self._emit(snap, FrameAction(nes_idle_action(), "mouth_idle"))
        return self._emit(snap, FrameAction(nes_action("UP"), "mouth_up"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": "RIGHT y=141 to x=176, LEFT+DOWN drop, LEFT x=48, UP west",
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
            "on_floor": self.on_floor,
        }


def make_exit75_controller() -> Level6Exit75Controller:
    """Leave cellar 0x75 with ADDR_ROD already set. Do not grant items."""
    return Level6Exit75Controller()


def level6_exit75_stages():
    """Rod leftover (136,141) → reverse inbound → play 0x09."""
    ctl = make_exit75_controller()
    return (
        ("level6_exit_0x75", ctl, ctl.max_frames),
    )


def level6_exit75_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x09 with ADDR_ROD. Do not require Gohma."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_ROD_WIZZ_ROOM
        and snap.triforce == 0x1F
        and snap.rod != 0
    )
