"""Play 0x23: west mouth (16,141) to backtrack44's first waypoint.

Pickup leftover is the west door. backtrack44 x-first to (176,117) would
RIGHT at y=141 through the water maze. Inland, UP the west column, north
band to the plus stem x=112, DOWN to y=117, RIGHT onto (176,117).
DOWN at x=176 y=93 is tile 244 water. Do not UP at x=16 (re-enter 0x22).
Clean M5 never runs this hop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B, axis_dir
from zelda_i.level1.finish import ROOM_KEY_GORIYA
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "REJOIN_DEST",
    "REJOIN_EAST_X",
    "REJOIN_MAX_FRAMES",
    "REJOIN_NORTH_Y",
    "REJOIN_PLUS_X",
    "REJOIN_WEST_COL",
    "Level1BowRejoinController",
    "level1_bow_rejoin_success",
    "make_bow_rejoin_controller",
]

LEVEL1 = 1
ALIGN = 2
REJOIN_WEST_COL = 32
REJOIN_NORTH_Y = 93
REJOIN_PLUS_X = 112
REJOIN_EAST_X = 176
REJOIN_DEST = (REJOIN_EAST_X, 117)
REJOIN_MAX_FRAMES = 2500
SAMPLE_PERIOD = 12


def make_bow_rejoin_controller() -> "Level1BowRejoinController":
    """Walk west-mouth leftover to (176,117). No poke."""
    return Level1BowRejoinController()


def level1_bow_rejoin_success(snap: ZeldaSnapshot) -> bool:
    """Play 0x23 at the first backtrack44 waypoint, bow already walked."""
    return (
        snap.level == LEVEL1
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == ROOM_KEY_GORIYA
        and int(snap.bow) != 0
        and abs(int(snap.link_x) - REJOIN_DEST[0]) <= ALIGN
        and abs(int(snap.link_y) - REJOIN_DEST[1]) <= ALIGN
    )


def _rejoin_stage(xy: tuple[int, int]) -> tuple[tuple[int, int], str, bool]:
    """RIGHT off the west mouth, plus-stem drop, east at y=117. Not x=176 DOWN."""
    x, y = xy
    tx, ty = REJOIN_DEST
    if abs(x - tx) <= ALIGN and abs(y - ty) <= ALIGN:
        return REJOIN_DEST, "rejoin_at", False
    if x < REJOIN_WEST_COL - ALIGN:
        return (REJOIN_WEST_COL, y), "rejoin_inland", False
    if y > REJOIN_NORTH_Y + ALIGN and x < REJOIN_PLUS_X - ALIGN:
        return (REJOIN_WEST_COL, REJOIN_NORTH_Y), "rejoin_up", True
    if y <= REJOIN_NORTH_Y + ALIGN and abs(x - REJOIN_PLUS_X) > ALIGN:
        return (REJOIN_PLUS_X, REJOIN_NORTH_Y), "rejoin_north", False
    if abs(x - REJOIN_PLUS_X) <= ALIGN + 4 and abs(y - ty) > ALIGN:
        return (REJOIN_PLUS_X, ty), "rejoin_drop", True
    if y >= ty - ALIGN and x < tx - ALIGN:
        return REJOIN_DEST, "rejoin_east", False
    return REJOIN_DEST, "rejoin_at", False


@dataclass
class Level1BowRejoinController(HopController):
    """Plus-stem reverse from 0x23 west mouth to (176,117)."""

    spec_id: str = "level1_bow_rejoin"
    room: int = ROOM_KEY_GORIYA
    max_frames: int = REJOIN_MAX_FRAMES
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "rejoin_at"
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_bow={int(snap.bow)}"
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"rejoin_{snap.link_x}_{snap.link_y}_keys={int(snap.keys)}"

    def _fields(self, snap: ZeldaSnapshot) -> dict[str, Any]:
        return {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "tile": int(snap.colliding_tile),
            "bow": int(snap.bow),
            "keys": int(snap.keys),
            "phase": "REJOIN",
        }

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or self.frames <= 2 or self.frames % SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    **self._fields(snap),
                    "reason": action.reason,
                }
            )
        self.leftover = self._fields(snap)
        return action

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return level1_bow_rejoin_success(snap)

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.level == 0:
            return self.mark_fail(
                f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.level != LEVEL1:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.mode != PLAY_MODE or snap.screen != ROOM_KEY_GORIYA:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if int(snap.bow) == 0:
            return self.mark_fail("bow_missing")
        dest, reason, y_first = _rejoin_stage(
            (int(snap.link_x), int(snap.link_y))
        )
        btn = axis_dir(
            (int(snap.link_x), int(snap.link_y)), dest, y_first=y_first, tol=ALIGN
        )
        if btn is None:
            return self.mark_done(snap)
        if reason == "rejoin_inland" and btn != "RIGHT":
            return FrameAction(nes_action("RIGHT"), "rejoin_inland")
        return FrameAction(nes_action(btn), reason)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "RIGHT off west mouth x=16; UP west column x=32; north band "
                "to plus x=112; DOWN y=117; RIGHT (176,117); no x=176 DOWN"
            ),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
        }
