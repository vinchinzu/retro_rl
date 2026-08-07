"""Short Clean combat kiting before damage-heavy overworld hops.

Used on 0x5A after heart farm + rejoin: clear/kite octoroks so the east
corridor keeps filled hearts for the 0x5C maze. No RAM health writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.nav_common import swing_action, track_stuck, unstick_wiggle, wake_or_wait_mode
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

DEFAULT_CLEAR_FRAMES = 201
DEFAULT_MAX_X = 160
ENEMY_TYPES = frozenset({3, 7, 13})  # tektite / octorok / similar OW
NEAR_DIST = 40
SWING_PERIOD = 6
SWING_HOLD = 3
STUCK_THRESHOLD = 40


class CorridorClearPhase(Enum):
    CLEAR = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class CorridorClearController:
    """Kite/swing on ``clear_screen`` for ``max_frames`` then yield done."""

    clear_screen: int = 0x5A
    max_frames: int = DEFAULT_CLEAR_FRAMES
    max_x: int = DEFAULT_MAX_X
    phase: CorridorClearPhase = CorridorClearPhase.CLEAR
    frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    start_filled: int = -1
    end_filled: int = -1

    def reset(self) -> None:
        self.phase = CorridorClearPhase.CLEAR
        self.frames = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()
        self.start_filled = -1
        self.end_filled = -1

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.start_filled < 0:
            self.start_filled = snap.filled_hearts
        self.end_filled = snap.filled_hearts

        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )

        if snap.mode == 17:
            self.phase = CorridorClearPhase.FAILED
            self.success = False
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "clear_death")

        if self.frames >= self.max_frames:
            self.phase = CorridorClearPhase.DONE
            self.success = True
            self.notes.append(
                f"clear_done_{self.start_filled}_to_{snap.filled_hearts}"
            )
            return FrameAction(nes_idle_action(), "clear_done")

        if snap.transitioning or snap.mode not in (PLAY_MODE, 8):
            if snap.mode not in (PLAY_MODE, 8, 6, 7, 16):
                return wake_or_wait_mode(self.frames, snap.mode)
            return FrameAction(nes_idle_action(), "clear_wait")

        if snap.screen != self.clear_screen:
            self.phase = CorridorClearPhase.DONE
            self.success = True
            self.notes.append(f"left_screen_{snap.screen:02x}")
            return FrameAction(nes_idle_action(), "clear_left_screen")

        if self.stuck > STUCK_THRESHOLD:
            action, self.stuck = unstick_wiggle(self.stuck, reason="clear_unstick")
            return action

        enemies = [
            o
            for o in snap.objects
            if o.slot >= 1
            and o.type_id in ENEMY_TYPES
            and 40 < o.y < 200
            and 8 < o.x < 248
        ]
        near = [
            o
            for o in enemies
            if abs(o.x - snap.link_x) + abs(o.y - snap.link_y) < NEAR_DIST
        ]
        if near:
            n = near[0]
            dx, dy = n.x - snap.link_x, n.y - snap.link_y
            if abs(dx) > abs(dy):
                d = "LEFT" if dx > 0 else "RIGHT"
            else:
                d = "UP" if dy > 0 else "DOWN"
            return swing_action(
                self.frames, d, "clear_kite", period=5, hold=SWING_HOLD
            )
        if enemies:
            n = min(
                enemies,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dx, dy = n.x - snap.link_x, n.y - snap.link_y
            if abs(dx) >= abs(dy):
                d = "RIGHT" if dx > 0 else "LEFT"
            else:
                d = "DOWN" if dy > 0 else "UP"
            if d == "RIGHT" and snap.link_x > self.max_x:
                d = "LEFT"
            return swing_action(
                self.frames, d, "clear_chase", period=SWING_PERIOD, hold=SWING_HOLD
            )
        d = "UP" if snap.link_y > 150 else "DOWN"
        return swing_action(
            self.frames, d, "clear_idle", period=12, hold=2
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "clear_screen": self.clear_screen,
            "start_filled": self.start_filled,
            "end_filled": self.end_filled,
            "notes": list(self.notes),
        }
