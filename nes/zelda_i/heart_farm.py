"""Clean overworld heart farming (no RAM health writes).

Used before damage-heavy corridors (Level 2 door path 0x5A→0x5C). Patrol
waypoints, swing sword, pick up dropped hearts. Assist must stay off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.nav_common import swing_action, track_stuck, unstick_wiggle, wake_or_wait_mode
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

# Default patrol on 0x4A — mid horizontal corridor (y≈140) is open; south
# wall blocks y≳160 and north pockets need the channel at x≈16–64.
DEFAULT_4A_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (48, 141),
    (96, 141),
    (144, 141),
    (192, 141),
    (208, 125),
    (176, 109),
    (128, 125),
    (80, 141),
    (40, 125),
)

DEFAULT_MAX_FRAMES = 3600
DEFAULT_STUCK_THRESHOLD = 40
FARM_SWING_PERIOD = 8
FARM_SWING_HOLD = 3
WAYPOINT_TOL = 6


class HeartFarmPhase(Enum):
    FARM = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class HeartFarmController:
    """Patrol a screen until ``filled_hearts >= min_filled`` (Clean combat only)."""

    min_filled: int = 3
    max_frames: int = DEFAULT_MAX_FRAMES
    farm_screen: int = 0x4A
    waypoints: tuple[tuple[int, int], ...] = DEFAULT_4A_WAYPOINTS
    phase: HeartFarmPhase = HeartFarmPhase.FARM
    frames: int = 0
    waypoint_index: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    start_filled: int = -1
    peak_filled: int = 0

    def reset(self) -> None:
        self.phase = HeartFarmPhase.FARM
        self.frames = 0
        self.waypoint_index = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()
        self.start_filled = -1
        self.peak_filled = 0

    def _set_done(self, note: str) -> FrameAction:
        self.success = True
        self.phase = HeartFarmPhase.DONE
        if note and (not self.notes or self.notes[-1] != note):
            self.notes.append(note)
        return FrameAction(nes_idle_action(), "farm_done")

    def _set_failed(self, note: str) -> FrameAction:
        self.success = False
        self.phase = HeartFarmPhase.FAILED
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def already_satisfied(self, snap: ZeldaSnapshot) -> bool:
        return snap.filled_hearts >= self.min_filled and self.min_filled > 0

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.start_filled < 0:
            self.start_filled = snap.filled_hearts
            self.peak_filled = snap.filled_hearts
        self.peak_filled = max(self.peak_filled, snap.filled_hearts)

        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )

        if self.min_filled <= 0:
            return self._set_done("farm_skipped")

        if snap.mode == 17:
            return self._set_failed("link_death")

        if self.frames >= self.max_frames:
            # Prefer min_filled; accept soft floor of 2 when we gained or held hearts
            # (door path needs filled_hearts > 0 through 0x5C).
            if snap.filled_hearts >= self.min_filled:
                return self._set_done("farm_timeout_ok")
            if snap.filled_hearts >= 2 and snap.filled_hearts >= self.start_filled:
                self.notes.append(
                    f"farm_soft_ok hearts={snap.filled_hearts}<{self.min_filled}"
                )
                return self._set_done("farm_soft_ok")
            self.notes.append(
                f"farm_timeout hearts={snap.filled_hearts}/{self.min_filled}"
            )
            self.phase = HeartFarmPhase.FAILED
            self.success = False
            return FrameAction(nes_idle_action(), "farm_timeout")

        if snap.filled_hearts >= self.min_filled:
            return self._set_done(
                f"farm_ok_{self.start_filled}_to_{snap.filled_hearts}"
            )

        if snap.transitioning or snap.mode not in (PLAY_MODE, 8):
            if snap.mode not in (PLAY_MODE, 8, 6, 7, 16):
                return wake_or_wait_mode(self.frames, snap.mode)
            return FrameAction(nes_idle_action(), "farm_wait_mode")

        # Drifted off farm screen — nudge back if possible; else fail soft.
        if snap.level != 0:
            return self._set_failed("left_overworld")
        if snap.screen != self.farm_screen:
            self.notes.append(f"left_screen_{snap.screen:02x}")
            # Soft fail: treat as done-with-current so door path can replan.
            self.phase = HeartFarmPhase.FAILED
            self.success = False
            return FrameAction(nes_idle_action(), "left_farm_screen")

        if self.stuck > DEFAULT_STUCK_THRESHOLD:
            action, self.stuck = unstick_wiggle(self.stuck, reason="farm_unstick")
            return action

        # Chase nearest live enemy on this screen (overworld slots 1+).
        enemies = [
            o
            for o in snap.objects
            if o.slot >= 1
            and o.type_id not in (0, 0xFF)
            and 40 < o.y < 220
            and 8 < o.x < 248
        ]
        if enemies:
            nearest = min(
                enemies,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            # Stay in the open mid band when possible (0x4A geometry).
            if snap.screen == self.farm_screen and snap.link_y < 120 and abs(dy) > 8:
                d = "DOWN"
            elif abs(dx) >= abs(dy) and abs(dx) > 4:
                d = "RIGHT" if dx > 0 else "LEFT"
            elif abs(dy) > 4:
                d = "DOWN" if dy > 0 else "UP"
            else:
                d = "RIGHT" if dx >= 0 else "LEFT"
            return swing_action(
                self.frames,
                d,
                "farm_chase",
                period=FARM_SWING_PERIOD,
                hold=FARM_SWING_HOLD,
            )

        if not self.waypoints:
            return swing_action(
                self.frames,
                "RIGHT",
                "farm_patrol",
                period=FARM_SWING_PERIOD,
                hold=FARM_SWING_HOLD,
            )

        tx, ty = self.waypoints[self.waypoint_index % len(self.waypoints)]
        if abs(snap.link_x - tx) <= WAYPOINT_TOL and abs(snap.link_y - ty) <= WAYPOINT_TOL:
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            self.stuck = 0
            tx, ty = self.waypoints[self.waypoint_index % len(self.waypoints)]

        # Prefer horizontal then vertical; keep swinging.
        if abs(snap.link_x - tx) > WAYPOINT_TOL:
            d = "RIGHT" if snap.link_x < tx else "LEFT"
        else:
            d = "DOWN" if snap.link_y < ty else "UP"
        return swing_action(
            self.frames,
            d,
            "farm",
            period=FARM_SWING_PERIOD,
            hold=FARM_SWING_HOLD,
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "min_filled": self.min_filled,
            "farm_screen": self.farm_screen,
            "start_filled": self.start_filled,
            "peak_filled": self.peak_filled,
            "waypoint_index": self.waypoint_index,
            "notes": list(self.notes),
        }
