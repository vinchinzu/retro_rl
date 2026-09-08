"""Clean overworld heart farming (no RAM health writes).

Patrol, swing, pick up drops. Overworld foes do not respawn until Link
leaves; a restock neighbor (kwargs or ``farm_at``) scrolls out and back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import overworld_threat_objects
from zelda_i.dungeon.ids import RUPEE_DROP_OBJECT_TYPE
from zelda_i.overworld.common import track_stuck, unstick_wiggle, wake_or_wait_mode, walk_or_swing
from zelda_i.overworld.locations import farm_at
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

# Screen-agnostic sweep for the low-heart hook: the open mid band plus one
# north lane. Any overworld screen the spine walks has a live y≈141 corridor
# (that is how the hop crossed it), so this patrols where enemies already are.
BAND_SWEEP_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (64, 141),
    (120, 141),
    (176, 141),
    (176, 109),
    (120, 109),
    (64, 109),
)

DEFAULT_MAX_FRAMES = 3600
DEFAULT_STUCK_THRESHOLD = 40
DEFAULT_EMPTY_WAIT_FRAMES = 90
FARM_SWING_PERIOD = 8
FARM_SWING_HOLD = 3
WAYPOINT_TOL = 6

_OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}


class HeartFarmPhase(Enum):
    FARM = auto()
    DONE = auto()
    FAILED = auto()


def _rupee_drops(snap: ZeldaSnapshot) -> tuple:
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot >= 1
        and int(obj.type_id) == RUPEE_DROP_OBJECT_TYPE
        and 40 < obj.y < 220
        and 8 < obj.x < 248
    )


@dataclass
class HeartFarmController:
    """Patrol a screen until ``filled_hearts >= min_filled`` (Clean combat only)."""

    min_filled: int = 3
    max_frames: int = DEFAULT_MAX_FRAMES
    farm_screen: int = 0x4A
    waypoints: tuple[tuple[int, int], ...] = DEFAULT_4A_WAYPOINTS
    restock_neighbor_screen: int | None = None
    restock_direction: str | None = None  # from farm_screen toward neighbor
    empty_wait_frames: int = DEFAULT_EMPTY_WAIT_FRAMES
    phase: HeartFarmPhase = HeartFarmPhase.FARM
    frames: int = 0
    waypoint_index: int = 0
    stuck: int = 0
    empty_frames: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    start_filled: int = -1
    peak_filled: int = 0

    def __post_init__(self) -> None:
        if self.restock_neighbor_screen is not None and self.restock_direction is not None:
            return
        spot = farm_at(self.farm_screen)
        if spot is None:
            return
        if self.restock_neighbor_screen is None:
            self.restock_neighbor_screen = spot.restock_neighbor
        if self.restock_direction is None:
            self.restock_direction = spot.restock_direction

    def reset(self) -> None:
        self.phase = HeartFarmPhase.FARM
        self.frames = 0
        self.waypoint_index = 0
        self.stuck = 0
        self.empty_frames = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()
        self.start_filled = -1
        self.peak_filled = 0

    def _has_restock(self) -> bool:
        return self.restock_neighbor_screen is not None and self.restock_direction is not None

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

    def _transition_direction(self, snap: ZeldaSnapshot) -> str:
        direction = self.restock_direction or "LEFT"
        returning_to_farm = (
            snap.screen == self.restock_neighbor_screen
            or snap.next_screen == self.farm_screen
        )
        if returning_to_farm:
            return _OPPOSITE[direction]
        return direction

    def _hearts_met(self, snap: ZeldaSnapshot) -> bool:
        return snap.filled_hearts >= self.min_filled

    def _chase(self, snap: ZeldaSnapshot, target: Any, reason: str) -> FrameAction:
        dx = target.x - snap.link_x
        dy = target.y - snap.link_y
        if reason == "farm_chase" and snap.screen == self.farm_screen and snap.link_y < 120 and abs(dy) > 8:
            d = "DOWN"
        elif abs(dx) >= abs(dy) and abs(dx) > 4:
            d = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 4:
            d = "DOWN" if dy > 0 else "UP"
        else:
            d = "RIGHT" if dx >= 0 else "LEFT"
        return walk_or_swing(
            self.frames,
            d,
            reason,
            snap,
            period=FARM_SWING_PERIOD,
            hold=FARM_SWING_HOLD,
        )

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

        restock = self._has_restock()
        recovered = self._hearts_met(snap)
        if recovered and not restock:
            return self._set_done(
                f"farm_ok_{self.start_filled}_to_{snap.filled_hearts}"
            )

        if snap.transitioning or snap.mode not in (PLAY_MODE, 8):
            if restock and snap.transitioning:
                reason = "farm_return_scroll" if recovered else "farm_scroll"
                return FrameAction(nes_action(self._transition_direction(snap)), reason)
            if snap.mode not in (PLAY_MODE, 8, 6, 7, 16):
                return wake_or_wait_mode(self.frames, snap.mode)
            return FrameAction(nes_idle_action(), "farm_wait_mode")

        if snap.level != 0:
            return self._set_failed("left_overworld")

        # Restock: stay FARM on the neighbor and walk back. Any other screen
        # still soft-fails. Do not invent a neighbor when the pair is missing.
        if restock and snap.screen == self.restock_neighbor_screen:
            return FrameAction(
                nes_action(_OPPOSITE[self.restock_direction or "LEFT"]),
                "farm_respawn",
            )
        if snap.screen != self.farm_screen:
            self.notes.append(f"left_screen_{snap.screen:02x}")
            self.phase = HeartFarmPhase.FAILED
            self.success = False
            return FrameAction(nes_idle_action(), "left_farm_screen")

        if recovered:
            return self._set_done(
                f"farm_ok_{self.start_filled}_to_{snap.filled_hearts}"
            )

        if self.stuck > DEFAULT_STUCK_THRESHOLD:
            action, self.stuck = unstick_wiggle(self.stuck, reason="farm_unstick")
            return action

        enemies = list(overworld_threat_objects(snap))
        if enemies:
            self.empty_frames = 0
            nearest = min(
                enemies,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            return self._chase(snap, nearest, "farm_chase")

        drops = _rupee_drops(snap)
        if drops:
            self.empty_frames = 0
            nearest = min(
                drops,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            return self._chase(snap, nearest, "farm_rupee")

        if restock:
            self.empty_frames += 1
            if self.empty_frames < self.empty_wait_frames:
                direction = "RIGHT" if snap.link_x < 160 else "LEFT"
                return FrameAction(nes_action(direction), "farm_wait")
            self.empty_frames = 0
            return FrameAction(nes_action(self.restock_direction or "LEFT"), "farm_leave")

        if not self.waypoints:
            return walk_or_swing(
                self.frames,
                "RIGHT",
                "farm_patrol",
                snap,
                period=FARM_SWING_PERIOD,
                hold=FARM_SWING_HOLD,
            )

        tx, ty = self.waypoints[self.waypoint_index % len(self.waypoints)]
        if abs(snap.link_x - tx) <= WAYPOINT_TOL and abs(snap.link_y - ty) <= WAYPOINT_TOL:
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            self.stuck = 0
            tx, ty = self.waypoints[self.waypoint_index % len(self.waypoints)]

        if abs(snap.link_x - tx) > WAYPOINT_TOL:
            d = "RIGHT" if snap.link_x < tx else "LEFT"
        else:
            d = "DOWN" if snap.link_y < ty else "UP"
        return walk_or_swing(
            self.frames,
            d,
            "farm",
            snap,
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
            "restock_neighbor_screen": self.restock_neighbor_screen,
            "restock_direction": self.restock_direction,
            "start_filled": self.start_filled,
            "peak_filled": self.peak_filled,
            "waypoint_index": self.waypoint_index,
            "notes": list(self.notes),
        }
