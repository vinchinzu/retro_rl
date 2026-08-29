"""Level 1 north Stalfos rooms 0x63 / 0x53 (clear hops).

Entrance first-key / unlock-north stay in ``level1.path``. Historical
names are re-exported from that module for composer strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import should_swing_at
from zelda_i.level1.path import (
    CLEAR_53_MAX_FRAMES,
    CLEAR_63_MAX_FRAMES,
    CLEAR_ENGAGE_DIST,
    CLEAR_SETTLE_ALL_DEAD,
    LEVEL_1,
    ROOM_53_KEY_X,
    ROOM_53_KEY_Y,
    ROOM_KEY_STALFOS,
    ROOM_NORTH_STALFOS,
    STALFOS_OBJECT_TYPE,
    SWORD_SWING_FRAMES,
    SWORD_SWING_PERIOD,
    Level1FirstKeyController,
)
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

_ROOM_63_PATROL: tuple[tuple[int, int], ...] = (
    (64, 117),
    (112, 117),
    (160, 117),
    (192, 117),
    (192, 149),
    (160, 149),
    (112, 149),
    (64, 149),
    (64, 181),
    (112, 181),
    (160, 181),
    (192, 181),
)

# Skirt the 0x63 diamond, then the open north doorway into 0x53.
_ROOM_63_TO_53_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (64, 101),
    (120, 101),
    (120, 93),
)


class Level1Clear63Phase(Enum):
    FIGHT = auto()
    DONE = auto()
    FAILED = auto()


class Level1Clear53Phase(Enum):
    ROUTE_NORTH = auto()
    ENTER_NORTH = auto()
    FIGHT = auto()
    COLLECT_KEY = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level1Clear63Controller:
    """Clear the three Stalfos in Level 1 room 0x63.

    Hybrid nearest-target chase inside engage range, otherwise box patrol
    with sword pulses. Room clear yields no inventory item.
    """

    phase: Level1Clear63Phase = Level1Clear63Phase.FIGHT
    frames: int = 0
    phase_frames: int = 0
    waypoint_index: int = 0
    notes: list[str] = field(default_factory=list)
    success: bool = False
    last_health: int = 0
    last_live_stalfos: int = 0
    max_live_stalfos: int = 0
    clear_settle_frames: int = 0

    def reset(self) -> None:
        self.phase = Level1Clear63Phase.FIGHT
        self.frames = 0
        self.phase_frames = 0
        self.waypoint_index = 0
        self.notes.clear()
        self.success = False
        self.last_health = 0
        self.last_live_stalfos = 0
        self.max_live_stalfos = 0
        self.clear_settle_frames = 0

    def _set_phase(self, phase: Level1Clear63Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            if note:
                self.notes.append(note)

    def _swing(
        self,
        snap: ZeldaSnapshot,
        direction: str,
        reason: str,
        *,
        enemies: tuple[ZeldaObject, ...],
        heavy: bool = False,
    ) -> FrameAction:
        period = 8 if heavy else SWORD_SWING_PERIOD
        hold = 4 if heavy else SWORD_SWING_FRAMES
        if (
            enemies
            and should_swing_at(snap.link_x, snap.link_y, direction, enemies)
            and self.frames % period < hold
        ):
            return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

    def _patrol(self, snap: ZeldaSnapshot) -> FrameAction:
        tx, ty = _ROOM_63_PATROL[self.waypoint_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= 6 and abs(dy) <= 6:
            self.waypoint_index = (self.waypoint_index + 1) % len(_ROOM_63_PATROL)
            tx, ty = _ROOM_63_PATROL[self.waypoint_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y
        if abs(dx) > 6 and abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 6:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "UP"
        return FrameAction(nes_action(direction), "clear_patrol")

    def _engage(self, snap: ZeldaSnapshot, target: ZeldaObject) -> FrameAction:
        dx = target.x - snap.link_x
        dy = target.y - snap.link_y
        if abs(dx) > 10:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 10:
            direction = "DOWN" if dy > 0 else "UP"
        elif abs(dx) >= abs(dy):
            direction = "RIGHT" if dx >= 0 else "LEFT"
        else:
            direction = "DOWN" if dy >= 0 else "UP"
        return self._swing(
            snap, direction, "clear_engage", enemies=(target,), heavy=True
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        self.last_health = snap.health
        live = Level1FirstKeyController._live_stalfos(snap)
        self.last_live_stalfos = len(live)
        self.max_live_stalfos = max(self.max_live_stalfos, len(live))

        if (
            snap.level == LEVEL_1
            and snap.screen == ROOM_NORTH_STALFOS
            and snap.mode == PLAY_MODE
            and not live
            and snap.room_all_dead >= CLEAR_SETTLE_ALL_DEAD
        ):
            self.clear_settle_frames += 1
            if self.clear_settle_frames >= 8:
                self.success = True
                self._set_phase(Level1Clear63Phase.DONE, "room_63_cleared")
                return FrameAction(nes_idle_action(), "done")
        else:
            self.clear_settle_frames = 0

        if self.frames >= CLEAR_63_MAX_FRAMES:
            self._set_phase(Level1Clear63Phase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.level != LEVEL_1:
            return FrameAction(nes_idle_action(), "wait_level1")

        if snap.transitioning:
            return FrameAction(nes_idle_action(), "transition_idle")

        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")

        if snap.mode not in (PLAY_MODE, 8):
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is Level1Clear63Phase.FIGHT:
            if live:
                nearest = min(
                    live,
                    key=lambda obj: abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y),
                )
                dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
                if dist < CLEAR_ENGAGE_DIST:
                    return self._engage(snap, nearest)
            return self._patrol(snap)

        if self.phase is Level1Clear63Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "failed")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "waypoint_index": self.waypoint_index,
            "last_health": self.last_health,
            "last_live_stalfos": self.last_live_stalfos,
            "max_live_stalfos": self.max_live_stalfos,
            "clear_settle_frames": self.clear_settle_frames,
        }


def level1_room_63_cleared(ram: np.ndarray) -> bool:
    """Stop predicate: room 0x63 cleared of its three Stalfos."""
    snap = read_snapshot(ram)
    live_stalfos = Level1FirstKeyController._live_stalfos(snap)
    return (
        snap.level == LEVEL_1
        and snap.screen == ROOM_NORTH_STALFOS
        and snap.mode == PLAY_MODE
        and len(live_stalfos) == 0
        and snap.room_all_dead >= CLEAR_SETTLE_ALL_DEAD
    )


@dataclass
class Level1Clear53Controller:
    """Enter room 0x53, clear five Stalfos, and collect its room key.

    Same chase/patrol combat as 0x63. Key is a fixed room-clear item at
    (128, 109), not the transient green-rupee object.
    """

    phase: Level1Clear53Phase = Level1Clear53Phase.ROUTE_NORTH
    frames: int = 0
    phase_frames: int = 0
    waypoint_index: int = 0
    initial_keys: int | None = None
    notes: list[str] = field(default_factory=list)
    success: bool = False
    last_health: int = 0
    last_live_stalfos: int = 0
    max_live_stalfos: int = 0
    clear_signal_seen: bool = False
    combat: Level1Clear63Controller = field(
        default_factory=Level1Clear63Controller,
        repr=False,
    )

    def reset(self) -> None:
        self.phase = Level1Clear53Phase.ROUTE_NORTH
        self.frames = 0
        self.phase_frames = 0
        self.waypoint_index = 0
        self.initial_keys = None
        self.notes.clear()
        self.success = False
        self.last_health = 0
        self.last_live_stalfos = 0
        self.max_live_stalfos = 0
        self.clear_signal_seen = False
        self.combat.reset()

    def _set_phase(self, phase: Level1Clear53Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            if note:
                self.notes.append(note)

    def _route_north(self, snap: ZeldaSnapshot) -> FrameAction:
        tx, ty = _ROOM_63_TO_53_WAYPOINTS[self.waypoint_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= 2 and abs(dy) <= 2:
            self.waypoint_index += 1
            if self.waypoint_index >= len(_ROOM_63_TO_53_WAYPOINTS):
                return FrameAction(nes_idle_action(), "route_room53_done")
            return FrameAction(nes_idle_action(), "route_room53_waypoint")
        if abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return FrameAction(nes_action(direction), "route_room53")

    @staticmethod
    def _collect_key(snap: ZeldaSnapshot) -> FrameAction:
        dx = ROOM_53_KEY_X - snap.link_x
        dy = ROOM_53_KEY_Y - snap.link_y
        if abs(dy) > 5:
            direction = "DOWN" if dy > 0 else "UP"
        elif abs(dx) > 5:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            return FrameAction(nes_idle_action(), "room53_key_wait")
        return FrameAction(nes_action(direction), "collect_room53_key")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        self.last_health = snap.health
        if self.initial_keys is None:
            self.initial_keys = snap.keys

        live = Level1FirstKeyController._live_stalfos(snap)
        self.last_live_stalfos = len(live)
        self.max_live_stalfos = max(self.max_live_stalfos, len(live))

        if (
            snap.level == LEVEL_1
            and snap.screen == ROOM_KEY_STALFOS
            and not live
            and snap.keys > self.initial_keys
        ):
            self.success = True
            self._set_phase(Level1Clear53Phase.DONE, "room_53_key_collected")
            return FrameAction(nes_idle_action(), "done")

        if self.frames >= CLEAR_53_MAX_FRAMES:
            self._set_phase(Level1Clear53Phase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.level != LEVEL_1:
            return FrameAction(nes_idle_action(), "wait_level1")

        if snap.screen == ROOM_KEY_STALFOS and self.phase in (
            Level1Clear53Phase.ROUTE_NORTH,
            Level1Clear53Phase.ENTER_NORTH,
        ):
            if snap.mode == PLAY_MODE:
                self._set_phase(Level1Clear53Phase.FIGHT, "room_53_playable")
            else:
                return FrameAction(nes_idle_action(), "settle_room53")

        if snap.transitioning:
            return FrameAction(nes_action("UP"), "room53_scroll")

        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode not in (PLAY_MODE, 8):
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is Level1Clear53Phase.ROUTE_NORTH:
            action = self._route_north(snap)
            if action.reason == "route_room53_done":
                self._set_phase(Level1Clear53Phase.ENTER_NORTH, "at_room53_door")
                return FrameAction(nes_action("UP"), "enter_room53")
            return action

        if self.phase is Level1Clear53Phase.ENTER_NORTH:
            return FrameAction(nes_action("UP"), "enter_room53")

        if self.phase is Level1Clear53Phase.FIGHT:
            if (
                not live
                and self.max_live_stalfos >= 5
                and snap.room_all_dead >= CLEAR_SETTLE_ALL_DEAD
            ):
                self.clear_signal_seen = True
                self._set_phase(
                    Level1Clear53Phase.COLLECT_KEY,
                    "room_53_cleared",
                )
                return self._collect_key(snap)
            action = self.combat.step(snap)
            return FrameAction(action.action, f"room53_{action.reason}")

        if self.phase is Level1Clear53Phase.COLLECT_KEY:
            return self._collect_key(snap)

        if self.phase is Level1Clear53Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "failed")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "waypoint_index": self.waypoint_index,
            "initial_keys": self.initial_keys,
            "last_health": self.last_health,
            "last_live_stalfos": self.last_live_stalfos,
            "max_live_stalfos": self.max_live_stalfos,
            "clear_signal_seen": self.clear_signal_seen,
            "combat_frames": self.combat.frames,
        }


def level1_room_53_cleared(ram: np.ndarray) -> bool:
    """Stop predicate: room 0x53 cleared and its fixed room key collected."""
    snap = read_snapshot(ram)
    live_stalfos = Level1FirstKeyController._live_stalfos(snap)
    return (
        snap.level == LEVEL_1
        and snap.screen == ROOM_KEY_STALFOS
        and snap.mode == PLAY_MODE
        and len(live_stalfos) == 0
        and snap.room_all_dead >= CLEAR_SETTLE_ALL_DEAD
        and snap.keys >= 1
    )
