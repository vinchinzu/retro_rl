"""Level 1 (Eagle) entrance, first-key, and early-room controllers.

Natural entry begins inside the tree transition produced by
``OverworldToLevel1Controller``. Controllers settle in entrance room 0x73,
take the open east door to room 0x74 for the first key, unlock north into
room 0x63, clear its three Stalfos, then clear room 0x53 and collect its key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.combat import should_swing_at
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

LEVEL_1 = 1
ROOM_ENTRANCE = 0x73
ROOM_FIRST_KEY = 0x74
ROOM_NORTH_STALFOS = 0x63
ROOM_NORTH_OF_63 = 0x53
ROOM_KEY_STALFOS = ROOM_NORTH_OF_63
STALFOS_OBJECT_TYPE = 0x2A
FIRST_KEY_ITEM_ID = 0x19
# Room 0x63 exposes RoomItemId=0x03 and drops no inventory item on clear.
ROOM_63_ITEM_ID = 0x03
SEGMENT_MAX_FRAMES = 6000
UNLOCK_NORTH_MAX_FRAMES = 3000
CLEAR_63_MAX_FRAMES = 6000
CLEAR_53_MAX_FRAMES = 6000
SWORD_SWING_PERIOD = 12
SWORD_SWING_FRAMES = 2
CLEAR_ENGAGE_DIST = 48
CLEAR_SETTLE_ALL_DEAD = 20
ROOM_53_KEY_X = 128
ROOM_53_KEY_Y = 109

# The statues block a direct center-to-east line. Approach below them, then
# rise into the east doorway.
_ENTRY_EAST_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (120, 149),
    (208, 149),
    (208, 141),
)

# Open lanes around the two diamond-shaped block clusters in room 0x74.
# Patrol when far; chase + hitbox-gated sword when a Stalfos is in range.
_FIRST_KEY_PATROL: tuple[tuple[int, int], ...] = (
    (48, 141),
    (48, 101),
    (112, 101),
    (112, 141),
    (112, 181),
    (208, 181),
    (208, 141),
    (208, 101),
    (112, 101),
    (112, 141),
    (112, 181),
    (48, 181),
    (48, 141),
)

# South-band exit (y≳125): drop to the south corridor then west door.
_RETURN_WEST_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (184, 181),
    (48, 181),
    (48, 141),
)
# North of the mid corridor (diamond y≈109): UP to y=101 first. Direct
# DOWN from (184, 109) eats the east diamond and stalls (live 6215f).
_RETURN_WEST_NORTH_Y = 125
_RETURN_WEST_FROM_NORTH: tuple[tuple[int, int], ...] = (
    (208, 101),
    (208, 181),
    (48, 181),
    (48, 141),
)


def return_west_waypoints(x: int, y: int) -> tuple[tuple[int, int], ...]:
    """Open-lane walk from a 0x74 pose to the west door (y≈141)."""
    if y <= _RETURN_WEST_NORTH_Y:
        return ((x, 101), *_RETURN_WEST_FROM_NORTH)
    return _RETURN_WEST_WAYPOINTS

_ENTRY_NORTH_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (208, 141),
    (208, 149),
    (120, 149),
    (120, 93),
)

# Hybrid patrol for room 0x63: box path while chasing nearby Stalfos.
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

# The room 0x63 clear policy finishes west of its central block diamond.
# Skirt the diamond above it, center on the open north doorway, and enter 0x53.
_ROOM_63_TO_53_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (64, 101),
    (120, 101),
    (120, 93),
)


class Level1KeyPhase(Enum):
    WAIT_ENTRANCE = auto()
    APPROACH_EAST = auto()
    ENTER_EAST = auto()
    FIGHT_KEY_CARRIER = auto()
    COLLECT_KEY = auto()
    DONE = auto()
    FAILED = auto()


class Level1NorthPhase(Enum):
    RETURN_WEST = auto()
    ENTER_WEST = auto()
    ROUTE_NORTH = auto()
    UNLOCK_NORTH = auto()
    DONE = auto()
    FAILED = auto()


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
class Level1FirstKeyController:
    """Frame policy from Level 1 entry transition through the first room key."""

    phase: Level1KeyPhase = Level1KeyPhase.WAIT_ENTRANCE
    frames: int = 0
    phase_frames: int = 0
    waypoint_index: int = 0
    initial_keys: int | None = None
    notes: list[str] = field(default_factory=list)
    success: bool = False
    last_health: int = 0
    saw_key_carrier: bool = False
    last_live_stalfos: int = 0

    def reset(self) -> None:
        self.phase = Level1KeyPhase.WAIT_ENTRANCE
        self.frames = 0
        self.phase_frames = 0
        self.waypoint_index = 0
        self.initial_keys = None
        self.notes.clear()
        self.success = False
        self.last_health = 0
        self.saw_key_carrier = False
        self.last_live_stalfos = 0

    def _set_phase(self, phase: Level1KeyPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            if note:
                self.notes.append(note)

    @staticmethod
    def _live_stalfos(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
        return tuple(
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 10
            and obj.type_id == STALFOS_OBJECT_TYPE
            and obj.hp > 0
        )

    def _move(
        self,
        snap: ZeldaSnapshot,
        direction: str,
        reason: str,
        *,
        enemies: tuple[ZeldaObject, ...] = (),
        allow_swing: bool = True,
    ) -> FrameAction:
        """Walk in ``direction``; slash only if allow_swing and enemy in hitbox."""
        if (
            allow_swing
            and enemies
            and should_swing_at(snap.link_x, snap.link_y, direction, enemies)
            and self.frames % SWORD_SWING_PERIOD < SWORD_SWING_FRAMES
        ):
            return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

    def _follow_waypoints(
        self,
        snap: ZeldaSnapshot,
        waypoints: tuple[tuple[int, int], ...],
        *,
        tolerance: int,
        loop: bool,
        reason: str,
        enemies: tuple[ZeldaObject, ...] = (),
        allow_swing: bool = False,
    ) -> FrameAction:
        tx, ty = waypoints[self.waypoint_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            self.waypoint_index += 1
            if self.waypoint_index >= len(waypoints):
                if not loop:
                    return FrameAction(nes_idle_action(), f"{reason}_done")
                self.waypoint_index = 0
            tx, ty = waypoints[self.waypoint_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y

        # Every leg in both tables is axis-aligned. Finish x before y only
        # when x is genuinely off the current lane.
        if abs(dx) > tolerance:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return self._move(
            snap, direction, reason, enemies=enemies, allow_swing=allow_swing
        )

    def _collect_key(self, snap: ZeldaSnapshot) -> FrameAction:
        # After the carried-key Stalfos dies, the engine clears its type but
        # leaves the dropped key position in object slot 1.
        carrier = snap.object_in_slot(1)
        tx = carrier.x if carrier and 24 <= carrier.x <= 224 else 107
        ty = carrier.y if carrier and 85 <= carrier.y <= 205 else 189
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dy) > 5:
            direction = "DOWN" if dy > 0 else "UP"
        elif abs(dx) > 5:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            return FrameAction(nes_idle_action(), "key_wait")
        return FrameAction(nes_action(direction), "collect_key")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        self.last_health = snap.health
        live_stalfos = Level1FirstKeyController._live_stalfos(snap)
        self.last_live_stalfos = len(live_stalfos)
        if self.initial_keys is None:
            self.initial_keys = snap.keys

        if snap.keys > self.initial_keys:
            self.success = True
            self._set_phase(Level1KeyPhase.DONE, "first_key_collected")
            return FrameAction(nes_idle_action(), "done")

        if self.frames >= SEGMENT_MAX_FRAMES:
            self._set_phase(Level1KeyPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.level != LEVEL_1:
            return FrameAction(nes_idle_action(), "wait_level1")

        if snap.screen == ROOM_FIRST_KEY and self.phase in (
            Level1KeyPhase.WAIT_ENTRANCE,
            Level1KeyPhase.APPROACH_EAST,
            Level1KeyPhase.ENTER_EAST,
        ):
            self._set_phase(
                Level1KeyPhase.FIGHT_KEY_CARRIER,
                "entered_key_room",
            )

        if self.phase is Level1KeyPhase.WAIT_ENTRANCE:
            if snap.screen == ROOM_ENTRANCE and snap.mode == PLAY_MODE:
                self._set_phase(Level1KeyPhase.APPROACH_EAST, "entrance_ready")
            else:
                return FrameAction(nes_idle_action(), "settle_entrance")

        if snap.transitioning:
            if self.phase in (
                Level1KeyPhase.ENTER_EAST,
                Level1KeyPhase.FIGHT_KEY_CARRIER,
            ):
                return FrameAction(nes_action("RIGHT"), "east_scroll")
            return FrameAction(nes_idle_action(), "transition_idle")

        if snap.mode not in (PLAY_MODE, 8):
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is Level1KeyPhase.APPROACH_EAST:
            action = self._follow_waypoints(
                snap,
                _ENTRY_EAST_WAYPOINTS,
                tolerance=2,
                loop=False,
                reason="entry_east",
            )
            if action.reason == "entry_east_done":
                self._set_phase(Level1KeyPhase.ENTER_EAST, "at_east_door")
                return FrameAction(nes_action("RIGHT"), "enter_east")
            return action

        if self.phase is Level1KeyPhase.ENTER_EAST:
            return FrameAction(nes_action("RIGHT"), "enter_east")

        if self.phase is Level1KeyPhase.FIGHT_KEY_CARRIER:
            carrier = snap.object_in_slot(1)
            carrier_alive = bool(
                carrier
                and carrier.type_id == STALFOS_OBJECT_TYPE
                and carrier.hp > 0
            )
            if carrier_alive:
                self.saw_key_carrier = True
            elif self.saw_key_carrier:
                self._set_phase(
                    Level1KeyPhase.COLLECT_KEY,
                    "key_carrier_defeated",
                )
                return self._collect_key(snap)
            # Chase nearest Stalfos when in range; else patrol box with
            # hitbox-gated sword (no air-swings).
            if live_stalfos:
                nearest = min(
                    live_stalfos,
                    key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
                )
                dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
                if dist < CLEAR_ENGAGE_DIST:
                    dx = nearest.x - snap.link_x
                    dy = nearest.y - snap.link_y
                    if abs(dx) > 10:
                        direction = "RIGHT" if dx > 0 else "LEFT"
                    elif abs(dy) > 10:
                        direction = "DOWN" if dy > 0 else "UP"
                    else:
                        direction = "RIGHT" if dx >= 0 else "LEFT"
                    return self._move(
                        snap,
                        direction,
                        "key_engage",
                        enemies=live_stalfos,
                        allow_swing=True,
                    )
            return self._follow_waypoints(
                snap,
                _FIRST_KEY_PATROL,
                tolerance=5,
                loop=True,
                reason="key_room_patrol",
                enemies=live_stalfos,
                allow_swing=True,
            )

        if self.phase is Level1KeyPhase.COLLECT_KEY:
            return self._collect_key(snap)

        if self.phase is Level1KeyPhase.DONE:
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
            "saw_key_carrier": self.saw_key_carrier,
            "last_live_stalfos": self.last_live_stalfos,
        }


@dataclass
class Level1UnlockNorthController:
    """Return from room 0x74, spend the first key, and enter room 0x63."""

    phase: Level1NorthPhase = Level1NorthPhase.RETURN_WEST
    frames: int = 0
    phase_frames: int = 0
    waypoint_index: int = 0
    initial_keys: int | None = None
    notes: list[str] = field(default_factory=list)
    success: bool = False
    last_health: int = 0
    north_ready_frames: int = 0
    west_waypoints: tuple[tuple[int, int], ...] | None = None

    def reset(self) -> None:
        self.phase = Level1NorthPhase.RETURN_WEST
        self.frames = 0
        self.phase_frames = 0
        self.waypoint_index = 0
        self.initial_keys = None
        self.notes.clear()
        self.success = False
        self.last_health = 0
        self.north_ready_frames = 0
        self.west_waypoints = None

    def _set_phase(self, phase: Level1NorthPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            if note:
                self.notes.append(note)

    def _follow_waypoints(
        self,
        snap: ZeldaSnapshot,
        waypoints: tuple[tuple[int, int], ...],
        reason: str,
    ) -> FrameAction:
        """Route without sword — unlock path has no combat requirement."""
        tx, ty = waypoints[self.waypoint_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= 2 and abs(dy) <= 2:
            self.waypoint_index += 1
            if self.waypoint_index >= len(waypoints):
                return FrameAction(nes_idle_action(), f"{reason}_done")
            tx, ty = waypoints[self.waypoint_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y
        if abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return FrameAction(nes_action(direction), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        self.last_health = snap.health
        if self.initial_keys is None:
            self.initial_keys = snap.keys

        live_stalfos = Level1FirstKeyController._live_stalfos(snap)
        if (
            snap.level == LEVEL_1
            and snap.screen == ROOM_NORTH_STALFOS
            and snap.mode == PLAY_MODE
            and len(live_stalfos) >= 3
        ):
            self.north_ready_frames += 1
            if self.north_ready_frames >= 30:
                self.success = True
                self._set_phase(Level1NorthPhase.DONE, "north_room_ready")
                return FrameAction(nes_idle_action(), "done")
        elif snap.screen != ROOM_NORTH_STALFOS:
            self.north_ready_frames = 0

        if self.frames >= UNLOCK_NORTH_MAX_FRAMES:
            self._set_phase(Level1NorthPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.level != LEVEL_1:
            return FrameAction(nes_idle_action(), "wait_level1")

        if snap.screen == ROOM_ENTRANCE and self.phase in (
            Level1NorthPhase.RETURN_WEST,
            Level1NorthPhase.ENTER_WEST,
        ):
            self._set_phase(Level1NorthPhase.ROUTE_NORTH, "back_in_entrance")

        if snap.transitioning:
            hold = (
                "LEFT"
                if self.phase in (
                    Level1NorthPhase.ENTER_WEST,
                    Level1NorthPhase.ROUTE_NORTH,
                )
                else "UP"
            )
            return FrameAction(nes_action(hold), "room_scroll")

        if snap.mode not in (PLAY_MODE, 8):
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is Level1NorthPhase.RETURN_WEST:
            if self.west_waypoints is None:
                self.west_waypoints = return_west_waypoints(snap.link_x, snap.link_y)
            action = self._follow_waypoints(
                snap,
                self.west_waypoints,
                "return_west",
            )
            if action.reason == "return_west_done":
                self._set_phase(Level1NorthPhase.ENTER_WEST, "at_west_door")
                return FrameAction(nes_action("LEFT"), "enter_west")
            return action

        if self.phase is Level1NorthPhase.ENTER_WEST:
            return FrameAction(nes_action("LEFT"), "enter_west")

        if self.phase is Level1NorthPhase.ROUTE_NORTH:
            action = self._follow_waypoints(
                snap,
                _ENTRY_NORTH_WAYPOINTS,
                "route_north",
            )
            if action.reason == "route_north_done":
                self._set_phase(Level1NorthPhase.UNLOCK_NORTH, "at_locked_door")
                return FrameAction(nes_action("UP"), "unlock_north")
            return action

        if self.phase is Level1NorthPhase.UNLOCK_NORTH:
            return FrameAction(nes_action("UP"), "unlock_north")

        if self.phase is Level1NorthPhase.DONE:
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
            "north_ready_frames": self.north_ready_frames,
        }


def level1_first_key_success(ram: np.ndarray) -> bool:
    """Stop predicate: at least one key owned inside Level 1."""
    snap = read_snapshot(ram)
    return snap.level == LEVEL_1 and snap.keys >= 1


def level1_north_room_success(ram: np.ndarray) -> bool:
    """Stop predicate: room 0x63 playable with its three Stalfos spawned."""
    snap = read_snapshot(ram)
    live_stalfos = Level1FirstKeyController._live_stalfos(snap)
    return (
        snap.level == LEVEL_1
        and snap.screen == ROOM_NORTH_STALFOS
        and snap.mode == PLAY_MODE
        and len(live_stalfos) >= 3
    )


@dataclass
class Level1Clear63Controller:
    """Clear the three Stalfos in Level 1 room 0x63.

    Probe-stable policy (2026-07-28): hybrid nearest-target chase inside engage
    range, otherwise box patrol with sword pulses. Room clear yields no key or
    inventory item (RoomItemId stays 0x03); south door remains open back to
    0x73 and the north door stays open into room 0x53.
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
        """Slash only when an enemy is in sword hitbox or contact-close."""
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
        """Walk patrol box without A — sword only on engage hitbox."""
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

        # Mode 8 is post-hit freeze; wait it out instead of buffering moves.
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

    Room 0x53 uses the same robust chase/patrol combat as room 0x63. Its key is
    a fixed room-clear item at (128, 109), not the transient green-rupee object
    that can appear at the final enemy position.
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
        # Finish y first so Link passes the west block column before moving
        # toward the fixed key tile in the open center lane.
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
