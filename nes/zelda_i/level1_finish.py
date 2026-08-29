"""Walkthrough-informed controllers for the required Level 1 route.

The guide supplies route hypotheses; room IDs, transitions, and stop predicates
remain emulator-verified before promotion into the natural-entry chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.combat import FACING_EAST, should_swing_at
from zelda_i.combat_behaviors import projectile_threats
from zelda_i.dungeon import AQUAMENTUS_OBJECT_TYPE
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, ZeldaObject

ROOM_GEL_SWITCH = 0x42
ROOM_OLD_MAN = 0x41
ROOM_MAP = 0x43
ROOM_KEY_GORIYA = 0x23
ROOM_KEY_STALFOS_MAZE = 0x33
ROOM_BOOMERANG_GORIYA = 0x44
ROOM_AQUAMENTUS = 0x35
ROOM_TRIFORCE = 0x36
ROOM_42_LEFT_DOOR_BIT = 0x02
ROOM_42_EXIT_MAX_FRAMES = 2400
BACKTRACK_TO_44_MAX_FRAMES = 3600
AQUAMENTUS_MAX_FRAMES = 6000
TRIFORCE_MAX_FRAMES = 1800
LEVEL1_TRIFORCE_BIT = 0x01
FIREBALL_OBJECT_TYPE = 0x55

_ROOM_42_EAST_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (32, 181),
    (208, 181),
    (208, 141),
)


class Room42ExitPhase(Enum):
    PUSH_BLOCK = auto()
    ENTER_HINT = auto()
    WAIT_HINT = auto()
    RETURN_FROM_HINT = auto()
    ROUTE_EAST = auto()
    ENTER_MAP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level1Room42ExitController:
    """Push the Gel-room block, visit the hint room, and enter map room 0x43."""

    phase: Room42ExitPhase = Room42ExitPhase.PUSH_BLOCK
    frames: int = 0
    phase_frames: int = 0
    waypoint_index: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Room42ExitPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            if note:
                self.notes.append(note)

    def _route_east(self, snap: ZeldaSnapshot) -> FrameAction:
        tx, ty = _ROOM_42_EAST_WAYPOINTS[self.waypoint_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= 2 and abs(dy) <= 2:
            self.waypoint_index += 1
            if self.waypoint_index >= len(_ROOM_42_EAST_WAYPOINTS):
                return FrameAction(nes_idle_action(), "east_route_done")
            tx, ty = _ROOM_42_EAST_WAYPOINTS[self.waypoint_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y
        if abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return FrameAction(nes_action(direction), "route_east")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if snap.mode == 17:
            self._set_phase(Room42ExitPhase.FAILED, "link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if self.frames >= ROOM_42_EXIT_MAX_FRAMES:
            self._set_phase(Room42ExitPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.screen == ROOM_MAP and snap.mode == PLAY_MODE:
            self.success = True
            self._set_phase(Room42ExitPhase.DONE, "map_room_entered")
            return FrameAction(nes_idle_action(), "done")

        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")

        if self.phase is Room42ExitPhase.PUSH_BLOCK:
            if snap.screen != ROOM_GEL_SWITCH:
                return FrameAction(nes_idle_action(), "wait_room42")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), "settle_room42")
            if snap.cur_opened_doors & ROOM_42_LEFT_DOOR_BIT:
                self._set_phase(Room42ExitPhase.ENTER_HINT, "center_block_pushed")
            else:
                if abs(snap.link_y - 149) > 2:
                    direction = "DOWN" if snap.link_y < 149 else "UP"
                    return FrameAction(
                        nes_action(direction),
                        "align_switch_block_y",
                    )
                if abs(snap.link_x - 112) > 2:
                    direction = "RIGHT" if snap.link_x < 112 else "LEFT"
                    return FrameAction(
                        nes_action(direction),
                        "align_switch_block_x",
                    )
                return FrameAction(nes_action("UP"), "push_center_block")

        if self.phase is Room42ExitPhase.ENTER_HINT:
            if snap.screen == ROOM_OLD_MAN and snap.mode == PLAY_MODE:
                self._set_phase(Room42ExitPhase.WAIT_HINT, "hint_room_entered")
                return FrameAction(nes_idle_action(), "settle_hint")
            if snap.transitioning:
                return FrameAction(nes_action("LEFT"), "hint_room_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), "wait_hint_door")
            if snap.link_y < 139:
                return FrameAction(nes_action("DOWN"), "align_hint_door")
            if snap.link_y > 143:
                return FrameAction(nes_action("UP"), "align_hint_door")
            return FrameAction(nes_action("LEFT"), "enter_hint_room")

        if self.phase is Room42ExitPhase.WAIT_HINT:
            if self.phase_frames < 180:
                return FrameAction(nes_idle_action(), "wait_hint_dialog")
            self._set_phase(
                Room42ExitPhase.RETURN_FROM_HINT,
                "hint_dialog_settled",
            )

        if self.phase is Room42ExitPhase.RETURN_FROM_HINT:
            if snap.screen == ROOM_GEL_SWITCH and snap.mode == PLAY_MODE:
                self._set_phase(Room42ExitPhase.ROUTE_EAST, "returned_room42")
            else:
                return FrameAction(nes_action("RIGHT"), "return_from_hint")

        if self.phase is Room42ExitPhase.ROUTE_EAST:
            if snap.transitioning:
                return FrameAction(nes_action("RIGHT"), "map_room_scroll")
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), "wait_room42_play")
            action = self._route_east(snap)
            if action.reason == "east_route_done":
                self._set_phase(Room42ExitPhase.ENTER_MAP, "at_map_room_door")
                return FrameAction(nes_action("RIGHT"), "enter_map_room")
            return action

        if self.phase is Room42ExitPhase.ENTER_MAP:
            return FrameAction(nes_action("RIGHT"), "enter_map_room")
        return FrameAction(nes_idle_action(), "done_or_failed")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
        }


class Backtrack44Phase(Enum):
    ROUTE_23_SOUTH = auto()
    ENTER_33 = auto()
    ROUTE_33_SOUTH = auto()
    ENTER_43 = auto()
    ROUTE_43_EAST = auto()
    ENTER_44 = auto()
    DONE = auto()
    FAILED = auto()


_ROOM_23_SOUTH_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (176, 117),
    (176, 149),
    (96, 149),
    (96, 189),
    (120, 189),
)

_ROOM_33_SOUTH_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (120, 93),
    (112, 93),
    (112, 133),
    (128, 133),
    (128, 173),
    (120, 173),
    (120, 189),
)

_ROOM_43_EAST_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (120, 93),
    (208, 93),
    (208, 141),
)


@dataclass
class Level1BacktrackTo44Controller:
    """Backtrack from room 0x23 and spend its key entering room 0x44."""

    phase: Backtrack44Phase = Backtrack44Phase.ROUTE_23_SOUTH
    frames: int = 0
    phase_frames: int = 0
    waypoint_index: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Backtrack44Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            if note:
                self.notes.append(note)

    def _follow(
        self,
        snap: ZeldaSnapshot,
        waypoints: tuple[tuple[int, int], ...],
        reason: str,
    ) -> FrameAction:
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

        if snap.mode == 17:
            self._set_phase(Backtrack44Phase.FAILED, "link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if self.frames >= BACKTRACK_TO_44_MAX_FRAMES:
            self._set_phase(Backtrack44Phase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.screen == ROOM_BOOMERANG_GORIYA and snap.mode == PLAY_MODE:
            self.success = True
            self._set_phase(Backtrack44Phase.DONE, "room44_entered")
            return FrameAction(nes_idle_action(), "done")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")

        if self.phase is Backtrack44Phase.ROUTE_23_SOUTH:
            if snap.screen != ROOM_KEY_GORIYA or snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), "wait_room23")
            action = self._follow(
                snap,
                _ROOM_23_SOUTH_WAYPOINTS,
                "route_room23_south",
            )
            if action.reason.endswith("_done"):
                self._set_phase(Backtrack44Phase.ENTER_33, "at_room23_south")
                return FrameAction(nes_action("DOWN"), "enter_room33")
            return action

        if self.phase is Backtrack44Phase.ENTER_33:
            if snap.screen == ROOM_KEY_STALFOS_MAZE and snap.mode == PLAY_MODE:
                self._set_phase(Backtrack44Phase.ROUTE_33_SOUTH, "room33_entered")
            else:
                return FrameAction(nes_action("DOWN"), "enter_room33")

        if self.phase is Backtrack44Phase.ROUTE_33_SOUTH:
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_action("DOWN"), "settle_room33")
            action = self._follow(
                snap,
                _ROOM_33_SOUTH_WAYPOINTS,
                "route_room33_south",
            )
            if action.reason.endswith("_done"):
                self._set_phase(Backtrack44Phase.ENTER_43, "at_room33_south")
                return FrameAction(nes_action("DOWN"), "enter_room43")
            return action

        if self.phase is Backtrack44Phase.ENTER_43:
            if snap.screen == ROOM_MAP and snap.mode == PLAY_MODE:
                self._set_phase(Backtrack44Phase.ROUTE_43_EAST, "room43_entered")
            else:
                return FrameAction(nes_action("DOWN"), "enter_room43")

        if self.phase is Backtrack44Phase.ROUTE_43_EAST:
            if snap.mode != PLAY_MODE:
                return FrameAction(nes_idle_action(), "settle_room43")
            action = self._follow(
                snap,
                _ROOM_43_EAST_WAYPOINTS,
                "route_room43_east",
            )
            if action.reason.endswith("_done"):
                self._set_phase(Backtrack44Phase.ENTER_44, "at_room43_east")
                return FrameAction(nes_action("RIGHT"), "enter_room44")
            return action

        if self.phase is Backtrack44Phase.ENTER_44:
            return FrameAction(nes_action("RIGHT"), "enter_room44")
        return FrameAction(nes_idle_action(), "done_or_failed")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
        }


class AquamentusPhase(Enum):
    ROUTE_ENTRY = auto()
    ENTER = auto()
    WAIT_BOSS = auto()
    ALIGN = auto()
    FACE = auto()
    ATTACK = auto()
    DODGE = auto()
    COLLECT_HEART = auto()
    DONE = auto()
    FAILED = auto()


_AQUAMENTUS_ENTRY_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (32, 189),
    (32, 93),
    (120, 93),
)
# Fallback only when the boss slot has no xy yet. Live combat closes to the
# dragon instead of camping this mid-room tile (wooden sword reach is ~20px).
_AQUAMENTUS_STANCE = (128, 125)
_AQUAMENTUS_HEART = (192, 141)
_AQUAMENTUS_STANCE_OFFSET_X = 16
_AQUAMENTUS_FLOOR_X = (48, 200)
_AQUAMENTUS_FLOOR_Y = (109, 173)


@dataclass
class Level1AquamentusController:
    """Defeat Aquamentus by closing to wooden-sword range.

    Survival (``tank_hits``) ignores fireballs and keeps slashing. Clean still
    sidesteps an imminent shot, then re-closes on the live boss xy instead of
    camping the old mid-room stance at (128, 125).
    """

    phase: AquamentusPhase = AquamentusPhase.ROUTE_ENTRY
    frames: int = 0
    phase_frames: int = 0
    waypoint_index: int = 0
    attack_frames: int = 0
    dodge_frames: int = 0
    dodge_direction: str = "DOWN"
    entry_delay_frames: int = 109
    entry_delay_waited: int = 0
    stance: tuple[int, int] = _AQUAMENTUS_STANCE
    stance_offset_x: int = _AQUAMENTUS_STANCE_OFFSET_X
    threat_radius: int = 20
    dodge_duration: int = 8
    tank_hits: bool = False
    initial_health: int | None = None
    initial_containers: int | None = None
    last_boss: tuple[int, int] | None = None
    boss_seen: bool = False
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: AquamentusPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            if note:
                self.notes.append(note)

    def _route_entry(self, snap: ZeldaSnapshot) -> FrameAction:
        tx, ty = _AQUAMENTUS_ENTRY_WAYPOINTS[self.waypoint_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= 2 and abs(dy) <= 2:
            self.waypoint_index += 1
            if self.waypoint_index >= len(_AQUAMENTUS_ENTRY_WAYPOINTS):
                return FrameAction(nes_idle_action(), "boss_route_done")
            return FrameAction(
                nes_idle_action(),
                "boss_entry_waypoint_idle",
            )
        if abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return FrameAction(nes_action(direction), "route_boss_door")

    @staticmethod
    def _move_to(
        snap: ZeldaSnapshot,
        target: tuple[int, int],
        reason: str,
    ) -> FrameAction | None:
        dx = target[0] - snap.link_x
        dy = target[1] - snap.link_y
        if abs(dy) > 3:
            direction = "DOWN" if dy > 0 else "UP"
            return FrameAction(nes_action(direction), reason)
        if abs(dx) > 3:
            direction = "RIGHT" if dx > 0 else "LEFT"
            return FrameAction(nes_action(direction), reason)
        return None

    def _live_bosses(self, snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
        return tuple(
            obj
            for obj in snap.objects
            if obj.type_id == AQUAMENTUS_OBJECT_TYPE and obj.hp > 0
        )

    def _approach_tile(
        self,
        snap: ZeldaSnapshot,
        bosses: tuple[ZeldaObject, ...],
    ) -> tuple[int, int]:
        if not bosses:
            return self.stance
        boss = min(
            bosses,
            key=lambda obj: abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y),
        )
        self.last_boss = (int(boss.x), int(boss.y))
        x = max(
            _AQUAMENTUS_FLOOR_X[0],
            min(_AQUAMENTUS_FLOOR_X[1], int(boss.x) - self.stance_offset_x),
        )
        y = max(
            _AQUAMENTUS_FLOOR_Y[0],
            min(_AQUAMENTUS_FLOOR_Y[1], int(boss.y)),
        )
        return (x, y)

    def _heart_collected(self, snap: ZeldaSnapshot) -> bool:
        if (
            self.initial_containers is not None
            and snap.heart_containers > self.initial_containers
        ):
            return True
        return (
            self.initial_health is not None and snap.health > self.initial_health
        )

    def _in_sword_range(
        self,
        snap: ZeldaSnapshot,
        bosses: tuple[ZeldaObject, ...],
    ) -> bool:
        return bool(bosses) and should_swing_at(
            snap.link_x,
            snap.link_y,
            "RIGHT",
            bosses,
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if snap.mode == 17:
            self._set_phase(AquamentusPhase.FAILED, "link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if self.frames >= AQUAMENTUS_MAX_FRAMES:
            self._set_phase(AquamentusPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        bosses: tuple[ZeldaObject, ...] = ()
        if snap.screen == ROOM_AQUAMENTUS and snap.mode == PLAY_MODE:
            if self.initial_health is None:
                self.initial_health = snap.health
                self.initial_containers = snap.heart_containers
            bosses = self._live_bosses(snap)
            self.boss_seen = self.boss_seen or bool(bosses)
            if (
                self.boss_seen
                and not bosses
                and self.phase
                not in (
                    AquamentusPhase.COLLECT_HEART,
                    AquamentusPhase.DONE,
                )
            ):
                self._set_phase(
                    AquamentusPhase.COLLECT_HEART,
                    "aquamentus_defeated",
                )
            if self.phase is AquamentusPhase.COLLECT_HEART and self._heart_collected(
                snap
            ):
                self.success = True
                self._set_phase(
                    AquamentusPhase.DONE,
                    "heart_container_collected",
                )
                return FrameAction(nes_idle_action(), "done")

        if snap.transitioning:
            return FrameAction(nes_action("UP"), "boss_room_scroll")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is AquamentusPhase.ROUTE_ENTRY:
            if (
                snap.screen == 0x45
                and self.entry_delay_waited < self.entry_delay_frames
            ):
                self.entry_delay_waited += 1
                return FrameAction(nes_idle_action(), "align_boss_rng")
            action = self._route_entry(snap)
            if action.reason == "boss_route_done":
                self._set_phase(AquamentusPhase.ENTER, "at_boss_door")
                return FrameAction(nes_action("UP"), "enter_boss_room")
            return action

        if self.phase is AquamentusPhase.ENTER:
            if snap.screen == ROOM_AQUAMENTUS:
                self._set_phase(AquamentusPhase.WAIT_BOSS, "boss_room_entered")
                return FrameAction(nes_idle_action(), "wait_boss_spawn")
            return FrameAction(nes_action("UP"), "enter_boss_room")

        if self.phase is AquamentusPhase.WAIT_BOSS:
            if self.boss_seen:
                self._set_phase(AquamentusPhase.ALIGN, "aquamentus_spawned")
            else:
                return FrameAction(
                    nes_action("UP"),
                    "preposition_while_boss_spawns",
                )

        if self.phase is AquamentusPhase.COLLECT_HEART:
            action = self._move_to(
                snap,
                _AQUAMENTUS_HEART,
                "collect_heart_container",
            )
            return action or FrameAction(nes_idle_action(), "wait_heart_pickup")

        if not self.tank_hits:
            projectiles = tuple(
                obj
                for obj in snap.objects
                if obj.type_id == FIREBALL_OBJECT_TYPE
            )
            if self.phase is AquamentusPhase.DODGE:
                if self.dodge_frames > 0:
                    self.dodge_frames -= 1
                    return FrameAction(
                        nes_action(self.dodge_direction),
                        "dodge_fireball",
                    )
                self._set_phase(AquamentusPhase.ALIGN, "fireball_dodged")

            # Imminent-hit band only. The old 48px ahead window kept a mid-room
            # stance in a dodge loop and would never release a close slash.
            threatening = projectile_threats(
                snap.link_x,
                snap.link_y,
                projectiles,
                direction="RIGHT",
                ahead=20,
                behind=8,
                half_width=min(12, self.threat_radius),
            )
            if threatening:
                nearest = min(
                    threatening,
                    key=lambda obj: abs(obj.x - snap.link_x)
                    + abs(obj.y - snap.link_y),
                )
                self.dodge_direction = (
                    "DOWN"
                    if nearest.y <= snap.link_y and snap.link_y < 173
                    else "UP"
                )
                self.dodge_frames = self.dodge_duration
                self._set_phase(AquamentusPhase.DODGE, "fireball_threat")
                return FrameAction(
                    nes_action(self.dodge_direction),
                    "dodge_fireball",
                )

        if self.phase in (
            AquamentusPhase.ALIGN,
            AquamentusPhase.FACE,
            AquamentusPhase.ATTACK,
        ):
            if self._in_sword_range(snap, bosses):
                if snap.facing != FACING_EAST:
                    self._set_phase(AquamentusPhase.FACE, "boss_in_range")
                    return FrameAction(nes_action("RIGHT"), "face_aquamentus")
                self._set_phase(AquamentusPhase.ATTACK, "facing_aquamentus")
                self.attack_frames += 1
                if self.attack_frames % 6 < 4:
                    return FrameAction(nes_action("A"), "attack_aquamentus")
                return FrameAction(nes_idle_action(), "attack_release")
            tile = self._approach_tile(snap, bosses)
            action = self._move_to(snap, tile, "align_boss_stance")
            if action is not None:
                if self.phase is not AquamentusPhase.ALIGN:
                    self._set_phase(AquamentusPhase.ALIGN, "boss_out_of_range")
                return action
            if bosses and snap.link_x < int(bosses[0].x) - 8:
                self._set_phase(AquamentusPhase.ALIGN, "close_sword_gap")
                return FrameAction(nes_action("RIGHT"), "close_sword_gap")
            self._set_phase(AquamentusPhase.FACE, "boss_stance_ready")
            return FrameAction(nes_action("RIGHT"), "face_aquamentus")

        return FrameAction(nes_idle_action(), "done_or_failed")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "attack_frames": self.attack_frames,
            "boss_seen": self.boss_seen,
            "initial_health": self.initial_health,
            "initial_containers": self.initial_containers,
            "entry_delay_frames": self.entry_delay_frames,
            "stance": list(self.stance),
            "stance_offset_x": self.stance_offset_x,
            "last_boss": list(self.last_boss) if self.last_boss else None,
            "tank_hits": self.tank_hits,
            "threat_radius": self.threat_radius,
            "dodge_duration": self.dodge_duration,
            "notes": list(self.notes),
        }


class TriforcePhase(Enum):
    ENTER_ROOM = auto()
    COLLECT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level1TriforceController:
    """Enter room 0x36, collect shard 1, and verify its persistent bit."""

    phase: TriforcePhase = TriforcePhase.ENTER_ROOM
    frames: int = 0
    waypoint_index: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: TriforcePhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            if note:
                self.notes.append(note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if snap.triforce & LEVEL1_TRIFORCE_BIT:
            self.success = True
            self._set_phase(TriforcePhase.DONE, "triforce_shard_1_collected")
            return FrameAction(nes_idle_action(), "done")
        if self.frames >= TRIFORCE_MAX_FRAMES:
            self._set_phase(TriforcePhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self._set_phase(TriforcePhase.FAILED, "link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")

        if snap.screen == ROOM_TRIFORCE:
            self._set_phase(TriforcePhase.COLLECT, "triforce_room_entered")

        if self.phase is TriforcePhase.ENTER_ROOM:
            if snap.screen != ROOM_AQUAMENTUS:
                return FrameAction(nes_idle_action(), "wait_boss_room")
            return FrameAction(nes_action("RIGHT"), "enter_triforce_room")

        if self.phase is TriforcePhase.COLLECT:
            if snap.transitioning or snap.mode != PLAY_MODE:
                return FrameAction(nes_action("RIGHT"), "settle_triforce_room")
            waypoints = (
                (32, 141),
                (32, 189),
                (120, 189),
                (128, 141),
            )
            if self.waypoint_index >= len(waypoints):
                return FrameAction(nes_idle_action(), "triforce_pickup_wait")
            tx, ty = waypoints[self.waypoint_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y
            if abs(dx) <= 3 and abs(dy) <= 3:
                self.waypoint_index += 1
                return FrameAction(
                    nes_idle_action(),
                    "triforce_waypoint_idle",
                )
            if abs(dx) > 3:
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:
                direction = "DOWN" if dy > 0 else "UP"
            return FrameAction(nes_action(direction), "collect_triforce")
        return FrameAction(nes_idle_action(), "done_or_failed")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
        }


def level1_triforce_stages(*, natural_entry: bool, survival: bool = False):
    """Controller table for the L1 west-route suffix through TF 0x01.

    Path geometry stays on the specs. ``survival`` swaps the Wallmaster room
    to the off-wall overlay and the x=208 collect; Clean M5 keeps
    ``ROOM_45_SPEC``.
    """
    from dataclasses import replace

    from zelda_i.dungeon import GenericDungeonRoomController
    from zelda_i.level1_dungeon import (
        ROOM_23_SPEC,
        ROOM_33_SPEC,
        ROOM_42_SPEC,
        ROOM_43_SPEC,
        ROOM_44_SPEC,
        ROOM_45_SPEC,
        ROOM_45_SURVIVAL_SPEC,
        ROOM_52_SPEC,
    )

    room33 = ROOM_33_SPEC
    room23 = ROOM_23_SPEC
    room44 = ROOM_44_SPEC
    room45 = ROOM_45_SURVIVAL_SPEC if survival else ROOM_45_SPEC
    boss_entry_delay = 109
    if not natural_entry:
        room33 = replace(
            room33,
            combat=replace(room33.combat, engage_distance=40, attack_phase=0),
        )
        room23 = replace(
            room23,
            combat=replace(room23.combat, engage_distance=64, attack_phase=0),
        )
        room44 = replace(
            room44,
            combat=replace(room44.combat, engage_distance=80, attack_phase=6),
        )
        room45 = replace(room45, combat=replace(room45.combat, attack_phase=2))
        boss_entry_delay = 0
    return (
        ("clear52", GenericDungeonRoomController(ROOM_52_SPEC), ROOM_52_SPEC.max_frames),
        ("clear42", GenericDungeonRoomController(ROOM_42_SPEC), ROOM_42_SPEC.max_frames),
        ("exit42", Level1Room42ExitController(), ROOM_42_EXIT_MAX_FRAMES),
        ("clear43", GenericDungeonRoomController(ROOM_43_SPEC), ROOM_43_SPEC.max_frames),
        (
            "clear33_key",
            GenericDungeonRoomController(room33),
            room33.max_frames,
        ),
        (
            "clear23_key",
            GenericDungeonRoomController(room23),
            room23.max_frames,
        ),
        (
            "backtrack44",
            Level1BacktrackTo44Controller(),
            BACKTRACK_TO_44_MAX_FRAMES,
        ),
        ("clear44", GenericDungeonRoomController(room44), room44.max_frames),
        (
            "clear45_key",
            GenericDungeonRoomController(room45),
            room45.max_frames,
        ),
        (
            "aquamentus_heart",
            Level1AquamentusController(
                entry_delay_frames=boss_entry_delay,
                tank_hits=survival,
            ),
            AQUAMENTUS_MAX_FRAMES,
        ),
        (
            "triforce_shard_1",
            Level1TriforceController(),
            TRIFORCE_MAX_FRAMES,
        ),
    )
