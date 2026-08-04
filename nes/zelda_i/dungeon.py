"""Data-driven early-dungeon combat and routing for Zelda I.

Keep this game-local until a second adventure game proves the API shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.level1 import (
    CLEAR_SETTLE_ALL_DEAD,
    LEVEL_1,
    PLAY_MODE,
    ROOM_KEY_STALFOS,
    ROOM_NORTH_STALFOS,
    STALFOS_OBJECT_TYPE,
)
from zelda_i.ram import ZeldaObject, ZeldaSnapshot, read_snapshot

KEESE_OBJECT_TYPE = 0x1B
GEL_OBJECT_TYPE = 0x15
GORIYA_OBJECT_TYPE = 0x06
WALLMASTER_OBJECT_TYPE = 0x27
AQUAMENTUS_OBJECT_TYPE = 0x3D


class AliveRule(str, Enum):
    """How an object type represents a living enemy."""

    TYPE = "type"
    TYPE_AND_HP = "hp"


class RewardKind(str, Enum):
    """Supported room-completion contracts."""

    CLEAR_ONLY = "clear"
    FIXED_INVENTORY = "fixed_inventory"


class DungeonPhase(Enum):
    ROUTE_ENTRY = auto()
    ENTER = auto()
    FIGHT = auto()
    COLLECT_REWARD = auto()
    DONE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class DoorRoute:
    direction: str
    waypoints: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        direction = self.direction.upper()
        if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError(f"unsupported door direction: {self.direction}")
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class CombatTuning:
    patrol: tuple[tuple[int, int], ...]
    engage_distance: int = 48
    engage_dominant_axis: bool = False
    engage_attack_period: int = 8
    engage_attack_hold: int = 4
    patrol_attack_period: int = 12
    patrol_attack_hold: int = 2
    attack_phase: int = 0
    tolerance: int = 6

    def __post_init__(self) -> None:
        if not self.patrol:
            raise ValueError("combat patrol must contain at least one waypoint")
        for period, hold in (
            (self.engage_attack_period, self.engage_attack_hold),
            (self.patrol_attack_period, self.patrol_attack_hold),
        ):
            if period <= 0 or not 0 <= hold <= period:
                raise ValueError("attack hold must be within a positive period")


@dataclass(frozen=True)
class RewardSpec:
    kind: RewardKind = RewardKind.CLEAR_ONLY
    inventory_field: str | None = None
    target: tuple[int, int] | None = None
    waypoints: tuple[tuple[int, int], ...] = ()
    settle_all_dead: int = CLEAR_SETTLE_ALL_DEAD
    y_first: bool = True

    def __post_init__(self) -> None:
        if self.kind is RewardKind.FIXED_INVENTORY:
            if not self.inventory_field or (
                self.target is None and not self.waypoints
            ):
                raise ValueError(
                    "fixed inventory rewards need a field and target or waypoints"
                )


@dataclass(frozen=True)
class DungeonRoomSpec:
    spec_id: str
    source_room: int
    room_id: int
    entry: DoorRoute
    enemy_types: tuple[int, ...]
    expected_enemy_count: int
    alive_rule: AliveRule
    combat: CombatTuning
    reward: RewardSpec = RewardSpec()
    room_item_id: int | None = None
    required_open_doors: int = 0
    exit_routes: tuple[DoorRoute, ...] = ()
    max_frames: int = 6000

    def live_enemies(self, snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
        enemies = tuple(
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 10 and obj.type_id in self.enemy_types
        )
        if self.alive_rule is AliveRule.TYPE_AND_HP:
            return tuple(obj for obj in enemies if obj.hp > 0)
        return enemies


_STALFOS_PATROL: tuple[tuple[int, int], ...] = (
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

_KEESE_54_PATROL: tuple[tuple[int, int], ...] = (
    (96, 101),
    (144, 101),
    (144, 141),
    (144, 181),
    (96, 181),
    (96, 141),
)

_KEESE_52_PATROL: tuple[tuple[int, int], ...] = (
    (96, 101),
    (144, 101),
    (176, 141),
    (144, 181),
    (96, 181),
    (64, 141),
)

_ROOM_42_PATROL: tuple[tuple[int, int], ...] = (
    (72, 109),
    (120, 109),
    (168, 109),
    (168, 157),
    (120, 181),
    (72, 157),
)

_ROOM_43_PATROL: tuple[tuple[int, int], ...] = (
    (48, 109),
    (96, 109),
    (144, 109),
    (192, 109),
    (192, 173),
    (144, 173),
    (96, 173),
    (48, 173),
)

_ROOM_44_PATROL: tuple[tuple[int, int], ...] = (
    (32, 141),
    (32, 101),
    (80, 101),
    (80, 93),
    (160, 93),
    (160, 101),
    (208, 101),
    (208, 141),
    (208, 181),
    (192, 181),
    (192, 189),
    (80, 189),
    (80, 181),
    (32, 181),
)

_WALLMASTER_PATROL: tuple[tuple[int, int], ...] = (
    (32, 141),
    (32, 109),
    (32, 173),
)

ROOM_53_SPEC = DungeonRoomSpec(
    spec_id="level1_room53",
    source_room=ROOM_NORTH_STALFOS,
    room_id=ROOM_KEY_STALFOS,
    entry=DoorRoute(
        "UP",
        ((64, 101), (120, 101), (120, 93)),
    ),
    enemy_types=(STALFOS_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(patrol=_STALFOS_PATROL),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(128, 109),
    ),
    room_item_id=0x19,
    exit_routes=(
        DoorRoute("DOWN", ((128, 189), (120, 189))),
        DoorRoute("LEFT", ((120, 93), (48, 93), (48, 141))),
        DoorRoute("RIGHT", ((120, 93), (208, 93), (208, 141))),
    ),
)

ROOM_54_SPEC = DungeonRoomSpec(
    spec_id="level1_room54",
    source_room=ROOM_KEY_STALFOS,
    room_id=0x54,
    entry=DoorRoute(
        "RIGHT",
        ((120, 93), (208, 93), (208, 141)),
    ),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_KEESE_54_PATROL,
        engage_distance=48,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x16,
    exit_routes=(
        DoorRoute("LEFT", ((128, 93), (48, 93), (48, 141))),
        DoorRoute("RIGHT", ((128, 93), (208, 93), (208, 141))),
    ),
)

ROOM_52_SPEC = DungeonRoomSpec(
    spec_id="level1_room52",
    source_room=ROOM_KEY_STALFOS,
    room_id=0x52,
    entry=DoorRoute(
        "LEFT",
        ((120, 93), (48, 93), (48, 141)),
    ),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=6,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_KEESE_52_PATROL,
        engage_distance=48,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("RIGHT", ((128, 93), (208, 93), (208, 141))),
        DoorRoute(
            "UP",
            ((176, 149), (176, 101), (120, 101), (120, 93)),
        ),
    ),
)

ROOM_42_SPEC = DungeonRoomSpec(
    spec_id="level1_room42",
    source_room=0x52,
    room_id=0x42,
    entry=DoorRoute(
        "UP",
        ((176, 149), (176, 101), (120, 101), (120, 93)),
    ),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_ROOM_42_PATROL,
        engage_distance=48,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
)

ROOM_43_SPEC = DungeonRoomSpec(
    spec_id="level1_room43",
    source_room=0x42,
    room_id=0x43,
    entry=DoorRoute(
        "RIGHT",
        ((32, 181), (208, 181), (208, 141)),
    ),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_ROOM_43_PATROL,
        engage_distance=56,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x17,
)

ROOM_33_SPEC = DungeonRoomSpec(
    spec_id="level1_room33",
    source_room=0x43,
    room_id=0x33,
    entry=DoorRoute(
        "UP",
        ((96, 133), (96, 93), (120, 93)),
    ),
    enemy_types=(STALFOS_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_STALFOS_PATROL,
        engage_distance=24,
        attack_phase=4,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(96, 173),
    ),
    room_item_id=0x19,
)

ROOM_23_SPEC = DungeonRoomSpec(
    spec_id="level1_room23",
    source_room=0x33,
    room_id=0x23,
    entry=DoorRoute(
        "UP",
        (
            (128, 173),
            (128, 133),
            (112, 133),
            (112, 93),
            (120, 93),
        ),
    ),
    enemy_types=(GORIYA_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_STALFOS_PATROL,
        engage_distance=96,
        attack_phase=2,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        waypoints=((176, 149), (176, 115), (112, 115)),
    ),
    room_item_id=0x19,
)

ROOM_44_SPEC = DungeonRoomSpec(
    spec_id="level1_room44",
    source_room=0x43,
    room_id=0x44,
    entry=DoorRoute(
        "RIGHT",
        ((120, 93), (208, 93), (208, 141)),
    ),
    enemy_types=(GORIYA_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_ROOM_44_PATROL,
        engage_distance=64,
        patrol_attack_period=8,
        patrol_attack_hold=4,
        attack_phase=7,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x1D,
)

ROOM_45_SPEC = DungeonRoomSpec(
    spec_id="level1_room45",
    source_room=0x44,
    room_id=0x45,
    entry=DoorRoute(
        "RIGHT",
        (
            (80, 101),
            (80, 93),
            (160, 93),
            (160, 101),
            (208, 101),
            (208, 141),
        ),
    ),
    enemy_types=(WALLMASTER_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_WALLMASTER_PATROL,
        # Dormant Wallmasters sit just outside the wall (x=0).  A wider
        # engage radius makes Link face and slash into the doorway instead of
        # walking a vertical patrol forever once only those slots remain.
        engage_distance=80,
        engage_dominant_axis=True,
        attack_phase=0,
        patrol_attack_period=8,
        patrol_attack_hold=4,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(160, 189),
    ),
    room_item_id=0x19,
    max_frames=9000,
)

ROOM_35_SPEC = DungeonRoomSpec(
    spec_id="level1_room35_aquamentus",
    source_room=0x45,
    room_id=0x35,
    entry=DoorRoute(
        "UP",
        ((32, 189), (32, 93), (120, 93)),
    ),
    enemy_types=(AQUAMENTUS_OBJECT_TYPE,),
    expected_enemy_count=1,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        patrol=_STALFOS_PATROL,
        engage_distance=64,
        engage_attack_period=6,
        engage_attack_hold=4,
        attack_phase=2,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="health",
        target=(192, 141),
    ),
    room_item_id=0x1A,
    max_frames=6000,
)

ROOM_SPECS: dict[int, DungeonRoomSpec] = {
    ROOM_23_SPEC.room_id: ROOM_23_SPEC,
    ROOM_33_SPEC.room_id: ROOM_33_SPEC,
    ROOM_42_SPEC.room_id: ROOM_42_SPEC,
    ROOM_43_SPEC.room_id: ROOM_43_SPEC,
    ROOM_44_SPEC.room_id: ROOM_44_SPEC,
    ROOM_45_SPEC.room_id: ROOM_45_SPEC,
    ROOM_35_SPEC.room_id: ROOM_35_SPEC,
    ROOM_52_SPEC.room_id: ROOM_52_SPEC,
    ROOM_53_SPEC.room_id: ROOM_53_SPEC,
    ROOM_54_SPEC.room_id: ROOM_54_SPEC,
}


def spec_for_room(room_id: int) -> DungeonRoomSpec:
    room_id = int(room_id)
    if room_id not in ROOM_SPECS:
        known = ", ".join(f"0x{room:02X}" for room in sorted(ROOM_SPECS))
        raise KeyError(f"no dungeon room spec for 0x{room_id:02X}; known: {known}")
    return ROOM_SPECS[room_id]


def dungeon_room_cleared(ram: np.ndarray, spec: DungeonRoomSpec) -> bool:
    """Stop predicate for a room whose enemies and clear counter are known."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL_1
        and snap.screen == spec.room_id
        and snap.mode == PLAY_MODE
        and not spec.live_enemies(snap)
        and snap.room_all_dead >= spec.reward.settle_all_dead
        and (
            not spec.required_open_doors
            or snap.cur_opened_doors & spec.required_open_doors
            == spec.required_open_doors
        )
    )


def override_room_spec(
    spec: DungeonRoomSpec,
    *,
    enemy_types: tuple[int, ...] | None = None,
    alive_rule: AliveRule | None = None,
    reward_kind: RewardKind | None = None,
    engage_distance: int | None = None,
    attack_phase: int | None = None,
) -> DungeonRoomSpec:
    """Create one sweep variant without mutating the canonical room spec."""
    combat = replace(
        spec.combat,
        engage_distance=(
            spec.combat.engage_distance
            if engage_distance is None
            else engage_distance
        ),
        attack_phase=(
            spec.combat.attack_phase if attack_phase is None else attack_phase
        ),
    )
    reward = spec.reward
    if reward_kind is not None and reward_kind is not reward.kind:
        reward = RewardSpec(kind=reward_kind)
    return replace(
        spec,
        enemy_types=spec.enemy_types if enemy_types is None else enemy_types,
        alive_rule=spec.alive_rule if alive_rule is None else alive_rule,
        combat=combat,
        reward=reward,
    )


@dataclass
class GenericDungeonRoomController:
    """Route into and clear one room described by ``DungeonRoomSpec``."""

    spec: DungeonRoomSpec
    phase: DungeonPhase = DungeonPhase.ROUTE_ENTRY
    frames: int = 0
    phase_frames: int = 0
    combat_frames: int = 0
    waypoint_index: int = 0
    patrol_index: int = 0
    initial_inventory: int | None = None
    max_live_enemies: int = 0
    last_live_enemies: int = 0
    clear_signal_seen: bool = False
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: DungeonPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            if note:
                self.notes.append(note)

    def _inventory_value(self, snap: ZeldaSnapshot) -> int:
        field_name = self.spec.reward.inventory_field
        return int(getattr(snap, field_name)) if field_name else 0

    def _follow_route(self, snap: ZeldaSnapshot, route: DoorRoute) -> FrameAction:
        tx, ty = route.waypoints[self.waypoint_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= 2 and abs(dy) <= 2:
            self.waypoint_index += 1
            if self.waypoint_index >= len(route.waypoints):
                return FrameAction(nes_idle_action(), "entry_route_done")
            return FrameAction(nes_idle_action(), "entry_waypoint_idle")
        if abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return FrameAction(nes_action(direction), "entry_route")

    def _swing(
        self,
        direction: str,
        reason: str,
        *,
        period: int,
        hold: int,
    ) -> FrameAction:
        active = (
            self.combat_frames + self.spec.combat.attack_phase
        ) % period < hold
        if active:
            return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

    def _patrol(self, snap: ZeldaSnapshot) -> FrameAction:
        tuning = self.spec.combat
        tx, ty = tuning.patrol[self.patrol_index]
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) <= tuning.tolerance and abs(dy) <= tuning.tolerance:
            self.patrol_index = (self.patrol_index + 1) % len(tuning.patrol)
            tx, ty = tuning.patrol[self.patrol_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y
        if abs(dx) > tuning.tolerance and abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > tuning.tolerance:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "UP"
        return self._swing(
            direction,
            "combat_patrol",
            period=tuning.patrol_attack_period,
            hold=tuning.patrol_attack_hold,
        )

    def _engage(self, snap: ZeldaSnapshot, target: ZeldaObject) -> FrameAction:
        dx = target.x - snap.link_x
        dy = target.y - snap.link_y
        if (
            self.spec.combat.engage_dominant_axis
            and abs(dy) > 10
            and abs(dy) > abs(dx)
        ):
            direction = "DOWN" if dy > 0 else "UP"
        elif abs(dx) > 10:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 10:
            direction = "DOWN" if dy > 0 else "UP"
        elif abs(dx) >= abs(dy):
            direction = "RIGHT" if dx >= 0 else "LEFT"
        else:
            direction = "DOWN" if dy >= 0 else "UP"
        tuning = self.spec.combat
        return self._swing(
            direction,
            "combat_engage",
            period=tuning.engage_attack_period,
            hold=tuning.engage_attack_hold,
        )

    def _combat(self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]) -> FrameAction:
        self.combat_frames += 1
        if live:
            nearest = min(
                live,
                key=lambda obj: abs(obj.x - snap.link_x)
                + abs(obj.y - snap.link_y),
            )
            distance = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            if distance < self.spec.combat.engage_distance:
                return self._engage(snap, nearest)
        return self._patrol(snap)

    def _collect_reward(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.spec.reward.waypoints:
            if self.waypoint_index >= len(self.spec.reward.waypoints):
                return FrameAction(nes_idle_action(), "reward_wait")
            tx, ty = self.spec.reward.waypoints[self.waypoint_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y
            if abs(dx) <= 2 and abs(dy) <= 2:
                self.waypoint_index += 1
                if self.waypoint_index >= len(self.spec.reward.waypoints):
                    return FrameAction(nes_idle_action(), "reward_wait")
                tx, ty = self.spec.reward.waypoints[self.waypoint_index]
                dx = tx - snap.link_x
                dy = ty - snap.link_y
            if abs(dx) > 2:
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:
                direction = "DOWN" if dy > 0 else "UP"
            return FrameAction(nes_action(direction), "collect_reward")

        target = self.spec.reward.target
        if target is None:
            return FrameAction(nes_idle_action(), "reward_wait")
        dx = target[0] - snap.link_x
        dy = target[1] - snap.link_y
        axes = ((dy, "DOWN", "UP"), (dx, "RIGHT", "LEFT"))
        if not self.spec.reward.y_first:
            axes = tuple(reversed(axes))
        for delta, positive, negative in axes:
            if abs(delta) > 5:
                return FrameAction(
                    nes_action(positive if delta > 0 else negative),
                    "collect_reward",
                )
        return FrameAction(nes_idle_action(), "reward_wait")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.initial_inventory is None and (
            self.spec.reward.kind is not RewardKind.FIXED_INVENTORY
            or (
                snap.screen == self.spec.room_id
                and snap.mode == PLAY_MODE
            )
        ):
            self.initial_inventory = self._inventory_value(snap)

        live = self.spec.live_enemies(snap)
        self.last_live_enemies = len(live)
        self.max_live_enemies = max(self.max_live_enemies, len(live))

        if snap.mode == 17:
            self._set_phase(DungeonPhase.FAILED, "link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if (
            self.spec.reward.kind is RewardKind.FIXED_INVENTORY
            and snap.screen == self.spec.room_id
            and not live
            and self.initial_inventory is not None
            and self._inventory_value(snap) > self.initial_inventory
            and (
                not self.spec.required_open_doors
                or snap.cur_opened_doors & self.spec.required_open_doors
                == self.spec.required_open_doors
            )
        ):
            self.success = True
            self._set_phase(DungeonPhase.DONE, "reward_collected")
            return FrameAction(nes_idle_action(), "done")

        if self.frames >= self.spec.max_frames:
            self._set_phase(DungeonPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.level != LEVEL_1:
            return FrameAction(nes_idle_action(), "wait_level1")

        if snap.screen == self.spec.room_id and self.phase in (
            DungeonPhase.ROUTE_ENTRY,
            DungeonPhase.ENTER,
        ):
            if snap.mode == PLAY_MODE:
                self._set_phase(DungeonPhase.FIGHT, "target_room_playable")
            else:
                return FrameAction(nes_idle_action(), "settle_target_room")

        if snap.transitioning:
            return FrameAction(
                nes_action(self.spec.entry.direction),
                "room_scroll",
            )
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is DungeonPhase.ROUTE_ENTRY:
            action = self._follow_route(snap, self.spec.entry)
            if action.reason == "entry_route_done":
                self._set_phase(DungeonPhase.ENTER, "at_entry_door")
                return FrameAction(
                    nes_action(self.spec.entry.direction),
                    "enter_target_room",
                )
            return action

        if self.phase is DungeonPhase.ENTER:
            return FrameAction(
                nes_action(self.spec.entry.direction),
                "enter_target_room",
            )

        if self.phase is DungeonPhase.FIGHT:
            if (
                not live
                and self.max_live_enemies >= self.spec.expected_enemy_count
                and snap.room_all_dead >= self.spec.reward.settle_all_dead
                and (
                    not self.spec.required_open_doors
                    or snap.cur_opened_doors & self.spec.required_open_doors
                    == self.spec.required_open_doors
                )
            ):
                self.clear_signal_seen = True
                if self.spec.reward.kind is RewardKind.CLEAR_ONLY:
                    self.success = True
                    self._set_phase(DungeonPhase.DONE, "room_cleared")
                    return FrameAction(nes_idle_action(), "done")
                self._set_phase(DungeonPhase.COLLECT_REWARD, "room_cleared")
                return self._collect_reward(snap)
            return self._combat(snap, live)

        if self.phase is DungeonPhase.COLLECT_REWARD:
            return self._collect_reward(snap)

        if self.phase is DungeonPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "failed")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "spec_id": self.spec.spec_id,
            "phase": self.phase.name,
            "frames": self.frames,
            "combat_frames": self.combat_frames,
            "max_live_enemies": self.max_live_enemies,
            "last_live_enemies": self.last_live_enemies,
            "clear_signal_seen": self.clear_signal_seen,
            "initial_inventory": self.initial_inventory,
            "notes": list(self.notes),
            "tuning": {
                "engage_distance": self.spec.combat.engage_distance,
                "engage_dominant_axis": (
                    self.spec.combat.engage_dominant_axis
                ),
                "attack_phase": self.spec.combat.attack_phase,
                "engage_attack_period": self.spec.combat.engage_attack_period,
                "engage_attack_hold": self.spec.combat.engage_attack_hold,
                "patrol_attack_period": self.spec.combat.patrol_attack_period,
                "patrol_attack_hold": self.spec.combat.patrol_attack_hold,
            },
        }
