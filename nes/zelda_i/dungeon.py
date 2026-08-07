"""Data-driven early-dungeon combat engine for Zelda I.

Room tables live in per-level modules (``level1_dungeon``, ``level2_dungeon``,
``level3_dungeon``, …). This module is the shared controller + registry API.

Keep this game-local until a second adventure game proves the API shape.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.combat import should_swing_at
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot

# Settle frames after last kill for CLEAR_ONLY stop (was level1.CLEAR_SETTLE_ALL_DEAD).
CLEAR_SETTLE_ALL_DEAD = 20

# Shared enemy type IDs (used by multiple dungeon levels / finish helpers).
KEESE_OBJECT_TYPE = 0x1B
GEL_OBJECT_TYPE = 0x15
# L2 boom room 0x4f uses type 0x05 (walkthrough blue Goriya; residual label).
# Red Goriya on 0x5e is 0x06.
BLUE_GORIYA_OBJECT_TYPE = 0x05
GORIYA_OBJECT_TYPE = 0x06
WALLMASTER_OBJECT_TYPE = 0x27
ROPE_OBJECT_TYPE = 0x28
AQUAMENTUS_OBJECT_TYPE = 0x3D
# Statue / fireball projectiles (Aquamentus + L2 0x4f); not room-clear targets.
FIREBALL_OBJECT_TYPE = 0x55


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
    level: int = 1

    def live_enemies(self, snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
        enemies = tuple(
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= 10 and obj.type_id in self.enemy_types
        )
        if self.alive_rule is AliveRule.TYPE_AND_HP:
            return tuple(obj for obj in enemies if obj.hp > 0)
        return enemies


# --- Spec registry: primary key (level, room_id); room_id-only for unique rooms ---

_ROOM_SPECS_BY_LEVEL: dict[tuple[int, int], DungeonRoomSpec] = {}
# Backward-compat room_id → spec when the room_id is unique across levels.
ROOM_SPECS: dict[int, DungeonRoomSpec] = {}
_DEFAULT_SPECS_LOADED = False


def register_room_spec(spec: DungeonRoomSpec) -> None:
    """Register a room spec under ``(level, room_id)`` and room_id if unique."""
    key = (int(spec.level), int(spec.room_id))
    _ROOM_SPECS_BY_LEVEL[key] = spec
    room_id = int(spec.room_id)
    existing = ROOM_SPECS.get(room_id)
    if existing is None or existing.level == spec.level:
        ROOM_SPECS[room_id] = spec
    # Ambiguous room_id across levels: leave prior room_id entry; use level=.


def ensure_default_specs() -> None:
    """Import built-in level room modules so specs self-register."""
    global _DEFAULT_SPECS_LOADED
    if _DEFAULT_SPECS_LOADED:
        return
    _DEFAULT_SPECS_LOADED = True
    # Import order is free; each module calls register_room_spec on load.
    import zelda_i.level1_dungeon  # noqa: F401
    import zelda_i.level2_dungeon  # noqa: F401
    import zelda_i.level3_dungeon  # noqa: F401
    import zelda_i.level5_dungeon  # noqa: F401
    import zelda_i.level6_dungeon  # noqa: F401


def spec_for_room(
    room_id: int, *, level: int | None = None
) -> DungeonRoomSpec:
    """Look up a registered room spec.

    Prefer ``level=`` when room IDs could collide across dungeons. Without
    ``level``, uses the room_id-only table (unique rooms only).
    """
    ensure_default_specs()
    room_id = int(room_id)
    if level is not None:
        key = (int(level), room_id)
        if key not in _ROOM_SPECS_BY_LEVEL:
            known = ", ".join(
                f"L{lvl}:0x{rid:02X}"
                for lvl, rid in sorted(_ROOM_SPECS_BY_LEVEL)
            )
            raise KeyError(
                f"no dungeon room spec for level={level} 0x{room_id:02X}; "
                f"known: {known}"
            )
        return _ROOM_SPECS_BY_LEVEL[key]
    if room_id not in ROOM_SPECS:
        known = ", ".join(f"0x{room:02X}" for room in sorted(ROOM_SPECS))
        raise KeyError(
            f"no dungeon room spec for 0x{room_id:02X}; known: {known}"
        )
    return ROOM_SPECS[room_id]


def dungeon_room_cleared(ram: np.ndarray, spec: DungeonRoomSpec) -> bool:
    """Stop predicate for a room whose enemies and clear counter are known."""
    snap = read_snapshot(ram)
    return (
        snap.level == spec.level
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


def inventory_reward_success(
    ram: np.ndarray,
    spec: DungeonRoomSpec,
    *,
    min_value: int | None = None,
) -> bool:
    """Stop predicate for FIXED_INVENTORY rooms.

    Requires level + room_id + PLAY_MODE + no live enemies + inventory field
    meets target. If ``min_value`` is set: field >= min_value. Else: field > 0
    (keys-style). Compass-style bitfields should use a thin wrapper with a
    bit-mask check instead of ``min_value``.
    """
    snap = read_snapshot(ram)
    if (
        snap.level != spec.level
        or snap.screen != spec.room_id
        or snap.mode != PLAY_MODE
        or spec.live_enemies(snap)
    ):
        return False
    field_name = spec.reward.inventory_field
    if not field_name:
        return False
    value = int(getattr(snap, field_name))
    if min_value is not None:
        return value >= min_value
    return value > 0


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
        """Walk patrol waypoints without pulsing A (sword only on engage hit)."""
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
        # Walk only: continuous A on patrol looked spasmodic and wasted frames.
        return FrameAction(nes_action(direction), "combat_patrol")

    def _engage(self, snap: ZeldaSnapshot, target: ZeldaObject) -> FrameAction:
        """Chase target; slash only when sword hitbox can hit or contact-close."""
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
        if should_swing_at(
            snap.link_x,
            snap.link_y,
            direction,
            (target,),
        ):
            return self._swing(
                direction,
                "combat_engage",
                period=tuning.engage_attack_period,
                hold=tuning.engage_attack_hold,
            )
        # Approach without slashing until in blade range.
        return FrameAction(nes_action(direction), "combat_engage")

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

        if snap.level != self.spec.level:
            return FrameAction(nes_idle_action(), f"wait_level_{self.spec.level}")

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


# ---------------------------------------------------------------------------
# Lazy re-exports (PEP 562) so ``from zelda_i.dungeon import ROOM_6D_SPEC``
# and L2 constants/preds keep working without circular imports at load time.
# ---------------------------------------------------------------------------

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Level 1 room specs
    "ROOM_23_SPEC": ("zelda_i.level1_dungeon", "ROOM_23_SPEC"),
    "ROOM_33_SPEC": ("zelda_i.level1_dungeon", "ROOM_33_SPEC"),
    "ROOM_35_SPEC": ("zelda_i.level1_dungeon", "ROOM_35_SPEC"),
    "ROOM_42_SPEC": ("zelda_i.level1_dungeon", "ROOM_42_SPEC"),
    "ROOM_43_SPEC": ("zelda_i.level1_dungeon", "ROOM_43_SPEC"),
    "ROOM_44_SPEC": ("zelda_i.level1_dungeon", "ROOM_44_SPEC"),
    "ROOM_45_SPEC": ("zelda_i.level1_dungeon", "ROOM_45_SPEC"),
    "ROOM_52_SPEC": ("zelda_i.level1_dungeon", "ROOM_52_SPEC"),
    "ROOM_53_SPEC": ("zelda_i.level1_dungeon", "ROOM_53_SPEC"),
    "ROOM_54_SPEC": ("zelda_i.level1_dungeon", "ROOM_54_SPEC"),
    # Level 2 constants + room specs + stop preds
    "LEVEL_2": ("zelda_i.level2_dungeon", "LEVEL_2"),
    "ROOM_L2_ENTRY": ("zelda_i.level2_dungeon", "ROOM_L2_ENTRY"),
    "ROOM_L2_ROPES": ("zelda_i.level2_dungeon", "ROOM_L2_ROPES"),
    "ROOM_L2_WEST_KEY": ("zelda_i.level2_dungeon", "ROOM_L2_WEST_KEY"),
    "ROOM_L2_EAST_KEY": ("zelda_i.level2_dungeon", "ROOM_L2_EAST_KEY"),
    "ROOM_L2_EAST_OF_ROPES": ("zelda_i.level2_dungeon", "ROOM_L2_EAST_OF_ROPES"),
    "ROOM_L2_COMPASS": ("zelda_i.level2_dungeon", "ROOM_L2_COMPASS"),
    "ROOM_L2_BOMB_N": ("zelda_i.level2_dungeon", "ROOM_L2_BOMB_N"),
    "ROOM_L2_GORIYA_WEST": ("zelda_i.level2_dungeon", "ROOM_L2_GORIYA_WEST"),
    "ROOM_L2_ROPES_NORTH": ("zelda_i.level2_dungeon", "ROOM_L2_ROPES_NORTH"),
    "ROOM_L2_BOOM_CANDIDATE": ("zelda_i.level2_dungeon", "ROOM_L2_BOOM_CANDIDATE"),
    "ROOM_L2_NORTH_OF_4E": ("zelda_i.level2_dungeon", "ROOM_L2_NORTH_OF_4E"),
    "ROOM_6D_LEFT_DOOR_BIT": ("zelda_i.level2_dungeon", "ROOM_6D_LEFT_DOOR_BIT"),
    "ROOM_7D_SPEC": ("zelda_i.level2_dungeon", "ROOM_7D_SPEC"),
    "ROOM_6D_SPEC": ("zelda_i.level2_dungeon", "ROOM_6D_SPEC"),
    "ROOM_6C_SPEC": ("zelda_i.level2_dungeon", "ROOM_6C_SPEC"),
    "ROOM_7E_SPEC": ("zelda_i.level2_dungeon", "ROOM_7E_SPEC"),
    "ROOM_6E_SPEC": ("zelda_i.level2_dungeon", "ROOM_6E_SPEC"),
    "ROOM_6F_SPEC": ("zelda_i.level2_dungeon", "ROOM_6F_SPEC"),
    "ROOM_5E_SPEC": ("zelda_i.level2_dungeon", "ROOM_5E_SPEC"),
    "ROOM_4E_SPEC": ("zelda_i.level2_dungeon", "ROOM_4E_SPEC"),
    "ROOM_4F_SPEC": ("zelda_i.level2_dungeon", "ROOM_4F_SPEC"),
    "level2_room_6d_cleared": ("zelda_i.level2_dungeon", "level2_room_6d_cleared"),
    "level2_room_6c_key_success": (
        "zelda_i.level2_dungeon",
        "level2_room_6c_key_success",
    ),
    "level2_room_7e_key_success": (
        "zelda_i.level2_dungeon",
        "level2_room_7e_key_success",
    ),
    "level2_room_6e_cleared": ("zelda_i.level2_dungeon", "level2_room_6e_cleared"),
    "level2_room_6f_compass_success": (
        "zelda_i.level2_dungeon",
        "level2_room_6f_compass_success",
    ),
    "level2_room_5f_ready": ("zelda_i.level2_dungeon", "level2_room_5f_ready"),
    "level2_room_5e_cleared": ("zelda_i.level2_dungeon", "level2_room_5e_cleared"),
    "level2_room_4e_key_success": (
        "zelda_i.level2_dungeon",
        "level2_room_4e_key_success",
    ),
    "level2_room_4f_ready": ("zelda_i.level2_dungeon", "level2_room_4f_ready"),
    "level2_room_4f_magic_boomerang_success": (
        "zelda_i.level2_dungeon",
        "level2_room_4f_magic_boomerang_success",
    ),
    "BOMB_N_STAND": ("zelda_i.level2_dungeon", "BOMB_N_STAND"),
    "BOOM_BOMB_N_STAND": ("zelda_i.level2_dungeon", "BOOM_BOMB_N_STAND"),
    "B_ITEM_BOMB": ("zelda_i.level2_dungeon", "B_ITEM_BOMB"),
    "BombNorthPhase": ("zelda_i.level2_dungeon", "BombNorthPhase"),
    "BoomBombNorthPhase": ("zelda_i.level2_dungeon", "BoomBombNorthPhase"),
    "Level2BombNorthController": (
        "zelda_i.level2_dungeon",
        "Level2BombNorthController",
    ),
    "Level2BoomBombNorthController": (
        "zelda_i.level2_dungeon",
        "Level2BoomBombNorthController",
    ),
    "make_bomb_north_controller": (
        "zelda_i.level2_dungeon",
        "make_bomb_north_controller",
    ),
    "make_boom_bomb_north_controller": (
        "zelda_i.level2_dungeon",
        "make_boom_bomb_north_controller",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(mod_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
