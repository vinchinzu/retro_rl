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
from zelda_i import dungeon_ids as _ids
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot
from zelda_i.walk_physics import OccupancyWalker

# Settle frames after last kill for CLEAR_ONLY stop (was level1.CLEAR_SETTLE_ALL_DEAD).
CLEAR_SETTLE_ALL_DEAD = 20
# Inland box for CombatTuning.avoid_walls (door/wall tiles grab).
_AVOID_WALL_X = (56, 200)
_AVOID_WALL_Y = (109, 173)

# Enemy type IDs come from dungeon_ids; names below are the engine re-exports.
AQUAMENTUS_OBJECT_TYPE = _ids.AQUAMENTUS_OBJECT_TYPE
FIREBALL_OBJECT_TYPE = _ids.FIREBALL_OBJECT_TYPE
GEL_OBJECT_TYPE = _ids.GEL_OBJECT_TYPE
BLUE_GORIYA_OBJECT_TYPE = _ids.GORIYA_BLUE_OBJECT_TYPE
GORIYA_OBJECT_TYPE = _ids.GORIYA_OBJECT_TYPE
KEESE_OBJECT_TYPE = _ids.KEESE_OBJECT_TYPE
MOLDORM_OBJECT_TYPE = _ids.MOLDORM_OBJECT_TYPE
ROPE_OBJECT_TYPE = _ids.ROPE_OBJECT_TYPE
WALLMASTER_OBJECT_TYPE = _ids.WALLMASTER_OBJECT_TYPE


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
    # LEFT/RIGHT doors sit at y≈141. x-first along y=109 walks the north
    # statue band (live 0x6c entry sat at 0x6d (48, 109) for 8000f).
    y_first: bool = False

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
    # When >0 and enemy closer than this (manhattan), step away one beat
    # before swinging (heart-safe Clean; rr-gjey L4 residual).
    contact_backstep: int = 0
    # Walk inland before engage/patrol. Wallmasters grab on door/wall tiles.
    avoid_walls: bool = False
    # Forced steps in the entry direction after the room is playable, so
    # Link is off the door before Wallmasters finish spawning.
    inland_dash: int = 0
    # Prefer patrol vertices on the same side of a mid-room gap (0x23 water).
    split_y: int | None = None
    # Predicted 1px walks; a miss blocks the cell ahead and BFS replans
    # around water (0x23 plus). No path → stand / next patrol vertex.
    occupancy_patrol: bool = False

    def __post_init__(self) -> None:
        if not self.patrol:
            raise ValueError("combat patrol must contain at least one waypoint")
        for period, hold in (
            (self.engage_attack_period, self.engage_attack_hold),
            (self.patrol_attack_period, self.patrol_attack_hold),
        ):
            if period <= 0 or not 0 <= hold <= period:
                raise ValueError("attack hold must be within a positive period")
        if self.contact_backstep < 0:
            raise ValueError("contact_backstep must be >= 0")


@dataclass(frozen=True)
class RewardSpec:
    kind: RewardKind = RewardKind.CLEAR_ONLY
    inventory_field: str | None = None
    target: tuple[int, int] | None = None
    waypoints: tuple[tuple[int, int], ...] = ()
    settle_all_dead: int = CLEAR_SETTLE_ALL_DEAD
    y_first: bool = True
    # Wallmaster key is on the floor from entry; do not wait for all-dead.
    reward_while_live: bool = False

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
    # Enemy types counted by presence even under TYPE_AND_HP (e.g. Vire split
    # 0x1c has HP=0 while alive; slots 11–12 also hold live combatants).
    type_only_enemy_types: tuple[int, ...] = ()
    # Inclusive object-slot range (Zelda uses 1–12 for room combatants).
    object_slot_max: int = 12

    def live_enemies(self, snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
        slot_max = max(1, int(self.object_slot_max))
        enemies = tuple(
            obj
            for obj in snap.objects
            if 1 <= obj.slot <= slot_max and obj.type_id in self.enemy_types
        )
        if self.alive_rule is AliveRule.TYPE_AND_HP:
            type_only = frozenset(self.type_only_enemy_types)
            return tuple(
                obj
                for obj in enemies
                if obj.hp > 0 or obj.type_id in type_only
            )
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
    import zelda_i.level4_dungeon  # noqa: F401
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
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    _stuck_frames: int = 0
    _stuck_xy: tuple[int, int] | None = None
    _collect_skips: int = 0

    def _set_phase(self, phase: DungeonPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            self._stuck_frames = 0
            self._stuck_xy = None
            self._collect_skips = 0
            if phase is DungeonPhase.FIGHT:
                self.walker = OccupancyWalker()
            if note:
                self.notes.append(note)

    def _snap_patrol_nearest(self, snap: ZeldaSnapshot) -> None:
        patrol = self.spec.combat.patrol
        x, y = int(snap.link_x), int(snap.link_y)
        idxs: list[int] | range = range(len(patrol))
        split = self.spec.combat.split_y
        if split is not None:
            south = y >= split
            same = [
                i for i, (_, py) in enumerate(patrol) if (py >= split) == south
            ]
            if same:
                idxs = same
        self.patrol_index = min(
            idxs,
            key=lambda i: abs(patrol[i][0] - x) + abs(patrol[i][1] - y),
        )

    def _update_stuck(self, snap: ZeldaSnapshot) -> None:
        xy = (int(snap.link_x), int(snap.link_y))
        if self._stuck_xy == xy:
            self._stuck_frames += 1
        else:
            self._stuck_xy = xy
            self._stuck_frames = 0

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
        y_first = route.y_first and abs(dy) > 2
        if not y_first and abs(dx) > 2:
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
        if tuning.occupancy_patrol:
            xy = (int(snap.link_x), int(snap.link_y))
            n = len(tuning.patrol)
            for _ in range(n):
                direction = self._occupancy_dir(xy, (tx, ty))
                if direction is not None:
                    return FrameAction(nes_action(direction), "combat_patrol")
                self.patrol_index = (self.patrol_index + 1) % n
                tx, ty = tuning.patrol[self.patrol_index]
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "combat_wait")
        if abs(dx) > tuning.tolerance and abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > tuning.tolerance:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "UP"
        # Walk only: continuous A on patrol looked spasmodic and wasted frames.
        return FrameAction(nes_action(direction), "combat_patrol")

    def _wall_step(self, x: int, y: int, direction: str) -> tuple[int, int]:
        if direction == "LEFT":
            return x - 1, y
        if direction == "RIGHT":
            return x + 1, y
        if direction == "UP":
            return x, y - 1
        return x, y + 1

    def _on_avoid_wall(self, x: int, y: int) -> bool:
        lo_x, hi_x = _AVOID_WALL_X
        lo_y, hi_y = _AVOID_WALL_Y
        return x < lo_x or x > hi_x or y < lo_y or y > hi_y

    def _occupancy_dir(
        self, xy: tuple[int, int], dest: tuple[int, int]
    ) -> str | None:
        dest_i = (int(dest[0]), int(dest[1]))
        if self.walker.goal != dest_i:
            self.walker.goal = dest_i
            self.walker.path = None
        return self.walker.next_dir(xy, dest_i)

    def _engage(
        self,
        snap: ZeldaSnapshot,
        target: ZeldaObject,
        direction: str | None = None,
    ) -> FrameAction:
        """Chase target; slash only when sword hitbox can hit or contact-close."""
        if direction is None:
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
        nx, ny = self._wall_step(int(snap.link_x), int(snap.link_y), direction)
        hold_inland = tuning.avoid_walls and self._on_avoid_wall(nx, ny)
        if hold_inland or should_swing_at(
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

    def _off_wall_step(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Step toward the playable interior when avoid_walls is set."""
        if not self.spec.combat.avoid_walls:
            return None
        x, y = int(snap.link_x), int(snap.link_y)
        tuning = self.spec.combat
        if x < _AVOID_WALL_X[0]:
            # Tunnel x<24 only accepts RIGHT. At the mouth (x≈32) the
            # door row y≈141 blocks eastbound movement — step off it first.
            if x >= 24 and 133 <= y <= 149:
                direction = "DOWN"
            else:
                direction = "RIGHT"
        elif x > _AVOID_WALL_X[1]:
            direction = "LEFT"
        elif y < _AVOID_WALL_Y[0]:
            direction = "DOWN"
        elif y > _AVOID_WALL_Y[1]:
            direction = "UP"
        else:
            return None
        if tuning.occupancy_patrol:
            self.walker.last_dir = direction
        return self._swing(
            direction,
            "leave_wall",
            period=tuning.engage_attack_period,
            hold=tuning.engage_attack_hold,
        )

    def _combat(self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]) -> FrameAction:
        self.combat_frames += 1
        occupancy = self.spec.combat.occupancy_patrol
        # Cleared (or not yet spawned): stand. Do not patrol-wiggle while waiting.
        if not live:
            if occupancy:
                self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "combat_wait")
        self._update_stuck(snap)
        if self.combat_frames == 1:
            self._snap_patrol_nearest(snap)
        if not occupancy and self._stuck_frames >= 24:
            # Blocked mid-fight: hop from the nearest vertex, not a stale
            # index that greedy-walks through water (0x23 north pocket).
            n = len(self.spec.combat.patrol)
            self._snap_patrol_nearest(snap)
            self.patrol_index = (self.patrol_index + 1) % n
            self._stuck_frames = 0
        off_wall = self._off_wall_step(snap)
        if off_wall is not None:
            return off_wall
        dash = self.spec.combat.inland_dash
        if dash > 0 and self.combat_frames <= dash:
            if occupancy:
                self.walker.last_dir = self.spec.entry.direction
            return self._swing(
                self.spec.entry.direction,
                "inland_dash",
                period=self.spec.combat.engage_attack_period,
                hold=self.spec.combat.engage_attack_hold,
            )
        nearest = min(
            live,
            key=lambda obj: abs(obj.x - snap.link_x)
            + abs(obj.y - snap.link_y),
        )
        distance = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
        back = self.spec.combat.contact_backstep
        # Intermittent backstep (2/6 frames) so we still land sword hits
        # while peeling contact damage (rr-gjey). Always-backstep starves kill.
        if (
            back > 0
            and distance < back
            and (self.combat_frames % 6) < 2
        ):
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) >= abs(dy):
                away = "LEFT" if dx > 0 else "RIGHT"
            else:
                away = "UP" if dy > 0 else "DOWN"
            if occupancy:
                self.walker.last_dir = away
            return FrameAction(nes_action(away), "combat_backstep")
        if occupancy:
            xy = (int(snap.link_x), int(snap.link_y))
            self.walker.observe(xy)
            direction = self._occupancy_dir(xy, (nearest.x, nearest.y))
            if direction is None and distance >= self.spec.combat.engage_distance:
                return self._patrol(snap)
            if distance < self.spec.combat.engage_distance:
                return self._engage(snap, nearest, direction=direction)
            return FrameAction(nes_action(direction), "combat_patrol")
        if distance < self.spec.combat.engage_distance:
            return self._engage(snap, nearest)
        return self._patrol(snap)

    def _collect_reward(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.spec.reward.waypoints:
            n = len(self.spec.reward.waypoints)
            # One hunt lap, then stand. Looping the grid was thousands of
            # LEFT/RIGHT/DOWN frames in place on blocked tiles.
            if self.waypoint_index >= n:
                self.waypoint_index = 0
            self._update_stuck(snap)
            xy = (int(snap.link_x), int(snap.link_y))
            tx, ty = self.spec.reward.waypoints[self.waypoint_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y
            reached = abs(dx) <= 2 and abs(dy) <= 2
            stuck = self._stuck_frames >= 24 and not reached
            if reached or stuck:
                if stuck:
                    self.notes.append(
                        f"collect_skip_{self.waypoint_index}_{xy[0]}_{xy[1]}"
                    )
                    self._collect_skips += 1
                self.waypoint_index = (self.waypoint_index + 1) % n
                self._stuck_frames = 0
                if self._collect_skips >= n:
                    return FrameAction(nes_idle_action(), "collect_wait")
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
            # 5px idled forever 5px off the 0x33 key (live 101,173 vs 96,173).
            if abs(delta) > 2:
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
            and (not live or self.spec.reward.reward_while_live)
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
                return FrameAction(
                    nes_action(self.spec.entry.direction),
                    "settle_target_room",
                )

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

        if (
            self.phase in (DungeonPhase.FIGHT, DungeonPhase.COLLECT_REWARD)
            and snap.screen != self.spec.room_id
        ):
            self._set_phase(DungeonPhase.FAILED, "left_target_room")
            return FrameAction(nes_idle_action(), "left_target_room")

        if self.phase is DungeonPhase.FIGHT:
            if (
                snap.screen == self.spec.room_id
                and not live
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
                "occupancy_patrol": self.spec.combat.occupancy_patrol,
                "occupancy_misses": self.walker.misses,
                "occupancy_blocked": len(self.walker.grid.blocked),
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
    "PostBoomBombNorthPhase": ("zelda_i.level2_bomb_path", "PostBoomBombNorthPhase"),
    "Level2PostBoomBombNorthController": (
        "zelda_i.level2_bomb_path",
        "Level2PostBoomBombNorthController",
    ),
    "level2_room_3f_ready": ("zelda_i.level2_dungeon", "level2_room_3f_ready"),
    "ROOM_3F_SPEC": ("zelda_i.level2_dungeon", "ROOM_3F_SPEC"),
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
    "BOMB_N_STAND": ("zelda_i.level2_bomb_path", "BOMB_N_STAND"),
    "BOOM_BOMB_N_STAND": ("zelda_i.level2_bomb_path", "BOOM_BOMB_N_STAND"),
    "B_ITEM_BOMB": ("zelda_i.dungeon_ops", "B_ITEM_BOMB"),
    "BombNorthPhase": ("zelda_i.level2_bomb_path", "BombNorthPhase"),
    "BoomBombNorthPhase": ("zelda_i.level2_bomb_path", "BoomBombNorthPhase"),
    # Canonical factories (make_*); class-named aliases still resolve via same module.
    "make_bomb_north_controller": (
        "zelda_i.level2_bomb_path",
        "make_bomb_north_controller",
    ),
    "make_boom_bomb_north_controller": (
        "zelda_i.level2_bomb_path",
        "make_boom_bomb_north_controller",
    ),
    "make_post_boom_bomb_north_controller": (
        "zelda_i.level2_bomb_path",
        "make_post_boom_bomb_north_controller",
    ),
    "make_bomb_north_1e_controller": (
        "zelda_i.level2_bomb_path",
        "make_bomb_north_1e_controller",
    ),
    "Level2BombNorthController": (
        "zelda_i.level2_bomb_path",
        "make_bomb_north_controller",
    ),
    "Level2BoomBombNorthController": (
        "zelda_i.level2_bomb_path",
        "make_boom_bomb_north_controller",
    ),
    "Level2PostBoomBombNorthController": (
        "zelda_i.level2_bomb_path",
        "make_post_boom_bomb_north_controller",
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
