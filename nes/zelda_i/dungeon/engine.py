"""Data-driven early-dungeon combat engine for Zelda I.

Room tables live in per-level modules (``level1.dungeon``, ``level2.dungeon``,
``level3.dungeon``, …). This module is the shared controller + registry API.

Keep this game-local until a second adventure game proves the API shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i import combat as _combat
from zelda_i.walk import live_env
from zelda_i.combat import CONTACT_CHEBYSHEV, chebyshev, manhattan, should_swing_at
from zelda_i.dungeon import ids as _ids
from zelda_i.dungeon.hop_controller import (
    inland_lattice_step,
    ladder_release,
    lattice_goto,
    lattice_goto_route,
)
from zelda_i.dungeon.behaviors import (
    blocked_by_projectile,
    fight_target,
    is_projectile,
)
from zelda_i.dungeon.ids import AliveRule
from zelda_i.dungeon.postmortem import DamageLog
# Re-exported so ``dungeon.engine`` stays the one import for a room lane.
from zelda_i.dungeon.route_entry import (  # noqa: F401
    ENTER_STALL_FRAMES,
    EntryRouteWalker,
    ROUTE_BOUNDS,
    ROUTE_STALL_FRAMES,
)
from zelda_i.dungeon.tilemap import (
    FLOOR_TILES,
    STAIR_TILES,
    blocked_link_cells,
    has_room_tile_map,
)
from zelda_i.beam import beam_aim, beam_ready
from zelda_i.dungeon.threat import MIN_DODGE_BODY, EvadeDecision, ReactiveEvader
from zelda_i.dungeon.tracking import ObjectTracker, TrackedObject
from zelda_i.ram import ADDR_LADDER, PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot
from zelda_i.walk.physics import (
    DEFAULT_BOUNDS,
    OccupancyGrid,
    OccupancyWalker,
    lattice_route,
    lattice_step,
    lattice_toward,
)

# Settle frames after last kill for CLEAR_ONLY stop (was level1.CLEAR_SETTLE_ALL_DEAD).
CLEAR_SETTLE_ALL_DEAD = 20
# Collect-phase idle reasons, and how long one may hold before the reward
# nudge closes the last pixels. See ``_collect_reward``.
_REWARD_IDLE_REASONS = frozenset(
    {"reward_wait", "collect_wait", "collect_skip_unreachable"}
)
REWARD_NUDGE_FRAMES = 24
# Frames of action tail kept on every controller for the failure report.
REASON_TAIL_FRAMES = 30
# Inland box for CombatTuning.avoid_walls (door/wall tiles grab).
_AVOID_WALL_X = (56, 200)
_AVOID_WALL_Y = (109, 173)
_SCOOP_RADIUS = 48
_SCOOP_REACH = 4
# Frames a cleared room may spend walking to its floor drops before leaving.
SWEEP_MAX_FRAMES = 240
_OCC_BODY_R = 8
# What Link may stand on *inside* a room he is clearing. Deliberately narrower
# than ``tilemap.LINK_WALKABLE_TILES``: that set also grades a bombed-open
# wall walkable, which is right for ``ROUTE_BOUNDS`` and the entry route,
# whose whole job is to walk out through one, and wrong here — a body parked
# by the hole would let the chase BFS path Link into it and scroll him out
# mid-clear. The L4/L5/L6 ``(16, 216, ...)`` boxes reach the west hole column
# at x=16..23 directly.
#
# It does *not* wall off an open doorway, and never did. An open mouth is
# plain ``FLOOR_TILES`` set into the wall ring (measured over 259 dungeon
# fixtures), so it is walkable under any set that contains floor. What bounds
# it is the fight box: ``DEFAULT_BOUNDS`` stops at y=205, one tile row into
# the south mouth and short of the void past it, so the BFS has nowhere
# further out to aim. Stairs stay walkable: they are route destinations the
# specs aim at, not an accidental exit.
_FIGHT_WALKABLE_TILES = FLOOR_TILES | STAIR_TILES
# Frames to hold a proven-unreachable scoop verdict before re-running the
# BFS. A live body (a slow Wallmaster) can clear the path within seconds;
# the tilemap itself never changes mid-room. Re-running the reverse-flood
# BFS every single frame while stuck is what turned a 27s ledger run into
# 2m48s (rr coordinator trace, L1 0x45) -- this bounds that cost without
# abandoning a goal that becomes reachable again.
_SCOOP_REPLAN_HOLD = 20
# Collect occupancy can walk a 3px y-loop around a statue and never
# trip the in-place stuck detector (L1 0x45 sat at (144,141) for 7666f).
# Skip the waypoint if manhattan to it has not dropped in this many frames.
_COLLECT_STALE = 48
# Lattice chase goals: every node this close to the nearest one.
LATTICE_GOAL_SLACK = 8
# Patrol frames without getting closer to the waypoint before the lattice.
PATROL_STALL_FRAMES = 24
# Combat frames with no engage before the patrol gives way to a lattice hunt.
PATROL_HUNT_FRAMES = 600
# Every live enemy still this long: a clock freeze (or a parked body) that
# the engage distance never reaches. Strike each from behind its facing --
# a Darknut's shield is its front (L8 0x1F, power-on gathered spine: three
# frozen Darknuts outlasted the 16000f clear while Link lapped the patrol).
STATIC_ENEMY_FRAMES = 120
OFF_WALL_SLACK = 3
STRIKE_SLASH_MIN = 10
STILL_NO_PROGRESS_FRAMES = 30
STILL_BACKOFF_FRAMES = 300
STRIKE_TURN_MIN = 16
STRIKE_TURN_MAX = 20
_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
_BEHIND = {0x08: (0, 16, "UP"), 0x04: (0, -16, "DOWN"), 0x01: (-16, 0, "RIGHT"), 0x02: (16, 0, "LEFT")}
# ``_boxed``: frames within this many px of one spot before a replan.
BOXED_PX = 8
BOXED_FRAMES = 24

# Enemy type IDs come from dungeon.ids; names below are the engine re-exports.
AQUAMENTUS_OBJECT_TYPE = _ids.AQUAMENTUS_OBJECT_TYPE
FIREBALL_OBJECT_TYPE = _ids.FIREBALL_OBJECT_TYPE
GEL_OBJECT_TYPE = _ids.GEL_OBJECT_TYPE
BLUE_GORIYA_OBJECT_TYPE = _ids.GORIYA_BLUE_OBJECT_TYPE
GORIYA_OBJECT_TYPE = _ids.GORIYA_OBJECT_TYPE
KEESE_OBJECT_TYPE = _ids.KEESE_OBJECT_TYPE
MOLDORM_OBJECT_TYPE = _ids.MOLDORM_OBJECT_TYPE
ROPE_OBJECT_TYPE = _ids.ROPE_OBJECT_TYPE
WALLMASTER_OBJECT_TYPE = _ids.WALLMASTER_OBJECT_TYPE


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
    waypoints: tuple[tuple[int, int], ...] | Any
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
    contact_backstep: int = 0  # manhattan peel before swing (rr-gjey)
    avoid_walls: bool = False  # inland first; Wallmasters grab door tiles
    # Playable box for ``avoid_walls`` as (x_lo, x_hi, y_lo, y_hi). Rooms
    # with a live south U-turn (L1 0x23) widen it; the default is the
    # common inland box.
    avoid_wall_bounds: tuple[int, int, int, int] = (56, 200, 109, 173)
    inland_dash: int = 0  # forced entry-dir steps after room is playable
    split_y: int | None = None  # same-side patrol vertices (0x23 water)
    occupancy_patrol: bool = False  # 1px predict; miss → block + BFS
    occupancy_bounds: tuple[int, int, int, int] | None = None
    occupancy_blocked: tuple[tuple[int, int], ...] = ()
    # Seed the walker from the live ``$6530`` tile map instead of (or as well
    # as) ``occupancy_blocked``. Measured geometry; hand-written boxes drift.
    occupancy_from_tilemap: bool = False
    # Closing BODY/shot: honor threat.decide before chase. Off by default;
    # 0x42's block-push leftover moved when every room peeled.
    evade: bool = False
    # Fire the full-health sword shot at a body already in a lane.
    beam: bool = True

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
        if self.kind == RewardKind.FIXED_INVENTORY:
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
        if self.alive_rule == AliveRule.TYPE_AND_HP:
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
    import zelda_i.level1.dungeon  # noqa: F401
    import zelda_i.level2.dungeon  # noqa: F401
    import zelda_i.level3.dungeon  # noqa: F401
    import zelda_i.level4.dungeon  # noqa: F401
    import zelda_i.level5.dungeon  # noqa: F401
    import zelda_i.level6.dungeon  # noqa: F401


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
    """FIXED_INVENTORY stop: level+room+PLAY_MODE, no live, field >0 or >=min."""
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


def _live_in_contact(
    snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
) -> bool:
    return any(
        chebyshev(snap.link_x, snap.link_y, obj.x, obj.y) <= CONTACT_CHEBYSHEV
        or manhattan(snap.link_x, snap.link_y, obj.x, obj.y) <= CONTACT_CHEBYSHEV
        for obj in live
    )


def _occupancy_bodies(
    snap: ZeldaSnapshot, target: ZeldaObject | None
) -> set[tuple[int, int]]:
    """Temp blocks: other live bodies + projectiles. Target cell stays open."""
    skip = None if target is None else int(target.slot)
    cells: set[tuple[int, int]] = set()
    for obj in snap.objects:
        if obj.slot < 1 or (skip is not None and int(obj.slot) == skip):
            continue
        proj = is_projectile(obj)
        if not proj and (int(obj.hp) <= 0 or int(obj.type_id) in (0, 0xFF)):
            continue
        if not proj and int(obj.type_id) in getattr(
            _combat, "FLOOR_DROP_TYPES", ()
        ):
            continue
        ox, oy = int(obj.x), int(obj.y)
        radius = 6 if proj else _OCC_BODY_R
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:
                    cells.add((ox + dx, oy + dy))
    return cells


@dataclass
class GenericDungeonRoomController(EntryRouteWalker):
    """Route into and clear one room described by ``DungeonRoomSpec``."""

    spec: DungeonRoomSpec
    phase: DungeonPhase = DungeonPhase.ROUTE_ENTRY
    frames: int = 0
    phase_frames: int = 0
    combat_frames: int = 0
    swings: int = 0
    swings_authorized: int = 0
    engage_frames: int = 0
    patrol_frames: int = 0
    backstep_frames: int = 0
    waypoint_index: int = 0
    patrol_index: int = 0
    initial_inventory: int | None = None
    max_live_enemies: int = 0
    last_live_enemies: int = 0
    clear_signal_seen: bool = False
    success: bool = False
    notes: list[str] = field(default_factory=list)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    tracker: ObjectTracker = field(default_factory=ObjectTracker)
    damage: DamageLog = field(default_factory=DamageLog)
    evader: ReactiveEvader = field(default_factory=ReactiveEvader)
    tracked: tuple[TrackedObject, ...] = ()
    last_reason: str = ""
    _stuck_frames: int = 0
    _stuck_xy: tuple[int, int] | None = None
    _collect_skips: int = 0
    _collect_best_dist: int | None = None
    _collect_no_progress: int = 0
    _env: Any = field(default=None, init=False, repr=False)
    _lattice: frozenset[tuple[int, int]] | None = field(default=None, init=False, repr=False)
    _lattice_room: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _patrol_goal: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _patrol_best: int | None = field(default=None, init=False, repr=False)
    _patrol_since: int = field(default=0, init=False, repr=False)
    _patrol_lattice: bool = field(default=False, init=False, repr=False)
    _last_engage_frame: int = field(default=0, init=False, repr=False)
    _still: dict[int, tuple[tuple[int, int], int]] = field(default_factory=dict, init=False, repr=False)
    _still_xy: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _still_idle: int = field(default=0, init=False, repr=False)
    _still_off_until: int = field(default=0, init=False, repr=False)
    _box_anchor: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _box_frames: int = field(default=0, init=False, repr=False)
    _beam_pressed: bool = field(default=False, init=False, repr=False)
    beam_presses: int = field(default=0, init=False)
    # Cached scoop-heart unreachable verdict (goal cell -> hold-until frame).
    # See ``_scoop_heart_occupancy``.
    _scoop_unreachable_goal: tuple[int, int] | None = field(
        default=None, init=False, repr=False
    )
    _scoop_unreachable_until: int = field(default=0, init=False, repr=False)
    _ladder_still: int = field(default=0, init=False, repr=False)
    _ladder_cross: str | None = field(default=None, init=False, repr=False)
    _ladder_xy: tuple[int, int] | None = field(default=None, init=False, repr=False)
    # Post-clear floor sweep: frames spent and drops given up as unreachable.
    sweep_frames: int = field(default=0, init=False)
    _sweep_skip: set[tuple[int, int]] = field(default_factory=set, init=False, repr=False)
    # Records-only action histogram + tail; see ``_record_reason``.
    _reason_counts: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _reason_tail: list[str] = field(default_factory=list, init=False, repr=False)
    _reward_idle_frames: int = field(default=0, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        """Keep the env so the walker can be seeded from the live tile map.

        Re-seeds immediately when the fight is already running, so binding is
        order-independent. Without that, a caller that entered FIGHT before
        binding silently gets a walker with no geometry at all — strictly
        worse than the hand-written blocks this replaced, and silent.
        """
        self._env = env
        if self.phase is DungeonPhase.FIGHT:
            self.walker = self._make_walker()

    def _measured_blocks(
        self, bounds: tuple[int, int, int, int]
    ) -> frozenset[tuple[int, int]]:
        """Live ``$6530`` geometry for ``bounds``; empty when unavailable.

        Graded against ``_FIGHT_WALKABLE_TILES``, not the module default:
        door mouths are solid to a room the controller is still clearing.
        """
        if not self.spec.combat.occupancy_from_tilemap or self._env is None:
            return frozenset()
        ram = self._env.get_ram()
        if not has_room_tile_map(ram):
            return frozenset()
        return blocked_link_cells(ram, bounds, walkable=_FIGHT_WALKABLE_TILES)

    def _set_phase(self, phase: DungeonPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.waypoint_index = 0
            self._stuck_frames = 0
            self._stuck_xy = None
            self._collect_skips = 0
            self._collect_best_dist = None
            self._collect_no_progress = 0
            self._reset_entry_route()
            self._scoop_unreachable_goal = None
            self._scoop_unreachable_until = 0
            self._reward_idle_frames = 0
            if phase is DungeonPhase.FIGHT:
                self.walker = self._make_walker()
            if note:
                self.notes.append(note)

    def _make_walker(self) -> OccupancyWalker:
        tuning = self.spec.combat
        bounds = tuning.occupancy_bounds or DEFAULT_BOUNDS
        blocked = set(tuning.occupancy_blocked) | self._measured_blocks(bounds)
        if tuning.occupancy_bounds is None and not blocked:
            return OccupancyWalker()
        xmin, xmax, ymin, ymax = bounds
        return OccupancyWalker(
            grid=OccupancyGrid(
                blocked=blocked, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
            )
        )

    def _relax_leftover_bounds(self) -> None:
        """Fresh leftover grid: spec seed only. Combat misses boxed the north door."""
        tuning = self.spec.combat
        xmin, xmax, ymin, ymax = DEFAULT_BOUNDS
        if tuning.occupancy_bounds is not None:
            xmin, xmax, ymin, _fight_ymax = tuning.occupancy_bounds
            ymax = DEFAULT_BOUNDS[3]
        blocked = set(tuning.occupancy_blocked) | self._measured_blocks(
            (xmin, xmax, ymin, ymax)
        )
        self.walker = OccupancyWalker(
            grid=OccupancyGrid(
                blocked=blocked, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
            )
        )

    def __post_init__(self) -> None:
        self.walker = self._make_walker()
        bounds = self.spec.combat.occupancy_bounds
        if bounds is None and self.spec.combat.avoid_walls:
            bounds = self.spec.combat.avoid_wall_bounds
        self.evader.bounds = bounds

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
            self.swings += 1
            return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

    def _patrol(self, snap: ZeldaSnapshot) -> FrameAction:
        """Walk patrol waypoints without pulsing A (sword only on engage hit)."""
        self.patrol_frames += 1
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
                direction = self.walker.next_dir(xy, (tx, ty))
                if direction is not None:
                    return FrameAction(nes_action(direction), "combat_patrol")
                self.patrol_index = (self.patrol_index + 1) % n
                tx, ty = tuning.patrol[self.patrol_index]
            # Pocket: occupancy miss-blocked every corridor. Greedy toward
            # the maze loop instead of standing (live 0x23 (99,157) 2 Goriyas).
            direction = lattice_toward(snap.link_x, snap.link_y, (tx, ty), tol=tuning.tolerance)
            if direction is None:
                self.walker.last_dir = None
                return FrameAction(nes_idle_action(), "combat_wait")
            self.walker.last_dir = direction
            return FrameAction(nes_action(direction), "combat_patrol")
        if self._patrol_stalled(snap, (tx, ty)):
            route = self._lattice_route(snap, (tx, ty))
            if route == []:
                # As close as the walls allow (the waypoint sits in a block,
                # e.g. L2 0x6E p4 (128,141)): that is arrival.
                self.patrol_index = (self.patrol_index + 1) % len(tuning.patrol)
                self._patrol_goal = None
                return FrameAction(nes_idle_action(), "combat_patrol_arrived")
            if route:
                step = lattice_step(int(snap.link_x), int(snap.link_y), route[0])
                if step is not None:
                    return FrameAction(nes_action(step), "combat_patrol_lattice")
        direction = lattice_toward(snap.link_x, snap.link_y, (tx, ty), tol=tuning.tolerance)
        if direction is None:
            return FrameAction(nes_idle_action(), "combat_wait")
        return FrameAction(nes_action(direction), "combat_patrol")

    def _beam_shot(
        self, snap: ZeldaSnapshot, live: tuple[ZeldaObject, ...]
    ) -> FrameAction | None:
        """Fire the full-health sword shot at a body already in a lane.

        The blade only swings inside reach, so a body parked on a raised
        block (L3 0x6B Zols, gathered spine: 3500 frames of ``combat_engage``
        under a Zol 24 px up a block) was never hit, though the White Sword
        at full hearts throws a screen-long shot. It fires only when Link
        already faces the lane, and A is an edge.
        """
        if not self.spec.combat.beam or not beam_ready(snap):
            self._beam_pressed = False
            return None
        if self._beam_pressed:
            self._beam_pressed = False
            return FrameAction(nes_idle_action(), "beam_release")
        if snap.objects and int(snap.objects[0].slot) == 0 and int(snap.objects[0].state) != 0:
            return None
        try:
            held = _combat.facing_to_direction(int(snap.facing))
        except ValueError:
            held = None
        aim = beam_aim(int(snap.link_x), int(snap.link_y), live, prefer=held)
        if aim is None:
            return None
        face, _body = aim
        if held != face:
            # No turn frames: a turn is a held direction, which walks Link
            # off the lane, and the chase walks him back (L3 0x59, gathered
            # spine: 8597 beam_turn / 8536 hunt frames at (96,116..117)).
            # The chase faces bodies on its own; fire when it has.
            return None
        self._beam_pressed = True
        self.beam_presses += 1
        return FrameAction(nes_action(face, "A"), "beam_fire")

    def _boxed(self, snap: ZeldaSnapshot) -> bool:
        """Link has stayed within ``BOXED_PX`` of one spot for ``BOXED_FRAMES``.

        Latches until he gets clear, so the replan it triggers is not undone
        the frame he takes his first step. Catches a one-pixel wall bounce,
        which a same-``xy`` counter does not.
        """
        xy = (int(snap.link_x), int(snap.link_y))
        anchor = self._box_anchor
        if anchor is None or abs(xy[0] - anchor[0]) + abs(xy[1] - anchor[1]) > BOXED_PX:
            self._box_anchor = xy
            self._box_frames = 0
            return False
        self._box_frames += 1
        return self._box_frames >= BOXED_FRAMES

    def _patrol_stalled(self, snap: ZeldaSnapshot, goal: tuple[int, int]) -> bool:
        """No progress toward this patrol waypoint for ``PATROL_STALL_FRAMES``.

        Position-equality stuck checks miss a wall that bounces Link one
        pixel each frame: L2 0x6E (gathered spine, 2026-09-22) flipped
        (80,189)<->(81,189) for 7511 frames of ``combat_patrol``. Distance
        to the goal is what does not move.
        """
        dist = abs(goal[0] - int(snap.link_x)) + abs(goal[1] - int(snap.link_y))
        if self._patrol_goal != goal or self._patrol_best is None or dist < self._patrol_best:
            if self._patrol_goal != goal:
                self._patrol_lattice = False
            self._patrol_goal = goal
            self._patrol_best = dist
            self._patrol_since = 0
            return self._patrol_lattice
        self._patrol_since += 1
        if self._patrol_since >= PATROL_STALL_FRAMES:
            self._patrol_lattice = True
        return self._patrol_lattice

    def _wall_step(self, x: int, y: int, direction: str) -> tuple[int, int]:
        if direction == "LEFT":
            return x - 1, y
        if direction == "RIGHT":
            return x + 1, y
        if direction == "UP":
            return x, y - 1
        return x, y + 1

    def _on_avoid_wall(self, x: int, y: int) -> bool:
        lo_x, hi_x, lo_y, hi_y = self.spec.combat.avoid_wall_bounds
        return x < lo_x or x > hi_x or y < lo_y or y > hi_y

    def _engage(
        self,
        snap: ZeldaSnapshot,
        target: ZeldaObject,
        direction: str | None = None,
    ) -> FrameAction:
        """Chase target; slash only when sword hitbox can hit or contact-close."""
        self.engage_frames += 1
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
        authorized = should_swing_at(
            snap.link_x, snap.link_y, direction, (target,)
        )
        if authorized:
            self.swings_authorized += 1
        if hold_inland or authorized:
            return self._swing(
                direction,
                "combat_engage",
                period=tuning.engage_attack_period,
                hold=tuning.engage_attack_hold,
            )
        # Approach without slashing until in blade range.
        return FrameAction(nes_action(direction), "combat_engage")

    def _owns_ladder(self) -> bool:
        env = self._env if self._env is not None else live_env.current()
        return env is not None and int(env.get_ram()[ADDR_LADDER]) != 0

    def _all_in_wall_zone(self, live) -> bool:
        """Every live body sits outside the avoid-wall band.

        Then the band is where Link cannot win: L6 0x68's last Zol parked at
        (208,96) above y=109 and leave_wall pushed Link back off every
        approach for 12000f (continuous power-on run 8).
        """
        if not live or not self.spec.combat.avoid_walls:
            return False
        # Only bodies that have held still: a teleporting Wizzrobe or a body
        # crossing the band is not a stalemate (L6 0x29 north door).
        counts = [self._still.get(int(o.slot)) for o in live]
        if any(c is None or c[1] < STATIC_ENEMY_FRAMES for c in counts):
            return False
        lo_x, hi_x, lo_y, hi_y = self.spec.combat.avoid_wall_bounds
        return all(
            not (lo_x <= int(o.x) <= hi_x and lo_y <= int(o.y) <= hi_y) for o in live
        )

    def _off_wall_step(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Step toward the playable interior when avoid_walls is set."""
        if not self.spec.combat.avoid_walls:
            return None
        x, y = int(snap.link_x), int(snap.link_y)
        tuning = self.spec.combat
        lo_x, hi_x, lo_y, hi_y = tuning.avoid_wall_bounds
        # Hysteresis: a patrol waypoint on the band edge (L6 0x39 (160,109))
        # walks Link to 108, and a 1 px leave_wall swapped with the patrol
        # every frame for 12515f.
        m = OFF_WALL_SLACK
        if lo_x - m <= x <= hi_x + m and lo_y - m <= y <= hi_y + m:
            return None
        if x < lo_x:
            # Tunnel x<24 only accepts RIGHT. At the mouth (x≈32) the
            # door row y≈141 blocks eastbound movement — step off it first.
            if x >= 24 and 133 <= y <= 149:
                direction = "DOWN"
            else:
                direction = "RIGHT"
        elif x > hi_x:
            direction = "LEFT"
        elif y < lo_y:
            direction = "DOWN"
        elif y > hi_y:
            direction = "UP"
        else:
            return None
        step = inland_lattice_step(x, y, direction, (lo_x, hi_x), (lo_y, hi_y))
        if step is not None:
            direction = step
        if tuning.occupancy_patrol:
            self.walker.last_dir = direction
        return self._swing(
            direction,
            "leave_wall",
            period=tuning.engage_attack_period,
            hold=tuning.engage_attack_hold,
        )

    def _parry(
        self,
        snap: ZeldaSnapshot,
        live: tuple[ZeldaObject, ...],
        *,
        only_slot: int | None = None,
    ) -> FrameAction | None:
        """Answer a closing body with the sword when the blade already reaches it.

        ``ReactiveEvader`` only reasons about feet: it buys frames by stepping.
        The sword outranges a body's contact pad, so inside blade range the
        cheapest dodge is the kill — stepping away there concedes the hit
        *and* lengthens the fight. Measured on L1 ``0x23``: evade-first spent
        2005 frames patrolling and landed four swings.

        ``only_slot`` keeps this honest when it pre-empts the evader: the
        sword may answer *the* inbound threat, never a different body while a
        boomerang is already in the air. Without it Link turned into a Goriya
        and took a shot from behind (L1 ``0x44``, dead in 243 frames).

        Never presses a direction while swinging; walking into the pad is the
        contact the evader was called to avoid.
        """
        target = fight_target(snap.link_x, snap.link_y, live)
        if target is None or is_projectile(target):
            return None
        if only_slot is not None and int(target.slot) != int(only_slot):
            return None
        lx, ly = int(snap.link_x), int(snap.link_y)
        dx, dy = int(target.x) - lx, int(target.y) - ly
        if abs(dx) >= abs(dy):
            face = "RIGHT" if dx > 0 else "LEFT"
        else:
            face = "DOWN" if dy > 0 else "UP"
        if not _combat.in_sword_hitbox(lx, ly, face, target.x, target.y):
            return None
        if self.spec.combat.occupancy_patrol:
            self.walker.last_dir = None
        if int(snap.facing) != _combat.direction_to_facing(face):
            # Link has no turn-in-place, so acquiring the facing also walks a
            # pixel that way. That pixel is worth it: the swing that follows
            # removes the threat, where a step only postpones it. Measured on
            # the Clean L1 run — 0x23 went 3 hits to 1 and 0x33 stayed at 0.
            return FrameAction(nes_action(face), "combat_parry_face")
        tuning = self.spec.combat
        if self.combat_frames % tuning.engage_attack_period < tuning.engage_attack_hold:
            self.swings += 1
            self.swings_authorized += 1
            return FrameAction(nes_action("A"), "combat_parry")
        return FrameAction(nes_idle_action(), "combat_parry_recover")

    def _evade_step(
        self,
        snap: ZeldaSnapshot,
        decision: EvadeDecision,
    ) -> FrameAction:
        """Walk the threat button. Do not slash-walk: that re-enters the pad."""
        reason = f"combat_{decision.reason}"
        direction = decision.direction
        if direction is None:
            if self.spec.combat.occupancy_patrol:
                self.walker.last_dir = None
            return FrameAction(nes_idle_action(), reason)
        if self.spec.combat.occupancy_patrol:
            self.walker.last_dir = direction
        return FrameAction(nes_action(direction), reason)

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
        # With the tile map bound, ``_patrol`` replans on the lattice; the
        # skip-a-waypoint rule would reset that replan every 24 frames.
        if not occupancy and self._stuck_frames >= 24 and not self._lattice_nodes(snap):
            n = len(self.spec.combat.patrol)
            self._snap_patrol_nearest(snap)
            self.patrol_index = (self.patrol_index + 1) % n
            self._stuck_frames = 0
        xy = (int(snap.link_x), int(snap.link_y))
        bodies: set[tuple[int, int]] = set()
        if occupancy:
            # Grade now: backstep/dash/off-wall skip next_dir (rr-8t4.4).
            bodies = _occupancy_bodies(snap, None)
            bodies.discard(xy)
            self.walker.observe(xy, transient_occupants=bodies)
        # Reactive first. Every position rule below — the off-wall step, the
        # entry dash, the patrol — is blind to what is inbound, so running one
        # ahead of the evader silences it for that frame. That is how L1 0x23
        # idled at (64,157) while a Goriya walked down the column, and how
        # 0x45 walked `leave_wall` into a Wallmaster. The evader still yields
        # (returns None) whenever standing is safe, so the rules keep driving.
        if self.spec.combat.evade:
            decision = self.evader.decide(snap, self.tracked)
            if decision is not None:
                if decision.direction is not None:
                    parry = self._parry(snap, live, only_slot=decision.source_slot)
                    return parry or self._evade_step(snap, decision)
                if decision.shield:
                    # ``shield_hold`` is a real answer: Link is already facing
                    # a blockable shot, so the idle frame *is* the block.
                    return self._evade_step(snap, decision)
                # ``evade_no_gain`` / ``evade_boxed_in``: the evader has no
                # step that buys a frame. Hand the frame back to the
                # positional rules (they still dash, slash and make progress)
                # instead of idling — same contract as
                # ``overworld.path._threat_action``. Consuming the frame here
                # is what stops ROOM_45_SPEC's ``inland_dash`` from ever
                # firing: entry lands Link at x=16 inside the west door
                # mouth, where a Wallmaster is always inside ``trigger_ttc``
                # and no sidestep clears the tunnel, so the grab lands.
                pass
        # The entry dash *is* an off-wall move, and a stronger one: it knows
        # which way the room was entered. Running the generic off-wall rule
        # first would consume every frame of it (0x45 enters at x=16, which
        # the off-wall rule owns, so the dash never fired).
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
        off_wall = None if self._all_in_wall_zone(live) else self._off_wall_step(snap)
        if off_wall is not None:
            return off_wall
        shot = self._beam_shot(snap, live)
        if shot is not None:
            return shot
        target = fight_target(snap.link_x, snap.link_y, live)
        if target is None:
            # Parked Wallmasters (and other illegal slots) stay in ``live``.
            return self._patrol(snap)
        still = self._all_still(live)
        if still is not None and self.combat_frames >= self._still_off_until:
            strike = self._strike_from_behind(snap, still)
            if strike is not None:
                xy = (int(snap.link_x), int(snap.link_y))
                moving = strike.reason in ("still_slash", "still_face") or xy != self._still_xy
                self._still_idle = 0 if moving else self._still_idle + 1
                self._still_xy = xy
                if self._still_idle < STILL_NO_PROGRESS_FRAMES:
                    return strike
                # A body that holds still is not always frozen (a Zol's
                # pause, a Like Like); an approach going nowhere hands back
                # to the ordinary fight (L4 0x32 pinned 24378f at (115,101)).
                self._still_idle = 0
                self._still_off_until = self.combat_frames + STILL_BACKOFF_FRAMES
        distance = abs(target.x - snap.link_x) + abs(target.y - snap.link_y)
        if self.spec.combat.evade and distance < MIN_DODGE_BODY:
            parry = self._parry(snap, live)
            if parry is not None:
                return parry
            dx = target.x - snap.link_x
            dy = target.y - snap.link_y
            if abs(dx) >= abs(dy):
                away = "LEFT" if dx > 0 else "RIGHT"
            else:
                away = "UP" if dy > 0 else "DOWN"
            if occupancy:
                self.walker.last_dir = away
            return FrameAction(nes_action(away), "combat_evade_body")
        back = self.spec.combat.contact_backstep
        # 2/6 peel so we still slash; always-backstep starves kill (rr-gjey).
        if back > 0 and distance < back and (self.combat_frames % 6) < 2:
            dx = target.x - snap.link_x
            dy = target.y - snap.link_y
            if abs(dx) >= abs(dy):
                away = "LEFT" if dx > 0 else "RIGHT"
            else:
                away = "UP" if dy > 0 else "DOWN"
            chase = "RIGHT" if dx >= 0 else "LEFT"
            if abs(dy) > abs(dx):
                chase = "DOWN" if dy >= 0 else "UP"
            if should_swing_at(snap.link_x, snap.link_y, chase, (target,)):
                self.swings_authorized += 1
            if occupancy:
                self.walker.last_dir = away
            self.backstep_frames += 1
            return FrameAction(nes_action(away), "combat_backstep")
        if occupancy:
            # extra_blocked skips target cell; bodies still grade it.
            extra = _occupancy_bodies(snap, target)
            extra.discard(xy)
            extra.discard((int(target.x), int(target.y)))
            direction = self.walker.next_dir(
                xy,
                self._chase_goal(target),
                extra_blocked=extra,
                transient_occupants=bodies,
            )
            direction = self._lattice_chase(snap, target) or direction
            if self._boxed(snap):
                # The one-pixel walker's step is into a block (L3 0x6B
                # diagonal blocks, gathered spine: 11929 engage frames at
                # (112,117)). The lattice knows both feet.
                direction = self._lattice_dir(snap, (int(target.x), int(target.y))) or direction
            blocked = direction is not None and blocked_by_projectile(
                snap.link_x, snap.link_y, direction, snap.objects
            )
            if blocked:
                self.walker.last_dir = None
                return FrameAction(nes_idle_action(), "combat_wait")
            if direction is None and distance >= self.spec.combat.engage_distance:
                return self._patrol(snap)
            if distance < self.spec.combat.engage_distance:
                return self._engage(snap, target, direction=direction)
            if should_swing_at(snap.link_x, snap.link_y, direction, (target,)):
                self.swings_authorized += 1
            self.patrol_frames += 1
            return FrameAction(nes_action(direction), "combat_patrol")
        if distance < self.spec.combat.engage_distance:
            self._last_engage_frame = self.combat_frames
            if self._boxed(snap):
                # The greedy engage axis is into a wall: L2 0x1E (gathered
                # spine) held DOWN at (120,93) for 19894 frames against a
                # block, a Goriya parked 64 px below. Walk round instead.
                step = self._lattice_dir(snap, (int(target.x), int(target.y)))
                if step is not None:
                    return self._engage(snap, target, direction=step)
            return self._engage(snap, target)
        if self.combat_frames - self._last_engage_frame >= PATROL_HUNT_FRAMES:
            # The patrol loop is a trap for a body that parks off it: L2 0x6E
            # (gathered spine) left a rope standing at (208,149) while Link
            # lapped x 96..152 for 7000 frames. Go to it.
            step = self._lattice_dir(snap, (int(target.x), int(target.y)))
            if step is not None:
                return FrameAction(nes_action(step), "combat_hunt_lattice")
        return self._patrol(snap)

    def _all_still(self, live) -> tuple | None:
        """All live bodies, once *every* one has held still ``STATIC_ENEMY_FRAMES``."""
        seen: dict[int, tuple[tuple[int, int], int]] = {}
        for obj in live:
            xy = (int(obj.x), int(obj.y))
            prev = self._still.get(int(obj.slot))
            seen[int(obj.slot)] = (xy, prev[1] + 1 if prev and prev[0] == xy else 0)
        self._still = seen
        if not seen or min(n for _, n in seen.values()) < STATIC_ENEMY_FRAMES:
            return None
        return tuple(live)

    def _strike_from_behind(self, snap: ZeldaSnapshot, still: tuple) -> FrameAction | None:
        """Walk to exactly 16 px behind a still body, turn, slash.

        The stand is off the 8 px lattice in general, but a lattice row
        (column) is free along its own axis: route to a node on the strike
        line, then slide along it. Closer than 16 px is body contact (L8 0x1F
        stood at 10 px and took hits instead of turning). Nearest body first;
        one with no reachable strike line is skipped.
        """
        nodes = self._lattice_nodes(snap)
        if not nodes:
            return None
        lx, ly = int(snap.link_x), int(snap.link_y)
        for target in sorted(still, key=lambda o: manhattan(lx, ly, o.x, o.y)):
            dx, dy, face = _BEHIND.get(int(target.facing), (0, 16, "UP"))
            sx, sy = int(target.x) + dx, int(target.y) + dy
            horizontal = face in ("LEFT", "RIGHT")
            if horizontal:
                line = [n for n in nodes if abs(n[1] - sy) <= 3 and abs(n[0] - sx) <= 8]
            else:
                line = [n for n in nodes if abs(n[0] - sx) <= 3 and abs(n[1] - sy) <= 8]
            if not line:
                continue
            node = min(line, key=lambda n: abs(n[0] - sx) + abs(n[1] - sy))
            on_line = abs(ly - node[1]) <= 1 if horizontal else abs(lx - node[0]) <= 1
            along = abs(lx - sx) <= 8 if horizontal else abs(ly - sy) <= 8
            if on_line and along:
                # Distance to the body along the strike axis, with hysteresis:
                # the turn press walks Link 1-2 px, so the slash window is
                # wider than the turn window (a single stand swapped align and
                # turn every frame at (96,101)/(96,102)).
                dist = abs(lx - int(target.x)) if horizontal else abs(ly - int(target.y))
                faced = int(snap.facing) == _combat.direction_to_facing(face)
                if faced and STRIKE_SLASH_MIN <= dist <= STRIKE_TURN_MAX + 2:
                    self.swings += 1
                    self.swings_authorized += 1
                    hold = (self.frames % 8) < 4
                    return FrameAction(nes_action("A") if hold else nes_idle_action(), "still_slash")
                if dist < STRIKE_TURN_MIN:
                    return FrameAction(nes_action(_OPPOSITE[face]), "still_back")
                if dist > STRIKE_TURN_MAX:
                    return FrameAction(nes_action(face), "still_close")
                return FrameAction(nes_action(face), "still_face")
            route = lattice_route(nodes, (lx, ly), {node})
            if not route:
                continue
            return FrameAction(nes_action(lattice_step(lx, ly, route[0])), "still_approach")
        return None

    def _lattice_nodes(self, snap: ZeldaSnapshot) -> frozenset[tuple[int, int]] | None:
        """ROM-collision lattice for this room, cached per screen."""
        env = self._env if self._env is not None else live_env.current()
        if env is None:
            return None
        room = (int(snap.level), int(snap.screen))
        if self._lattice_room == room and self._lattice is not None:
            return self._lattice
        ram = env.get_ram()
        if not has_room_tile_map(ram):
            return None
        from zelda_i.dungeon.tilemap import ow_walkable_nodes

        xmin, xmax, ymin, ymax = self.spec.combat.occupancy_bounds or DEFAULT_BOUNDS
        self._lattice = frozenset(
            (x, y)
            for x, y in ow_walkable_nodes(ram, overworld=False)
            if xmin <= x <= xmax and ymin <= y <= ymax
        )
        self._lattice_room = room
        return self._lattice

    def _lattice_chase(self, snap: ZeldaSnapshot, target: ZeldaObject) -> str | None:
        """First step of the ROM-collision route to the target's nearest node.

        The occupancy walker samples one pixel under Link, so it reads a
        block's edge as floor. L1 0x23 (rung 2, 2026-09-22): a red Goriya
        walked the inner ring's bottom row while Link pushed DOWN into the
        water from the top row; the only vertical passages are x=64 and
        x=176, and the walker never went round. The lattice tests both feet
        tiles the way ``GetCollidingTileMoving`` does. Only rooms that
        already measure walls (``occupancy_from_tilemap``) take it.
        """
        if not self.spec.combat.occupancy_from_tilemap:
            return None
        return self._lattice_dir(snap, (int(target.x), int(target.y)))

    def _lattice_dir(self, snap: ZeldaSnapshot, goal: tuple[int, int]) -> str | None:
        """First lattice step toward the nodes nearest ``goal``; ``None`` if none."""
        route = self._lattice_route(snap, goal)
        if not route:
            return None
        return lattice_step(int(snap.link_x), int(snap.link_y), route[0])

    def _lattice_route(
        self, snap: ZeldaSnapshot, goal: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        """Lattice corners to the nodes nearest ``goal``; ``[]`` on one, ``None`` if none."""
        nodes = self._lattice_nodes(snap)
        if not nodes:
            return None
        tx, ty = int(goal[0]), int(goal[1])
        near = sorted(nodes, key=lambda n: abs(n[0] - tx) + abs(n[1] - ty))
        if not near:
            return None
        best = abs(near[0][0] - tx) + abs(near[0][1] - ty)
        goals = {n for n in near[:8] if abs(n[0] - tx) + abs(n[1] - ty) <= best + LATTICE_GOAL_SLACK}
        return lattice_route(nodes, (int(snap.link_x), int(snap.link_y)), goals)

    def _chase_goal(self, target) -> tuple[int, int]:
        """The enemy's cell, or the nearest open cell to it.

        An enemy standing on geometry is normal -- a goriya walks the block
        rows of L1 ``0x23`` constantly -- and its cell is then not passable,
        so the chase BFS correctly reports *no path* to it. Handing that
        ``None`` on to ``_engage`` is what killed Clean M5: at
        ``distance < engage_distance`` Link parried in place while the body
        closed, and took all three hits standing (``f1453 body 0x06 (goriya)
        from N ... action=combat_parry``). Chasing the nearest cell that
        exists reaches sword range instead.

        Deliberately local to the chase. ``OccupancyWalker`` has the same
        rule behind ``retarget_blocked_goal``, and it stays off here: turning
        it on for the *walker* also retargets route and collect waypoints,
        which shifts arrival frames, and the L1 chain is frame-perfect.
        """
        goal = (int(target.x), int(target.y))
        if self.walker.grid.passable(*goal):
            return goal
        open_goal = self.walker.grid.nearest_open(*goal)
        return goal if open_goal is None else open_goal

    def _collect_reward(self, snap: ZeldaSnapshot) -> FrameAction:
        """Collect policy, plus an escape from standing next to the pickup.

        ``_collect_policy`` stops walking once Link is within the 2px
        waypoint tolerance of the reward target, and then idles waiting for
        the inventory to change. On the tile that is right; 2px off it is a
        stall with no way out — L2 ``0x7e`` spent 6752 of its 8000 frames on
        ``reward_wait`` at ``(138,141)`` with the key at ``(136,141)``, which
        is what ``Level6EastKeyController._go_key`` already wiggles for
        locally. Close the last pixels instead of idling.
        """
        action = self._collect_policy(snap)
        if action.reason not in _REWARD_IDLE_REASONS:
            self._reward_idle_frames = 0
            return action
        self._reward_idle_frames += 1
        if self._reward_idle_frames < REWARD_NUDGE_FRAMES:
            return action
        return self._reward_nudge(snap, action)

    def _reward_nudge(
        self, snap: ZeldaSnapshot, idle: FrameAction
    ) -> FrameAction:
        """Walk the last 1-2px onto the reward tile, tolerance ignored."""
        target = self.spec.reward.target
        if target is None:
            waypoints = self.spec.reward.waypoints
            if not waypoints:
                return idle
            target = waypoints[self.waypoint_index % len(waypoints)]
        dx = int(target[0]) - int(snap.link_x)
        dy = int(target[1]) - int(snap.link_y)
        if dx:
            return FrameAction(
                nes_action("RIGHT" if dx > 0 else "LEFT"), "reward_nudge"
            )
        if dy:
            return FrameAction(
                nes_action("DOWN" if dy > 0 else "UP"), "reward_nudge"
            )
        # Exactly on the tile and still nothing: rock one pixel so the
        # pickup check runs again on a fresh position.
        direction = "RIGHT" if (self.frames % 8) < 4 else "LEFT"
        return FrameAction(nes_action(direction), "reward_nudge_wiggle")

    def _collect_policy(self, snap: ZeldaSnapshot) -> FrameAction:
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
            dist = abs(int(dx)) + abs(int(dy))
            if self._collect_best_dist is None or dist < self._collect_best_dist:
                self._collect_best_dist = dist
                self._collect_no_progress = 0
            else:
                self._collect_no_progress += 1
            stale = self._collect_no_progress >= _COLLECT_STALE
            if reached or stuck or stale:
                if stuck or stale:
                    self.notes.append(
                        f"collect_skip_{self.waypoint_index}_{xy[0]}_{xy[1]}"
                    )
                    self._collect_skips += 1
                self.waypoint_index = (self.waypoint_index + 1) % n
                self._stuck_frames = 0
                self._collect_best_dist = None
                self._collect_no_progress = 0
                if self.spec.combat.occupancy_patrol:
                    self.walker.path = None
                if self._collect_skips >= n:
                    return FrameAction(nes_idle_action(), "collect_wait")
                tx, ty = self.spec.reward.waypoints[self.waypoint_index]
                dx = tx - snap.link_x
                dy = ty - snap.link_y
            if self._collect_skips:
                # Hand walks can disagree with the ROM: L1 0x23 (power-on
                # gathered spine) planned down through the (192,100) block and
                # jittered at (192,93) for 5000f; L2 0x6f sat at (120,165) for
                # 11000f. Only after a skip: lattice-first broke 0x23's hunt.
                step = self._lattice_dir(snap, (int(tx), int(ty)))
                if step is not None:
                    return FrameAction(nes_action(step), "collect_reward_lattice")
            if self.spec.combat.occupancy_patrol:
                # A drop can land on geometry, and a waypoint can be written
                # onto it. The BFS then correctly reports no path; walking to
                # the nearest cell that exists is what picks the drop up, and
                # is local to collect for the same reason ``_chase_goal`` is.
                goal = (int(tx), int(ty))
                if not self.walker.grid.passable(*goal):
                    open_goal = self.walker.grid.nearest_open(*goal)
                    if open_goal is not None:
                        goal = open_goal
                direction = self.walker.next_dir(xy, goal)
                if direction is not None:
                    return FrameAction(nes_action(direction), "collect_reward")
                # Still nothing: skip, and *count* it. Advancing the index
                # without counting is why this loop had no end -- 0x45 sat at
                # (40,157) alternating `collect_skip_2` and `collect_skip_3`
                # for the whole 9000-frame budget while ``_collect_skips``
                # stayed under one lap.
                self.notes.append(
                    f"collect_skip_{self.waypoint_index}_{xy[0]}_{xy[1]}"
                )
                self._collect_skips += 1
                self.waypoint_index = (self.waypoint_index + 1) % n
                self.walker.path = None
                self._collect_best_dist = None
                self._collect_no_progress = 0
                if self._collect_skips >= n:
                    return FrameAction(nes_idle_action(), "collect_wait")
                return FrameAction(nes_idle_action(), "collect_skip_unreachable")

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
                direction = positive if delta > 0 else negative
                if self.spec.reward.reward_while_live and (self.frames % 6) < 3:
                    return FrameAction(
                        nes_action(direction, "A"),
                        "collect_reward_slash",
                    )
                return FrameAction(
                    nes_action(direction),
                    "collect_reward",
                )
        return FrameAction(nes_idle_action(), "reward_wait")

    def _finish_clear_leftover(self, snap: ZeldaSnapshot) -> FrameAction:
        target = self.spec.reward.target
        if target is None:
            return self._collect_reward(snap)
        x, y = int(snap.link_x), int(snap.link_y)
        tx, ty = target
        if abs(x - tx) <= 2 and abs(y - ty) <= 2:
            self.success = True
            self._set_phase(DungeonPhase.DONE, "leftover")
            return FrameAction(nes_idle_action(), "done")
        # ROM lattice first: the waist-elbow cardinals below held RIGHT+DOWN
        # into a wall for 13653f (L6 0x29, power-on gathered spine resume).
        # Whole-room lattice: the fight lattice is clipped to the occupancy
        # bounds, and a leftover like (120,189) sits outside them. On a
        # deployed stepladder only its axis moves (``ladder_release``); the
        # old vertical-only rule pressed DOWN beside a horizontal ladder for
        # 14269 frames (0x29, R19 resume).
        step = ladder_release(snap, lattice_goto(self._env, snap, (int(tx), int(ty)), slack=0))
        if step is not None:
            return FrameAction(nes_action(step), "leftover_lattice")
        ladder = next(
            (o for o in snap.objects if int(o.type_id) == _ids.STEPLADDER_OBJECT_TYPE),
            None,
        )
        if ladder is None:
            self._ladder_cross = None
        else:
            # Mid-crossing with no lattice route (the far bank is only reachable
            # over this water): keep going onto the ladder's far side. R19 0x29
            # sat on the island's east ladder pressing DOWN, then LEFT back.
            ldx = int(ladder.x) - x
            ldy = int(ladder.y) - 3 - y
            if max(abs(ldx), abs(ldy)) <= 16:
                # The heading is chosen once, when the crossing starts: past
                # the ladder's centre "toward it" points back (190<->193).
                if self._ladder_cross is not None or (not ldx and not ldy):
                    cross = self._ladder_cross
                elif abs(ldx) >= abs(ldy):
                    cross = "RIGHT" if ldx > 0 else "LEFT"
                else:
                    cross = "DOWN" if ldy > 0 else "UP"
                if cross is not None:
                    self._ladder_cross = cross
                    return FrameAction(nes_action(cross), "leftover_ladder_cross")
        if self._owns_ladder() and (abs(ty - y) > 2 or abs(tx - x) > 2):
            # No lattice route: the fight crossed a moat on the stepladder
            # (L6 0x29 island, 13273f of leftover_clip). A straight press
            # toward the target re-deploys it; the lattice resumes beyond.
            # One axis can be walled (R19: DOWN at (184,141) for 14269f), so
            # swap axes after 12 frames without moving.
            self._ladder_still = self._ladder_still + 1 if (x, y) == self._ladder_xy else 0
            self._ladder_xy = (x, y)
            vertical = "DOWN" if ty > y else "UP"
            horizontal = "RIGHT" if tx > x else "LEFT"
            first, second = (vertical, horizontal) if abs(ty - y) > 2 else (horizontal, vertical)
            direction = first if (self._ladder_still // 12) % 2 == 0 else second
            return FrameAction(nes_action(direction), "leftover_ladder")
        waypoints = self.spec.reward.waypoints
        if waypoints:
            # Waist elbow first; cardinals cannot round the plus from the north.
            _elbow_x, waist_y = waypoints[0]
            if y < waist_y - 2:
                return FrameAction(nes_action("RIGHT", "DOWN"), "leftover_clip")
            if abs(x - tx) > 2:
                horiz = "LEFT" if x > tx else "RIGHT"
                return FrameAction(nes_action(horiz), "leftover_align")
            if y < ty - 2:
                return FrameAction(nes_action("DOWN"), "leftover_south")
            return FrameAction(nes_action("DOWN"), "leftover_push")
        if y < _AVOID_WALL_Y[0]:
            return FrameAction(nes_action("DOWN"), "leftover_inland")
        return self._collect_reward(snap)

    def _scoop_standable_goal(
        self, drop: ZeldaObject
    ) -> tuple[int, int] | None:
        """Nearest cell within scoop reach the occupancy grid calls passable.

        ``drop.(x, y)`` is where the item is DRAWN; Link's feet collide
        ``LINK_FOOT_OFFSET`` px lower (tilemap.py), so the drop's own pixel
        is frequently "solid" in the measured grid even though the drop
        plainly rests on real floor — the grid is answering "can Link stand
        with his stored (x, y) here", not "is this pixel floor". Search the
        small diamond ``_SCOOP_REACH`` already accepts as close-enough,
        nearest first: any hit is inside pickup range, so aiming the BFS at
        it is exactly as good as the exact pixel once Link is standing there.
        """
        grid = self.walker.grid
        ox, oy = int(drop.x), int(drop.y)
        if grid.passable(ox, oy):
            return (ox, oy)
        candidates: list[tuple[int, int, int]] = []
        for dx in range(-_SCOOP_REACH, _SCOOP_REACH + 1):
            for dy in range(-_SCOOP_REACH, _SCOOP_REACH + 1):
                r = abs(dx) + abs(dy)
                if 0 < r <= _SCOOP_REACH and grid.passable(ox + dx, oy + dy):
                    candidates.append((r, ox + dx, oy + dy))
        if not candidates:
            return None
        candidates.sort()
        _, cx, cy = candidates[0]
        return (cx, cy)

    def _scoop_heart_occupancy(
        self, snap: ZeldaSnapshot, drop: ZeldaObject
    ) -> FrameAction | None:
        """Occupancy-walk toward ``drop``; never idle on an unreachable one.

        Idling and re-planning every frame are the two measured failure
        modes here (rr coordinator trace, L1 0x45): idling pinned Link while
        a slow Wallmaster walked into him, and re-running the reverse-flood
        BFS every frame turned a 27s ledger run into 2m48s. Returning None
        lets the room policy (off-wall, evade, combat) drive instead, and
        the cached verdict below bounds the BFS cost while still adapting
        once a blocking body moves on.
        """
        xy = (int(snap.link_x), int(snap.link_y))
        goal = self._scoop_standable_goal(drop)
        if goal is None:
            # No cell within reach is even nominally passable -- do not
            # spend a BFS proving what the grid already answered.
            self.walker.last_dir = None
            return None
        if (
            self._scoop_unreachable_goal == goal
            and self.frames < self._scoop_unreachable_until
        ):
            return None
        bodies = _occupancy_bodies(snap, None)
        bodies.discard(xy)
        bodies.discard(goal)
        direction = self.walker.next_dir(
            xy, goal, extra_blocked=bodies, transient_occupants=bodies
        )
        if direction is None:
            self.walker.last_dir = None
            self._scoop_unreachable_goal = goal
            self._scoop_unreachable_until = self.frames + _SCOOP_REPLAN_HOLD
            return None
        self._scoop_unreachable_goal = None
        return FrameAction(nes_action(direction), "scoop_heart")

    def _sweep_drops(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Walk the lattice onto a cleared room's floor drops before leaving.

        Rupees, bombs and clocks always; hearts and fairies only when hurt.
        The run left 51 of 81 rupees and 6 of 9 bomb drops on the floor while
        Survival poked 82 bombs (full_poweron12 ledger).
        """
        if self.sweep_frames >= SWEEP_MAX_FRAMES:
            return None
        hurt = not snap.health_is_full
        drops = [
            d
            for d in _combat.floor_drops(snap)
            if (int(d.x), int(d.y)) not in self._sweep_skip
            and (hurt or int(d.state) not in _combat.HEART_OR_FAIRY_STATES)
        ]
        if not drops:
            return None
        lx, ly = int(snap.link_x), int(snap.link_y)
        drop = min(drops, key=lambda d: manhattan(lx, ly, int(d.x), int(d.y)))
        goal = (int(drop.x), int(drop.y))
        self.sweep_frames += 1
        if manhattan(lx, ly, *goal) <= _SCOOP_REACH:
            return FrameAction(nes_idle_action(), "sweep_drop")
        route = lattice_goto_route(None, snap, goal)
        if route is None:
            self._sweep_skip.add(goal)
            return None
        # On the drop's nearest node: the last pixels are a direct press.
        step = lattice_step(lx, ly, route[0] if route else goal)
        if step is None:
            self._sweep_skip.add(goal)
            return None
        return FrameAction(nes_action(step), "sweep_drop")

    def _scoop_heart(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.health_is_full or snap.filled_hearts >= snap.heart_containers:
            return None
        drop = _combat.nearest_heart_or_fairy(snap)
        if drop is None:
            self._scoop_unreachable_goal = None
            return None
        dist = manhattan(snap.link_x, snap.link_y, drop.x, drop.y)
        max_dist = _SCOOP_RADIUS if self.spec.live_enemies(snap) else 120
        if dist > max_dist:
            self._scoop_unreachable_goal = None
            return None
        occupancy = self.spec.combat.occupancy_patrol
        if dist <= _SCOOP_REACH:
            if occupancy:
                self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "scoop_heart")
        if _combat.scoop_exits_room(
            snap.link_x, snap.link_y, drop, bounds=DEFAULT_BOUNDS
        ):
            self._scoop_unreachable_goal = None
            return None
        if occupancy:
            return self._scoop_heart_occupancy(snap, drop)
        dx = int(drop.x) - int(snap.link_x)
        dy = int(drop.y) - int(snap.link_y)
        if abs(dx) >= abs(dy) and abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 2:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            return FrameAction(nes_idle_action(), "scoop_heart")
        return FrameAction(nes_action(direction), "scoop_heart")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        """Observe (idempotent per snap), decide; last_reason is hit blame."""
        self.tracked = self.tracker.observe(snap)
        self.damage.observe(
            snap,
            self.tracked,
            action=self.last_reason,
            phase=self.phase.name,
        )
        action = self._step_policy(snap)
        self.last_reason = action.reason
        self._record_reason(snap, action)
        return action

    def _record_reason(self, snap: ZeldaSnapshot, action: FrameAction) -> None:
        """Records only. No action is ever chosen from these.

        A timed-out stage that reports one tile costs the next sitting a whole
        trial working out what held the frames. ``rr-d6v`` spent one finding
        that 11443 of 12000 frames were ``combat_patrol`` in a 1px loop; the
        Survival ``clear45_key`` timeout was 8999 frames of ``entry_route``.
        The histogram names the rule, the tail shows the pose it held.
        This was ``Level6EastKeyController``'s local trace; every room has the
        same failure mode, so it belongs on the engine.
        """
        self._reason_counts[action.reason] = (
            self._reason_counts.get(action.reason, 0) + 1
        )
        patrol = self.spec.combat.patrol
        vertex = patrol[self.patrol_index % len(patrol)] if patrol else None
        self._reason_tail.append(
            f"f{self.frames} {self.phase.name} "
            f"({int(snap.link_x)},{int(snap.link_y)}) "
            f"room=0x{int(snap.screen):02x} {action.reason} "
            f"p{self.patrol_index}{vertex}"
        )
        del self._reason_tail[:-REASON_TAIL_FRAMES]

    def _step_policy(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.initial_inventory is None and (
            self.spec.reward.kind != RewardKind.FIXED_INVENTORY
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
            self.spec.reward.kind == RewardKind.FIXED_INVENTORY
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
            # The door push is a held button with no escape by design (a key
            # door eats frames while it opens). Do not steer it — but do
            # leave the pose behind, so a stage that dies here reports where
            # rather than only "timeout".
            self._note_enter_stall(snap, self.spec.entry.direction)
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
            if not _live_in_contact(snap, live):
                scooped = self._scoop_heart(snap)
                if scooped is not None:
                    return scooped
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
                if self.spec.reward.kind == RewardKind.CLEAR_ONLY:
                    # Key rooms keep their hunt pose; a sweep there moved
                    # Link into the 0x45 west pocket the hunt cannot leave.
                    swept = self._sweep_drops(snap)
                    if swept is not None:
                        return swept
                self.clear_signal_seen = True
                if self.spec.reward.kind == RewardKind.CLEAR_ONLY:
                    if self.spec.reward.target is not None:
                        self._set_phase(DungeonPhase.COLLECT_REWARD, "room_cleared")
                        self._relax_leftover_bounds()
                        return self._finish_clear_leftover(snap)
                    self.success = True
                    self._set_phase(DungeonPhase.DONE, "room_cleared")
                    return FrameAction(nes_idle_action(), "done")
                self._set_phase(DungeonPhase.COLLECT_REWARD, "room_cleared")
                # Combat observe() scars inferred cells that boxed the
                # leftover (L1 0x45 sat at (144,141) for 7666 collect
                # frames after Wallmasters died). CLEAR_ONLY already
                # rebuilds; key hunt uses the same walker.
                self._relax_leftover_bounds()
                return self._collect_reward(snap)
            return self._combat(snap, live)

        if self.phase is DungeonPhase.COLLECT_REWARD:
            scooped = self._scoop_heart(snap)
            if scooped is not None:
                return scooped
            if self.spec.reward.kind == RewardKind.CLEAR_ONLY:
                return self._finish_clear_leftover(snap)
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
            "sweep_frames": self.sweep_frames,
            "max_live_enemies": self.max_live_enemies,
            "last_live_enemies": self.last_live_enemies,
            "clear_signal_seen": self.clear_signal_seen,
            "initial_inventory": self.initial_inventory,
            "notes": list(self.notes),
            "reason_counts": dict(
                sorted(self._reason_counts.items(), key=lambda kv: -kv[1])
            ),
            "tail": list(self._reason_tail),
            "tuning": {
                "engage_distance": self.spec.combat.engage_distance,
                "engage_dominant_axis": self.spec.combat.engage_dominant_axis,
                "attack_phase": self.spec.combat.attack_phase,
                "engage_attack_period": self.spec.combat.engage_attack_period,
                "engage_attack_hold": self.spec.combat.engage_attack_hold,
                "patrol_attack_period": self.spec.combat.patrol_attack_period,
                "patrol_attack_hold": self.spec.combat.patrol_attack_hold,
                "occupancy_patrol": self.spec.combat.occupancy_patrol,
                "occupancy_misses": self.walker.misses,
                "occupancy_blocked": len(self.walker.grid.blocked),
                "evades": self.evader.evades,
                "off_line_steps": self.evader.off_line_steps,
            },
            "damage": dict(
                self.damage.report(),
                swings=self.swings,
                swings_authorized=self.swings_authorized,
                engage_frames=self.engage_frames,
                patrol_frames=self.patrol_frames,
                backstep_frames=self.backstep_frames,
            ),
        }
