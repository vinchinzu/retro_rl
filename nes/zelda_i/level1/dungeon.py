"""Level 1 (Eagle) dungeon room specs.

Uses ``dungeon.DungeonRoomSpec`` / engine helpers read-only. Specs register
themselves on import so ``dungeon.spec_for_room`` can find them.
"""

from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import (
    chebyshev,
    direction_to_facing,
    in_sword_hitbox,
    nearest_heart_or_fairy,
)
from zelda_i.dungeon.behaviors import fight_target
from zelda_i.dungeon.threat import MIN_DODGE_BODY
from zelda_i.dungeon.engine import (
    AQUAMENTUS_OBJECT_TYPE,
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    _FIGHT_WALKABLE_TILES,
    GEL_OBJECT_TYPE,
    GORIYA_OBJECT_TYPE,
    GenericDungeonRoomController,
    KEESE_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.level1.east_dungeon import (
    ROOM_44_SPEC,
    ROOM_44_SURVIVAL_SPEC,
    ROOM_45_SPEC,
    ROOM_45_SURVIVAL_SPEC,
    Room44SurvivalController,
)
from zelda_i.level1.path import (
    LEVEL_1,
    ROOM_ENTRANCE,
    ROOM_KEY_STALFOS,
    ROOM_NORTH_STALFOS,
    ROOM_WEST_KEY,
    STALFOS_OBJECT_TYPE,
)
from zelda_i.dungeon.tilemap import blocked_link_cells, has_room_tile_map
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import DEFAULT_BOUNDS, OccupancyGrid, OccupancyWalker

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

# Q1 west-of-entrance key: 3 Keese, floor key after clear (Zelda Dungeon / IGN).
_KEESE_72_PATROL: tuple[tuple[int, int], ...] = _KEESE_54_PATROL

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

ROOM_72_SPEC = DungeonRoomSpec(
    spec_id="level1_room72",
    source_room=ROOM_ENTRANCE,
    room_id=ROOM_WEST_KEY,
    entry=DoorRoute(
        "LEFT",
        ((120, 149), (48, 149), (48, 141)),
    ),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_KEESE_72_PATROL,
        engage_distance=48,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    # The key shows at (160, 192) once the Keese are dead ($6530/room item
    # read 2026-09-24); the room is open floor, so one lattice node reaches it.
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(160, 189),
        waypoints=((160, 189),),
    ),
    room_item_id=0x19,
    exit_routes=(
        DoorRoute("RIGHT", ((128, 141), (208, 141))),
    ),
    level=LEVEL_1,
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
    combat=CombatTuning(patrol=_STALFOS_PATROL, contact_backstep=24),
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
    level=LEVEL_1,
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
    level=LEVEL_1,
)

def _room_42_entry_waypoints(snap: ZeldaSnapshot) -> tuple[tuple[int, int], ...]:
    if snap.link_y <= 101:
        return ((120, 101), (120, 93))
    if snap.link_x <= 120:
        return ((96, 101), (120, 101), (120, 93))
    return ((176, 101), (120, 101), (120, 93))


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
        contact_backstep=16,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    exit_routes=(
        DoorRoute("RIGHT", ((128, 93), (208, 93), (208, 141))),
        DoorRoute(
            "UP",
            _room_42_entry_waypoints,
            y_first=True,
        ),
    ),
    level=LEVEL_1,
)

ROOM_42_SPEC = DungeonRoomSpec(
    spec_id="level1_room42",
    source_room=0x52,
    room_id=0x42,
    entry=DoorRoute(
        "UP",
        _room_42_entry_waypoints,
        y_first=True,
    ),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=3,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=_ROOM_42_PATROL,
        # Open floor: Gels are slow. 48px chase waited on the patrol loop
        # (Survival 1770 combat f). 160 covers the interior from center.
        engage_distance=160,
        patrol_attack_period=10,
        patrol_attack_hold=3,
        contact_backstep=16,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    level=LEVEL_1,
)

# 0x52 center 2x2 diamond. West lip leftover (103,165) was 1px outside
# x=104..135; peel until aisle x<=96 then occupancy north.
_ROOM_52_AISLE_X = 96
_ROOM_52_DIAMOND: frozenset[tuple[int, int]] = frozenset(
    (x, y) for x in range(97, 136) for y in range(149, 181)
)


class Room42EntryController(GenericDungeonRoomController):
    """0x52 diamond: peel to an aisle, occupancy around, then north door."""

    def _follow_route(self, snap: ZeldaSnapshot, route: DoorRoute) -> FrameAction:
        if int(snap.screen) != 0x52:
            return super()._follow_route(snap, route)
        x, y = int(snap.link_x), int(snap.link_y)
        dest = (120, 93)
        if abs(x - dest[0]) <= 2 and abs(y - dest[1]) <= 2:
            return FrameAction(nes_idle_action(), "entry_route_done")
        xy = (x, y)
        on_diamond = (
            149 <= y <= 180 and _ROOM_52_AISLE_X < x < 144
        )
        if on_diamond:
            direction = "LEFT" if x <= 120 else "RIGHT"
            self.walker.last_dir = direction
            return FrameAction(nes_action(direction), "entry_route")
        extra = set(_ROOM_52_DIAMOND)
        direction = self.walker.next_dir(xy, dest, extra_blocked=extra)
        if direction is None:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "combat_wait")
        self.walker.last_dir = direction
        return FrameAction(nes_action(direction), "entry_route")


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
        # Same open-floor Gel chase as 0x42. 56px left Clean 2323f of variance.
        engage_distance=160,
        patrol_attack_period=10,
        patrol_attack_hold=3,
        contact_backstep=16,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x17,
    level=LEVEL_1,
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
        contact_backstep=24,
        evade=True,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(96, 173),
    ),
    room_item_id=0x19,
    level=LEVEL_1,
)


ROOM33_LOW_HEARTS = 2


class Room33ScoopController(GenericDungeonRoomController):
    """0x33: do not key-DONE while lo < hi. Walk key-tile after clear; else fail-closed."""

    last_health: int = 0
    _key_walker: OccupancyWalker | None = None
    heart_wait: int = 0
    heart_wait_limit: int = 180
    # North patrol row. Dump 1962f sandwiched at (80,173) while a Stalfos
    # sat at cheb 18 in the UP box. Do not chase onto the key row.
    _hold: tuple[int, int] = (120, 117)

    def _occupancy_walk(
        self, snap: ZeldaSnapshot, dest: tuple[int, int], reason: str
    ) -> FrameAction:
        """BFS to dest; miss → block + replan. No path → stand, do not wiggle."""
        xy = (int(snap.link_x), int(snap.link_y))
        tx, ty = int(dest[0]), int(dest[1])
        if abs(xy[0] - tx) <= 2 and abs(xy[1] - ty) <= 2:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), reason)
        direction = self.walker.next_dir(xy, (tx, ty))
        if direction is None:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), reason)
        return FrameAction(nes_action(direction), reason)

    @staticmethod
    def _low(snap: ZeldaSnapshot) -> bool:
        """Two whole hearts or fewer. With 3 containers that is "not full",
        the rule this was written for; with the gathered 6 a one-heart chip
        is not a reason to wait on a drop (and then fail holding the key).
        """
        return not snap.health_is_full and int(snap.whole_hearts) <= ROOM33_LOW_HEARTS

    def _scoop_if_low(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if not self._low(snap):
            return None
        drop = nearest_heart_or_fairy(snap)
        if drop is None:
            return None
        return self._occupancy_walk(snap, (int(drop.x), int(drop.y)), "scoop_heart")

    def _tick_low(self, snap: ZeldaSnapshot) -> None:
        self.frames += 1
        self.phase_frames += 1
        live = self.spec.live_enemies(snap)
        self.last_live_enemies = len(live)
        self.max_live_enemies = max(self.max_live_enemies, len(live))
        if self.initial_inventory is None and snap.screen == self.spec.room_id:
            self.initial_inventory = self._inventory_value(snap)

    def _cleared_low(self, snap: ZeldaSnapshot) -> bool:
        """Heart-wait only after the room is empty. Live Stalfos keep fighting."""
        if not self._low(snap) or snap.screen != self.spec.room_id:
            return False
        live = self.spec.live_enemies(snap)
        if live:
            return False
        key_got = (
            self.initial_inventory is not None
            and self._inventory_value(snap) > self.initial_inventory
        )
        cleared = self.max_live_enemies >= self.spec.expected_enemy_count
        return bool(key_got or cleared)

    def _key_xy(self, snap: ZeldaSnapshot) -> tuple[int, int]:
        """Where the key lies: slot 1 once the room is dead, else the spec tile.

        A Stalfos carries it. The engine clears the carrier's type and leaves
        the drop's position in slot 1 (``Level1FirstKeyController`` rule).
        (96, 173) is only where the wooden-sword tape killed it: the White
        Sword beam killed it at (128, 148) and the fixed nudge sat 6000f.
        """
        fixed = self.spec.reward.target or (96, 173)
        if self.spec.live_enemies(snap):
            return fixed
        slot = snap.object_in_slot(1)
        if slot and slot.type_id == 0 and 24 <= slot.x <= 224 and 85 <= slot.y <= 205:
            return (int(slot.x), int(slot.y))
        return fixed

    def _on_key_tile(self, snap: ZeldaSnapshot) -> bool:
        target = self._key_xy(snap)
        return (
            abs(int(snap.link_x) - int(target[0])) <= 2
            and abs(int(snap.link_y) - int(target[1])) <= 2
        )

    def _walk_key_tile(self, snap: ZeldaSnapshot) -> FrameAction:
        """Sit on the key/item tile so a 0x60 heart/fairy can be scooped.

        The fight walker is blind here (no tilemap seed), so the key walk
        seeds its own from the live ``$6530`` map once the room is dead.
        """
        if self._key_walker is None and self._env is not None:
            ram = self._env.get_ram()
            if has_room_tile_map(ram):
                blocked = blocked_link_cells(ram, DEFAULT_BOUNDS, walkable=_FIGHT_WALKABLE_TILES)
                xmin, xmax, ymin, ymax = DEFAULT_BOUNDS
                self._key_walker = OccupancyWalker(
                    grid=OccupancyGrid(
                        blocked=set(blocked), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
                    )
                )
        if self._key_walker is not None:
            self.walker = self._key_walker
        return self._occupancy_walk(snap, self._key_xy(snap), "scoop_key_tile")

    def _combat(self, snap: ZeldaSnapshot, live: tuple) -> FrameAction:
        """Slash in place at sword reach. Never walk into the 16px pad."""
        self.combat_frames += 1
        if not live:
            return FrameAction(nes_idle_action(), "combat_wait")
        lx, ly = int(snap.link_x), int(snap.link_y)
        hold = self._hold
        target = fight_target(lx, ly, live)
        if target is None:
            return self._occupancy_walk(snap, hold, "combat_hold")
        cheb = chebyshev(lx, ly, target.x, target.y)
        dx = int(target.x) - lx
        dy = int(target.y) - ly
        if abs(dx) >= abs(dy):
            face = "RIGHT" if dx > 0 else "LEFT"
            away = "LEFT" if dx > 0 else "RIGHT"
        else:
            face = "DOWN" if dy > 0 else "UP"
            away = "UP" if dy > 0 else "DOWN"
        if cheb < MIN_DODGE_BODY:
            return FrameAction(nes_action(away), "combat_evade_body")
        if in_sword_hitbox(lx, ly, face, target.x, target.y):
            if int(snap.facing) == direction_to_facing(face):
                self.swings += 1
                self.swings_authorized += 1
                return FrameAction(nes_action("A"), "combat_hold_slash")
            return FrameAction(nes_action(face), "combat_hold_face")
        if abs(lx - hold[0]) > 2 or abs(ly - hold[1]) > 2:
            return self._occupancy_walk(snap, hold, "combat_hold")
        if abs(dx) > 12:
            step = "RIGHT" if dx > 0 else "LEFT"
            return FrameAction(nes_action(step), "combat_hold_align")
        return FrameAction(nes_action(face), "combat_hold_close")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.last_health = int(snap.health)
        if snap.mode == 17 or snap.transitioning or snap.mode != PLAY_MODE:
            return super().step(snap)
        if snap.level != self.spec.level:
            return super().step(snap)
        scooped = self._scoop_if_low(snap)
        if scooped is not None:
            self._tick_low(snap)
            return scooped
        if self._cleared_low(snap):
            self._tick_low(snap)
            if self._on_key_tile(snap):
                self.heart_wait += 1
                if self.heart_wait >= self.heart_wait_limit:
                    self._set_phase(DungeonPhase.FAILED, "0x33_needs_heart")
                    return FrameAction(nes_idle_action(), "0x33_needs_heart")
            return self._walk_key_tile(snap)
        self.heart_wait = 0
        if self.phase is DungeonPhase.COLLECT_REWARD and snap.screen == self.spec.room_id:
            action = super().step(snap)
            if self.phase is DungeonPhase.COLLECT_REWARD:
                return self._walk_key_tile(snap)
            return action
        return super().step(snap)

    def report(self) -> dict:
        out = super().report()
        out["last_health"] = self.last_health
        out["heart_wait"] = self.heart_wait
        return out


# Water-maze walkable loop. The mid-row y=129..151 is blocked by water
# between cols 65 and 175; safe cross-passages are col 64 and col 176.
_ROOM_23_MAZE: tuple[tuple[int, int], ...] = (
    (120, 93),
    (120, 125),
    (64, 125),
    (64, 157),
    (120, 157),
    (176, 157),
    (176, 125),
    (120, 125),
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
        patrol=_ROOM_23_MAZE,
        engage_distance=80,
        attack_phase=2,
        avoid_walls=True,
        avoid_wall_bounds=(56, 200, 88, 192),
        split_y=141,
        occupancy_patrol=True,
        occupancy_from_tilemap=True,
        contact_backstep=16,
        evade=True,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        # Key drops at (128, 117) on the north channel. Loop routes through
        # the east and west channels avoiding center water.
        waypoints=(
            (128, 117),
            (120, 125),
            (64, 125),
            (64, 157),
            (120, 157),
            (176, 157),
            (176, 125),
            (120, 125),
        ),
    ),
    room_item_id=0x19,
    level=LEVEL_1,
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
    level=LEVEL_1,
)

for _spec in (
    ROOM_23_SPEC,
    ROOM_33_SPEC,
    ROOM_35_SPEC,
    ROOM_42_SPEC,
    ROOM_43_SPEC,
    ROOM_52_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
    ROOM_72_SPEC,
):
    register_room_spec(_spec)

__all__ = [
    "ROOM_23_SPEC",
    "Room33ScoopController",
    "ROOM_33_SPEC",
    "ROOM_35_SPEC",
    "Room42EntryController",
    "ROOM_42_SPEC",
    "ROOM_43_SPEC",
    "ROOM_44_SPEC",
    "ROOM_44_SURVIVAL_SPEC",
    "ROOM_45_SPEC",
    "ROOM_45_SURVIVAL_SPEC",
    "ROOM_52_SPEC",
    "ROOM_53_SPEC",
    "ROOM_54_SPEC",
    "ROOM_72_SPEC",
    "Room44SurvivalController",
]
