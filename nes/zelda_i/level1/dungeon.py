"""Level 1 (Eagle) dungeon room specs.

Uses ``dungeon.DungeonRoomSpec`` / engine helpers read-only. Specs register
themselves on import so ``dungeon.spec_for_room`` can find them.
"""

from __future__ import annotations

from dataclasses import replace

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import nearest_heart_or_fairy
from zelda_i.dungeon.engine import (
    AQUAMENTUS_OBJECT_TYPE,
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
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
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

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
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(128, 141),
        waypoints=(
            (128, 141),
            (96, 141),
            (160, 141),
            (128, 109),
            (128, 173),
        ),
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
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(96, 173),
    ),
    room_item_id=0x19,
    level=LEVEL_1,
)


class Room33ScoopController(GenericDungeonRoomController):
    """0x33: do not key-DONE while lo < hi. Walk key-tile after clear; else fail-closed."""

    last_health: int = 0
    heart_wait: int = 0
    heart_wait_limit: int = 180

    @staticmethod
    def _scoop_if_low(snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.health_is_full:
            return None
        drop = nearest_heart_or_fairy(snap)
        if drop is None:
            return None
        dx = int(drop.x) - int(snap.link_x)
        dy = int(drop.y) - int(snap.link_y)
        if abs(dx) <= 2 and abs(dy) <= 2:
            return FrameAction(nes_idle_action(), "scoop_heart")
        if abs(dx) >= abs(dy) and abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return FrameAction(nes_action(direction), "scoop_heart")

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
        if snap.health_is_full or snap.screen != self.spec.room_id:
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

    def _on_key_tile(self, snap: ZeldaSnapshot) -> bool:
        target = self.spec.reward.target or (96, 173)
        return (
            abs(int(snap.link_x) - int(target[0])) <= 2
            and abs(int(snap.link_y) - int(target[1])) <= 2
        )

    def _walk_key_tile(self, snap: ZeldaSnapshot) -> FrameAction:
        """Sit on the key/item tile so a 0x60 heart/fairy can be scooped."""
        target = self.spec.reward.target or (96, 173)
        dx = int(target[0]) - int(snap.link_x)
        dy = int(target[1]) - int(snap.link_y)
        if abs(dx) <= 2 and abs(dy) <= 2:
            return FrameAction(nes_idle_action(), "scoop_key_tile")
        if abs(dx) >= abs(dy) and abs(dx) > 2:
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return FrameAction(nes_action(direction), "scoop_key_tile")

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

_ROOM_23_BLOCKED: tuple[tuple[int, int], ...] = (
    # Top wall
    *(
        (x, y)
        for x in range(32, 224)
        for y in range(80, 88)
    ),
    # East wall
    *(
        (x, y)
        for x in range(209, 224)
        for y in range(88, 193)
    ),
    # West block
    *(
        (x, y)
        for x in range(33, 96)
        for y in (*range(97, 120), *range(161, 184))
    ),
    *(
        (x, y)
        for x in range(33, 64)
        for y in range(120, 161)
    ),
    # Center water: 16px cell row at y=128, inset from west/east passages.
    # x=65..175 / y=129..151 boxed the west 16px column (x=64..79) and the
    # south corridor — live leftover (144,149) and death (78,157) were floor.
    *(
        (x, y)
        for x in range(80, 160)
        for y in range(128, 144)
    ),
    # East block
    *(
        (x, y)
        for x in range(145, 208)
        for y in (*range(97, 120), *range(161, 184))
    ),
    *(
        (x, y)
        for x in range(177, 208)
        for y in range(120, 161)
    ),
    # South wall & exterior
    *(
        (x, y)
        for x in (*range(32, 120), *range(121, 224))
        for y in range(193, 201)
    ),
    *(
        (x, y)
        for x in range(32, 224)
        for y in range(201, 208)
    ),
)

# South U-turn only. 1-heart hold must not cycle the maze into (128, 117).
_ROOM_23_SOUTH: tuple[tuple[int, int], ...] = tuple(
    xy for xy in _ROOM_23_MAZE if xy[1] >= 149
)
_ROOM_23_HOLD_Y = 149
_ROOM_23_CORRIDOR_Y = 157


class Room23HeartSafeController(GenericDungeonRoomController):
    """1-heart 0x23: hold y=157; do not chase the north plus-stem.

    Peel DOWN off y=149; UP from the south mouth back to the corridor.
    Mask UP on 141<y<=157 (boomerang plus-stem). Never DOWN at y>=157.
    Occupancy miss → block → replan; no path → stand on the corridor.
    lo>=2 restores the full maze.
    """

    def _combat(self, snap: ZeldaSnapshot, live: tuple) -> FrameAction:
        if not live or int(snap.filled_hearts) > 1:
            return super()._combat(snap, live)
        y = int(snap.link_y)
        south = tuple(obj for obj in live if int(obj.y) >= _ROOM_23_CORRIDOR_Y)
        old = self.spec
        self.spec = replace(
            old, combat=replace(old.combat, patrol=_ROOM_23_SOUTH)
        )
        try:
            if self.patrol_index >= len(_ROOM_23_SOUTH):
                self._snap_patrol_nearest(snap)
            if y > _ROOM_23_CORRIDOR_Y:
                self.combat_frames += 1
                action = self._return_corridor()
            elif y < _ROOM_23_CORRIDOR_Y:
                self.combat_frames += 1
                if y >= _ROOM_23_HOLD_Y:
                    action = self._peel_south()
                else:
                    action = self._retreat_south(snap)
            elif south:
                action = super()._combat(snap, south)
            else:
                self.combat_frames += 1
                action = self._patrol(snap)
        finally:
            self.spec = old
        if _action_is(action, "UP") and y <= _ROOM_23_CORRIDOR_Y:
            if y < _ROOM_23_CORRIDOR_Y:
                return self._peel_south()
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "combat_wait")
        if _action_is(action, "DOWN") and y >= _ROOM_23_CORRIDOR_Y:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "combat_wait")
        return action

    def _peel_south(self) -> FrameAction:
        self.walker.last_dir = "DOWN"
        return FrameAction(nes_action("DOWN"), "heart_safe_peel_south")

    def _return_corridor(self) -> FrameAction:
        self.walker.last_dir = "UP"
        return FrameAction(nes_action("UP"), "heart_safe_return_corridor")

    def _retreat_south(self, snap: ZeldaSnapshot) -> FrameAction:
        x, y = int(snap.link_x), int(snap.link_y)
        direction = self.walker.next_dir((x, y), (120, 157))
        if direction is None or direction == "UP":
            if y >= _ROOM_23_HOLD_Y:
                return self._peel_south()
            direction = "LEFT" if x >= 120 else "RIGHT"
        self.walker.last_dir = direction
        return FrameAction(nes_action(direction), "combat_patrol")


def _action_is(action: FrameAction, direction: str) -> bool:
    a = list(action.action)
    return a == list(nes_action(direction)) or a == list(
        nes_action(direction, "A")
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
        engage_distance=24,
        attack_phase=2,
        avoid_walls=True,
        avoid_wall_bounds=(56, 200, 88, 192),
        split_y=141,
        occupancy_patrol=True,
        occupancy_blocked=_ROOM_23_BLOCKED,
        contact_backstep=16,
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
    "Room23HeartSafeController",
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
