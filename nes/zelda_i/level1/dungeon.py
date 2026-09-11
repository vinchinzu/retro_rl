"""Level 1 (Eagle) dungeon room specs.

Uses ``dungeon.DungeonRoomSpec`` / engine helpers read-only. Specs register
themselves on import so ``dungeon.spec_for_room`` can find them.
"""

from __future__ import annotations

from zelda_i.dungeon.engine import (
    AQUAMENTUS_OBJECT_TYPE,
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GEL_OBJECT_TYPE,
    GORIYA_OBJECT_TYPE,
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
    level=LEVEL_1,
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
        # Open floor: Gels are slow. 48px chase waited on the patrol loop
        # (Survival 1770 combat f). 160 covers the interior from center.
        engage_distance=160,
        patrol_attack_period=10,
        patrol_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x03,
    level=LEVEL_1,
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
        # Same open-floor Gel chase as 0x42. 56px left Clean 2323f of variance.
        engage_distance=160,
        patrol_attack_period=10,
        patrol_attack_hold=3,
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
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(96, 173),
    ),
    room_item_id=0x19,
    level=LEVEL_1,
)

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
    # Center water bar
    *(
        (x, y)
        for x in range(65, 176)
        for y in range(129, 152)
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
    "ROOM_33_SPEC",
    "ROOM_35_SPEC",
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
