"""Level 1 (Eagle) dungeon room specs.

Uses ``dungeon.DungeonRoomSpec`` / engine helpers read-only. Specs register
themselves on import so ``dungeon.spec_for_room`` can find them.
"""

from __future__ import annotations

from dataclasses import replace

from zelda_i.dungeon import (
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
    WALLMASTER_OBJECT_TYPE,
    register_room_spec,
)
from zelda_i.level1 import (
    LEVEL_1,
    ROOM_KEY_STALFOS,
    ROOM_NORTH_STALFOS,
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

# Stay inland. Dormant Wallmasters at x=0 still grab on the west door
# (x=32) after TYPE_AND_HP treats them as dead.
_WALLMASTER_PATROL: tuple[tuple[int, int], ...] = (
    (32, 117),
    (80, 117),
    (120, 117),
    (160, 117),
    (80, 117),
    (32, 141),
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
        engage_distance=48,
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
        engage_distance=56,
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
        # Live key stand is (114, 117) on the east-to-west upper channel.
        # y=149 is the only greedy join from the west pocket; north combat
        # cannot X-first to (176, 149), so the list continues through the
        # south U-turn (128,181)→(96,181)→(96,149) after a stuck skip.
        waypoints=(
            (176, 149),
            (176, 117),
            (114, 117),
            (128, 133),
            (128, 173),
            (128, 181),
            (96, 181),
            (96, 149),
        ),
    ),
    room_item_id=0x19,
    level=LEVEL_1,
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
    level=LEVEL_1,
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
        # Single south-wall target. Hunt the live stand (152, 189) via the
        # east column. Do not linger on the south wall.
        waypoints=(
            (160, 141),
            (160, 173),
            (152, 189),
            (120, 189),
            (80, 141),
            (120, 141),
        ),
    ),
    room_item_id=0x19,
    max_frames=9000,
    level=LEVEL_1,
)

# Survival overlay only. Clean M5 keeps ROOM_45_SPEC (x=160 east-column hunt).
# Off-wall fight avoids the grab-to-entrance. Continuous combat ends in the
# y=149–157 band; south of that at x=80/120/160 is solid, so collect first
# walks the free east column at x=208 (same column the entry route uses).
ROOM_45_SURVIVAL_SPEC = replace(
    ROOM_45_SPEC,
    spec_id="level1_room45_survival",
    combat=replace(
        ROOM_45_SPEC.combat,
        engage_distance=56,
        contact_backstep=16,
        avoid_walls=True,
        inland_dash=48,
    ),
    reward=replace(
        ROOM_45_SPEC.reward,
        waypoints=(
            (208, 157),
            (208, 189),
            (152, 189),
            (208, 141),
            (160, 141),
            (32, 157),
            (32, 189),
        ),
    ),
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
    ROOM_44_SPEC,
    ROOM_45_SPEC,
    ROOM_52_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
):
    register_room_spec(_spec)

__all__ = [
    "ROOM_23_SPEC",
    "ROOM_33_SPEC",
    "ROOM_35_SPEC",
    "ROOM_42_SPEC",
    "ROOM_43_SPEC",
    "ROOM_44_SPEC",
    "ROOM_45_SPEC",
    "ROOM_52_SPEC",
    "ROOM_53_SPEC",
    "ROOM_54_SPEC",
]
