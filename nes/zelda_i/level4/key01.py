"""Level 4 leftover 0x11 → bomb-UP 0x01 natural key (no live BFS).

bomb-UP v2 leftover play 0x11 (120,189). v1 hold-UP leftover (120,93): north wall, not a free door.
v2 bomb-UP 377f entered 0x01; Keese clear then hunt at (96,125) missed the
floor key east of leftover (96,135). Isolated bomb-probe stand (120,105).
Type 0x35 stays live. Predecessor keys=4 until the pickup. Ignore 0x2b/0x68.
"""

from __future__ import annotations

from zelda_i.dungeon.bomb_wall import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    KEESE_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.level4.dungeon import (
    LEVEL4,
    ROOM_ITEM_SMALL_KEY,
    ROOM_L4_KEY_01,
    ROOM_L4_MID_11,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "BOMB_11_NORTH_FACE",
    "BOMB_11_NORTH_STAND",
    "BombWall11North",
    "KEY_01_PICKUP_XY",
    "ROOM_01_SPEC",
    "level4_key01_stages",
    "level4_key01_success",
    "make_bomb_11_north_controller",
    "make_room_01_key_controller",
]

BOMB_11_NORTH_STAND = (120, 105)
BOMB_11_NORTH_FACE = "UP"
# v2 leftover (96,135): key is east of Link, not at (96,125). Center column.
KEY_01_PICKUP_XY = (120, 141)
_PATROL_01: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
)
_KEY_01_WAYPOINTS: tuple[tuple[int, int], ...] = (
    KEY_01_PICKUP_XY,
    (128, 141),
    (112, 141),
    (120, 133),
    (136, 141),
    (104, 141),
    (120, 149),
    (128, 133),
    (112, 133),
)

ROOM_01_SPEC = DungeonRoomSpec(
    spec_id="level4_room01_keese_key",
    source_room=ROOM_L4_MID_11,
    room_id=ROOM_L4_KEY_01,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE,  # Keese HP stays 0 while alive
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_01,
        engage_distance=56,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        settle_all_dead=0,
        reward_while_live=True,
        target=KEY_01_PICKUP_XY,
        waypoints=_KEY_01_WAYPOINTS,
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(DoorRoute("DOWN", ((120, 189), (120, 205))),),
    max_frames=16000,
    level=LEVEL4,
)
register_room_spec(ROOM_01_SPEC)


class BombWall11North:
    """Geometry stand for ``BombWallController``: 0x11 bomb-UP → 0x01."""

    room = ROOM_L4_MID_11
    stand = BOMB_11_NORTH_STAND
    face = BOMB_11_NORTH_FACE
    opens_to = ROOM_L4_KEY_01


def make_bomb_11_north_controller() -> BombWallController:
    """0x11 leftover → bomb north → 0x01. No 0x35 clear."""
    return BombWallController(
        wall=BombWall11North(),
        level=LEVEL4,
        stand_tol=2,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=8000,
    )


def make_room_01_key_controller() -> GenericDungeonRoomController:
    """Collect 0x01 key 0x19. Keese may stay live (reward_while_live)."""
    return GenericDungeonRoomController(ROOM_01_SPEC)


def level4_key01_stages():
    """0x11 leftover → bomb-UP 0x01 → natural key (keys 4→5)."""
    bomb = make_bomb_11_north_controller()
    key = make_room_01_key_controller()
    key.phase = DungeonPhase.FIGHT
    return (
        ("level4_bomb_north_0x11", bomb, bomb.max_frames),
        ("level4_key_0x01", key, ROOM_01_SPEC.max_frames),
    )


def level4_key01_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    """Play-ready 0x01 with a natural key delta. Keese/0x35 may stay live."""
    return (
        snap.level == LEVEL4
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_L4_KEY_01
        and not snap.transitioning
        and snap.keys > keys_before
    )
