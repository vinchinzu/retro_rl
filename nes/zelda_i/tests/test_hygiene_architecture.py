"""Layer contracts: anchors, bomb-wall engine, door stands, dungeon IDs."""

from __future__ import annotations

from zelda_i.anchors import (
    ENTRANCES,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL5_DOOR,
    SCREEN_LEVEL5_ENTRANCE,
    TF_BIT_L3,
    TRIFORCE_BITS_BY_LEVEL,
)
from zelda_i.bomb_wall_path import BombWallController, BombWallPhase
from zelda_i.door_graph.level2_exits import _BOMB_STAND_6F_N
from zelda_i.level2_bomb_path import (
    make_bomb_north_1e_controller,
    make_bomb_north_controller,
    make_boom_bomb_north_controller,
    make_post_boom_bomb_north_controller,
)
from zelda_i.level2_puzzles import BOMB_WALL_6F_NORTH
from zelda_i.later_nodes import SCREEN_LEVEL3_ENTRANCE as LN_L3


def test_anchors_are_single_source() -> None:
    assert SCREEN_LEVEL3_ENTRANCE == 0x74
    assert SCREEN_LEVEL5_DOOR is SCREEN_LEVEL5_ENTRANCE
    assert TRIFORCE_BITS_BY_LEVEL[3] == TF_BIT_L3 == 0x04
    assert ENTRANCES[3].verified
    assert ENTRANCES[4].verified  # rr-0fx live entry
    assert ENTRANCES[4].entry_room == 0x71
    assert LN_L3 == SCREEN_LEVEL3_ENTRANCE


def test_bomb_wall_factories_share_engine() -> None:
    c6 = make_bomb_north_controller()
    c5 = make_boom_bomb_north_controller(clear_gels=False)
    c4 = make_post_boom_bomb_north_controller()
    c1 = make_bomb_north_1e_controller()
    assert isinstance(c6, BombWallController)
    assert c6.phase is BombWallPhase.SETTLE
    assert c6.wall.opens_to == 0x5F
    assert c5.wall.opens_to == 0x4F
    assert c4.wall.opens_to == 0x3F
    assert c1.south_band_first
    assert c1.wall.opens_to == 0x0E


def test_door_graph_stands_match_puzzle_catalog() -> None:
    assert _BOMB_STAND_6F_N == BOMB_WALL_6F_NORTH.stand


def test_level3_dungeon_enemy_types_come_from_dungeon_ids() -> None:
    from zelda_i import dungeon as eng
    from zelda_i import dungeon_ids as ids
    from zelda_i.level3_dungeon import (
        DARKNUT_OBJECT_TYPE,
        INVULN_MOVER_0X2B,
        KEESE_OBJECT_TYPE,
        MANHANDLA_OBJECT_TYPE,
        ZOL_OBJECT_TYPE,
    )

    assert ZOL_OBJECT_TYPE is ids.ZOL_OBJECT_TYPE
    assert DARKNUT_OBJECT_TYPE is ids.DARKNUT_OBJECT_TYPE
    assert KEESE_OBJECT_TYPE is ids.KEESE_OBJECT_TYPE
    assert MANHANDLA_OBJECT_TYPE is ids.MANHANDLA_OBJECT_TYPE
    assert INVULN_MOVER_0X2B is ids.INVULN_MOVER_OBJECT_TYPE
    assert eng.KEESE_OBJECT_TYPE is ids.KEESE_OBJECT_TYPE
    assert eng.GORIYA_OBJECT_TYPE is ids.GORIYA_OBJECT_TYPE


def test_level4_dungeon_enemy_types_come_from_dungeon_ids() -> None:
    from zelda_i import dungeon_ids as ids
    from zelda_i.level4_dungeon import (
        GEL_OBJECT_TYPE,
        GLEEOK_OBJECT_TYPE,
        LIKE_LIKE_OBJECT_TYPE,
        VIRE_OBJECT_TYPE,
        ZOL_OBJECT_TYPE,
    )

    assert VIRE_OBJECT_TYPE is ids.VIRE_OBJECT_TYPE
    assert ZOL_OBJECT_TYPE is ids.ZOL_OBJECT_TYPE
    assert GEL_OBJECT_TYPE is ids.GEL_OBJECT_TYPE
    assert LIKE_LIKE_OBJECT_TYPE is ids.LIKE_LIKE_OBJECT_TYPE
    assert GLEEOK_OBJECT_TYPE is ids.GLEEOK_OBJECT_TYPE


def test_level5_dungeon_enemy_types_come_from_dungeon_ids() -> None:
    from zelda_i import dungeon_ids as ids
    from zelda_i.level5_dungeon import (
        BUBBLE_OBJECT_TYPE,
        GIBDO_OBJECT_TYPE,
        POLS_VOICE_OBJECT_TYPE,
        ZOL_OBJECT_TYPE,
    )

    assert GIBDO_OBJECT_TYPE is ids.GIBDO_OBJECT_TYPE
    assert POLS_VOICE_OBJECT_TYPE is ids.POLS_VOICE_OBJECT_TYPE
    assert BUBBLE_OBJECT_TYPE is ids.BUBBLE_OBJECT_TYPE
    assert ZOL_OBJECT_TYPE is ids.ZOL_OBJECT_TYPE


def test_level4_boss_combat_gleeok_types_come_from_dungeon_ids() -> None:
    from zelda_i import dungeon_ids as ids
    from zelda_i.level4_boss_combat import (
        GLEEOK_FIREBALL_TYPE,
        GLEEOK_HEAD_OBJECT_TYPE,
        GLEEOK_OBJECT_TYPE,
    )

    assert GLEEOK_OBJECT_TYPE is ids.GLEEOK_OBJECT_TYPE
    assert GLEEOK_HEAD_OBJECT_TYPE is ids.GLEEOK_HEAD_OBJECT_TYPE
    assert GLEEOK_FIREBALL_TYPE is ids.MANHANDLA_PROJECTILE_TYPE
    assert ids.GLEEOK_3HEAD_OBJECT_TYPE == 0x44


def test_dungeon_ids_has_l4_l5_enemy_types() -> None:
    from zelda_i import dungeon_ids as ids

    assert ids.LIKE_LIKE_OBJECT_TYPE == 0x17
    assert ids.POLS_VOICE_OBJECT_TYPE == 0x16
    assert ids.GIBDO_OBJECT_TYPE == 0x30
    assert ids.BUBBLE_OBJECT_TYPE == 0x40
    assert ids.GORIYA_BOOMERANG_OBJECT_TYPE == 0x5C
    assert ids.OBJECT_NAMES[0x5C] == "boomerang_projectile"
