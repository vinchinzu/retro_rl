"""Smoke tests for architecture hygiene refactors."""

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


def test_level2_dungeon_reexports_bomb_path() -> None:
    from zelda_i.level2_dungeon import BOMB_N_STAND, make_bomb_north_controller as C

    assert BOMB_N_STAND == (120, 101)
    assert C().wall.room == 0x6F


def test_level3_dungeon_is_sole_raft_shim() -> None:
    """Canonical: level3_raft_path. One shim: level3_dungeon. Not level3_path."""
    from zelda_i.level3_dungeon import Level3RaftPathController
    from zelda_i.level3_geometry import RAFT_CHANNEL_X as GEO_CHANNEL
    from zelda_i.level3_raft_path import (
        RAFT_PATH_MAX_FRAMES as RAFT_MAX,
        Level3RaftPathController as Canonical,
    )
    import zelda_i.level3_dungeon as l3d
    import zelda_i.level3_path as l3p

    assert Level3RaftPathController is Canonical
    assert "RAFT_PATH_MAX_FRAMES" not in l3d.__dict__
    assert l3d.RAFT_PATH_MAX_FRAMES is RAFT_MAX
    assert l3d.RAFT_CHANNEL_X is GEO_CHANNEL
    # level3_path must not re-export raft (rr-iji5).
    assert not hasattr(l3p, "Level3RaftPathController") or "Level3RaftPathController" not in getattr(
        l3p, "__all__", ()
    )
    try:
        getattr(l3p, "Level3RaftPathController")
        # If attribute exists via accidental module attr, fail if it came from __getattr__
        # After rr-iji5 there is no __getattr__ raft path.
        assert "Level3RaftPathController" in l3p.__dict__
    except AttributeError:
        pass  # expected: no raft re-export


def test_level3_dungeon_enemy_types_come_from_dungeon_ids() -> None:
    from zelda_i import dungeon_ids as ids
    from zelda_i import dungeon as eng
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


def test_level4_dungeon_reexports_path_controllers() -> None:
    import zelda_i.level4_dungeon as l4d
    import zelda_i.level4_path as l4p

    assert "make_entry_up_controller" not in l4d.__dict__
    assert l4d.make_entry_up_controller is l4p.make_entry_up_controller
    ctl = l4d.make_entry_up_controller()
    assert isinstance(ctl, l4p.Level4EntryUpController)


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


def test_level6_stairs3a_is_own_module_not_dungeon() -> None:
    import zelda_i.level6_dungeon as l6d
    import zelda_i.level6_path as l6p
    import zelda_i.level6_spine as l6s
    import zelda_i.level6_stairs3a as stairs

    assert "make_stairs_3a_controller" not in l6d.__dict__
    assert "make_stairs_3a_controller" not in l6p.__dict__
    assert stairs.make_stairs_3a_controller().spec_id == "level6_stairs_0x3a"
    assert l6s.level6_stairs3a_success is stairs.level6_stairs3a_success
    spine_lines = open(l6s.__file__, encoding="utf-8").read().count("\n") + 1
    assert spine_lines < 800, spine_lines


def test_level6_exit_ow_is_own_module_and_spine_under_800() -> None:
    import zelda_i.level4_spine as l4s
    import zelda_i.level6_dungeon as l6d
    import zelda_i.level6_exit_ow as exit_ow
    import zelda_i.level6_inland29 as inland29
    import zelda_i.level6_path as l6p
    import zelda_i.level6_spine as l6s
    import zelda_i.level6_spine_suffix as suffix
    import zelda_i.level6_west39 as west39
    import zelda_i.level6_west39_reband as west39_reband
    import zelda_i.level6_stairs3a_71 as stairs3a_71
    import zelda_i.level6_clear39_west as clear39_west
    import zelda_i.level6_west19 as west19
    import zelda_i.level6_south18 as south18
    import zelda_i.level6_aisle_west28 as aisle_west28
    import zelda_i.level6_west28 as west28
    import zelda_i.level6_east28 as east28
    import zelda_i.level6_clear28_south as clear28_south
    import zelda_i.level6_east38 as east38
    import zelda_i.level6_east38_lane as east38_lane
    import zelda_i.level6_west38 as west38
    import zelda_i.level6_bomb38_south as bomb38_south
    import zelda_i.level6_south38 as south38
    import zelda_i.level6_clear38_south as clear38_south
    import zelda_i.level6_aisle28 as aisle28
    import zelda_i.level6_south28 as south28
    import zelda_i.survival_spine as spine

    assert "make_exit_ow_controller" not in l6d.__dict__
    assert "make_exit_ow_controller" not in l6p.__dict__
    assert "make_west39_controller" not in l6d.__dict__
    assert "make_west39_controller" not in l6p.__dict__
    assert "make_west39_reband_controller" not in l6d.__dict__
    assert "make_west39_reband_controller" not in l6p.__dict__
    assert "make_clear39_west_controller" not in l6d.__dict__
    assert "make_clear39_west_controller" not in l6p.__dict__
    assert "make_inland29_controller" not in l6d.__dict__
    assert "make_inland29_controller" not in l6p.__dict__
    assert "make_west19_controller" not in l6d.__dict__
    assert "make_west19_controller" not in l6p.__dict__
    assert "make_south18_controller" not in l6d.__dict__
    assert "make_south18_controller" not in l6p.__dict__
    assert "make_aisle_west28_controller" not in l6d.__dict__
    assert "make_aisle_west28_controller" not in l6p.__dict__
    assert "make_west28_controller" not in l6d.__dict__
    assert "make_west28_controller" not in l6p.__dict__
    assert "make_east28_controller" not in l6d.__dict__
    assert "make_east28_controller" not in l6p.__dict__
    assert "make_clear28_south_controller" not in l6d.__dict__
    assert "make_clear28_south_controller" not in l6p.__dict__
    assert "make_east38_controller" not in l6d.__dict__
    assert "make_east38_controller" not in l6p.__dict__
    assert "make_east38_lane_controller" not in l6d.__dict__
    assert "make_east38_lane_controller" not in l6p.__dict__
    assert "make_west38_controller" not in l6d.__dict__
    assert "make_west38_controller" not in l6p.__dict__
    assert "make_bomb38_south_controller" not in l6d.__dict__
    assert "make_bomb38_south_controller" not in l6p.__dict__
    assert "make_south38_controller" not in l6d.__dict__
    assert "make_south38_controller" not in l6p.__dict__
    assert "make_clear38_south_controller" not in l6d.__dict__
    assert "make_clear38_south_controller" not in l6p.__dict__
    assert "make_aisle28_controller" not in l6d.__dict__
    assert "make_aisle28_controller" not in l6p.__dict__
    assert "make_south28_controller" not in l6d.__dict__
    assert "make_south28_controller" not in l6p.__dict__
    assert exit_ow.make_exit_ow_controller().spec_id == "level6_exit_ow_0x22"
    west39_reband_ctl = west39_reband.make_west39_reband_controller()
    assert west39_reband_ctl.spec_id == "level6_west39_reband_0x39"
    assert west39_reband.WEST_DOOR == (32, 141)
    assert west39_reband.LANE_Y == 141
    assert west39_reband.DATED_LEFT6 == (125, 133)
    assert west39_reband.DATED_LEFT7 == (127, 133)
    assert west39_reband.DATED_LEFT8 == (128, 133)
    assert west39_reband_ctl.room == 0x3A
    assert west39_reband_ctl._goal() == (48, 141)
    assert not hasattr(west39_reband_ctl, "bomb")
    west39_ctl = west39.make_west39_controller()
    assert west39_ctl.spec_id == "level6_west_0x39"
    assert west39.WEST_DOOR == (32, 141)
    assert west39_ctl.room == 0x3A
    assert west39_ctl._goal() == (48, 141)
    assert not hasattr(west39_ctl, "bomb")
    clear39_west_ctl = clear39_west.make_clear39_west_controller()
    assert clear39_west_ctl.spec_id == "level6_clear39_west_0x39"
    assert clear39_west.WEST_DOOR == (32, 141)
    assert clear39_west.LANE_Y == 141
    assert clear39_west.DATED_DOWN == (144, 109)
    assert clear39_west.DATED_LEFT == (142, 141)
    assert clear39_west.DATED_LEFT2 == (139, 141)
    assert clear39_west_ctl.room == 0x3A
    assert clear39_west_ctl._goal() == (48, 141)
    assert not hasattr(clear39_west_ctl, "bomb")
    assert exit_ow.make_north39_controller().spec_id == "level6_north39_0x29"
    assert inland29.make_inland29_controller().spec_id == "level6_inland_0x29"
    assert west19.make_west19_controller().spec_id == "level6_west_0x19"
    assert south18.make_south18_controller().spec_id == "level6_south_0x18"
    assert aisle_west28.make_aisle_west28_controller().spec_id == (
        "level6_aisle_west_0x28"
    )
    assert aisle_west28.AISLE_X == 64
    assert aisle_west28.AISLE_Y == 141
    assert aisle_west28.WEST_DOOR == (32, 141)
    assert aisle_west28.WEST_XMAX == 120
    assert aisle_west28.make_aisle_west28_controller().aisle == (
        aisle_west28.AISLE_X,
        aisle_west28.AISLE_Y,
    )
    assert aisle_west28.make_aisle_west28_controller().goal == aisle_west28.WEST_DOOR
    assert not hasattr(aisle_west28.make_aisle_west28_controller(), "bomb")
    assert west28.make_west28_controller().spec_id == "level6_west_0x28"
    assert west28.WEST_DOOR == (32, 141)
    assert west28.WEST_XMAX == 120
    assert west28.CLIP_PAST_Y == 109
    assert west28.make_west28_controller().goal == west28.WEST_DOOR
    assert not hasattr(west28.make_west28_controller(), "bomb")
    assert east28.make_east28_controller().spec_id == "level6_east_0x28"
    assert east28.EAST_DOOR == (208, 141)
    assert east28.make_east28_controller().goal == east28.EAST_DOOR
    assert not hasattr(east28.make_east28_controller(), "bomb")
    assert clear28_south.make_clear28_south_controller().spec_id == (
        "level6_clear_south_0x28"
    )
    assert east38.make_east38_controller().spec_id == "level6_east_0x38"
    assert east38.CLIP_BOX == (128, 141)
    assert east38.EAST_DOOR == (208, 141)
    assert east38.make_east38_controller().goal == east38.EAST_DOOR
    assert not hasattr(east38.make_east38_controller(), "bomb")
    assert east38_lane.make_east38_lane_controller().spec_id == "level6_east_lane_0x38"
    assert east38_lane.LANE_X == 136
    assert east38_lane.LANE_Y == 141
    assert east38_lane.make_east38_lane_controller().goal == east38.EAST_DOOR
    assert not hasattr(east38_lane.make_east38_lane_controller(), "bomb")
    assert west38.make_west38_controller().spec_id == "level6_west_0x38"
    assert west38.WEST_DOOR == (32, 141)
    assert west38.WEST_XMAX == 120
    assert west38.NORTH_BAND_Y == 109
    assert west38.make_west38_controller().goal == west38.WEST_DOOR
    assert not hasattr(west38.make_west38_controller(), "bomb")
    assert bomb38_south.make_bomb38_south_controller().spec_id == (
        "level6_bomb_south_0x38"
    )
    assert bomb38_south.BOMB_38_SOUTH_STAND == (120, 173)
    assert bomb38_south.BOMB_38_SOUTH_STAND != (120, 181)
    assert bomb38_south.EAST_DOOR == (208, 141)
    assert bomb38_south.WEST_DOOR == (32, 141)
    assert bomb38_south.make_bomb38_south_controller().goal == bomb38_south.EAST_DOOR
    assert bomb38_south.make_bomb38_south_controller().phase == "east_path"
    assert not hasattr(bomb38_south.make_bomb38_south_controller(), "bomb")
    assert bomb38_south.BOMB_WALL_38_SOUTH.face == "DOWN"
    assert bomb38_south.BOMB_WALL_38_SOUTH.live is False
    assert south38.make_south38_controller().spec_id == "level6_south_0x38"
    assert clear38_south.make_clear38_south_controller().spec_id == (
        "level6_clear_south_0x38"
    )
    assert aisle28.make_aisle28_controller().spec_id == "level6_aisle_0x28"
    assert south28.make_south28_controller().spec_id == "level6_south_0x28"
    assert l6s.level6_exit_ow_success is exit_ow.level6_exit_ow_success
    assert suffix.level6_exit_ow_success is exit_ow.level6_exit_ow_success
    assert l6s.level6_stairs3a_71_success is stairs3a_71.level6_stairs3a_71_success
    assert suffix.level6_stairs3a_71_success is stairs3a_71.level6_stairs3a_71_success
    assert stairs3a_71.make_stairs_3a_71_controller().spec_id == "level6_stairs_0x3a_71"
    assert l6s.level6_west39_reband_success is west39_reband.level6_west39_reband_success
    assert suffix.level6_west39_reband_success is west39_reband.level6_west39_reband_success
    assert l6s.level6_west39_success is west39.level6_west39_success
    assert suffix.level6_west39_success is west39.level6_west39_success
    assert l6s.level6_clear39_west_success is clear39_west.level6_clear39_west_success
    assert suffix.level6_clear39_west_success is clear39_west.level6_clear39_west_success
    assert l6s.level6_north39_success is exit_ow.level6_north39_success
    assert suffix.level6_north39_success is exit_ow.level6_north39_success
    assert l6s.level6_inland29_success is inland29.level6_inland29_success
    assert suffix.level6_inland29_success is inland29.level6_inland29_success
    assert l6s.level6_west19_success is west19.level6_west19_success
    assert suffix.level6_west19_success is west19.level6_west19_success
    assert l6s.level6_south18_success is south18.level6_south18_success
    assert suffix.level6_south18_success is south18.level6_south18_success
    assert l6s.level6_aisle_west28_success is aisle_west28.level6_aisle_west28_success
    assert suffix.level6_aisle_west28_success is aisle_west28.level6_aisle_west28_success
    assert l6s.level6_west28_success is west28.level6_west28_success
    assert suffix.level6_west28_success is west28.level6_west28_success
    assert l6s.level6_east28_success is east28.level6_east28_success
    assert suffix.level6_east28_success is east28.level6_east28_success
    assert l6s.level6_clear28_south_success is clear28_south.level6_clear28_south_success
    assert suffix.level6_clear28_south_success is clear28_south.level6_clear28_south_success
    assert l6s.level6_east38_success is east38.level6_east38_success
    assert suffix.level6_east38_success is east38.level6_east38_success
    assert l6s.level6_east38_lane_success is east38_lane.level6_east38_lane_success
    assert suffix.level6_east38_lane_success is east38_lane.level6_east38_lane_success
    assert l6s.level6_west38_success is west38.level6_west38_success
    assert suffix.level6_west38_success is west38.level6_west38_success
    assert l6s.level6_bomb38_south_success is bomb38_south.level6_bomb38_south_success
    assert suffix.level6_bomb38_south_success is bomb38_south.level6_bomb38_south_success
    assert l6s.level6_south38_success is south38.level6_south38_success
    assert suffix.level6_south38_success is south38.level6_south38_success
    assert l6s.level6_clear38_south_success is clear38_south.level6_clear38_south_success
    assert suffix.level6_clear38_south_success is clear38_south.level6_clear38_south_success
    assert l6s.level6_aisle28_success is aisle28.level6_aisle28_success
    assert suffix.level6_aisle28_success is aisle28.level6_aisle28_success
    assert l6s.level6_south28_success is south28.level6_south28_success
    assert suffix.level6_south28_success is south28.level6_south28_success
    src = open(suffix.__file__, encoding="utf-8").read()
    assert '"level6-stairs3a"' in src and "True)" in src
    assert '"level6-stairs3a-71"' in src and "True)" in src
    assert '"level6-west39-reband"' in src
    assert '"level6-west39"' in src
    assert '"level6-clear39-west"' in src
    assert '"level6-north39"' in src
    assert '"level6-inland29"' in src
    assert '"level6-west19"' in src
    assert '"level6-south18"' in src
    assert '"level6-aisle-west28"' in src
    assert '"level6-west28"' in src
    assert '"level6-east28"' in src
    assert '"level6-clear28-south"' in src
    assert '"level6-east38"' in src
    assert '"level6-east38-lane"' in src
    assert '"level6-west38"' in src
    assert '"level6-bomb38-south"' in src
    assert '"level6-south38"' in src
    assert '"level6-clear38-south"' in src
    assert '"level6-aisle28"' in src
    assert '"level6-south28"' in src
    assert "level6_stairs3a_71_success, True)" in src
    assert "level6_west39_reband_success, True)" in src
    assert "level6_west39_success, True)" in src
    assert "level6_clear39_west_success, True)" in src
    assert "level6_east38_success, True)" in src
    assert "level6_east38_lane_success, True)" in src
    assert "level6_aisle_west28_success, True)" in src
    assert "level6_west28_success, True)" in src
    assert "level6_east28_success, True)" in src
    assert "level6_clear28_south_success, True)" in src
    assert "level6_west38_success, True)" in src
    assert "level6_bomb38_south_success, True)" in src
    assert "level6_south38_success, True)" in src
    assert "level6_clear38_south_success, True)" in src
    assert "level6_aisle28_success, True)" in src
    assert "level6_south28_success, True)" in src
    assert "level6_exit_ow_success, True)" in src
    for mod in (spine, l6s, suffix, l4s):
        lines = open(mod.__file__, encoding="utf-8").read().count("\n") + 1
        assert lines < 800, (mod.__name__, lines)


def test_dungeon_ids_has_l4_l5_enemy_types() -> None:
    from zelda_i import dungeon_ids as ids

    assert ids.LIKE_LIKE_OBJECT_TYPE == 0x17
    assert ids.POLS_VOICE_OBJECT_TYPE == 0x16
    assert ids.GIBDO_OBJECT_TYPE == 0x30
    assert ids.BUBBLE_OBJECT_TYPE == 0x40
