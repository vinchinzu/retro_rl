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
    assert "level6_stairs3a_success" not in l6s.__all__
    spine_lines = open(l6s.__file__, encoding="utf-8").read().count("\n") + 1
    assert spine_lines < 800, spine_lines


def test_level6_north39_is_own_module_and_spine_under_800() -> None:
    import zelda_i.level4_spine as l4s
    import zelda_i.level6_dungeon as l6d
    import zelda_i.level6_north39 as north39
    import zelda_i.level6_inland29 as inland29
    import zelda_i.level6_path as l6p
    import zelda_i.level6_spine as l6s
    import zelda_i.level6_spine_suffix as suffix
    import zelda_i.level6_stairs3a_warp as stairs3a_warp
    import zelda_i.level6_cellar08 as cellar08
    import zelda_i.level6_east3a as east3a
    import zelda_i.level6_west19 as west19
    import zelda_i.level6_south18 as south18
    import zelda_i.level6_south1d as south1d
    import zelda_i.level6_west2d as west2d
    import zelda_i.level6_north2c as north2c
    import zelda_i.survival_spine as spine

    assert "make_north39_controller" not in l6d.__dict__
    assert "make_north39_controller" not in l6p.__dict__
    assert "make_stairs_3a_warp_controller" not in l6d.__dict__
    assert "make_stairs_3a_warp_controller" not in l6p.__dict__
    assert "make_cellar08_controller" not in l6d.__dict__
    assert "make_cellar08_controller" not in l6p.__dict__
    assert "make_east3a_controller" not in l6d.__dict__
    assert "make_east3a_controller" not in l6p.__dict__
    assert "make_inland29_controller" not in l6d.__dict__
    assert "make_inland29_controller" not in l6p.__dict__
    assert "make_west19_controller" not in l6d.__dict__
    assert "make_west19_controller" not in l6p.__dict__
    assert "make_south18_controller" not in l6d.__dict__
    assert "make_south18_controller" not in l6p.__dict__
    assert "make_south1d_controller" not in l6d.__dict__
    assert "make_south1d_controller" not in l6p.__dict__
    assert "make_west2d_controller" not in l6d.__dict__
    assert "make_west2d_controller" not in l6p.__dict__
    assert "make_north2c_controller" not in l6d.__dict__
    assert "make_north2c_controller" not in l6p.__dict__
    assert north39.make_north39_controller().spec_id == "level6_north39_0x29"
    assert inland29.make_inland29_controller().spec_id == "level6_inland_0x29"
    assert west19.make_west19_controller().spec_id == "level6_west_0x19"
    assert south18.make_south18_controller().spec_id == "level6_south_0x18"
    assert south1d.make_south1d_controller().spec_id == "level6_south_0x1d"
    assert west2d.make_west2d_controller().spec_id == "level6_west_0x2d"
    assert north2c.make_north2c_controller().spec_id == "level6_north_0x2c"
    assert l6s.level6_stairs3a_warp_success is stairs3a_warp.level6_stairs3a_warp_success
    assert suffix.level6_stairs3a_warp_success is stairs3a_warp.level6_stairs3a_warp_success
    assert stairs3a_warp.make_stairs_3a_warp_controller().spec_id == (
        "level6_stairs_0x3a_warp"
    )
    assert l6s.level6_cellar08_success is cellar08.level6_cellar08_success
    assert suffix.level6_cellar08_success is cellar08.level6_cellar08_success
    assert cellar08.make_cellar08_controller().spec_id == "level6_cellar_0x08"
    assert l6s.level6_east3a_success is east3a.level6_east3a_success
    assert suffix.level6_east3a_success is east3a.level6_east3a_success
    assert east3a.make_east3a_controller().spec_id == "level6_east_0x3a"
    assert l6s.level6_north39_success is north39.level6_north39_success
    assert suffix.level6_north39_success is north39.level6_north39_success
    assert l6s.level6_inland29_success is inland29.level6_inland29_success
    assert suffix.level6_inland29_success is inland29.level6_inland29_success
    assert l6s.level6_west19_success is west19.level6_west19_success
    assert suffix.level6_west19_success is west19.level6_west19_success
    assert l6s.level6_south18_success is south18.level6_south18_success
    assert suffix.level6_south18_success is south18.level6_south18_success
    assert l6s.level6_south1d_success is south1d.level6_south1d_success
    assert suffix.level6_south1d_success is south1d.level6_south1d_success
    assert l6s.level6_west2d_success is west2d.level6_west2d_success
    assert suffix.level6_west2d_success is west2d.level6_west2d_success
    assert l6s.level6_north2c_success is north2c.level6_north2c_success
    assert suffix.level6_north2c_success is north2c.level6_north2c_success
    src = open(suffix.__file__, encoding="utf-8").read()
    assert '"level6-stairs3a-warp"' in src and "True)" in src
    assert '"level6-cellar08"' in src and "True)" in src
    assert '"level6-south1d"' in src and "True)" in src
    assert '"level6-west2d"' in src and "True)" in src
    assert '"level6-north2c"' in src and "True)" in src
    assert '"level6-east3a"' in src and "True)" in src
    assert '"level6-north39"' in src
    assert '"level6-inland29"' in src
    assert '"level6-west19"' in src
    assert '"level6-south18"' in src
    assert '"level6-stairs3a"' not in src
    assert '"level6-center3a"' not in src
    assert '"level6-exit-ow"' not in src
    assert "level6_stairs3a_warp_success, True)" in src
    assert "level6_cellar08_success, True)" in src
    assert "level6_south1d_success, True)" in src
    assert "level6_west2d_success, True)" in src
    assert "level6_north2c_success, True)" in src
    assert "level6_east3a_success, True)" in src
    assert "level6_exit_ow_success, True)" not in src
    for mod in (spine, l6s, suffix, l4s):
        lines = open(mod.__file__, encoding="utf-8").read().count("\n") + 1
        assert lines < 800, (mod.__name__, lines)


def test_level1_bow_is_own_module() -> None:
    import zelda_i.dungeon as dungeon
    import zelda_i.level1_bow as l1bow
    import zelda_i.level1_bow_cellar as l1cellar
    import zelda_i.level1_finish as l1f

    assert "make_bow22_controller" not in dungeon.__dict__
    assert "Level1Bow22Controller" not in l1f.__dict__
    assert "make_bow_cellar_controller" not in l1bow.__dict__
    assert l1bow.make_bow22_controller().spec_id == "level1_bow_0x22"
    assert l1cellar.make_bow_cellar_controller().spec_id == "level1_bow_cellar"
    assert open(l1bow.__file__, encoding="utf-8").read().count("\n") + 1 < 500
    assert open(l1cellar.__file__, encoding="utf-8").read().count("\n") + 1 < 500


def test_dungeon_ids_has_l4_l5_enemy_types() -> None:
    from zelda_i import dungeon_ids as ids

    assert ids.LIKE_LIKE_OBJECT_TYPE == 0x17
    assert ids.POLS_VOICE_OBJECT_TYPE == 0x16
    assert ids.GIBDO_OBJECT_TYPE == 0x30
    assert ids.BUBBLE_OBJECT_TYPE == 0x40


def test_level6_door_hop_is_shared_controller() -> None:
    import zelda_i.level6_door_hop as door_hop
    import zelda_i.level6_east29 as east29
    import zelda_i.level6_east39 as east39
    import zelda_i.level6_south09 as south09
    import zelda_i.level6_south18 as south18
    import zelda_i.level6_south19 as south19
    import zelda_i.level6_south1d as south1d
    import zelda_i.level6_south29 as south29
    import zelda_i.level6_west19 as west19
    import zelda_i.level6_west2d as west2d
    import zelda_i.level6_north2c as north2c

    hops = (
        south09, south19, south29, east29, east39, west19, south18, south1d,
        west2d, north2c,
    )
    makers = (
        south09.make_south09_controller,
        south19.make_south19_controller,
        south29.make_south29_controller,
        east29.make_east29_controller,
        east39.make_east39_controller,
        west19.make_west19_controller,
        south18.make_south18_controller,
        south1d.make_south1d_controller,
        west2d.make_west2d_controller,
        north2c.make_north2c_controller,
    )
    for mod, make in zip(hops, makers, strict=True):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "level6_door_hop" in src
        ctl = make()
        assert isinstance(ctl, door_hop.Level6DoorHopController)
    lines = open(door_hop.__file__, encoding="utf-8").read().count("\n") + 1
    assert lines < 500, lines
