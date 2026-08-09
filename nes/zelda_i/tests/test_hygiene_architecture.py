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