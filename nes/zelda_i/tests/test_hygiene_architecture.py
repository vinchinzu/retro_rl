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
    Level2BombNorth1EController,
    Level2BombNorthController,
    Level2BoomBombNorthController,
    Level2PostBoomBombNorthController,
)
from zelda_i.level2_puzzles import BOMB_WALL_6F_NORTH
from zelda_i.later_nodes import SCREEN_LEVEL3_ENTRANCE as LN_L3


def test_anchors_are_single_source() -> None:
    assert SCREEN_LEVEL3_ENTRANCE == 0x74
    assert SCREEN_LEVEL5_DOOR is SCREEN_LEVEL5_ENTRANCE
    assert TRIFORCE_BITS_BY_LEVEL[3] == TF_BIT_L3 == 0x04
    assert ENTRANCES[3].verified
    assert not ENTRANCES[4].verified
    assert LN_L3 == SCREEN_LEVEL3_ENTRANCE


def test_bomb_wall_factories_share_engine() -> None:
    c6 = Level2BombNorthController()
    c5 = Level2BoomBombNorthController(clear_gels=False)
    c4 = Level2PostBoomBombNorthController()
    c1 = Level2BombNorth1EController()
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
    from zelda_i.level2_dungeon import BOMB_N_STAND, Level2BombNorthController as C

    assert BOMB_N_STAND == (120, 101)
    assert C().wall.room == 0x6F


def test_level3_dungeon_reexports_path() -> None:
    from zelda_i.level3_dungeon import Level3RaftPathController

    assert Level3RaftPathController is not None
