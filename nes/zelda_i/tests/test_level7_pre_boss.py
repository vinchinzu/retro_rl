"""L7 0x29 east BOMB wall hop (`level7.pre_boss`).

Probe constants from ``scratch/probe_l7_7b_cellar_cross.py``: stand (208,141)
face RIGHT, approach (96,189)->(208,189)->(208,141), dest live 0x2A.
Factory configures ``BombWallController``; it does not write bombs.
"""

from __future__ import annotations

import ast

from zelda_i.dungeon.bomb_wall import BombWallController
from zelda_i.level7.cellar import DEST_ROOM
from zelda_i.level7.graph import LEVEL7_ROOM_BY_ID, PRE_BOSS
from zelda_i.level7.pre_boss import (
    L7_ROOM29_EAST_APPROACH,
    L7_ROOM29_EAST_BOMB,
    make_room29_east_bomb_controller,
)
from zelda_i.level7.stairs import AQUAMENTUS_ROM


def test_wall_geometry_matches_probe() -> None:
    assert DEST_ROOM == 0x29
    assert LEVEL7_ROOM_BY_ID[PRE_BOSS].ram_id == DEST_ROOM
    assert AQUAMENTUS_ROM == 0x2A
    assert L7_ROOM29_EAST_BOMB.room == DEST_ROOM == 0x29
    assert L7_ROOM29_EAST_BOMB.stand == (208, 141)
    assert L7_ROOM29_EAST_BOMB.face == "RIGHT"
    assert L7_ROOM29_EAST_BOMB.opens_to == AQUAMENTUS_ROM == 0x2A
    assert L7_ROOM29_EAST_APPROACH == ((96, 189), (208, 189), (208, 141))


def test_factory_returns_configured_bomb_wall_controller() -> None:
    ctl = make_room29_east_bomb_controller()
    assert isinstance(ctl, BombWallController)
    assert ctl.level == 7
    assert ctl.wall is L7_ROOM29_EAST_BOMB
    assert ctl.wall.room == 0x29
    assert ctl.wall.opens_to == 0x2A
    assert ctl.stand == (208, 141)
    assert ctl.face == "RIGHT"
    assert ctl.approach_waypoints == L7_ROOM29_EAST_APPROACH
    assert ctl.approach_waypoints[0] == (96, 189)
    assert ctl.approach_tol == 4


def test_factory_never_shares_instances() -> None:
    assert make_room29_east_bomb_controller() is not (
        make_room29_east_bomb_controller()
    )


def test_factory_does_not_write_bombs() -> None:
    import zelda_i.level7.pre_boss as pre_boss

    source_path = pre_boss.__file__
    assert source_path is not None
    tree = ast.parse(open(source_path, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            imported.add(node.id)
    assert "apply_owned_inventory" not in imported
    assert "ADDR_BOMBS" not in imported
    assert "BombWallPhase" not in imported
