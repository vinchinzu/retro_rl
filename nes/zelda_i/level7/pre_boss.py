"""Level 7 pre-boss 0x29 east BOMB wall → live Aquamentus 0x2A.

Geometry from ``scratch/probe_l7_7b_cellar_cross.py`` (``EAST_BOMB_STAND`` /
``EAST_BOMB_APPROACH``). Configures ``BombWallController``; no new phase
machine. Factory does not write bombs — Survival top-up stays the
probe/spine's job.
"""

from __future__ import annotations

from zelda_i.dungeon.bomb_wall import BombWallController
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS
from zelda_i.level7.cellar import DEST_ROOM
from zelda_i.level7.path import Level7BombWall
from zelda_i.level7.stairs import AQUAMENTUS_ROM

__all__ = [
    "L7_ROOM29_EAST_APPROACH",
    "L7_ROOM29_EAST_BOMB",
    "make_room29_east_bomb_controller",
]

LEVEL7 = 7

# 0x29 PRE_BOSS (cellar DEST_ROOM / graph ram_id) east BOMB wall -> live 0x2A.
# South-around from the cellar leftover: (96,189)->(208,189)->(208,141) face
# RIGHT. Probe ``EAST_BOMB_STAND`` / ``EAST_BOMB_APPROACH``.
L7_ROOM29_EAST_BOMB = Level7BombWall(
    room=DEST_ROOM, stand=(208, 141), face="RIGHT", opens_to=AQUAMENTUS_ROM
)
L7_ROOM29_EAST_APPROACH = ((96, 189), (208, 189), (208, 141))


def make_room29_east_bomb_controller() -> BombWallController:
    """0x29 PRE_BOSS east BOMB wall → live 0x2A Aquamentus.

    South-around to stand (208,141) face RIGHT. Needs bombs + bomb on B.
    Factory does not write bombs. Recon-wired only.
    """
    return BombWallController(
        wall=L7_ROOM29_EAST_BOMB,
        level=LEVEL7,
        approach_waypoints=L7_ROOM29_EAST_APPROACH,
        approach_tol=4,
        select_item=B_SLOT_BOMBS,
    )
