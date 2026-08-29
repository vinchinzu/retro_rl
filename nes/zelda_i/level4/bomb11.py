"""Level 4 map leftover → bomb-UP 0x11 (waypoints, no live BFS).

v15 leftover play 0x21 (208,181) after ADDR_MAP|0x08. Inbound was spawn
RIGHT+UP to (48,93), RIGHT+DOWN clip, then east and DOWN onto the map.
bomb v1 leftover (192,109): cardinal LEFT at y=109 is a 16px pillar.
North-around y=93 then LEFT; isolated MAP_21_SAMPLE_PATH is not this tape.
"""

from __future__ import annotations

from zelda_i.dungeon.bomb_wall import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.level4.dungeon import (
    BOMB_21_NORTH_FACE,
    BOMB_21_NORTH_STAND,
    BOMB_21_OPENS_TO,
    LEVEL4,
    ROOM_L4_MAP_21,
    ROOM_L4_MID_11,
)
from zelda_i.level4.occupancy import ROOM_21_BOMB_WAYPOINTS
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "BombWall21North",
    "level4_bomb11_stages",
    "level4_bomb11_success",
    "make_bomb_21_north_controller",
]


class BombWall21North:
    """Geometry stand for ``BombWallController``: 0x21 bomb-UP → 0x11."""

    room = ROOM_L4_MAP_21
    stand = BOMB_21_NORTH_STAND
    face = BOMB_21_NORTH_FACE
    opens_to = BOMB_21_OPENS_TO


def make_bomb_21_north_controller() -> BombWallController:
    """0x21 leftover → waypoint corridor → bomb north → 0x11. No gel clear."""
    return BombWallController(
        wall=BombWall21North(),
        level=LEVEL4,
        approach_waypoints=ROOM_21_BOMB_WAYPOINTS,
        approach_tol=2,
        stand_tol=2,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=8000,
    )


def level4_bomb11_stages():
    bomb = make_bomb_21_north_controller()
    return (("level4_bomb_north_0x21", bomb, bomb.max_frames),)


def level4_bomb11_success(snap: ZeldaSnapshot) -> bool:
    """Exact enter-0x11 stop; type-0x35 cluster may stay live."""
    return (
        snap.level == LEVEL4
        and snap.screen == ROOM_L4_MID_11
        and snap.mode == PLAY_MODE
        and not snap.transitioning
    )
