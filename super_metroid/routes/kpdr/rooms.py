"""KPDR room ids and shared segment helpers."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.runtime import ControllerSession

# Private aliases matching historical controller style.
_hold = hold
_require_room = require_room
_select_weapon = select_weapon
_unmorph = unmorph
_wait_ordinary_room = wait_ordinary_room


ROOM_GHZ = 0x9E52
ROOM_NOOB = 0x9FBA
ROOM_RED_TOWER = 0xA253
ROOM_BAT = 0xA3DD
ROOM_BIG_PINK = 0x9D19
ROOM_BELOW_SPAZER = 0xA408
ROOM_WEST_TUNNEL = 0xCF54
ROOM_GLASS = 0xCEFB
ROOM_EAST_TUNNEL = 0xCF80
ROOM_WAREHOUSE = 0xA6A1
ROOM_BUSINESS = 0xA7DE
ROOM_HJ_SHAFT = 0xAA41
ROOM_HJ = 0xA9E5
ROOM_ZEELA = 0xA471
ROOM_WAREHOUSE_KIHUNTER = 0xA4DA
ROOM_BABY_KRAID = 0xA521
ROOM_KRAID_EYE = 0xA56B
ROOM_KRAID = 0xA59F

ITEM_HI_JUMP = 0x0100

