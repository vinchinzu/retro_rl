"""KPDR room ids, Super-collect evidence, and shared segment helpers.

Includes the early post-Spore rooms (Super / Farming / Pink PB) that are part
of the continuous KPDR spine (K0–K1), not a separate vanilla-first route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

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

# K0 suffix / K1 prefix (Spore Super → Pink Brinstar)
ROOM_SUPER = 0x9B5B
ROOM_FARMING = 0xA0A4
ROOM_BIG_PINK = 0x9D19
ROOM_PINK_PB = 0x9E11

# K1–K2 Brinstar → Kraid
ROOM_GHZ = 0x9E52
ROOM_NOOB = 0x9FBA
ROOM_RED_TOWER = 0xA253
ROOM_BAT = 0xA3DD
ROOM_BELOW_SPAZER = 0xA408
ROOM_SPAZER = 0xA447
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
ROOM_VARIA = 0xA6E2

# K4: Varia return → Business → Bubble Mountain → Speed / Wave / Ice
# First Bubble visit is Cathedral climb (no Speed). Frog Speedway is post-Speed.
ROOM_FROG_SAVE = 0xB167
ROOM_FROG_SPEEDWAY = 0xB106
ROOM_UPPER_NORFAIR_FARM = 0xAF72
ROOM_CATHEDRAL_ENTRANCE = 0xA7B3
ROOM_CATHEDRAL = 0xA788
ROOM_RISING_TIDE = 0xAFA3
ROOM_BUBBLE = 0xACB3
ROOM_BAT_CAVE = 0xB07A
ROOM_SPEED_HALL = 0xACF0
ROOM_SPEED = 0xAD1B
ROOM_SINGLE_CHAMBER = 0xAD5E
ROOM_DOUBLE_CHAMBER = 0xADAD
ROOM_WAVE = 0xADDE
ROOM_ICE_GATE = 0xA815
ROOM_ICE_TUTORIAL = 0xA865
ROOM_ICE_SNAKE = 0xA8B9
ROOM_ICE = 0xA890

ITEM_HI_JUMP = 0x0100
ITEM_VARIA = 0x0001
ITEM_SPEED = 0x2000


@dataclass(frozen=True)
class SuperCollectEvidence:
    entry_frame: int
    collect_frame: int
    exit_frame: int | None
    max_super_missiles: int
    final_room_id: int
    samus_x: int
    samus_y: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PowerBombEvidence:
    entry_frame: int
    collect_frame: int | None
    max_super_missiles: int
    max_power_bombs: int
    final_room_id: int
    samus_x: int
    samus_y: int
    reached_big_pink: bool
    reached_pb_room: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

