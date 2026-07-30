"""KPDR pure-controller segments (split by route geography).

Public surface matches the historical ``kpdr_controller`` module. Prefer
importing from ``super_metroid.routes.kpdr`` or the thin re-export
``super_metroid.routes.kpdr_controller``.
"""

from __future__ import annotations

from super_metroid.routes.controller_common import play_run_shoot_exit
from super_metroid.routes.kpdr.big_pink import play_big_pink_to_ghz
from super_metroid.routes.kpdr.green_hill import (
    play_ghz_to_noob,
    play_noob_to_red_tower,
)
from super_metroid.routes.kpdr.hijump import (
    play_business_to_hj_shaft,
    play_business_to_warehouse,
    play_hijump_to_warehouse,
    play_hj_room_collect,
    play_hj_room_to_shaft,
    play_hj_shaft_to_business,
    play_hj_shaft_to_hj_room,
    play_warehouse_to_hijump,
)
from super_metroid.routes.kpdr.kraid_approach import (
    play_baby_kraid_to_eye,
    play_eye_to_kraid,
    play_kihunter_to_baby_kraid,
    play_warehouse_hijump_kraid,
    play_warehouse_to_kraid_with_hijump,
    play_warehouse_to_zeela_with_hijump,
    play_zeela_to_kihunter,
)
from super_metroid.routes.kpdr.red_tower import (
    play_bat_to_below_spazer,
    play_below_spazer_to_west,
    play_east_to_warehouse,
    play_glass_to_east,
    play_red_tower_to_bat,
    play_red_tower_to_warehouse,
    play_west_to_glass,
)
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS, get_segment
from super_metroid.routes.kpdr.rooms import (
    ITEM_HI_JUMP,
    ROOM_BABY_KRAID,
    ROOM_BAT,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BUSINESS,
    ROOM_EAST_TUNNEL,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_NOOB,
    ROOM_RED_TOWER,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
)
from super_metroid.routes.kpdr.warehouse import (
    play_warehouse_to_business,
    play_warehouse_wall_to_lower_lip,
)

__all__ = [
    "ITEM_HI_JUMP",
    "KPDR_SEGMENTS",
    "ROOM_BABY_KRAID",
    "ROOM_BAT",
    "ROOM_BELOW_SPAZER",
    "ROOM_BIG_PINK",
    "ROOM_BUSINESS",
    "ROOM_EAST_TUNNEL",
    "ROOM_GHZ",
    "ROOM_GLASS",
    "ROOM_HJ",
    "ROOM_HJ_SHAFT",
    "ROOM_KRAID",
    "ROOM_KRAID_EYE",
    "ROOM_NOOB",
    "ROOM_RED_TOWER",
    "ROOM_WAREHOUSE",
    "ROOM_WAREHOUSE_KIHUNTER",
    "ROOM_WEST_TUNNEL",
    "ROOM_ZEELA",
    "get_segment",
    "play_baby_kraid_to_eye",
    "play_bat_to_below_spazer",
    "play_below_spazer_to_west",
    "play_big_pink_to_ghz",
    "play_business_to_hj_shaft",
    "play_business_to_warehouse",
    "play_east_to_warehouse",
    "play_eye_to_kraid",
    "play_ghz_to_noob",
    "play_glass_to_east",
    "play_hijump_to_warehouse",
    "play_hj_room_collect",
    "play_hj_room_to_shaft",
    "play_hj_shaft_to_business",
    "play_hj_shaft_to_hj_room",
    "play_kihunter_to_baby_kraid",
    "play_noob_to_red_tower",
    "play_red_tower_to_bat",
    "play_red_tower_to_warehouse",
    "play_run_shoot_exit",
    "play_warehouse_hijump_kraid",
    "play_warehouse_to_business",
    "play_warehouse_to_hijump",
    "play_warehouse_to_kraid_with_hijump",
    "play_warehouse_to_zeela_with_hijump",
    "play_warehouse_wall_to_lower_lip",
    "play_west_to_glass",
    "play_zeela_to_kihunter",
]
