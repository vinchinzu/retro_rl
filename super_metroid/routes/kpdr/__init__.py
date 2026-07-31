"""KPDR pure-controller segments (full continuous spine after power-on).

Includes Spore Super collect → Big Pink (formerly ``post_spore/``) through
Hi-Jump and Kraid entry. Public surface matches the historical
``kpdr_controller`` / ``post_spore_controller`` modules. Prefer importing
from ``super_metroid.routes.kpdr`` or the thin re-exports
``super_metroid.routes.kpdr_controller`` /
``super_metroid.routes.post_spore_controller``.
"""

from __future__ import annotations

from super_metroid.routes.controller_common import (
    MORPH_POSES,
    ensure_morph,
    is_morph,
    play_run_shoot_exit,
    wait_until,
)
from super_metroid.routes.kpdr.big_pink import play_big_pink_to_ghz
from super_metroid.routes.kpdr.big_pink_shaft import (
    play_big_pink_bomb_to_walkway_edge,
    play_big_pink_clear_super_block,
    play_big_pink_crest_pocket,
    play_big_pink_drop_to_pocket,
    play_big_pink_into_main_shaft,
    play_big_pink_morph_to_tunnel,
    play_big_pink_tunnel_west,
)
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
    play_kraid_entry_to_varia,
    play_warehouse_hijump_kraid,
    play_warehouse_to_kraid_with_hijump,
    play_warehouse_to_zeela_with_hijump,
    play_zeela_to_kihunter,
)
from super_metroid.routes.kpdr.kraid_return import (
    play_baby_to_kihunter_return,
    play_eye_to_baby_return,
    play_kihunter_to_zeela_return,
    play_zeela_to_warehouse_return,
)
from super_metroid.routes.kpdr.morph_bomb_roll import (
    MorphBombRollPhase,
    bomb_roll_left_safe,
)
from super_metroid.routes.kpdr.pb_door import (
    play_big_pink_enter_pb_door_from_sill,
    play_big_pink_enter_pb_door_from_top_ledge,
)
from super_metroid.routes.kpdr.pink_pb_maze import (
    play_pink_pb_break_maze_wall,
    play_pink_pb_from_left_zone,
    play_pink_pb_mid_maze_to_collect,
    play_pink_pb_morph_bomb_collect,
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
    ROOM_FARMING,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_NOOB,
    ROOM_PINK_PB,
    ROOM_RED_TOWER,
    ROOM_SUPER,
    ROOM_VARIA,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
    PowerBombEvidence,
    SuperCollectEvidence,
)
from super_metroid.routes.kpdr.super_room import (
    play_farming_to_big_pink,
    play_post_spore_supers,
    play_super_room_collect,
    play_super_room_to_farming,
)
from super_metroid.routes.kpdr.varia_return import (
    play_kraid_to_eye_return,
    play_varia_to_kraid,
)
from super_metroid.routes.kpdr.warehouse import (
    play_warehouse_to_business,
    play_warehouse_wall_to_lower_lip,
)

__all__ = [
    "ITEM_HI_JUMP",
    "KPDR_SEGMENTS",
    "MORPH_POSES",
    "MorphBombRollPhase",
    "PowerBombEvidence",
    "ROOM_BABY_KRAID",
    "ROOM_BAT",
    "ROOM_BELOW_SPAZER",
    "ROOM_BIG_PINK",
    "ROOM_BUSINESS",
    "ROOM_EAST_TUNNEL",
    "ROOM_FARMING",
    "ROOM_GHZ",
    "ROOM_GLASS",
    "ROOM_HJ",
    "ROOM_HJ_SHAFT",
    "ROOM_KRAID",
    "ROOM_KRAID_EYE",
    "ROOM_NOOB",
    "ROOM_PINK_PB",
    "ROOM_RED_TOWER",
    "ROOM_SUPER",
    "ROOM_VARIA",
    "ROOM_WAREHOUSE",
    "ROOM_WAREHOUSE_KIHUNTER",
    "ROOM_WEST_TUNNEL",
    "ROOM_ZEELA",
    "SuperCollectEvidence",
    "bomb_roll_left_safe",
    "ensure_morph",
    "get_segment",
    "is_morph",
    "play_baby_kraid_to_eye",
    "play_bat_to_below_spazer",
    "play_below_spazer_to_west",
    "play_big_pink_bomb_to_walkway_edge",
    "play_big_pink_clear_super_block",
    "play_big_pink_crest_pocket",
    "play_big_pink_drop_to_pocket",
    "play_big_pink_enter_pb_door_from_sill",
    "play_big_pink_enter_pb_door_from_top_ledge",
    "play_big_pink_into_main_shaft",
    "play_big_pink_morph_to_tunnel",
    "play_big_pink_to_ghz",
    "play_big_pink_tunnel_west",
    "play_business_to_hj_shaft",
    "play_business_to_warehouse",
    "play_east_to_warehouse",
    "play_eye_to_kraid",
    "play_eye_to_baby_return",
    "play_kraid_entry_to_varia",
    "play_kraid_to_eye_return",
    "play_farming_to_big_pink",
    "play_ghz_to_noob",
    "play_glass_to_east",
    "play_hijump_to_warehouse",
    "play_hj_room_collect",
    "play_hj_room_to_shaft",
    "play_hj_shaft_to_business",
    "play_hj_shaft_to_hj_room",
    "play_kihunter_to_baby_kraid",
    "play_kihunter_to_zeela_return",
    "play_noob_to_red_tower",
    "play_pink_pb_break_maze_wall",
    "play_pink_pb_from_left_zone",
    "play_pink_pb_mid_maze_to_collect",
    "play_pink_pb_morph_bomb_collect",
    "play_post_spore_supers",
    "play_red_tower_to_bat",
    "play_red_tower_to_warehouse",
    "play_run_shoot_exit",
    "play_super_room_collect",
    "play_super_room_to_farming",
    "play_varia_to_kraid",
    "play_baby_to_kihunter_return",
    "play_warehouse_hijump_kraid",
    "play_warehouse_to_business",
    "play_warehouse_to_hijump",
    "play_warehouse_to_kraid_with_hijump",
    "play_warehouse_to_zeela_with_hijump",
    "play_warehouse_wall_to_lower_lip",
    "play_west_to_glass",
    "play_zeela_to_kihunter",
    "play_zeela_to_warehouse_return",
    "wait_until",
]
