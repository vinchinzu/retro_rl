"""KPDR segment registry: tracker id ↔ controller callable.

Maps living ``KPDR_TRACKER.csv`` / hop ids to pure controller entry points
where a 1:1 controller exists. Includes early post-Spore Super collect
through Big Pink (K0–K1). Dev door-warp hops without a pure segment are
omitted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from super_metroid.routes.kpdr.pink_to_ghz import play_big_pink_to_ghz
from super_metroid.routes.kpdr.pink_shaft import (
    play_big_pink_bomb_to_walkway_edge,
    play_big_pink_clear_super_block,
    play_big_pink_crest_pocket,
    play_big_pink_drop_to_pocket,
    play_big_pink_into_main_shaft,
    play_big_pink_morph_to_tunnel,
    play_big_pink_tunnel_west,
)
from super_metroid.routes.kpdr.ghz_to_red import (
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
from super_metroid.routes.kpdr.to_kraid import (
    play_baby_kraid_to_eye,
    play_eye_to_kraid,
    play_kihunter_to_baby_kraid,
    play_kraid_entry_to_varia,
    play_warehouse_hijump_kraid,
    play_warehouse_to_kraid_with_hijump,
    play_warehouse_to_zeela_with_hijump,
    play_zeela_to_kihunter,
)
from super_metroid.routes.kpdr.from_kraid import (
    play_baby_to_kihunter_return,
    play_eye_to_baby_return,
    play_kihunter_to_zeela_return,
    play_zeela_to_warehouse_return,
)
from super_metroid.routes.kpdr.to_bat_cave import (
    play_bubble_to_bat_cave,
)
from super_metroid.routes.kpdr.k4_norfair import (
    play_business_to_cathedral_entrance,
    play_business_to_frog_save,
    play_cathedral_entrance_to_cathedral,
    play_cathedral_to_rising_tide,
    play_farm_to_bubble,
    play_frog_save_to_business,
    play_frog_save_to_speedway,
    play_rising_tide_to_bubble,
    play_speedway_to_farm,
)
from super_metroid.routes.kpdr.pb_door import (
    play_big_pink_enter_pb_door_from_sill,
    play_big_pink_enter_pb_door_from_top_ledge,
)
from super_metroid.routes.kpdr.pink_pb import (
    play_pink_pb_break_maze_wall,
    play_pink_pb_from_left_zone,
    play_pink_pb_mid_maze_to_collect,
    play_pink_pb_morph_bomb_collect,
)
from super_metroid.routes.kpdr.red_stack import (
    play_bat_to_below_spazer,
    play_below_spazer_to_west,
    play_east_to_warehouse,
    play_glass_to_east,
    play_red_tower_to_bat,
    play_red_tower_to_warehouse,
    play_west_to_glass,
)
from super_metroid.routes.kpdr.super_collect import (
    play_farming_to_big_pink,
    play_post_spore_supers,
    play_super_room_collect,
    play_super_room_to_farming,
)
from super_metroid.routes.kpdr.varia_return import (
    play_kraid_to_eye_return,
    play_varia_to_kraid,
)
from super_metroid.routes.kpdr.warehouse_stack import (
    play_warehouse_to_business,
    play_warehouse_wall_to_lower_lip,
)

SegmentFn = Callable[..., Any]

# Segment ids used by pure probes / tracker correlation.
# Order roughly follows continuous KPDR spine (K0 Super → K2 Kraid).
KPDR_SEGMENTS: dict[str, SegmentFn] = {
    # K0: Spore Super collect → Big Pink main (was post_spore/)
    "super_room_collect": play_super_room_collect,
    "super_room_to_farming": play_super_room_to_farming,
    "farming_to_big_pink": play_farming_to_big_pink,
    "post_spore_supers": play_post_spore_supers,
    "big_pink_crest_pocket": play_big_pink_crest_pocket,
    "big_pink_clear_super_block": play_big_pink_clear_super_block,
    "big_pink_morph_to_tunnel": play_big_pink_morph_to_tunnel,
    "big_pink_tunnel_west": play_big_pink_tunnel_west,
    "big_pink_drop_to_pocket": play_big_pink_drop_to_pocket,
    "big_pink_bomb_to_walkway_edge": play_big_pink_bomb_to_walkway_edge,
    "big_pink_into_main_shaft": play_big_pink_into_main_shaft,
    # Parked Pink PB (not competitive KPDR; optional backfill)
    "big_pink_enter_pb_door_from_sill": play_big_pink_enter_pb_door_from_sill,
    "big_pink_enter_pb_door_from_top_ledge": play_big_pink_enter_pb_door_from_top_ledge,
    "pink_pb_break_maze_wall": play_pink_pb_break_maze_wall,
    "pink_pb_morph_bomb_collect": play_pink_pb_morph_bomb_collect,
    "pink_pb_mid_maze_to_collect": play_pink_pb_mid_maze_to_collect,
    "pink_pb_from_left_zone": play_pink_pb_from_left_zone,
    # K1: Big Pink → Red Tower
    "big_pink_to_ghz": play_big_pink_to_ghz,
    "ghz_to_noob": play_ghz_to_noob,
    "noob_to_red_tower": play_noob_to_red_tower,
    # K2: Red Tower → Hi-Jump → Kraid
    "red_tower_to_bat": play_red_tower_to_bat,
    "bat_to_below_spazer": play_bat_to_below_spazer,
    "below_spazer_to_west": play_below_spazer_to_west,
    "west_to_glass": play_west_to_glass,
    "glass_to_east": play_glass_to_east,
    "east_to_warehouse": play_east_to_warehouse,
    "red_tower_to_warehouse": play_red_tower_to_warehouse,
    "warehouse_wall_to_lower_lip": play_warehouse_wall_to_lower_lip,
    "warehouse_to_business": play_warehouse_to_business,
    "business_to_hj_shaft": play_business_to_hj_shaft,
    "hj_shaft_to_hj_room": play_hj_shaft_to_hj_room,
    "hj_room_collect": play_hj_room_collect,
    "warehouse_to_hijump": play_warehouse_to_hijump,
    "hj_room_to_shaft": play_hj_room_to_shaft,
    "hj_shaft_to_business": play_hj_shaft_to_business,
    "business_to_warehouse": play_business_to_warehouse,
    "hijump_to_warehouse": play_hijump_to_warehouse,
    "warehouse_to_zeela_with_hijump": play_warehouse_to_zeela_with_hijump,
    "zeela_to_kihunter": play_zeela_to_kihunter,
    "kihunter_to_baby_kraid": play_kihunter_to_baby_kraid,
    "baby_kraid_to_eye": play_baby_kraid_to_eye,
    "eye_to_kraid": play_eye_to_kraid,
    "kraid_entry_to_varia": play_kraid_entry_to_varia,
    "warehouse_to_kraid_with_hijump": play_warehouse_to_kraid_with_hijump,
    "warehouse_hijump_kraid": play_warehouse_hijump_kraid,
    # K4 prefix: first natural post-Varia door (pure/controller_dev)
    "varia_to_kraid": play_varia_to_kraid,
    "kraid_to_eye_return": play_kraid_to_eye_return,
    "eye_to_baby_return": play_eye_to_baby_return,
    "baby_to_kihunter_return": play_baby_to_kihunter_return,
    "kihunter_to_zeela_return": play_kihunter_to_zeela_return,
    "zeela_to_warehouse_return": play_zeela_to_warehouse_return,
    # K4.0: continuous through Frog Save (save milestone).
    "business_to_frog_save": play_business_to_frog_save,
    "frog_save_to_business": play_frog_save_to_business,
    # First Bubble visit: Cathedral climb (no Speed).
    "business_to_cathedral_entrance": play_business_to_cathedral_entrance,
    "cathedral_entrance_to_cathedral": play_cathedral_entrance_to_cathedral,
    "cathedral_to_rising_tide": play_cathedral_to_rising_tide,
    "rising_tide_to_bubble": play_rising_tide_to_bubble,
    "bubble_to_bat_cave": play_bubble_to_bat_cave,
    # Post-Speed shortcut only (Boost Blocks).
    "frog_save_to_speedway": play_frog_save_to_speedway,
    "speedway_to_farm": play_speedway_to_farm,
    "farm_to_bubble": play_farm_to_bubble,
}


def get_segment(segment_id: str) -> SegmentFn:
    """Return the controller callable for ``segment_id`` or raise ``KeyError``."""
    return KPDR_SEGMENTS[segment_id]
