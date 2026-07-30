"""KPDR segment registry: tracker id ↔ controller callable.

Maps living ``KPDR_TRACKER.csv`` / hop ids to pure controller entry points
where a 1:1 controller exists. Dev door-warp hops without a pure segment
are omitted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
from super_metroid.routes.kpdr.warehouse import (
    play_warehouse_to_business,
    play_warehouse_wall_to_lower_lip,
)

SegmentFn = Callable[..., Any]

# Segment ids used by pure probes / tracker correlation.
KPDR_SEGMENTS: dict[str, SegmentFn] = {
    "big_pink_to_ghz": play_big_pink_to_ghz,
    "ghz_to_noob": play_ghz_to_noob,
    "noob_to_red_tower": play_noob_to_red_tower,
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
    "warehouse_to_kraid_with_hijump": play_warehouse_to_kraid_with_hijump,
    "warehouse_hijump_kraid": play_warehouse_hijump_kraid,
}


def get_segment(segment_id: str) -> SegmentFn:
    """Return the controller callable for ``segment_id`` or raise ``KeyError``."""
    return KPDR_SEGMENTS[segment_id]
