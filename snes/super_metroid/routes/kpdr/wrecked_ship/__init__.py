"""K6 Alpha Power Bomb return toward Crateria and Moat."""

from super_metroid.routes.kpdr.wrecked_ship.alpha_pb_escape import play_alpha_pb_to_caterpillar
from super_metroid.routes.kpdr.wrecked_ship.caterpillar_climb import play_caterpillar_to_elevator
from super_metroid.routes.kpdr.wrecked_ship.elevator_to_kihunter import play_elevator_to_kihunter
from super_metroid.routes.kpdr.wrecked_ship.gravity_collect import (
    play_attic_to_west_ocean,
    play_bowling_to_gravity,
    play_gravity_collect,
    play_homing_geemer_to_bowling,
    play_pancakes_to_homing_geemer,
    play_west_ocean_to_pancakes,
)
from super_metroid.routes.kpdr.wrecked_ship.kihunter_to_moat import play_kihunter_to_moat
from super_metroid.routes.kpdr.wrecked_ship.phantoon_fight import play_phantoon_room_fight
from super_metroid.routes.kpdr.wrecked_ship.phantoon_leave import play_phantoon_loot_exit
from super_metroid.routes.kpdr.wrecked_ship.ws_basement import play_ws_basement_to_phantoon
from super_metroid.routes.kpdr.wrecked_ship.ws_basement_return import play_ws_basement_to_main
from super_metroid.routes.kpdr.wrecked_ship.ws_entrance import play_ws_entrance_to_main
from super_metroid.routes.kpdr.wrecked_ship.ws_main import play_ws_main_to_basement
from super_metroid.routes.kpdr.wrecked_ship.ws_main_climb import play_ws_main_to_attic

__all__ = [
    "play_alpha_pb_to_caterpillar",
    "play_attic_to_west_ocean",
    "play_bowling_to_gravity",
    "play_caterpillar_to_elevator",
    "play_elevator_to_kihunter",
    "play_gravity_collect",
    "play_homing_geemer_to_bowling",
    "play_kihunter_to_moat",
    "play_pancakes_to_homing_geemer",
    "play_phantoon_loot_exit",
    "play_phantoon_room_fight",
    "play_west_ocean_to_pancakes",
    "play_ws_basement_to_main",
    "play_ws_basement_to_phantoon",
    "play_ws_entrance_to_main",
    "play_ws_main_to_attic",
    "play_ws_main_to_basement",
]
