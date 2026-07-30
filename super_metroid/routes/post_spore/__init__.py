"""Post-Spore Super collect and Pink PB controllers (split modules)."""

from __future__ import annotations

from super_metroid.routes.controller_common import (
    MORPH_POSES,
    ensure_morph,
    is_morph,
    wait_until,
)
from super_metroid.routes.post_spore.morph_bomb_roll import (
    MorphBombRollPhase,
    bomb_roll_left_safe,
)
from super_metroid.routes.post_spore.pb_door import (
    play_big_pink_enter_pb_door_from_sill,
    play_big_pink_enter_pb_door_from_top_ledge,
)
from super_metroid.routes.post_spore.pink_pb_maze import (
    play_pink_pb_break_maze_wall,
    play_pink_pb_from_left_zone,
    play_pink_pb_mid_maze_to_collect,
    play_pink_pb_morph_bomb_collect,
)
from super_metroid.routes.post_spore.rooms import (
    ROOM_BIG_PINK,
    ROOM_FARMING,
    ROOM_PINK_PB,
    ROOM_SUPER,
    PowerBombEvidence,
    SuperCollectEvidence,
)
from super_metroid.routes.post_spore.supers_collect import (
    play_big_pink_bomb_to_walkway_edge,
    play_big_pink_clear_super_block,
    play_big_pink_crest_pocket,
    play_big_pink_drop_to_pocket,
    play_big_pink_into_main_shaft,
    play_big_pink_morph_to_tunnel,
    play_big_pink_tunnel_west,
    play_farming_to_big_pink,
    play_post_spore_supers,
    play_super_room_collect,
    play_super_room_to_farming,
)

__all__ = [
    "MORPH_POSES",
    "MorphBombRollPhase",
    "PowerBombEvidence",
    "ROOM_BIG_PINK",
    "ROOM_FARMING",
    "ROOM_PINK_PB",
    "ROOM_SUPER",
    "SuperCollectEvidence",
    "bomb_roll_left_safe",
    "ensure_morph",
    "is_morph",
    "play_big_pink_bomb_to_walkway_edge",
    "play_big_pink_clear_super_block",
    "play_big_pink_crest_pocket",
    "play_big_pink_drop_to_pocket",
    "play_big_pink_enter_pb_door_from_sill",
    "play_big_pink_enter_pb_door_from_top_ledge",
    "play_big_pink_into_main_shaft",
    "play_big_pink_morph_to_tunnel",
    "play_big_pink_tunnel_west",
    "play_farming_to_big_pink",
    "play_pink_pb_break_maze_wall",
    "play_pink_pb_from_left_zone",
    "play_pink_pb_mid_maze_to_collect",
    "play_pink_pb_morph_bomb_collect",
    "play_post_spore_supers",
    "play_super_room_collect",
    "play_super_room_to_farming",
    "wait_until",
]
