"""Planning node ids + door screen candidates for Zelda I L3–L9.

**Not route-ready.** Screen hex values are source path arithmetic (see
``docs/OVERWORLD_DOORS.md``) until live fceumm probes confirm door tiles and
entry rooms. Do not import these into Clean STATUS or natural-entry claims.

L1/L2 anchors stay in ``zelda_i.overworld`` / ``zelda_i.ram`` — this module
avoids hot-path edits to those files.
"""

from __future__ import annotations

# --- Door screens (live assisted 2026-08-06 unless marked source) ---
SCREEN_LEVEL3_ENTRANCE = 0x74  # Manji — live; entry room 0x7c
SCREEN_LEVEL3_ENTRY_ROOM = 0x7C
# L4: source hyp only (needs raft from L3)
SCREEN_LEVEL4_ENTRANCE = 0x45  # Snake island candidate after raft
SCREEN_LEVEL4_RAFT_DOCK = 0x55  # mainland dock (source short path)
SCREEN_LEVEL5_ENTRANCE = 0x0B  # Lizard — live; entry room 0x76
SCREEN_LEVEL5_LOST_HILLS = 0x1B  # live ↑×4
SCREEN_LEVEL5_ENTRY_ROOM = 0x76
SCREEN_LEVEL6_ENTRANCE = 0x22  # Dragon — live; entry room 0x79
SCREEN_LEVEL6_ENTRY_ROOM = 0x79
SCREEN_LEVEL7_ENTRANCE = 0x42  # Demon; whistle pond (source)
SCREEN_LEVEL7_BAIT_SHOP = 0x34
SCREEN_LEVEL8_BUSH = 0x6D  # Lion bush pocket — live; needs candle to enter
SCREEN_LEVEL8_ENTRANCE = 0x6D  # same screen after burn (entry room TBD)
SCREEN_LEVEL9_ENTRANCE = 0x05  # Death Mountain; bomb rock (source)
# Blue Candle shop O-6 — live assisted OW path (rr-ccx); natural buy residual
SCREEN_CANDLE_SHOP = 0x5E

# --- OW capability candidates (source; TBD live) ---
SCREEN_BRACELET_ARMOS = 0x24
SCREEN_MAGICAL_SWORD_GRAVE = 0x21
SCREEN_RAFT_HEART_DOCK = 0x3F
SCREEN_LADDER_HEART = 0x5F

# --- Triforce bits (matches DUNGEON_WALKTHROUGHS / OVERWORLD_DOORS) ---
TF_BIT_L1 = 0x01
TF_BIT_L2 = 0x02
TF_BIT_L3 = 0x04
TF_BIT_L4 = 0x08
TF_BIT_L5 = 0x10
TF_BIT_L6 = 0x20
TF_BIT_L7 = 0x40
TF_BIT_L8 = 0x80
TF_BITS_ALL = 0xFF

TRIFORCE_BITS_BY_LEVEL: dict[int, int] = {
    1: TF_BIT_L1,
    2: TF_BIT_L2,
    3: TF_BIT_L3,
    4: TF_BIT_L4,
    5: TF_BIT_L5,
    6: TF_BIT_L6,
    7: TF_BIT_L7,
    8: TF_BIT_L8,
}

# --- Named graph nodes (planning placeholders) ---
NODE_LEVEL3_ENTRANCE = "ow_74_level3"
NODE_LEVEL3_DUNGEON = "dungeon_level3"
NODE_LEVEL3_COMPLETE = "level3_triforce_shard_3"

NODE_LEVEL4_ENTRANCE = "ow_45_level4"
NODE_LEVEL4_DUNGEON = "dungeon_level4"
NODE_LEVEL4_COMPLETE = "level4_triforce_shard_4"

NODE_LEVEL5_ENTRANCE = "ow_0b_level5"
NODE_LEVEL5_DUNGEON = "dungeon_level5"
NODE_LEVEL5_COMPLETE = "level5_triforce_shard_5"

NODE_LEVEL6_ENTRANCE = "ow_22_level6"
NODE_LEVEL6_DUNGEON = "dungeon_level6"
NODE_LEVEL6_COMPLETE = "level6_triforce_shard_6"

NODE_LEVEL7_ENTRANCE = "ow_42_level7"
NODE_LEVEL7_DUNGEON = "dungeon_level7"
NODE_LEVEL7_COMPLETE = "level7_triforce_shard_7"

NODE_LEVEL8_ENTRANCE = "ow_6d_level8"
NODE_LEVEL8_DUNGEON = "dungeon_level8"
NODE_LEVEL8_COMPLETE = "level8_triforce_shard_8"

NODE_LEVEL9_ENTRANCE = "ow_05_level9"
NODE_LEVEL9_DUNGEON = "dungeon_level9"
NODE_LEVEL9_GANON = "level9_ganon"
NODE_LEVEL9_ZELDA = "level9_zelda_ending"

# Optional OW prep nodes (planning)
NODE_BRACELET_ARMOS = "ow_24_bracelet_armos"
NODE_MAGICAL_SWORD = "ow_21_magical_sword"
NODE_BAIT_SHOP = "ow_34_bait_shop"
NODE_CANDLE_SHOP = "ow_5e_candle_shop"
NODE_WHISTLE_POND = "ow_42_whistle_pond"
NODE_LOST_HILLS = "ow_1b_lost_hills"
NODE_RAFT_L4_DOCK = "ow_55_raft_dock_l4"

DOOR_SCREEN_BY_LEVEL: dict[int, int] = {
    3: SCREEN_LEVEL3_ENTRANCE,
    4: SCREEN_LEVEL4_ENTRANCE,
    5: SCREEN_LEVEL5_ENTRANCE,
    6: SCREEN_LEVEL6_ENTRANCE,
    7: SCREEN_LEVEL7_ENTRANCE,
    8: SCREEN_LEVEL8_ENTRANCE,
    9: SCREEN_LEVEL9_ENTRANCE,
}

# SCREEN_LABELS-style map for docs / probes (do not merge into overworld.py yet)
LATER_SCREEN_LABELS: dict[int, str] = {
    SCREEN_LEVEL3_ENTRANCE: "level3_entrance_manji_source",
    SCREEN_LEVEL4_RAFT_DOCK: "level4_raft_dock_0x55_source",
    SCREEN_LEVEL4_ENTRANCE: "level4_island_door_0x45_source",
    SCREEN_LEVEL5_LOST_HILLS: "lost_hills_maze_source",
    SCREEN_LEVEL5_ENTRANCE: "level5_entrance_lizard_source",
    SCREEN_LEVEL6_ENTRANCE: "level6_entrance_dragon_source",
    SCREEN_LEVEL7_BAIT_SHOP: "bait_shop_armos_source",
    SCREEN_LEVEL7_ENTRANCE: "level7_entrance_demon_pond_source",
    SCREEN_LEVEL8_ENTRANCE: "level8_entrance_lion_bush_source",
    SCREEN_LEVEL9_ENTRANCE: "level9_entrance_bomb_rock_source",
    SCREEN_CANDLE_SHOP: "candle_shop_5e_live",
    SCREEN_BRACELET_ARMOS: "bracelet_armos_source",
    SCREEN_MAGICAL_SWORD_GRAVE: "magical_sword_grave_source",
    SCREEN_RAFT_HEART_DOCK: "raft_heart_dock_source",
    SCREEN_LADDER_HEART: "ladder_heart_coast_source",
}
