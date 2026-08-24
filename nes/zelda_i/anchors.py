"""Canonical OW entrance / capability / triforce anchors for Zelda I L3–L9.

Single source of truth for door screens, entry rooms, and TF bits. Level
overworld modules and ``later_nodes`` re-export from here — do not redefine
hex constants elsewhere.

``verified=True`` means live assisted recon; ``False`` is source hypothesis only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntranceAnchor:
    """Dungeon (or gated OW) entrance geometry."""

    level: int
    door_screen: int
    entry_room: int | None
    verified: bool
    notes: str = ""
    # Optional secondary screens (maze, dock, bush, shop).
    approach_screen: int | None = None
    label: str = ""


# --- Triforce bits (first quest) ---
TF_BIT_L1 = 0x01
TF_BIT_L2 = 0x02
TF_BIT_L3 = 0x04
TF_BIT_L4 = 0x08
TF_BIT_L5 = 0x10
TF_BIT_L6 = 0x20
TF_BIT_L7 = 0x40
TF_BIT_L8 = 0x80
TF_BITS_ALL = 0xFF
FULL_TRIFORCE = 0xFF

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

# --- Door / entry screens ---
SCREEN_LEVEL3_ENTRANCE = 0x74  # Manji — live
SCREEN_LEVEL3_ENTRY_ROOM = 0x7C
SCREEN_LEVEL4_ENTRANCE = 0x45  # Snake island door — live (rr-0fx)
SCREEN_LEVEL4_RAFT_DOCK = 0x55  # mainland raft dock — live (rr-0fx)
SCREEN_LEVEL4_ENTRY_ROOM = 0x71  # Snake entry room — live (rr-0fx)
SCREEN_LEVEL5_ENTRANCE = 0x0B  # Lizard — live
SCREEN_LEVEL5_LOST_HILLS = 0x1B  # live ↑×4
SCREEN_LEVEL5_ENTRY_ROOM = 0x76
SCREEN_LEVEL5_TF_ROOM = 0x14  # Digdogger north; TF bit 0x10
SCREEN_LEVEL6_ENTRANCE = 0x22  # Dragon — live
SCREEN_LEVEL6_ENTRY_ROOM = 0x79
SCREEN_LEVEL7_ENTRANCE = 0x42  # Demon pond (source)
SCREEN_LEVEL7_BAIT_SHOP = 0x34
SCREEN_LEVEL8_BUSH = 0x6D  # Lion bush — live; needs candle
SCREEN_LEVEL8_ENTRANCE = 0x6D
SCREEN_LEVEL9_ENTRANCE = 0x05  # Death Mountain bomb rock (source)
ROOM_LEVEL9_62 = 0x62  # candidate predecessor of final Patra 0x52
ROOM_LEVEL9_72 = 0x72  # hypothesized south neighbor / loader source
SCREEN_CANDLE_SHOP = 0x5E  # Blue Candle O-6 — live assisted

# --- OW capability candidates ---
SCREEN_BRACELET_ARMOS = 0x24
SCREEN_MAGICAL_SWORD_GRAVE = 0x21
SCREEN_RAFT_HEART_DOCK = 0x3F
SCREEN_LADDER_HEART = 0x5F

# Canonical entrance table
ENTRANCES: dict[int, EntranceAnchor] = {
    3: EntranceAnchor(
        level=3,
        door_screen=SCREEN_LEVEL3_ENTRANCE,
        entry_room=SCREEN_LEVEL3_ENTRY_ROOM,
        verified=True,
        label="manji",
        notes="Live assisted 2026-08-06; entry room 0x7c",
    ),
    4: EntranceAnchor(
        level=4,
        door_screen=SCREEN_LEVEL4_ENTRANCE,
        entry_room=SCREEN_LEVEL4_ENTRY_ROOM,
        verified=True,
        approach_screen=SCREEN_LEVEL4_RAFT_DOCK,
        label="snake",
        notes="Live assisted 2026-08-08 rr-0fx; dock 0x55 raft→0x45 door→room 0x71",
    ),
    5: EntranceAnchor(
        level=5,
        door_screen=SCREEN_LEVEL5_ENTRANCE,
        entry_room=SCREEN_LEVEL5_ENTRY_ROOM,
        verified=True,
        approach_screen=SCREEN_LEVEL5_LOST_HILLS,
        label="lizard",
        notes="Lost Hills UP×4 then door UP @x≈112",
    ),
    6: EntranceAnchor(
        level=6,
        door_screen=SCREEN_LEVEL6_ENTRANCE,
        entry_room=SCREEN_LEVEL6_ENTRY_ROOM,
        verified=True,
        label="dragon",
    ),
    7: EntranceAnchor(
        level=7,
        door_screen=SCREEN_LEVEL7_ENTRANCE,
        entry_room=None,
        verified=False,
        approach_screen=SCREEN_LEVEL7_BAIT_SHOP,
        label="demon",
        notes="Whistle pond drain (source)",
    ),
    8: EntranceAnchor(
        level=8,
        door_screen=SCREEN_LEVEL8_ENTRANCE,
        entry_room=None,
        verified=True,
        label="lion",
        notes="Bush 0x6d live; candle required to enter",
    ),
    9: EntranceAnchor(
        level=9,
        door_screen=SCREEN_LEVEL9_ENTRANCE,
        entry_room=None,
        verified=False,
        label="death_mountain",
        notes="Bomb rock; full triforce for interior progress",
    ),
}

DOOR_SCREEN_BY_LEVEL: dict[int, int] = {
    lv: a.door_screen for lv, a in ENTRANCES.items()
}

# Doc / probe labels
LATER_SCREEN_LABELS: dict[int, str] = {
    SCREEN_LEVEL3_ENTRANCE: "level3_entrance_manji_live",
    SCREEN_LEVEL4_RAFT_DOCK: "level4_raft_dock_0x55_live",
    SCREEN_LEVEL4_ENTRANCE: "level4_island_door_0x45_live",
    SCREEN_LEVEL4_ENTRY_ROOM: "level4_entry_room_0x71_live",
    SCREEN_LEVEL5_LOST_HILLS: "lost_hills_maze_live",
    SCREEN_LEVEL5_ENTRANCE: "level5_entrance_lizard_live",
    SCREEN_LEVEL6_ENTRANCE: "level6_entrance_dragon_live",
    SCREEN_LEVEL7_BAIT_SHOP: "bait_shop_armos_source",
    SCREEN_LEVEL7_ENTRANCE: "level7_entrance_demon_pond_source",
    SCREEN_LEVEL8_ENTRANCE: "level8_entrance_lion_bush_live",
    SCREEN_LEVEL9_ENTRANCE: "level9_entrance_bomb_rock_source",
    SCREEN_CANDLE_SHOP: "candle_shop_5e_live",
    SCREEN_BRACELET_ARMOS: "bracelet_armos_source",
    SCREEN_MAGICAL_SWORD_GRAVE: "magical_sword_grave_source",
    SCREEN_RAFT_HEART_DOCK: "raft_heart_dock_source",
    SCREEN_LADDER_HEART: "ladder_heart_coast_source",
}

# Aliases kept for modules that used different names historically.
SCREEN_LEVEL5_DOOR = SCREEN_LEVEL5_ENTRANCE
SCREEN_LOST_HILLS = SCREEN_LEVEL5_LOST_HILLS
LEVEL5_ENTRY_ROOM = SCREEN_LEVEL5_ENTRY_ROOM
LEVEL5_TF_ROOM = SCREEN_LEVEL5_TF_ROOM
LEVEL6_ENTRY_ROOM = SCREEN_LEVEL6_ENTRY_ROOM
SCREEN_LEVEL4_DOCK_HYP = SCREEN_LEVEL4_RAFT_DOCK
SCREEN_LEVEL4_ISLAND_HYP = SCREEN_LEVEL4_ENTRANCE
SCREEN_RAFT_HEART_DOCK_HYP = SCREEN_RAFT_HEART_DOCK
SCREEN_LEVEL7_POND_HYP = SCREEN_LEVEL7_ENTRANCE
SCREEN_LEVEL7_BAIT_SHOP_HYP = SCREEN_LEVEL7_BAIT_SHOP
SCREEN_LEVEL9_ROCK_HYP = SCREEN_LEVEL9_ENTRANCE


__all__ = [
    "DOOR_SCREEN_BY_LEVEL",
    "ENTRANCES",
    "EntranceAnchor",
    "FULL_TRIFORCE",
    "LATER_SCREEN_LABELS",
    "LEVEL5_ENTRY_ROOM",
    "LEVEL5_TF_ROOM",
    "LEVEL6_ENTRY_ROOM",
    "SCREEN_BRACELET_ARMOS",
    "SCREEN_CANDLE_SHOP",
    "SCREEN_LADDER_HEART",
    "SCREEN_LEVEL3_ENTRANCE",
    "SCREEN_LEVEL3_ENTRY_ROOM",
    "SCREEN_LEVEL4_DOCK_HYP",
    "SCREEN_LEVEL4_ENTRANCE",
    "SCREEN_LEVEL4_ENTRY_ROOM",
    "SCREEN_LEVEL4_ISLAND_HYP",
    "SCREEN_LEVEL4_RAFT_DOCK",
    "SCREEN_LEVEL5_DOOR",
    "SCREEN_LEVEL5_ENTRANCE",
    "SCREEN_LEVEL5_ENTRY_ROOM",
    "SCREEN_LEVEL5_LOST_HILLS",
    "SCREEN_LEVEL5_TF_ROOM",
    "SCREEN_LEVEL6_ENTRANCE",
    "SCREEN_LEVEL6_ENTRY_ROOM",
    "SCREEN_LEVEL7_BAIT_SHOP",
    "SCREEN_LEVEL7_BAIT_SHOP_HYP",
    "SCREEN_LEVEL7_ENTRANCE",
    "SCREEN_LEVEL7_POND_HYP",
    "SCREEN_LEVEL8_BUSH",
    "SCREEN_LEVEL8_ENTRANCE",
    "SCREEN_LEVEL9_ENTRANCE",
    "SCREEN_LEVEL9_ROCK_HYP",
    "ROOM_LEVEL9_62",
    "ROOM_LEVEL9_72",
    "SCREEN_LOST_HILLS",
    "SCREEN_MAGICAL_SWORD_GRAVE",
    "SCREEN_RAFT_HEART_DOCK",
    "SCREEN_RAFT_HEART_DOCK_HYP",
    "TF_BITS_ALL",
    "TF_BIT_L1",
    "TF_BIT_L2",
    "TF_BIT_L3",
    "TF_BIT_L4",
    "TF_BIT_L5",
    "TF_BIT_L6",
    "TF_BIT_L7",
    "TF_BIT_L8",
    "TRIFORCE_BITS_BY_LEVEL",
]
