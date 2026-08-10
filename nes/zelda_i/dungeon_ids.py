"""Probe-verified and source-correlated IDs used by the dungeon laboratory."""

from __future__ import annotations

from zelda_i import ram

OBJECT_NAMES: dict[int, str] = {
    0x05: "goriya_blue_or_residual",  # L2 boom room 0x4f
    0x06: "goriya",
    0x0B: "darknut",  # L3 0x5b/0x59/0x69 live
    0x12: "vire",  # L4 0x61/0x50 live (rr-5lu); HP64; sword splits → 0x1c
    0x13: "zol",
    0x14: "gel_or_zol_split_residual",  # L3 0x4b after wooden-sword hits
    0x15: "gel",
    0x16: "pols_voice",
    0x17: "like_like",  # L4 0x32 live (rr-resv); avoid contact (shield loss)
    0x1B: "keese",
    0x1C: "vire_split_keese",  # L4 Vire split residual (live rr-5lu; not 0x1b)
    0x23: "wizzrobe_blue_walkthrough_correlated",
    0x24: "wizzrobe_orange",
    0x27: "wallmaster",
    0x28: "rope",
    0x30: "gibdo",
    0x2B: "invuln_mover_residual",  # L3 0x49/0x5d HP240; sword/bomb no dmg (not Manhandla)
    0x32: "dodongo",  # L2 boss room 0x0e (live rr-n5i 2026-08-07)
    0x35: "l4_mid_11_cluster",  # L4 room 0x11 live rr-rvae
    0x3C: "manhandla",  # L3 room 0x4d assisted kill 2/2 (rr-vpl 2026-08-07); L4 0x10
    0x3D: "aquamentus",
    0x2A: "stalfos",
    0x40: "bubble",
    0x43: "gleeok",  # L4 boss room 0x13 live rr-rvae (2-head)
    0x46: "gleeok_head",  # L4 0x13 detached head mid-fight (rr-rvae dual)
    0x49: "blade_trap",  # L4 room 0x02 live rr-rvae
    0x4D: "old_man_or_npc",
    0x4e: "trap_or_fire_residual",
    0x55: "fireball_or_statue_projectile",  # L2 0x4f statues
    0x56: "manhandla_projectile_residual",  # L3 Manhandla + L4 Gleeok fireball
    0x60: "green_rupee_drop",
}


# Canonical object type IDs (prefer these over redefining in dungeon_ops / level modules).
GORIYA_BLUE_OBJECT_TYPE = 0x05
GORIYA_OBJECT_TYPE = 0x06
DARKNUT_OBJECT_TYPE = 0x0B
VIRE_OBJECT_TYPE = 0x12  # L4 live rr-5lu
ZOL_OBJECT_TYPE = 0x13
GEL_SPLIT_OBJECT_TYPE = 0x14  # wooden-sword Zol split residual
GEL_OBJECT_TYPE = 0x15
KEESE_OBJECT_TYPE = 0x1B
VIRE_SPLIT_KEESE_TYPE = 0x1C  # L4 Vire → red Keese-like split
WIZZROBE_ORANGE_OBJECT_TYPE = 0x24
WALLMASTER_OBJECT_TYPE = 0x27
ROPE_OBJECT_TYPE = 0x28
INVULN_MOVER_OBJECT_TYPE = 0x2B
DODONGO_OBJECT_TYPE = 0x32
L4_MID_11_OBJECT_TYPE = 0x35  # L4 0x11 live rr-rvae
MANHANDLA_OBJECT_TYPE = 0x3C
AQUAMENTUS_OBJECT_TYPE = 0x3D
MOLDORM_OBJECT_TYPE = 0x41
GLEEOK_OBJECT_TYPE = 0x43  # L4 0x13 live rr-rvae
GLEEOK_HEAD_OBJECT_TYPE = 0x46  # L4 0x13 detached head (rr-rvae dual)
BLADE_TRAP_OBJECT_TYPE = 0x49  # L4 0x02 live rr-rvae
FIREBALL_OBJECT_TYPE = 0x55
MANHANDLA_PROJECTILE_TYPE = 0x56  # also Gleeok fireball residual

ROOM_ITEM_NAMES: dict[int, str] = {
    0x03: "no_inventory_reward_observed",
    0x0C: "raft_room_item_live",  # L3 mode-9 passage 0x0f (assisted 2026-08-07)
    0x16: "compass_walkthrough_correlated",
    0x17: "dungeon_map_walkthrough_correlated",
    0x19: "small_key",
    0x1A: "heart_container",
    0x1B: "triforce_or_residual_room_item",  # L2 0x0d after boss (rr-n5i); not collected yet
    0x1D: "boomerang_walkthrough_correlated",
    0x1E: "magical_boomerang_room_item_residual",  # live on L2 0x4f (rr-cjf)
}

MODE_NAMES: dict[int, str] = {
    4: "dungeon_room_settle",
    5: "play",
    6: "scroll_prepare",
    7: "scroll",
    8: "hurt_freeze_or_game_over_menu",
    9: "dungeon_underworld_passage",  # L3 Raft stairs 0x0f live
    10: "dungeon_stairs_exit_residual",
    11: "cave_play",
    16: "cave_enter",
    17: "link_death",
}

RAM_SYMBOLS: dict[int, str] = {
    ram.ADDR_LEVEL: "level",
    ram.ADDR_MODE: "mode",
    ram.ADDR_DIALOG_TIMER: "dialog_timer",
    ram.ADDR_LINK_X: "link_x",
    ram.ADDR_LINK_Y: "link_y",
    ram.ADDR_LINK_FACING: "link_facing",
    ram.ADDR_SCREEN: "screen",
    ram.ADDR_NEXT_SCREEN: "next_screen",
    ram.ADDR_COLLIDING_TILE: "colliding_tile",
    ram.ADDR_ROOM_ITEM_ID: "room_item_id",
    ram.ADDR_CUR_OPENED_DOORS: "cur_opened_doors",
    ram.ADDR_OPEN_DOORWAY_MASK: "open_doorway_mask",
    ram.ADDR_ROOM_ALL_DEAD: "room_all_dead",
    ram.ADDR_ROOM_OBJ_COUNT: "room_obj_count",
    ram.ADDR_SWORD: "sword",
    ram.ADDR_BOMBS: "bombs",
    ram.ADDR_ARROWS: "arrows",
    ram.ADDR_BOW: "bow",
    ram.ADDR_CANDLE: "candle",
    ram.ADDR_WHISTLE: "whistle",
    ram.ADDR_FOOD: "food",
    ram.ADDR_POTION: "potion",
    ram.ADDR_ROD: "rod",
    ram.ADDR_RAFT: "raft",
    ram.ADDR_BOOK: "book",
    ram.ADDR_RING: "ring",
    ram.ADDR_LADDER: "ladder",
    ram.ADDR_MAGIC_KEY: "magic_key",
    ram.ADDR_BRACELET: "bracelet",
    ram.ADDR_LETTER: "letter",
    ram.ADDR_COMPASS: "compass",
    ram.ADDR_MAP: "map",
    ram.ADDR_RUPEES: "rupees",
    ram.ADDR_KEYS: "keys",
    ram.ADDR_HEALTH: "health",
    ram.ADDR_TRIFORCE: "triforce",
    ram.ADDR_BOOMERANG: "boomerang",
    ram.ADDR_MAGIC_BOOMERANG: "magical_boomerang",
    ram.ADDR_MAGIC_SHIELD: "magic_shield",
}


def object_name(type_id: int) -> str:
    """Return a stable symbolic label without pretending unknown IDs are known."""
    value = int(type_id) & 0xFF
    return OBJECT_NAMES.get(value, f"unknown_object_0x{value:02x}")


def room_item_name(item_id: int) -> str:
    """Return the verified room-item label, or an explicit unknown label."""
    value = int(item_id) & 0xFF
    return ROOM_ITEM_NAMES.get(value, f"unknown_room_item_0x{value:02x}")


def mode_name(mode: int) -> str:
    value = int(mode) & 0xFF
    return MODE_NAMES.get(value, f"unknown_mode_{value}")


def ram_symbol(address: int) -> str | None:
    """Name known scalar and object-array addresses for RAM delta reports."""
    address = int(address)
    if address in RAM_SYMBOLS:
        return RAM_SYMBOLS[address]
    if ram.ADDR_OBJ_TYPE <= address < ram.ADDR_OBJ_TYPE + 16:
        return f"obj_type[{address - ram.ADDR_OBJ_TYPE}]"
    if ram.ADDR_OBJ_HP <= address < ram.ADDR_OBJ_HP + 13:
        return f"obj_hp[{address - ram.ADDR_OBJ_HP}]"
    if ram.ADDR_LINK_X < address < ram.ADDR_LINK_X + 13:
        return f"obj_x[{address - ram.ADDR_LINK_X}]"
    if ram.ADDR_LINK_Y < address < ram.ADDR_LINK_Y + 13:
        return f"obj_y[{address - ram.ADDR_LINK_Y}]"
    if ram.ADDR_LINK_FACING < address < ram.ADDR_LINK_FACING + 13:
        return f"obj_facing[{address - ram.ADDR_LINK_FACING}]"
    return None
