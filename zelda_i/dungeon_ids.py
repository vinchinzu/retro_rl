"""Probe-verified symbolic IDs used by the Zelda I dungeon laboratory."""

from __future__ import annotations

from zelda_i import ram

OBJECT_NAMES: dict[int, str] = {
    0x06: "goriya",
    0x15: "gel",
    0x1B: "keese",
    0x27: "wallmaster",
    0x3D: "aquamentus",
    0x2A: "stalfos",
    0x60: "green_rupee_drop",
}

ROOM_ITEM_NAMES: dict[int, str] = {
    0x03: "no_inventory_reward_observed",
    0x16: "unknown_room_item_16",
    0x19: "small_key",
    0x1A: "heart_container",
}

MODE_NAMES: dict[int, str] = {
    4: "dungeon_room_settle",
    5: "play",
    6: "scroll_prepare",
    7: "scroll",
    8: "hurt_freeze_or_game_over_menu",
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
