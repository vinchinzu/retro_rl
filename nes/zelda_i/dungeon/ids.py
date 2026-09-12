"""Probe-verified and source-correlated IDs used by the dungeon laboratory."""

from __future__ import annotations

from enum import Enum

from zelda_i import ram


class AliveRule(str, Enum):
    """How an object type represents a living enemy."""

    TYPE = "type"
    TYPE_AND_HP = "hp"

OBJECT_NAMES: dict[int, str] = {
    0x01: "lynel_blue",  # aldonunez UpdateObject_JumpTable
    0x02: "lynel",
    0x03: "moblin_blue",  # not Octorok; level8/overworld.py comment is spawn-id mixup
    0x04: "moblin",
    0x05: "goriya_blue_or_residual",  # L2 boom room 0x4f
    0x06: "goriya",
    0x07: "octorok",  # red slow; screens 0x59–0x5E
    0x08: "octorok_fast",
    0x09: "octorok_blue",
    0x0A: "octorok_blue_fast",
    0x0B: "darknut",  # L3 0x5b/0x59/0x69 live
    0x0D: "tektite_blue",
    0x0E: "tektite",
    0x0F: "leever_blue",
    0x10: "leever",
    0x11: "zora",
    0x12: "vire",  # L4 0x61/0x50 live (rr-5lu); HP64; sword splits → 0x1c
    0x13: "zol",
    0x14: "gel_or_zol_split_residual",  # L3 0x4b after wooden-sword hits
    0x15: "gel",
    0x16: "pols_voice",
    0x17: "like_like",  # L4 0x32 live (rr-resv); avoid contact (shield loss)
    0x1A: "peahat",
    0x1B: "keese",
    0x1C: "vire_split_keese",  # L4 Vire split residual (live rr-5lu; not 0x1b)
    0x1E: "armos",  # awake object; statue is tile $66/$67
    0x20: "boulder",  # falling mountain rock (same updater as Tektite)
    0x21: "ghini",
    0x22: "ghini_flying",
    0x23: "wizzrobe_blue_walkthrough_correlated",
    0x24: "wizzrobe_orange",
    0x25: "patra_eye",  # L9 final Patra 0x52 live (rr-sz8.2)
    0x27: "wallmaster",
    0x28: "rope",
    0x30: "gibdo",
    0x2B: "invuln_mover_residual",  # L3 0x49/0x5d HP240; sword/bomb no dmg (not Manhandla)
    0x32: "dodongo",  # L2 boss room 0x0e (live rr-n5i 2026-08-07)
    0x33: "gohma_red",  # L6 0x1C; one wooden arrow to open eye
    0x34: "gohma_blue",  # L8 Gohma (3 arrows); not this hop
    0x35: "l4_mid_11_cluster",  # L4 room 0x11 live rr-rvae
    0x3C: "manhandla",  # L3 room 0x4d assisted kill 2/2 (rr-vpl 2026-08-07); L4 0x10
    0x3D: "aquamentus",
    0x2A: "stalfos",
    0x40: "bubble",
    0x43: "gleeok",  # L4 boss room 0x13 live rr-rvae (2-head)
    0x44: "gleeok_3head",  # L6 0x18 live settle (not 0x43)
    0x46: "gleeok_head",  # L4 0x13 detached head mid-fight (rr-rvae dual)
    0x47: "patra",  # L9 final Patra room 0x52 live (rr-sz8.2)
    0x49: "blade_trap",  # L4 room 0x02 live rr-rvae
    0x4D: "old_man_or_npc",
    0x4e: "trap_or_fire_residual",
    0x53: "rock_projectile",  # Octorok spit (UpdateOctorock TryShooting)
    0x55: "fireball_or_statue_projectile",  # L2 0x4f statues; Zora spit
    0x56: "manhandla_projectile_residual",  # L3 Manhandla + L4 Gleeok fireball
    0x57: "lynel_sword_shot",
    0x5B: "moblin_arrow",
    0x5C: "boomerang_projectile",  # L1 0x44 Goriya throw (lab); not type 0x06
    0x60: "floor_drop",  # live At4A: rupee/heart/5-rupee/clock; item in ObjState
}


# Canonical object type IDs (prefer these over redefining in dungeon_ops / level modules).
# Overworld types: aldonunez/zelda1-disassembly UpdateObject_JumpTable (matches
# dungeon IDs already here). level8/overworld.py "Octoroks (type 0x03)" is Blue Moblin.
LYNEL_BLUE_OBJECT_TYPE = 0x01
LYNEL_OBJECT_TYPE = 0x02
MOBLIN_BLUE_OBJECT_TYPE = 0x03
MOBLIN_OBJECT_TYPE = 0x04
GORIYA_BLUE_OBJECT_TYPE = 0x05
GORIYA_OBJECT_TYPE = 0x06
OCTOROK_OBJECT_TYPE = 0x07  # red slow; 0x08 fast, 0x09 blue, 0x0A blue fast
OCTOROK_FAST_OBJECT_TYPE = 0x08
OCTOROK_BLUE_OBJECT_TYPE = 0x09
OCTOROK_BLUE_FAST_OBJECT_TYPE = 0x0A
DARKNUT_OBJECT_TYPE = 0x0B
TEKTITE_BLUE_OBJECT_TYPE = 0x0D
TEKTITE_OBJECT_TYPE = 0x0E
LEEVER_BLUE_OBJECT_TYPE = 0x0F
LEEVER_OBJECT_TYPE = 0x10
ZORA_OBJECT_TYPE = 0x11
VIRE_OBJECT_TYPE = 0x12  # L4 live rr-5lu
ZOL_OBJECT_TYPE = 0x13
GEL_SPLIT_OBJECT_TYPE = 0x14  # wooden-sword Zol split residual
GEL_OBJECT_TYPE = 0x15
POLS_VOICE_OBJECT_TYPE = 0x16  # L5 0x77/0x25/0x27 live
LIKE_LIKE_OBJECT_TYPE = 0x17  # L4 0x32 live (rr-resv); avoid contact
PEAHAT_OBJECT_TYPE = 0x1A
KEESE_OBJECT_TYPE = 0x1B
VIRE_SPLIT_KEESE_TYPE = 0x1C  # L4 Vire → red Keese-like split
ARMOS_OBJECT_TYPE = 0x1E  # awake; statue is tile $66/$67 not an object slot
BOULDER_OBJECT_TYPE = 0x20  # falling mountain rock
GHINI_OBJECT_TYPE = 0x21
GHINI_FLYING_OBJECT_TYPE = 0x22
WIZZROBE_BLUE_OBJECT_TYPE = 0x23  # walkthrough-correlated; L6 0x38
WIZZROBE_ORANGE_OBJECT_TYPE = 0x24
PATRA_EYE_OBJECT_TYPE = 0x25
WALLMASTER_OBJECT_TYPE = 0x27
ROPE_OBJECT_TYPE = 0x28
INVULN_MOVER_OBJECT_TYPE = 0x2B
GIBDO_OBJECT_TYPE = 0x30  # L5 0x66/0x65/0x26/0x34 live
DODONGO_OBJECT_TYPE = 0x32
GOHMA_OBJECT_TYPE = 0x33  # L6 red Gohma; source + Data Crystal ObjType
GOHMA_BLUE_OBJECT_TYPE = 0x34  # L8 blue Gohma; not this hop
L4_MID_11_OBJECT_TYPE = 0x35  # L4 0x11 live rr-rvae
MANHANDLA_OBJECT_TYPE = 0x3C
AQUAMENTUS_OBJECT_TYPE = 0x3D
BUBBLE_OBJECT_TYPE = 0x40  # L5 0x67 live; sword-immune residual
MOLDORM_OBJECT_TYPE = 0x41
GLEEOK_OBJECT_TYPE = 0x43  # L4 0x13 live rr-rvae
GLEEOK_3HEAD_OBJECT_TYPE = 0x44  # L6 0x18 live settle census
GLEEOK_HEAD_OBJECT_TYPE = 0x46  # L4 0x13 detached head (rr-rvae dual)
PATRA_OBJECT_TYPE = 0x47  # L9 room 0x52 final Patra
BLADE_TRAP_OBJECT_TYPE = 0x49  # L4 0x02 live rr-rvae
ROCK_PROJECTILE_TYPE = 0x53  # Octorok spit
FIREBALL_OBJECT_TYPE = 0x55
MANHANDLA_PROJECTILE_TYPE = 0x56  # also Gleeok fireball residual
LYNEL_SWORD_SHOT_TYPE = 0x57
MOBLIN_ARROW_OBJECT_TYPE = 0x5B
GORIYA_BOOMERANG_OBJECT_TYPE = 0x5C  # L1 0x44 lab traces; HP 0/144; not 0x06
# Floor drops share ObjType 0x60 (live At4A 2026-09-11: tektite 0x0D → 0x60).
# Item code is ObjState, not ObjType. Data Crystal 0x22/0x23/0x18 are item IDs;
# 0x22 as ObjType is ghini_flying. Drops: hp 0 (flash 0x80), slot 1–10.
RUPEE_DROP_OBJECT_TYPE = 0x60
RUPEE_DROP_STATE = 0x18  # live +1 rupee / rupees_to_add
HEART_DROP_OBJECT_TYPE = 0x60  # live pickup +1 HeartValues; state 0x22; not ghini
HEART_DROP_STATE = 0x22
FAIRY_DROP_OBJECT_TYPE = 0x60  # item code 0x23; $0627==16 force (not live-picked)
FAIRY_DROP_STATE = 0x23
FIVE_RUPEE_DROP_OBJECT_TYPE = 0x60  # live +5 rupees_to_add
FIVE_RUPEE_DROP_STATE = 0x0F
CLOCK_DROP_OBJECT_TYPE = 0x60  # live no inventory delta
CLOCK_DROP_STATE = 0x21

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
    ram.ADDR_HELP_DROP_COUNT: "help_drop_count",
    ram.ADDR_HELP_DROP_VALUE: "help_drop_value",
    ram.ADDR_WORLD_KILL_COUNT: "world_kill_count",
    ram.ADDR_HEALTH: "health",
    ram.ADDR_HEART_PARTIAL: "heart_partial",
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
