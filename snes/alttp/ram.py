"""ALTTP WRAM readers and readiness helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# stable-retro get_ram() index mapping:
#   WRAM offset < 0x2000 → index = offset (block 0 mirror)
#   WRAM offset >= 0x2000 → index = 16384 + offset
WRAM_IDX = 16384

MODULE = 0x10
SUBMODULE = 0x11
INDOORS = 0x1B
ROOM_ID = 0xA0
LINK_Y = 0x20
LINK_X = 0x22
LINK_DIRECTION = 0x2F
LINK_ACTION = 0x5D  # link_player_handler_state; 0 at rest, 2 during recoil, 21 hold-up
# link_incapacitated_timer. CastleB1Guard sword-hit counted 18,17,… while $5D stayed 2.
LINK_KNOCKBACK = 0x0046
SCREEN_ID = 0x008A
# dung_cur_floor. CastleMain 0x00 (F1), CastleB1* 0xFF (B1), CastleB3 / cell 0xFD (B3).
DUNGEON_FLOOR = 0x00A4
DARK_WORLD_FLAG = 0x0FFF
BG2_VOFS = 0x00E8
CAMERA_X = 0xE2
# link_is_on_lower_level. 0 on CastleMain and CastleB1Guard; 1 on CastleB1Pit.
LINK_LAYER = 0x00EE
# countdown_for_blink. Same hit counted 57,56,…; CastleB1Pit still held 0x11 at rest.
LINK_IFRAMES = 0x031F
# cur_palace_index_x2. Castle 0x02, EasternPalaceEntry 0x04, overworld / FighterSword 0xFF.
# High byte $040D was 0 on those pins.
DUNGEON_ID = 0x040C
# Orange/blue peg state, low byte of the $C172 word. HeraEntry room 0x77,
# crystal (3840, 3968), no RAM writes: 0 while the west alcove stopped at
# (3792, 3979), 1 after slash+spin with the alcove open at (3704, 3979),
# then 0 and the alcove stopped at (3792, 3979) again. $C173 and $0642 stayed 0.
PEG_STATE = 0xC172
LINK_HP = 0xF36D
LINK_MAX_HP = 0xF36C
EQUIP_SWORD = 0xF359
LINK_ITEM_LAMP = 0xF34A
NUM_KEYS = 0xF36F
FOLLOWER = 0xF3CC  # tagalong / follower_indicator
# link_ability_flags. Dash A-press mask is bit 0x04 (see the ability field).
LINK_ABILITY = 0xF379
ABILITY_DASH = 0x04
# sram_progress_indicator. 0 on LinksHouseWake / FirstAction / HyruleCastleGrounds.
# 1 on the sword-route pins, including CastleZeldaFollower. Value 3 is only on
# the late dungeon-entry pins (Ice, PoD, Swamp, Thieves, YazeSlot000).
SRAM_PROGRESS = 0xF3C5
# sram_progress_flags. Bit 0x01 is clear on HyruleCastleGrounds (0x10) and set
# on CourtyardSecretPocket and FighterSword (0x11). Bit 0x10 is already set
# on the pre-sword pins; this scan never saw it clear.
SRAM_PROGRESS_FLAGS = 0xF3C6
PROGRESS_FLAG_UNCLE_GEAR = 0x01

HYRULE_CASTLE_SCREEN = 0x1B
LINKS_HOUSE_SCREEN = 0x2C
LINKS_HOUSE_ROOM = 0x0004
SANCTUARY_SCREEN = 0x13
SANCTUARY_WEST_SCREEN = 0x12
NORTH_FIELD_SCREEN = 0x11
KAKARIKO_NW_SCREEN = 0x18
KAKARIKO_NE_SCREEN = 0x19
KAKARIKO_SW_SCREEN = 0x20
KAKARIKO_SE_SCREEN = 0x21
KAKARIKO_SCREEN = KAKARIKO_NW_SCREEN

# Kakariko Village interior rooms
KAKARIKO_WELL_ROOM = 0x2F
BLINDS_HIDEOUT_ROOM = 0x11D
TAVERN_ROOM = 0x103
CHICKEN_HOUSE_ROOM = 0x108
KAKARIKO_SHOP_ROOM = 0x11F

# Opening-route dungeon/cave rooms (stable-retro ``$A0`` base ids).
# Prefer the descriptive names in logs/docs; hex is RAM only.

# Bush-hole drop east of the castle: multi-screen "Hyrule Castle Secret
# Entrance" (Uncle + fighter sword, then combat/key path). RAM base = 0x55.
HYRULE_CASTLE_SECRET_ENTRANCE_ROOM = 0x55
# Aliases kept for existing imports (same room id).
SECRET_PASSAGE_ROOM = HYRULE_CASTLE_SECRET_ENTRANCE_ROOM
UNCLE_ROOM = HYRULE_CASTLE_SECRET_ENTRANCE_ROOM

# Main south hall (ordinary castle entrance) — post-sword / alternate path.
HYRULE_CASTLE_MAIN_HALL_ROOM = 0x61
# West of main hall (Zelda-path edge from room 0x61; measured 2026-07-31).
HYRULE_CASTLE_MAIN_WEST_ROOM = 0x60
# East of main hall (alternate edge from room 0x61).
HYRULE_CASTLE_MAIN_EAST_ROOM = 0x62
# North of main west (Zelda path hop from 0x60; isolated 2026-07-31).
HYRULE_CASTLE_NW_ROOM = 0x50
# North connector / NE / throne approach (measured maps).
HYRULE_CASTLE_NORTH_CONNECTOR_ROOM = 0x01
HYRULE_CASTLE_THRONE_APPROACH_ROOM = 0x51
HYRULE_CASTLE_NE_ROOM = 0x52
# B1 rooms on work-queue / Zelda path (measured maps; not all continuous).
HYRULE_CASTLE_B1_NORTH_ZELDA_WEST_ROOM = 0x71
HYRULE_CASTLE_B1_PIT_ROOM = 0x72
HYRULE_CASTLE_B1_WEST_ROOM = 0x81
HYRULE_CASTLE_B1_EAST_ROOM = 0x82
# Zelda's cell (opening rescue target; confirm on real ROM before claiming).
ZELDA_CELL_ROOM = 0x80
# Sanctuary indoor room after escort (confirm on real ROM before claiming).
SANCTUARY_ROOM = 0x12

# Human labels for room_base_id (hex only as fallback).
ROOM_LABELS: dict[int, str] = {
    LINKS_HOUSE_ROOM & 0xFF: "links_house",
    HYRULE_CASTLE_SECRET_ENTRANCE_ROOM: "hyrule_castle_secret_entrance",
    HYRULE_CASTLE_NORTH_CONNECTOR_ROOM: "hyrule_castle_north_connector",
    HYRULE_CASTLE_NW_ROOM: "hyrule_castle_nw",
    HYRULE_CASTLE_THRONE_APPROACH_ROOM: "hyrule_castle_throne_approach",
    HYRULE_CASTLE_NE_ROOM: "hyrule_castle_ne",
    HYRULE_CASTLE_MAIN_WEST_ROOM: "hyrule_castle_main_west",
    HYRULE_CASTLE_MAIN_HALL_ROOM: "hyrule_castle_main_hall",
    HYRULE_CASTLE_MAIN_EAST_ROOM: "hyrule_castle_main_east",
    HYRULE_CASTLE_B1_NORTH_ZELDA_WEST_ROOM: "hyrule_castle_b1_north_zelda_west",
    HYRULE_CASTLE_B1_PIT_ROOM: "hyrule_castle_b1_pit",
    ZELDA_CELL_ROOM: "zelda_cell",
    HYRULE_CASTLE_B1_WEST_ROOM: "hyrule_castle_b1_west",
    HYRULE_CASTLE_B1_EAST_ROOM: "hyrule_castle_b1_east",
    SANCTUARY_ROOM: "sanctuary",
}


def room_label(room_base_id: int) -> str:
    """Descriptive name for a room base id, or ``room_XX`` hex fallback."""
    rid = int(room_base_id) & 0xFF
    return ROOM_LABELS.get(rid, f"room_{rid:02x}")

# Fighter sword equip level at $F359 (0=none, 1=fighter, …).
FIGHTER_SWORD_LEVEL = 1

# follower_indicator values used on the opening route.
FOLLOWER_NONE = 0
FOLLOWER_ZELDA = 1

# link_player_handler_state ($5D) — kPlayerState_HoldUpItem after item get.
LINK_ACTION_HOLD_UP_ITEM = 21

# Yaze-exported overworld hole for entrance 0x7D (logic/nav target only;
# gameplay proof still requires indoors room / inventory evidence).
SECRET_HOLE_WORLD_X = 2432
SECRET_HOLE_WORLD_Y = 1696
SECRET_HOLE_APPROACH_TOLERANCE = 48


def wram_index(offset: int) -> int:
    """Map a WRAM byte offset to a stable-retro get_ram() index."""
    off = int(offset)
    if off < 0x2000:
        return off
    return WRAM_IDX + off


def assign_u8(env: object, offset: int, value: int) -> None:
    """Write one WRAM byte. Refuses follower ``$F3CC``. Does not refresh RAM.

    Bus address is ``0x7E0000 + offset`` when ``offset >= 0x2000``, else the
    offset. Uses ``env.unwrapped`` when that attribute is present.
    """
    off = int(offset)
    if off == int(FOLLOWER):
        raise SystemExit("refusing to write $F3CC")
    bus = 0x7E0000 + off if off >= 0x2000 else off
    unwrapped = getattr(env, "unwrapped", env)
    unwrapped.data.memory.assign(bus, "|u1", int(value) & 0xFF)


@dataclass(frozen=True)
class AlttpSnapshot:
    """Frame snapshot of the fields needed for title→castle / opening routing."""

    game_mode: int
    submodule: int
    room_id: int
    indoors: bool
    screen_id: int
    link_x: int
    link_y: int
    link_direction: int
    link_action: int
    camera_x: int
    camera_y: int
    dark_world: bool
    sword_level: int = 0
    lamp_level: int = 0
    num_keys: int = 0xFF
    follower: int = 0
    dungeon_id: int = 0
    floor: int = 0
    layer: int = 0
    hp: int = 0
    max_hp: int = 0
    iframes: int = 0
    knockback: int = 0

    @property
    def room_base_id(self) -> int:
        return int(self.room_id) & 0x00FF

    @property
    def has_control(self) -> bool:
        return self.game_mode in (0x07, 0x09) and self.submodule == 0x00

    @property
    def is_text_mode(self) -> bool:
        return self.game_mode == 0x0E

    @property
    def is_file_select(self) -> bool:
        return self.game_mode == 0x02

    @property
    def is_title_screen(self) -> bool:
        return self.game_mode == 0x01

    @property
    def is_hold_up_item(self) -> bool:
        """True during post-item hold-up pose ($5D == 21). Blocks normal control."""
        return int(self.link_action) == LINK_ACTION_HOLD_UP_ITEM

    @property
    def link_state(self) -> int:
        """Player handler state ($5D). Same byte as ``link_action``."""
        return int(self.link_action)

    @property
    def on_castle_grounds(self) -> bool:
        return (
            (not self.indoors)
            and (not self.dark_world)
            and self.screen_id == HYRULE_CASTLE_SCREEN
            and self.has_control
        )

    @property
    def in_secret_passage(self) -> bool:
        """Indoors in the secret-entrance / uncle passage room family."""
        return (
            self.indoors
            and (not self.dark_world)
            and self.room_base_id == SECRET_PASSAGE_ROOM
        )

    @property
    def in_castle_interior(self) -> bool:
        """Any indoor non-DW room on the opening castle path (passage or hall)."""
        return self.indoors and (not self.dark_world)

    @property
    def in_zelda_cell(self) -> bool:
        return (
            self.indoors
            and (not self.dark_world)
            and self.room_base_id == ZELDA_CELL_ROOM
        )

    @property
    def in_sanctuary(self) -> bool:
        return (
            self.indoors
            and (not self.dark_world)
            and self.room_base_id == SANCTUARY_ROOM
        )

    @property
    def has_fighter_sword(self) -> bool:
        return int(self.sword_level) >= FIGHTER_SWORD_LEVEL

    @property
    def has_lamp(self) -> bool:
        return int(self.lamp_level) != 0

    @property
    def has_zelda_follower(self) -> bool:
        return int(self.follower) == FOLLOWER_ZELDA

    @property
    def dungeon_key_count(self) -> int | None:
        """Current dungeon keys, or None when HUD uses the 0xFF 'blank' sentinel."""
        k = int(self.num_keys)
        if k == 0xFF:
            return None
        return k

    @property
    def near_secret_hole(self) -> bool:
        """True when outdoors near the Yaze 0x7D hole world coords."""
        if self.indoors or self.dark_world:
            return False
        if self.screen_id != HYRULE_CASTLE_SCREEN:
            return False
        return (
            abs(self.link_x - SECRET_HOLE_WORLD_X) <= SECRET_HOLE_APPROACH_TOLERANCE
            and abs(self.link_y - SECRET_HOLE_WORLD_Y) <= SECRET_HOLE_APPROACH_TOLERANCE
        )


def read_u8(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr])


def read_u16(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def read_u8_safe(ram: np.ndarray, addr: int) -> int:
    if addr < 0 or addr >= len(ram):
        return 0
    return int(ram[addr])


# save_dung_info: one u16 per room, $F000 + 2*room, 0x140 rooms ($F000..$F27F).
# Pass the full $A0 id. Link's house is 0x104; room_base_id masks that to 0x04.
SAVE_DUNG_INFO = 0xF000
SAVE_ROOM_COUNT = 0x140
# Chest 0. FighterSword -> FighterSwordLamp sets it on room 0x104 (lamp).
# CastleB3 -> CastleB3Boomerang sets the same bit on room 0x71.
ROOM_FLAG_CHEST = 0x0010
# Key or freestanding item. CastleB1Key live $0402 was already 0x4000 in room
# 0x72; CastleB1SecondKey stored ($0402 >> 4) as this bit after leaving.
ROOM_FLAG_KEY = 0x0400
# One door bit. Same pair, room 0x72. CastleB1Pit live $0400 is 0x8000.
# The other three door bits in 0x7000 did not change on the scanned pairs.
ROOM_FLAG_DOOR = 0x8000
# Boss-dead (diagram bit 0x0800) did not change on these pairs. Not decoded.


@dataclass(frozen=True)
class RoomFlagDiff:
    """One save-word bit that differed between two pins."""

    before_pin: str
    after_pin: str
    room: int
    bit: int
    name: str


# Bits that actually flipped. Quadrant visits are listed so they are not
# mistaken for chest / key / door. Boss-dead is absent.
ROOM_FLAG_PIN_DIFFS: tuple[RoomFlagDiff, ...] = (
    RoomFlagDiff("FighterSword", "FighterSwordLamp", 0x104, ROOM_FLAG_CHEST, "chest"),
    RoomFlagDiff("CastleB3", "CastleB3Boomerang", 0x71, ROOM_FLAG_CHEST, "chest"),
    RoomFlagDiff("CastleB1Key", "CastleB1SecondKey", 0x72, ROOM_FLAG_KEY, "key"),
    RoomFlagDiff("CastleB1Key", "CastleB1SecondKey", 0x72, ROOM_FLAG_DOOR, "door"),
    RoomFlagDiff("CastleB1Key", "CastleB1SecondKey", 0x72, 0x0003, "quadrants"),
    RoomFlagDiff("CastleB1Key", "CastleB1SecondKey", 0x71, 0x0003, "quadrants"),
    RoomFlagDiff("CastleB1Key", "CastleB1SecondKey", 0x81, 0x000F, "quadrants"),
    RoomFlagDiff("CastleB1Key", "CastleB1SecondKey", 0x82, 0x000F, "quadrants"),
)


def room_flags(ram: np.ndarray, room: int) -> int:
    """Save-RAM room word ``$F000 + 2*room`` (little-endian)."""
    rid = int(room)
    if rid < 0 or rid >= SAVE_ROOM_COUNT:
        raise ValueError(f"room {rid:#x} outside save_dung_info")
    addr = wram_index(SAVE_DUNG_INFO + 2 * rid)
    return read_u8_safe(ram, addr) | (read_u8_safe(ram, addr + 1) << 8)


def format_room_flag_diffs(
    diffs: tuple[RoomFlagDiff, ...] = ROOM_FLAG_PIN_DIFFS,
) -> str:
    """Report lines: which bit changed on which pin pair."""
    lines = []
    for diff in diffs:
        lines.append(
            f"{diff.before_pin} -> {diff.after_pin} "
            f"room=0x{diff.room:03X} bit=0x{diff.bit:04X} {diff.name}"
        )
    return "\n".join(lines)


# save_ow_event_info: one byte per screen, $F280 + screen, screens 0x00..0x7F.
# Bit 0x02 is the bombable secondary overlay. Bit 0x20 is the event overlay.
# The castle bush hole sets neither. It is a memorized map16 change.
SAVE_OW_EVENT_INFO = 0xF280
OW_SCREEN_COUNT = 0x80
OW_FLAG_BOMB_OVERLAY = 0x02
OW_FLAG_EVENT_OVERLAY = 0x20
# num_memorized_tiles counts bytes (+2 per change). Addrs $F800, values
# $FA00, both u16. The value table stops at $FA40.
MEMORIZED_TILE_COUNT = 0x04AC
MEMORIZED_TILE_ADDR = 0xF800
MEMORIZED_TILE_VALUE = 0xFA00
MEMORIZED_TILE_MAX = 0x20
# kTileBelow[0]. Secret byte 0x80 is "Hole". 0x212 is Warp (secret 0x82).
OW_TILE_HOLE = 0x0DCC
# ROM screen 0x1B secret list: pos 0x0570, data 0x80. Live lift wrote it.
CASTLE_BUSH_HOLE_POS = 0x0570
CASTLE_BUSH_HOLE_SCREEN = HYRULE_CASTLE_SCREEN


def ow_flags(ram: np.ndarray, screen: int) -> int:
    """``save_ow_event_info[screen]``. Not the opened bush-hole record."""
    sid = int(screen) & 0xFF
    if sid >= OW_SCREEN_COUNT:
        raise ValueError(f"screen {sid:#x} outside save_ow_event_info")
    return read_u8_safe(ram, wram_index(SAVE_OW_EVENT_INFO + sid))


def memorized_tiles(ram: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Map16 changes kept for this overworld visit. ``(pos, value)``."""
    count_at = wram_index(MEMORIZED_TILE_COUNT)
    count = read_u8_safe(ram, count_at) | (read_u8_safe(ram, count_at + 1) << 8)
    n = min(count >> 1, MEMORIZED_TILE_MAX)
    found: list[tuple[int, int]] = []
    for i in range(n):
        addr_at = wram_index(MEMORIZED_TILE_ADDR + 2 * i)
        value_at = wram_index(MEMORIZED_TILE_VALUE + 2 * i)
        pos = read_u8_safe(ram, addr_at) | (read_u8_safe(ram, addr_at + 1) << 8)
        value = read_u8_safe(ram, value_at) | (read_u8_safe(ram, value_at + 1) << 8)
        found.append((pos, value))
    return tuple(found)


# Save block $F340..$F37B. Offsets are snesrev/zelda3 ``variables.h``
# (``link_item_*`` and the following words). ``verified_on`` is a pin that
# showed a distinctive live value, or ``unverified`` when the scanned pins
# only read 0.
INV_BLOCK_LO = 0xF340
INV_BLOCK_HI = 0xF37B


@dataclass(frozen=True)
class InventoryField:
    name: str
    offset: int
    size: int = 1
    verified_on: str = "unverified"


INVENTORY_FIELDS: tuple[InventoryField, ...] = (
    InventoryField("bow", 0xF340),
    InventoryField("boomerang", 0xF341, verified_on="CastleB3Boomerang"),
    InventoryField("hookshot", 0xF342),
    InventoryField("bombs", 0xF343),
    InventoryField("powder", 0xF344),
    InventoryField("fire_rod", 0xF345),
    InventoryField("ice_rod", 0xF346),
    InventoryField("bombos", 0xF347),
    InventoryField("ether", 0xF348),
    InventoryField("quake", 0xF349),
    InventoryField("lamp", LINK_ITEM_LAMP, verified_on="FighterSwordLamp"),
    InventoryField("hammer", 0xF34B),
    InventoryField("flute", 0xF34C),
    InventoryField("net", 0xF34D),
    InventoryField("book", 0xF34E),
    InventoryField("somaria", 0xF350),
    InventoryField("byrna", 0xF351),
    InventoryField("cape", 0xF352),
    InventoryField("mirror", 0xF353),
    InventoryField("gloves", 0xF354),
    InventoryField("boots", 0xF355),
    InventoryField("flippers", 0xF356),
    InventoryField("pearl", 0xF357),
    InventoryField("sword", EQUIP_SWORD, verified_on="FighterSword"),
    InventoryField("shield", 0xF35A, verified_on="FighterSword"),
    InventoryField("armor", 0xF35B),
    InventoryField("bottle_1", 0xF35C),
    InventoryField("bottle_2", 0xF35D),
    InventoryField("bottle_3", 0xF35E),
    InventoryField("bottle_4", 0xF35F),
    InventoryField("rupees", 0xF362, 2),
    InventoryField("compass", 0xF364, 2),
    InventoryField("big_key", 0xF366, 2, "CastleMainZeldaBoomerang"),
    InventoryField("dungeon_map", 0xF368, 2),
    InventoryField("heart_pieces", 0xF36B),
    InventoryField("max_hp", LINK_MAX_HP, verified_on="CastleB1Key"),
    InventoryField("hp", LINK_HP, verified_on="CastleB1Key"),
    InventoryField("magic", 0xF36E, verified_on="IcePalaceEntry"),
    InventoryField("keys", NUM_KEYS, verified_on="CastleB1Key"),
    InventoryField("pendants", 0xF374),
    InventoryField("arrows", 0xF377),
    # $07:9C2F loads $9BA2,X and ANDs this byte. Dash is action X=2, mask 0x04.
    # Boots receipt Y=$4B ORs that bit at $09:8654 and writes $F355=1.
    InventoryField("ability", LINK_ABILITY),
    InventoryField("crystals", 0xF37A),
)


@dataclass(frozen=True)
class AlttpInventory:
    """Item block ``$F340..$F37B``. Multi-byte fields are little-endian."""

    bow: int
    boomerang: int
    hookshot: int
    bombs: int
    powder: int
    fire_rod: int
    ice_rod: int
    bombos: int
    ether: int
    quake: int
    lamp: int
    hammer: int
    flute: int
    net: int
    book: int
    somaria: int
    byrna: int
    cape: int
    mirror: int
    gloves: int
    boots: int
    flippers: int
    pearl: int
    sword: int
    shield: int
    armor: int
    bottle_1: int
    bottle_2: int
    bottle_3: int
    bottle_4: int
    rupees: int
    compass: int
    big_key: int
    dungeon_map: int
    heart_pieces: int
    max_hp: int
    hp: int
    magic: int
    keys: int
    pendants: int
    arrows: int
    ability: int
    crystals: int


def _read_inventory_value(ram: np.ndarray, field: InventoryField) -> int:
    idx = wram_index(field.offset)
    if field.size == 1:
        return read_u8_safe(ram, idx)
    if field.size == 2:
        return read_u8_safe(ram, idx) | (read_u8_safe(ram, idx + 1) << 8)
    raise ValueError(f"{field.name}: size {field.size}")


def read_inventory(ram: np.ndarray) -> AlttpInventory:
    """Read the save-block inventory from a stable-retro RAM buffer."""
    values = {
        field.name: _read_inventory_value(ram, field) for field in INVENTORY_FIELDS
    }
    return AlttpInventory(**values)


def read_sword_level(ram: np.ndarray) -> int:
    """Read equip sword level from high WRAM ($F359)."""
    return read_u8_safe(ram, wram_index(EQUIP_SWORD))


def read_snapshot(ram: np.ndarray) -> AlttpSnapshot:
    """Read a routing snapshot from a stable-retro RAM buffer."""
    return AlttpSnapshot(
        game_mode=read_u8(ram, MODULE),
        submodule=read_u8(ram, SUBMODULE),
        room_id=read_u16(ram, ROOM_ID),
        indoors=bool(read_u8(ram, INDOORS)),
        screen_id=read_u8(ram, SCREEN_ID),
        link_x=read_u16(ram, LINK_X),
        link_y=read_u16(ram, LINK_Y),
        link_direction=read_u8(ram, LINK_DIRECTION),
        link_action=read_u8(ram, LINK_ACTION),
        camera_x=read_u16(ram, CAMERA_X),
        camera_y=read_u16(ram, BG2_VOFS),
        dark_world=bool(read_u8(ram, DARK_WORLD_FLAG)),
        sword_level=read_sword_level(ram),
        lamp_level=read_u8_safe(ram, wram_index(LINK_ITEM_LAMP)),
        num_keys=read_u8_safe(ram, wram_index(NUM_KEYS)),
        follower=read_u8_safe(ram, wram_index(FOLLOWER)),
        dungeon_id=read_u8(ram, DUNGEON_ID),
        floor=read_u8(ram, DUNGEON_FLOOR),
        layer=read_u8(ram, LINK_LAYER),
        hp=read_u8_safe(ram, wram_index(LINK_HP)),
        max_hp=read_u8_safe(ram, wram_index(LINK_MAX_HP)),
        iframes=read_u8(ram, LINK_IFRAMES),
        knockback=read_u8(ram, LINK_KNOCKBACK),
    )


def snapshot_to_diag(snapshot: AlttpSnapshot) -> dict[str, object]:
    """Structured diagnostics for route reports (no invented milestones)."""
    return {
        "game_mode": snapshot.game_mode,
        "submodule": snapshot.submodule,
        "room_id": snapshot.room_id,
        "room_base_id": snapshot.room_base_id,
        "room_hex": f"0x{snapshot.room_base_id:02X}",
        "indoors": snapshot.indoors,
        "screen_id": snapshot.screen_id,
        "screen_hex": f"0x{snapshot.screen_id:02X}",
        "link_x": snapshot.link_x,
        "link_y": snapshot.link_y,
        "link_direction": snapshot.link_direction,
        "link_action": snapshot.link_action,
        "link_state": snapshot.link_state,
        "dungeon_id": snapshot.dungeon_id,
        "floor": snapshot.floor,
        "layer": snapshot.layer,
        "hp": snapshot.hp,
        "max_hp": snapshot.max_hp,
        "iframes": snapshot.iframes,
        "knockback": snapshot.knockback,
        "dark_world": snapshot.dark_world,
        "has_control": snapshot.has_control,
        "is_text_mode": snapshot.is_text_mode,
        "on_castle_grounds": snapshot.on_castle_grounds,
        "in_secret_passage": snapshot.in_secret_passage,
        "in_castle_interior": snapshot.in_castle_interior,
        "near_secret_hole": snapshot.near_secret_hole,
        "sword_level": snapshot.sword_level,
        "has_fighter_sword": snapshot.has_fighter_sword,
        "lamp_level": snapshot.lamp_level,
        "has_lamp": snapshot.has_lamp,
        "num_keys": snapshot.num_keys,
        "dungeon_key_count": snapshot.dungeon_key_count,
        "follower": snapshot.follower,
        "has_zelda_follower": snapshot.has_zelda_follower,
        "is_hold_up_item": snapshot.is_hold_up_item,
        "in_zelda_cell": snapshot.in_zelda_cell,
        "in_sanctuary": snapshot.in_sanctuary,
    }


def format_pin_probe(name: str, snapshot: AlttpSnapshot) -> str:
    """One pin line: room plus the AP-007 snapshot extras."""
    return (
        f"{name} room=0x{snapshot.room_base_id:02X} "
        f"dungeon=0x{snapshot.dungeon_id:02X} floor=0x{snapshot.floor:02X} "
        f"layer={snapshot.layer} state={snapshot.link_state} "
        f"hp=0x{snapshot.hp:02X} iframes={snapshot.iframes} "
        f"knockback={snapshot.knockback}"
    )


def player_has_control(env: object, _info: dict | None = None) -> bool:
    """Readiness predicate for shared StartupPlan runners."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return read_snapshot(ram).has_control


def on_hyrule_castle_grounds(env: object, _info: dict | None = None) -> bool:
    """True when Link is controllable on light-world castle screen 0x1B."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return read_snapshot(ram).on_castle_grounds


def in_secret_passage(env: object, _info: dict | None = None) -> bool:
    """True when Link is indoors in secret-passage room 0x55."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return read_snapshot(ram).in_secret_passage


def has_fighter_sword(env: object, _info: dict | None = None) -> bool:
    """True when equip sword level is at least the fighter sword."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return read_snapshot(ram).has_fighter_sword


def castle_entry_accepted(snapshot: AlttpSnapshot) -> bool:
    """Acceptance: entered castle interior (secret passage or other indoor)."""
    return snapshot.in_castle_interior and (not snapshot.dark_world)


def uncle_sword_event_accepted(snapshot: AlttpSnapshot) -> bool:
    """Acceptance: fighter sword present in inventory/equip RAM."""
    return snapshot.has_fighter_sword


def secret_passage_accepted(snapshot: AlttpSnapshot) -> bool:
    """Acceptance: indoors in secret-passage room 0x55."""
    return snapshot.in_secret_passage


def zelda_rescued_accepted(snapshot: AlttpSnapshot) -> bool:
    """Acceptance: Zelda is following (tagalong id 1)."""
    return snapshot.has_zelda_follower


def sanctuary_accepted(snapshot: AlttpSnapshot) -> bool:
    """Acceptance: indoors in Sanctuary room (post-escort)."""
    return snapshot.in_sanctuary


def has_zelda_follower(env: object, _info: dict | None = None) -> bool:
    """True when follower_indicator is Zelda."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return read_snapshot(ram).has_zelda_follower
