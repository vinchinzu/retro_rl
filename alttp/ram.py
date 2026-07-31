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
LINK_ACTION = 0x5D
SCREEN_ID = 0x008A
DARK_WORLD_FLAG = 0x0FFF
BG2_VOFS = 0x00E8
CAMERA_X = 0xE2
LINK_HP = 0xF36D
LINK_MAX_HP = 0xF36C
EQUIP_SWORD = 0xF359
LINK_ITEM_LAMP = 0xF34A
NUM_KEYS = 0xF36F
FOLLOWER = 0xF3CC  # tagalong / follower_indicator

HYRULE_CASTLE_SCREEN = 0x1B
LINKS_HOUSE_SCREEN = 0x2C
LINKS_HOUSE_ROOM = 0x0004
SANCTUARY_SCREEN = 0x13

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
# Zelda's cell (opening rescue target; confirm on real ROM before claiming).
ZELDA_CELL_ROOM = 0x80
# Sanctuary indoor room after escort (confirm on real ROM before claiming).
SANCTUARY_ROOM = 0x12

# Human labels for room_base_id (hex only as fallback).
ROOM_LABELS: dict[int, str] = {
    LINKS_HOUSE_ROOM & 0xFF: "links_house",
    HYRULE_CASTLE_SECRET_ENTRANCE_ROOM: "hyrule_castle_secret_entrance",
    HYRULE_CASTLE_MAIN_HALL_ROOM: "hyrule_castle_main_hall",
    ZELDA_CELL_ROOM: "zelda_cell",
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
