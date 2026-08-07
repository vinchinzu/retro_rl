"""RAM fields and snapshots for Zelda I (NES).

Addresses verified against Data Crystal and live fceumm probes (2026-07-27).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retro_harness.ram_state import GameMode, GameState

# --- Core engine ---
ADDR_LEVEL = 0x0010  # 0 = overworld; 1-9 = dungeon
ADDR_MODE = 0x0012  # 5=play, 6/7=scroll, 11=cave/underworld play, 16=cave enter
ADDR_DIALOG_TIMER = 0x0029
ADDR_LINK_X = 0x0070
ADDR_LINK_Y = 0x0084
ADDR_LINK_FACING = 0x0098  # $08 N, $04 S, $01 E, $02 W
ADDR_SCREEN = 0x00EB  # overworld: y_nibble<<4 | x_nibble (16x8)
ADDR_NEXT_SCREEN = 0x00EC
ADDR_COLLIDING_TILE = 0x049E

# --- Dungeon room / object state ---
ADDR_ROOM_ITEM_ID = 0x00AB
ADDR_CUR_OPENED_DOORS = 0x00EE  # bit0=R bit1=L bit2=D bit3=U
ADDR_OPEN_DOORWAY_MASK = 0x033F
ADDR_ROOM_ALL_DEAD = 0x034D
ADDR_ROOM_OBJ_COUNT = 0x034E
ADDR_OBJ_TYPE = 0x034F  # 16 slots
ADDR_OBJ_HP = 0x0485  # 13 gameplay slots used by the engine

# --- Inventory / progress (file slot mirrored in WRAM) ---
ADDR_SWORD = 0x0657  # 0=none, 1=wooden, 2=white, 3=magical
ADDR_BOMBS = 0x0658
ADDR_ARROWS = 0x0659
ADDR_BOW = 0x065A  # present when non-zero with arrows
ADDR_CANDLE = 0x065B
ADDR_WHISTLE = 0x065C
ADDR_FOOD = 0x065D
ADDR_POTION = 0x065E
ADDR_ROD = 0x065F
ADDR_RAFT = 0x0660
ADDR_BOOK = 0x0661
ADDR_RING = 0x0662
ADDR_LADDER = 0x0663
ADDR_MAGIC_KEY = 0x0664
ADDR_BRACELET = 0x0665
ADDR_LETTER = 0x0666
ADDR_COMPASS = 0x0667
ADDR_MAP = 0x0668
ADDR_RUPEES = 0x066D
ADDR_KEYS = 0x066E
ADDR_HEALTH = 0x066F  # high nibble = containers-1, low = filled hearts
ADDR_TRIFORCE = 0x0671
# Boomerangs sit after triforce in the file-slot inventory block (Data Crystal).
# Magical (0x0675) overrides wooden (0x0674) when both would apply.
ADDR_BOOMERANG = 0x0674  # wooden; 0=false, 1=true
ADDR_MAGIC_BOOMERANG = 0x0675  # magical full-screen; 0=false, 1=true
ADDR_MAGIC_SHIELD = 0x0676

# Overworld start + first milestones
SCREEN_START = 0x77
SCREEN_NORTH_OF_START = 0x67
SCREEN_LEVEL1_ENTRANCE = 0x37
SCREEN_LEVEL2_ENTRANCE = 0x3C  # walkthrough-correlated Moon overworld door
SCREEN_LEVEL2_ENTRY_ROOM = 0x7D  # Moon dungeon south mouth (live settle)
CAVE_MODE = 11
PLAY_MODE = 5

SWORD_WOODEN = 1


@dataclass(frozen=True)
class ZeldaObject:
    """One live engine object slot used by dungeon combat policies."""

    slot: int
    type_id: int
    x: int
    y: int
    facing: int
    hp: int
    state: int


@dataclass(frozen=True)
class ZeldaSnapshot:
    """Frame snapshot for routing and segment stop predicates."""

    mode: int
    level: int
    screen: int
    next_screen: int
    link_x: int
    link_y: int
    facing: int
    sword: int
    bombs: int
    rupees: int
    keys: int
    health: int
    triforce: int
    compass: int  # ADDR_COMPASS bitfield: one bit per dungeon level
    dialog_timer: int
    colliding_tile: int
    room_item_id: int
    room_all_dead: int
    room_obj_count: int
    cur_opened_doors: int
    open_doorway_mask: int
    objects: tuple[ZeldaObject, ...]

    @property
    def overworld(self) -> bool:
        return self.level == 0 and self.mode == PLAY_MODE

    @property
    def in_cave(self) -> bool:
        return self.mode == CAVE_MODE and self.level == 0

    @property
    def transitioning(self) -> bool:
        return self.mode in (6, 7, 16)

    @property
    def has_sword(self) -> bool:
        return self.sword >= SWORD_WOODEN

    @property
    def screen_col(self) -> int:
        return int(self.screen) & 0x0F

    @property
    def screen_row(self) -> int:
        return (int(self.screen) >> 4) & 0x0F

    @property
    def heart_containers(self) -> int:
        return ((int(self.health) >> 4) & 0x0F) + 1

    @property
    def filled_hearts(self) -> int:
        return int(self.health) & 0x0F

    @property
    def health_is_full(self) -> bool:
        """True when the low nibble is the full-health sentinel ``0xF``."""
        return (int(self.health) & 0x0F) >= 0x0F

    def object_in_slot(self, slot: int) -> ZeldaObject | None:
        return next((obj for obj in self.objects if obj.slot == slot), None)


def full_health_byte(health: int) -> int:
    """Preserve heart containers (high nibble); set filled hearts to full (``0xF``).

    Does **not** grant containers. Used by the survival assist only.
    """
    return (int(health) & 0xF0) | 0x0F


def read_u8(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr])


def read_snapshot(ram: np.ndarray) -> ZeldaSnapshot:
    """Read a routing snapshot from stable-retro NES RAM."""
    objects = tuple(
        ZeldaObject(
            slot=slot,
            type_id=read_u8(ram, ADDR_OBJ_TYPE + slot),
            x=read_u8(ram, ADDR_LINK_X + slot),
            y=read_u8(ram, ADDR_LINK_Y + slot),
            facing=read_u8(ram, ADDR_LINK_FACING + slot),
            hp=read_u8(ram, ADDR_OBJ_HP + slot),
            state=read_u8(ram, 0x00AC + slot),
        )
        for slot in range(13)
    )
    return ZeldaSnapshot(
        mode=read_u8(ram, ADDR_MODE),
        level=read_u8(ram, ADDR_LEVEL),
        screen=read_u8(ram, ADDR_SCREEN),
        next_screen=read_u8(ram, ADDR_NEXT_SCREEN),
        link_x=read_u8(ram, ADDR_LINK_X),
        link_y=read_u8(ram, ADDR_LINK_Y),
        facing=read_u8(ram, ADDR_LINK_FACING),
        sword=read_u8(ram, ADDR_SWORD),
        bombs=read_u8(ram, ADDR_BOMBS),
        rupees=read_u8(ram, ADDR_RUPEES),
        keys=read_u8(ram, ADDR_KEYS),
        health=read_u8(ram, ADDR_HEALTH),
        triforce=read_u8(ram, ADDR_TRIFORCE),
        compass=read_u8(ram, ADDR_COMPASS),
        dialog_timer=read_u8(ram, ADDR_DIALOG_TIMER),
        colliding_tile=read_u8(ram, ADDR_COLLIDING_TILE),
        room_item_id=read_u8(ram, ADDR_ROOM_ITEM_ID),
        room_all_dead=read_u8(ram, ADDR_ROOM_ALL_DEAD),
        room_obj_count=read_u8(ram, ADDR_ROOM_OBJ_COUNT),
        cur_opened_doors=read_u8(ram, ADDR_CUR_OPENED_DOORS),
        open_doorway_mask=read_u8(ram, ADDR_OPEN_DOORWAY_MASK),
        objects=objects,
    )


def is_level1_ready(ram, obs_mean: float | None = None) -> bool:
    """True on controllable overworld start (not title/file/inventory-dark)."""
    snap = read_snapshot(ram)
    if snap.mode != PLAY_MODE:
        return False
    if snap.level != 0:
        return False
    if snap.health <= 0:
        return False
    if obs_mean is not None and obs_mean <= 50.0:
        return False
    return True


def is_sword_obtained(ram) -> bool:
    return read_u8(ram, ADDR_SWORD) >= SWORD_WOODEN


def is_on_start_overworld(ram) -> bool:
    snap = read_snapshot(ram)
    return snap.overworld and snap.screen == SCREEN_START


def capabilities_from_ram(ram) -> frozenset[str]:
    """Inventory / event capabilities for route planning."""
    snap = read_snapshot(ram)
    caps: set[str] = set()
    if snap.sword >= 1:
        caps.add("wooden_sword")
    if snap.sword >= 2:
        caps.add("white_sword")
    if snap.sword >= 3:
        caps.add("magical_sword")
    if snap.bombs > 0:
        caps.add("bombs")
    if snap.keys > 0:
        caps.add("keys")
    if read_u8(ram, ADDR_LADDER):
        caps.add("ladder")
    if read_u8(ram, ADDR_RAFT):
        caps.add("raft")
    if read_u8(ram, ADDR_BRACELET):
        caps.add("bracelet")
    if read_u8(ram, ADDR_CANDLE):
        caps.add("candle")
    if read_u8(ram, ADDR_MAGIC_BOOMERANG):
        caps.add("magical_boomerang")
    elif read_u8(ram, ADDR_BOOMERANG):
        caps.add("boomerang")
    return frozenset(caps)


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project confirmed fields into ``GameState``."""
    snap = read_snapshot(ram)
    ready = is_level1_ready(ram)
    if snap.in_cave:
        mode = GameMode.PLAYING
    elif snap.overworld:
        mode = GameMode.PLAYING
    elif snap.transitioning:
        mode = GameMode.CUTSCENE
    elif snap.mode in (0, 1):
        mode = GameMode.TITLE
    else:
        mode = GameMode.MENU if not ready else GameMode.PLAYING

    extras = {
        "level1_ready": ready,
        "ram_map_partial": False,
        "mode_raw": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "screen_col": snap.screen_col,
        "screen_row": snap.screen_row,
        "sword": snap.sword,
        "bombs": snap.bombs,
        "rupees": snap.rupees,
        "keys": snap.keys,
        "triforce": snap.triforce,
        "room_item_id": snap.room_item_id,
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "open_doorway_mask": snap.open_doorway_mask,
        "in_cave": snap.in_cave,
        "overworld": snap.overworld,
        "facing": snap.facing,
        "capabilities": sorted(capabilities_from_ram(ram)),
    }
    return GameState(
        frame=frame,
        mode=mode,
        stage=snap.level,
        room=snap.screen,
        player_x=snap.link_x,
        player_y=snap.link_y,
        health=snap.health,
        lives=0,
        enemies=(),
        extras=extras,
    )
