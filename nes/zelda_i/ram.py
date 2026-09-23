"""RAM fields and snapshots for Zelda I (NES).

Addresses verified against Data Crystal and live fceumm probes (2026-07-27).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retro_harness.ram_state import GameMode, GameState

# --- Core engine ---
ADDR_LEVEL = 0x0010  # 0 = overworld; 1-9 = dungeon
ADDR_IS_UPDATING_MODE = 0x0011  # 0=mode init, nonzero=ordinary update loop
ADDR_MODE = 0x0012  # 5=play, 6/7=scroll, 9=passage, 11=cave, 16=cave enter
ADDR_SUBMODE = 0x0013  # mode-local phase; ending uses 3=credits, 4=final screen
ADDR_DIALOG_TIMER = 0x0029
# Forced drops (aldonunez Link_BeHarmed / redcandle). Zeroed on Link-enemy
# *collision* — bubbles and the recorder whirlwind included — not on a
# $066F change. A wooden-sword octorok chip is $0670 only ($80 of partial);
# Survival assist writes $0670 back to $FF the same frame, so a post-assist
# snapshot can read full health with $50/$627 already 0. $04F0 (Link's
# ObjInvincibilityTimer) is the collision that survives the refill.
ADDR_HELP_DROP_COUNT = 0x0050  # 10 kills → forced 5-rupee (or bomb if VALUE set)
ADDR_HELP_DROP_VALUE = 0x0051  # nonzero → the 10-kill force is a bomb
ADDR_WORLD_KILL_COUNT = 0x0627  # 16 kills → forced fairy
ADDR_LINK_IFRAMES = 0x04F0  # ObjInvincibilityTimer[0]; 24 on Link_BeHarmed
ADDR_WORLD_KILL_CYCLE = 0x052A  # 0-9 random-drop table column; not a streak
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
ADDR_OBJ_STATE = 0x00AC  # 13 slots; floor-drop item code lives here
ADDR_OBJ_HP = 0x0485  # 13 gameplay slots used by the engine
# Link's own weapon slots sit above the 13 the engine hands enemies: $0D is
# the blade, and at blade state 3 ``Z_07 MakeSwordShot`` puts the shot in $0E.
SWORD_SHOT_SLOT = 0x0E

# --- Inventory / progress (file slot mirrored in WRAM) ---
ADDR_SELECTED_ITEM = 0x0656  # B-item slot: 1=bombs, 2=arrows, 4=candle
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
ADDR_HEALTH = 0x066F  # HeartValues: hi = containers−1, lo = whole hearts
ADDR_HEART_PARTIAL = 0x0670  # HeartPartial: $FF = current heart full
ADDR_TRIFORCE = 0x0671
# Boomerangs sit after triforce in the file-slot inventory block (Data Crystal).
# Magical (0x0675) overrides wooden (0x0674) when both would apply.
ADDR_BOOMERANG = 0x0674  # wooden; 0=false, 1=true
ADDR_MAGIC_BOOMERANG = 0x0675  # magical full-screen; 0=false, 1=true
ADDR_MAGIC_SHIELD = 0x0676
ADDR_MAX_BOMBS = 0x067C
# aldonunez WorldFlags: 128-byte array indexed by overworld screen / dungeon
# room. Save-file copy lives in battery RAM ($6092+). One bit per OW screen
# records whether that screen's secret is found (ZeldaHacks).
ADDR_WORLD_FLAGS = 0x067F
ADDR_RUPEES_TO_ADD = 0x067D  # pending credit; HUD counts up toward $066D
ADDR_RUPEES_TO_SUBTRACT = 0x067E
WORLD_FLAG_ITEM = 0x10  # UW item taken; OW secret revealed
WORLD_FLAG_VISITED = 0x20
WORLD_FLAG_KILLS = 0xC0  # 0/1/2+ kills packed in the high bits

# Overworld start + first milestones
SCREEN_START = 0x77
SCREEN_NORTH_OF_START = 0x67
SCREEN_LEVEL1_ENTRANCE = 0x37
SCREEN_LEVEL2_ENTRANCE = 0x3C  # walkthrough-correlated Moon overworld door
SCREEN_LEVEL2_ENTRY_ROOM = 0x7D  # Moon dungeon south mouth (live settle)
CAVE_MODE = 11
PLAY_MODE = 5
PASSAGE_MODE = 9

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
    # Boomerang inventory (Data Crystal); magical overrides wooden when set.
    # Defaults keep older ZeldaSnapshot(...) test constructors working.
    boomerang: int = 0
    magical_boomerang: int = 0
    submode: int = 0
    is_updating_mode: int = 0
    heart_partial: int = 0xFF
    raft: int = 0
    ladder: int = 0
    map: int = 0  # ADDR_MAP bitfield: one bit per dungeon level
    rod: int = 0  # ADDR_ROD; default 0 so older constructors still work
    bow: int = 0  # ADDR_BOW
    arrows: int = 0  # ADDR_ARROWS; wooden=1 silver=2
    candle: int = 0  # ADDR_CANDLE; blue=1 red=2
    food: int = 0  # ADDR_FOOD (meat)
    letter: int = 0  # ADDR_LETTER; 1 once the 0x0E old man's letter is taken
    magic_shield: int = 0  # ADDR_MAGIC_SHIELD; blocks fireballs when owned
    # Forced-drop kill counters. Link_BeHarmed (collision) zeros all three.
    # Defaults keep older ZeldaSnapshot(...) test constructors working.
    world_kill_count: int = 0  # ADDR_WORLD_KILL_COUNT; 16 → fairy
    help_drop_count: int = 0  # ADDR_HELP_DROP_COUNT; 10 → 5-rupee (or bomb)
    help_drop_value: int = 0  # ADDR_HELP_DROP_VALUE; nonzero → bomb at 10
    link_iframes: int = 0  # ADDR_LINK_IFRAMES; 0→24 is the collision that zeros them
    # Slot $0E, read apart from ``objects``: the 13-slot census is enemies, and
    # a weapon slot inside it would read as prey (its type byte is not an enemy).
    sword_shot: ZeldaObject = field(
        default_factory=lambda: ZeldaObject(
            slot=SWORD_SHOT_SLOT, type_id=0, x=0, y=0, facing=0, hp=0, state=0
        )
    )

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
        """``$066F`` low nibble. This is **whole hearts minus one**.

        Kept as the raw nibble because the L1 chain is frame-perfect against
        it. Use :attr:`whole_hearts` for anything that reasons about how much
        life is left — ``filled_hearts <= 1`` reads "one heart" and means two.
        """
        return int(self.health) & 0x0F

    @property
    def whole_hearts(self) -> int:
        """Whole hearts Link is actually holding (``$066F`` low nibble + 1).

        Symmetric with :attr:`heart_containers` (high nibble + 1): a full
        3-container Link is ``0x22`` — three containers and three hearts — and
        the live pre-L1 walk ended alive on ``0x20``, which is one heart, not
        zero. ``health_is_full`` is ``lo == hi`` for the same reason.
        """
        return (int(self.health) & 0x0F) + 1

    @property
    def health_is_full(self) -> bool:
        """True when whole hearts match containers (low nibble == high nibble)."""
        hv = int(self.health)
        return (hv & 0x0F) == ((hv >> 4) & 0x0F)

    def object_in_slot(self, slot: int) -> ZeldaObject | None:
        return next((obj for obj in self.objects if obj.slot == slot), None)


def hearts_held(snap: "ZeldaSnapshot") -> float:
    """Life as a float in hearts: the low nibble plus the partial heart.

    ``$066F`` low nibble counts whole hearts minus one and ``$0670`` is the
    heart being eaten (``$FF`` full), so ``0x22``/``$FF`` is 3.0.
    """
    return (int(snap.health) & 0x0F) + int(snap.heart_partial) / 255.0


def ram_hearts(ram: np.ndarray) -> float:
    """:func:`hearts_held` straight from RAM, without a snapshot."""
    return (int(ram[ADDR_HEALTH]) & 0x0F) + int(ram[ADDR_HEART_PARTIAL]) / 255.0


def full_health_byte(health: int) -> int:
    """Full ``HeartValues`` for this container count: low nibble == high nibble.

    Zelda 1 (aldonunez ``CompareHeartsToContainers`` / ``World_FillHearts``):
    low nibble is whole hearts, **not** a ``0xF`` full flag. Writing ``0xF``
    makes the triforce/potion fill ``INC HeartValues`` until the nibbles
    match, which grants extra containers.
    """
    n = (int(health) >> 4) & 0x0F
    return (n << 4) | n


def health_byte_is_coherent(health: int) -> bool:
    """True when ``$066F`` is a byte normal play can hold.

    ``hi = containers - 1`` and ``lo = whole hearts``, so ``lo <= hi`` always:
    Link cannot hold more filled hearts than he has containers. The ROM does
    not clamp a byte that breaks this — it just decrements the low nibble
    happily — so an incoherent pin silently hands a lane several times the
    damage budget a real Link has, and every heart number measured from it is
    against a fake denominator (``Level6Entrance`` held ``0x2F``: 15 hearts in
    3 containers, and four sittings of L6 room tuning were graded against it).
    Check this on any pin before quoting hearts from it.
    """
    hv = int(health) & 0xFF
    return (hv & 0x0F) <= ((hv >> 4) & 0x0F)


def health_byte_for_containers(containers: int, *, filled: int | None = None) -> int:
    """Encode ``HeartValues`` from a known container count (never from a glitch)."""
    n = (max(1, int(containers)) - 1) & 0x0F
    lo = n if filled is None else max(0, int(filled) - 1) & 0x0F
    return (n << 4) | lo


def read_u8(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr])


def read_object(ram: np.ndarray, slot: int) -> ZeldaObject:
    """One engine object slot. Weapon slots ($0D/$0E) read the same way."""
    return ZeldaObject(
        slot=slot,
        type_id=read_u8(ram, ADDR_OBJ_TYPE + slot),
        x=read_u8(ram, ADDR_LINK_X + slot),
        y=read_u8(ram, ADDR_LINK_Y + slot),
        facing=read_u8(ram, ADDR_LINK_FACING + slot),
        hp=read_u8(ram, ADDR_OBJ_HP + slot),
        state=read_u8(ram, ADDR_OBJ_STATE + slot),
    )


def read_snapshot(ram: np.ndarray) -> ZeldaSnapshot:
    """Read a routing snapshot from stable-retro NES RAM."""
    objects = tuple(read_object(ram, slot) for slot in range(13))
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
        heart_partial=read_u8(ram, ADDR_HEART_PARTIAL),
        triforce=read_u8(ram, ADDR_TRIFORCE),
        compass=read_u8(ram, ADDR_COMPASS),
        map=read_u8(ram, ADDR_MAP),
        dialog_timer=read_u8(ram, ADDR_DIALOG_TIMER),
        colliding_tile=read_u8(ram, ADDR_COLLIDING_TILE),
        room_item_id=read_u8(ram, ADDR_ROOM_ITEM_ID),
        room_all_dead=read_u8(ram, ADDR_ROOM_ALL_DEAD),
        room_obj_count=read_u8(ram, ADDR_ROOM_OBJ_COUNT),
        cur_opened_doors=read_u8(ram, ADDR_CUR_OPENED_DOORS),
        open_doorway_mask=read_u8(ram, ADDR_OPEN_DOORWAY_MASK),
        objects=objects,
        boomerang=read_u8(ram, ADDR_BOOMERANG),
        magical_boomerang=read_u8(ram, ADDR_MAGIC_BOOMERANG),
        submode=read_u8(ram, ADDR_SUBMODE),
        is_updating_mode=read_u8(ram, ADDR_IS_UPDATING_MODE),
        magic_shield=read_u8(ram, ADDR_MAGIC_SHIELD),
        raft=read_u8(ram, ADDR_RAFT),
        ladder=read_u8(ram, ADDR_LADDER),
        rod=read_u8(ram, ADDR_ROD),
        bow=read_u8(ram, ADDR_BOW),
        arrows=read_u8(ram, ADDR_ARROWS),
        candle=read_u8(ram, ADDR_CANDLE),
        food=read_u8(ram, ADDR_FOOD),
        letter=read_u8(ram, ADDR_LETTER),
        world_kill_count=read_u8(ram, ADDR_WORLD_KILL_COUNT),
        help_drop_count=read_u8(ram, ADDR_HELP_DROP_COUNT),
        help_drop_value=read_u8(ram, ADDR_HELP_DROP_VALUE),
        link_iframes=read_u8(ram, ADDR_LINK_IFRAMES),
        sword_shot=read_object(ram, SWORD_SHOT_SLOT),
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


def world_flag(ram: np.ndarray, screen: int) -> int:
    """``WorldFlags[screen]`` at ``$067F+screen``. Screen 0x00..0x7F."""
    return read_u8(ram, ADDR_WORLD_FLAGS + (int(screen) & 0x7F))


def ow_secret_taken(ram: np.ndarray, screen: int) -> bool:
    """True when this overworld screen's secret/item bit (``$10``) is set."""
    return bool(world_flag(ram, screen) & WORLD_FLAG_ITEM)


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
