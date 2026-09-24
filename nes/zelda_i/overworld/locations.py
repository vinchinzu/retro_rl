"""First-quest overworld cave / secret / enemy-farm catalog.

ROM (iNES+16, Data Crystal / aldonunez):

- ``$18480`` ``LevelBlockAttrsB``: ``%LLLL LLPP`` — cave dest in bits 7-2
  (``byte >> 2``). 1-9 = dungeons; ``$10+`` = caves.
- ``$18680`` ``LevelBlockAttrsF``: bit7 = ignore in Q1, bit6 = ignore in Q2.
- ``$18500`` monster placement: ``%CCMM MMMM`` — count index / monster id.
- ``$18580`` bit7 = monster id is a mixed group, not an ObjType.
- ``$19324`` count table: index 0..3 → 1, 4, 5, 6 foes.
- WRAM ``WorldFlags[$067F+screen]`` bit ``$10`` = secret found / item taken.
- WRAM ``$50`` / ``$51`` / ``$627`` = forced 5-rupee / bomb / fairy kill counters.

Standard z1r cave shuffle moves ``AttrsB`` dests; the screen's open method
stays. Enemy shuffle (when on) rewrites ``$18500``. ``locations_from_rom`` /
``spawns_from_rom`` reread those tables.

Grid: column A-P = 0-F, row 1-8 = north to south. Screen ``(row-1)<<4 | col``.
Start is H8 = ``0x77``.
"""

from __future__ import annotations

from dataclasses import dataclass

# iNES header; Data Crystal offsets are ROM-file offsets without it.
INES_HEADER = 16
ROM_OW_ATTRS_B = 0x18480
ROM_OW_ATTRS_F = 0x18680
ROM_OW_SPAWN = 0x18500
ROM_OW_LAYOUT = 0x18580
ROM_OW_COUNTS = 0x19324
Q1_IGNORE = 0x80
Q2_IGNORE = 0x40
# $19324 first four bytes (vanilla). count_idx 0..3 → foe count.
SPAWN_COUNTS: tuple[int, ...] = (1, 4, 5, 6)

# AttrsB >> 2 (first quest)
CAVE_NONE = 0x00
CAVE_DUNGEON_1 = 0x01
CAVE_DUNGEON_2 = 0x02
CAVE_DUNGEON_3 = 0x03
CAVE_DUNGEON_4 = 0x04
CAVE_DUNGEON_5 = 0x05
CAVE_DUNGEON_6 = 0x06
CAVE_DUNGEON_7 = 0x07
CAVE_DUNGEON_8 = 0x08
CAVE_DUNGEON_9 = 0x09
CAVE_WOOD_SWORD = 0x10
CAVE_TAKE_ANY = 0x11
CAVE_WHITE_SWORD = 0x12
CAVE_MAGICAL_SWORD = 0x13
CAVE_WARP = 0x14
CAVE_HINT = 0x15
CAVE_GAMBLE = 0x16
CAVE_DOOR_REPAIR = 0x17
CAVE_LETTER = 0x18
CAVE_UNIQUE_19 = 0x19
CAVE_POTION = 0x1A
CAVE_PAID_HINT = 0x1B  # three rupee slots, 5/10/20
CAVE_PAID_HINT_B = 0x1C  # three rupee slots, 10/30/50 (not a shop)
CAVE_SHOP_ARROWS = 0x1D  # shield 130 / bombs 20 / arrows 80 (0x4A, 0x6F)
CAVE_SHOP_CANDLE = 0x1E  # shield 160 / key 100 / blue candle 60 (0x0C, 0x5E)
CAVE_SHOP_ALT = 0x1F  # shield 90 / bait 100 / heart 10
CAVE_SHOP_SPECIAL = 0x20  # 0x34 only: key 80 / blue ring 250 / bait 60
# "It's a secret to everybody": one rupee pedestal whose price is the payout.
CAVE_RUPEES_30 = 0x21
CAVE_RUPEES_100 = 0x22
CAVE_RUPEES_10 = 0x23

CAVE_KIND: dict[int, str] = {
    CAVE_NONE: "none",
    CAVE_DUNGEON_1: "dungeon",
    CAVE_DUNGEON_2: "dungeon",
    CAVE_DUNGEON_3: "dungeon",
    CAVE_DUNGEON_4: "dungeon",
    CAVE_DUNGEON_5: "dungeon",
    CAVE_DUNGEON_6: "dungeon",
    CAVE_DUNGEON_7: "dungeon",
    CAVE_DUNGEON_8: "dungeon",
    CAVE_DUNGEON_9: "dungeon",
    CAVE_WOOD_SWORD: "sword",
    CAVE_TAKE_ANY: "take_any",
    CAVE_WHITE_SWORD: "sword",
    CAVE_MAGICAL_SWORD: "sword",
    CAVE_WARP: "warp",
    CAVE_HINT: "hint",
    CAVE_GAMBLE: "gamble",
    CAVE_DOOR_REPAIR: "door_repair",
    CAVE_LETTER: "letter",
    CAVE_UNIQUE_19: "hint",
    CAVE_POTION: "potion",
    CAVE_PAID_HINT: "hint",
    CAVE_PAID_HINT_B: "hint",
    CAVE_SHOP_ARROWS: "shop",
    CAVE_SHOP_CANDLE: "shop",
    CAVE_SHOP_ALT: "shop",
    CAVE_SHOP_SPECIAL: "shop",
    CAVE_RUPEES_30: "rupees",
    CAVE_RUPEES_100: "rupees",
    CAVE_RUPEES_10: "rupees",
}

OPEN_OPEN = "open"
OPEN_BOMB = "bomb"
OPEN_BURN = "burn"
OPEN_ARMOS = "armos"
OPEN_PUSH_GRAVE = "push_grave"
OPEN_RECORDER = "recorder"
OPEN_RAFT = "raft"
OPEN_LADDER = "ladder"
OPEN_PUSH_ROCK = "push_rock"  # needs the Power Bracelet
OPEN_SECRET = "secret"  # hidden, method not yet measured

EVIDENCE_VERIFIED = "verified"
EVIDENCE_SOURCE = "source"

COLS = "ABCDEFGHIJKLMNOP"


def grid_name(screen: int) -> str:
    """FAQ grid label (A1 north-west … P8 south-east) for an OW screen id."""
    col = int(screen) & 0x0F
    row = (int(screen) >> 4) & 0x07
    return f"{COLS[col]}{row + 1}"


def cave_id_from_attrs_b(attrs_b: int) -> int:
    return (int(attrs_b) >> 2) & 0x3F


@dataclass(frozen=True)
class OwScreenAttrs:
    """One overworld screen's ROM cave-dest / quest-ignore bits."""

    screen: int
    cave_id: int
    q1_ignore: bool
    q2_ignore: bool


@dataclass(frozen=True)
class OwLocation:
    """One first-quest overworld location (cave, dungeon mouth, or OW item)."""

    name: str
    screen: int
    cave_id: int
    kind: str
    vanilla: str
    open: str
    evidence: str
    shop_slots: int = 0  # z1r: each shop pedestal is a location

    @property
    def grid(self) -> str:
        return grid_name(self.screen)

    @property
    def q1_item_location(self) -> bool:
        """True if this screen is an item/shop/heart location (not a dungeon)."""
        return self.kind not in ("dungeon", "warp", "hint", "door_repair", "gamble", "fairy")


def decode_ow_attrs(rom: bytes) -> tuple[OwScreenAttrs, ...]:
    """Decode 128 overworld cave dests from an iNES or headerless PRG dump."""
    data = rom[INES_HEADER:] if rom[:4] == b"NES\x1a" else rom
    attrs_b = data[ROM_OW_ATTRS_B : ROM_OW_ATTRS_B + 128]
    attrs_f = data[ROM_OW_ATTRS_F : ROM_OW_ATTRS_F + 128]
    if len(attrs_b) != 128 or len(attrs_f) != 128:
        raise ValueError("ROM too short for overworld AttrsB/F tables")
    return tuple(
        OwScreenAttrs(
            screen=i,
            cave_id=cave_id_from_attrs_b(attrs_b[i]),
            q1_ignore=bool(attrs_f[i] & Q1_IGNORE),
            q2_ignore=bool(attrs_f[i] & Q2_IGNORE),
        )
        for i in range(128)
    )


# Cave wares, ROM file offset $18610 (iNES header included): three item
# bytes per cave type $10..$23, then the three prices at $1864C. Item low six
# bits are the item id ($3F = empty slot, $18 = rupee); a secret cave is one
# rupee in the middle slot whose "price" is the payout. Decoded 2026-09-23
# and matched by the live 0x0F (100R) and 0x48 (30R) takes.
ROM_CAVE_ITEMS = 0x18610
ROM_CAVE_PRICES = ROM_CAVE_ITEMS + 60
CAVE_FIRST = 0x10
CAVE_COUNT = 20
ITEM_NONE = 0x3F
ITEM_RUPEE = 0x18
ITEM_BLUE_POTION = 0x1F
ITEM_RED_POTION = 0x20


@dataclass(frozen=True)
class CaveWare:
    """One cave pedestal: item id (low six bits) and its price in rupees."""

    item: int
    price: int

    @property
    def empty(self) -> bool:
        return self.item == ITEM_NONE


def decode_cave_wares(rom: bytes) -> dict[int, tuple[CaveWare, CaveWare, CaveWare]]:
    """Cave type -> its three pedestals (left, middle, right), from the ROM file."""
    items = rom[ROM_CAVE_ITEMS : ROM_CAVE_ITEMS + 3 * CAVE_COUNT]
    prices = rom[ROM_CAVE_PRICES : ROM_CAVE_PRICES + 3 * CAVE_COUNT]
    if len(items) != 3 * CAVE_COUNT or len(prices) != 3 * CAVE_COUNT:
        raise ValueError("ROM too short for the cave wares tables")
    return {
        CAVE_FIRST + i: tuple(
            CaveWare(item=items[3 * i + k] & 0x3F, price=prices[3 * i + k]) for k in range(3)
        )
        for i in range(CAVE_COUNT)
    }


def secret_payout(wares: tuple[CaveWare, CaveWare, CaveWare]) -> int | None:
    """Rupees a one-rupee secret cave pays, else ``None``."""
    left, middle, right = wares
    if left.empty and middle.item == ITEM_RUPEE and right.empty:
        return int(middle.price)
    return None


# Rupees each secret cave type pays (the ROM table above).
SECRET_PAYOUT: dict[int, int] = {CAVE_RUPEES_30: 30, CAVE_RUPEES_100: 100, CAVE_RUPEES_10: 10}


def cave_keeper(cave_id: int) -> int:
    """Object type of the cave's keeper: cave type + $5A.

    Measured: take-any $11 -> $6B, letter $18 -> $72, 30R $21 -> $7B,
    100R $22 -> $7C. The keeper's arrival is when the cave has loaded.
    """
    return int(cave_id) + 0x5A


# Hidden-secret tile objects: RAM object slot 11 the frame the screen loads
# (``Z_04`` ``UpdateTree`` / ``UpdateRockWall``). The reveal sets $80 in the
# screen's world flag. Stands: a bomb 5-8 px below the rock facing UP (0x7B
# rock (144, 80) opens from (144, 88), 0x2C's (144, 160) from (144, 165)),
# preferably the lattice node. A candle flame spawns 16 px ahead of Link
# and walks 16 more: facing DOWN from tree y - 27 it stands at tree y + 5
# (swept 2026-09-23 on 0x28/0x56/0x5B/0x6B; from y - 19 0x28 revealed too
# late and Link slid off the stairs), or 20 px beside the tree on row
# tree y - 3 facing it (0x48 tree (208, 96) opens from (188, 93) RIGHT).
SECRET_ROCK = 0x63
SECRET_TREE = 0x64


@dataclass(frozen=True)
class OwSecret:
    """A hidden cave mouth: tile object, where Link opens it from, and the pay."""

    screen: int
    cave_id: int
    obj: int  # SECRET_ROCK (bomb) or SECRET_TREE (candle)
    x: int
    y: int
    stand: tuple[int, int]
    face: str

    @property
    def rupees(self) -> int:
        return SECRET_PAYOUT.get(self.cave_id, 0)

    @property
    def keeper(self) -> int:
        return cave_keeper(self.cave_id)

    @property
    def uses_bomb(self) -> bool:
        return self.obj == SECRET_ROCK


# Measured 2026-09-23 from BFS_<screen> pins and a walk to 0x62: the slot-11
# object xy. Stands follow the rules above and sit on the ROM turn lattice.
# 0x3D (30R) and 0x4E (10R) open by touching the right-hand Armos, not a
# tile object, so they are not rows here.
SECRET_RUPEE_CAVES: dict[int, OwSecret] = {
    s.screen: s
    for s in (
        # Rock + 5 on a lattice node, as 0x2C's measured (144, 165) under
        # its rock at y=160. (80, 88) is off-node: UP and the approach's
        # DOWN swapped 1 px for 2000 frames (2026-09-23).
        OwSecret(0x2D, CAVE_RUPEES_30, SECRET_ROCK, 80, 80, (80, 85), "UP"),
        # Live +30/+10 from what-if items (docs/research/SECRET_CAVES.md).
        # 0x13 is west of the 0x17 river (stepladder); 0x67 is one hop
        # north of the start screen.
        OwSecret(0x13, CAVE_RUPEES_30, SECRET_ROCK, 32, 80, (32, 85), "UP"),
        OwSecret(0x67, CAVE_RUPEES_30, SECRET_ROCK, 112, 80, (112, 85), "UP"),
        OwSecret(0x71, CAVE_RUPEES_30, SECRET_ROCK, 80, 80, (80, 85), "UP"),
        OwSecret(0x51, CAVE_RUPEES_10, SECRET_TREE, 144, 160, (144, 141), "DOWN"),
        OwSecret(0x28, CAVE_RUPEES_30, SECRET_TREE, 208, 160, (208, 133), "DOWN"),
        OwSecret(0x48, CAVE_RUPEES_30, SECRET_TREE, 208, 96, (188, 93), "RIGHT"),
        OwSecret(0x56, CAVE_RUPEES_10, SECRET_TREE, 160, 160, (160, 133), "DOWN"),
        OwSecret(0x5B, CAVE_RUPEES_10, SECRET_TREE, 32, 160, (32, 133), "DOWN"),
        # The tree is one bush of a full-height column (x 128..143) that
        # splits 0x62; the east half (entered from 0x63) burns it facing LEFT.
        OwSecret(0x62, CAVE_RUPEES_100, SECRET_TREE, 128, 96, (148, 93), "LEFT"),
        OwSecret(0x6B, CAVE_RUPEES_100, SECRET_TREE, 128, 160, (128, 133), "DOWN"),
    )
}


# (screen, cave_id, name, vanilla, open, evidence, shop_slots)
# cave_id is the vanilla Q1 AttrsB>>2; kind comes from CAVE_KIND.
# Open method is screen geometry (stays put in a standard cave shuffle).
_Q1_CAVES: tuple[tuple[int, int, str, str, str, str, int], ...] = (
    (0x01, CAVE_DOOR_REPAIR, "door_repair_b1", "pay_20", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x03, CAVE_DOOR_REPAIR, "door_repair_d1", "pay_20", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x04, CAVE_POTION, "potion_e1", "potion", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x05, CAVE_DUNGEON_9, "dungeon_9", "dungeon_9", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x07, CAVE_DOOR_REPAIR, "door_repair_h1", "pay_20", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x0A, CAVE_WHITE_SWORD, "white_sword", "white_sword", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x0B, CAVE_DUNGEON_5, "dungeon_5", "dungeon_5", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x0C, CAVE_SHOP_CANDLE, "shop_m1", "shop", OPEN_OPEN, EVIDENCE_SOURCE, 3),
    (0x0D, CAVE_POTION, "potion_n1", "potion", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x0E, CAVE_LETTER, "letter", "letter", OPEN_OPEN, EVIDENCE_SOURCE, 0),
    (0x0F, CAVE_RUPEES_100, "rupees_100_p1", "rupees_100", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x10, CAVE_GAMBLE, "gamble_a2", "gamble", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x12, CAVE_SHOP_ALT, "shop_c2", "shop", OPEN_BOMB, EVIDENCE_SOURCE, 3),
    (0x13, CAVE_RUPEES_30, "rupees_d2", "rupees_30", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x14, CAVE_DOOR_REPAIR, "door_repair_e2", "pay_20", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x16, CAVE_GAMBLE, "gamble_g2", "gamble", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x1A, CAVE_PAID_HINT, "paid_hint_k2", "hint", OPEN_OPEN, EVIDENCE_SOURCE, 0),
    (0x1C, CAVE_HINT, "hint_m2", "hint", OPEN_ARMOS, EVIDENCE_SOURCE, 0),
    (0x1D, CAVE_WARP, "warp_n2", "warp", OPEN_PUSH_ROCK, EVIDENCE_SOURCE, 0),
    (0x1E, CAVE_DOOR_REPAIR, "door_repair_o2", "pay_20", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x1F, CAVE_GAMBLE, "gamble_p2", "gamble", OPEN_OPEN, EVIDENCE_SOURCE, 0),
    (0x21, CAVE_MAGICAL_SWORD, "magical_sword", "magical_sword", OPEN_PUSH_GRAVE, EVIDENCE_SOURCE, 0),
    (0x22, CAVE_DUNGEON_6, "dungeon_6", "dungeon_6", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x23, CAVE_WARP, "warp_d3", "warp", OPEN_PUSH_ROCK, EVIDENCE_SOURCE, 0),
    (0x25, CAVE_SHOP_ARROWS, "shop_f3", "shop", OPEN_OPEN, EVIDENCE_SOURCE, 3),
    (0x26, CAVE_SHOP_ALT, "shop_g3", "shop", OPEN_BOMB, EVIDENCE_SOURCE, 3),
    (0x27, CAVE_POTION, "potion_h3", "potion", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x28, CAVE_RUPEES_30, "rupees_i3", "rupees_30", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x2C, CAVE_TAKE_ANY, "heart_m3", "heart_container", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x2D, CAVE_RUPEES_30, "rupees_n3", "rupees_30", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x2F, CAVE_TAKE_ANY, "raft_heart", "heart_container", OPEN_RAFT, EVIDENCE_SOURCE, 0),
    (0x33, CAVE_POTION, "potion_d4", "potion", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x34, CAVE_SHOP_SPECIAL, "special_shop_e4", "bait_or_blue_ring", OPEN_ARMOS, EVIDENCE_SOURCE, 3),
    (0x37, CAVE_DUNGEON_1, "dungeon_1", "dungeon_1", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x3C, CAVE_DUNGEON_2, "dungeon_2", "dungeon_2", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x3D, CAVE_RUPEES_30, "rupees_n4", "rupees_30", OPEN_ARMOS, EVIDENCE_VERIFIED, 0),
    (0x42, CAVE_DUNGEON_7, "dungeon_7", "dungeon_7", OPEN_RECORDER, EVIDENCE_VERIFIED, 0),
    (0x44, CAVE_SHOP_ARROWS, "shop_e5", "shop", OPEN_OPEN, EVIDENCE_SOURCE, 3),
    (0x45, CAVE_DUNGEON_4, "dungeon_4", "dungeon_4", OPEN_RAFT, EVIDENCE_VERIFIED, 0),
    (0x46, CAVE_SHOP_ALT, "shop_g5", "shop", OPEN_BURN, EVIDENCE_SOURCE, 3),
    (0x47, CAVE_TAKE_ANY, "heart_h5", "heart_container", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x48, CAVE_RUPEES_30, "rupees_i5", "rupees_30", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x49, CAVE_WARP, "warp_j5", "warp", OPEN_PUSH_ROCK, EVIDENCE_SOURCE, 0),
    (0x4A, CAVE_SHOP_ARROWS, "arrow_shop", "arrows_80", OPEN_OPEN, EVIDENCE_VERIFIED, 3),
    (0x4B, CAVE_POTION, "potion_l5", "potion", OPEN_BURN, EVIDENCE_SOURCE, 0),
    (0x4D, CAVE_SHOP_ALT, "shop_n5", "shop", OPEN_BURN, EVIDENCE_SOURCE, 3),
    (0x4E, CAVE_RUPEES_10, "rupees_10_o5", "rupees_10", OPEN_ARMOS, EVIDENCE_VERIFIED, 0),
    (0x51, CAVE_RUPEES_10, "rupees_10_b6", "rupees_10", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x56, CAVE_RUPEES_10, "rupees_10_g6", "rupees_10", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x5B, CAVE_RUPEES_10, "rupees_10_l6", "rupees_10", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x5E, CAVE_SHOP_CANDLE, "candle_shop", "candle_60", OPEN_OPEN, EVIDENCE_VERIFIED, 3),
    (0x62, CAVE_RUPEES_100, "rupees_100_c7", "rupees_100", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x63, CAVE_DOOR_REPAIR, "door_repair_d7", "pay_20", OPEN_BURN, EVIDENCE_SOURCE, 0),
    (0x64, CAVE_POTION, "potion_e7", "potion", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x66, CAVE_SHOP_CANDLE, "shop_g7", "shop", OPEN_OPEN, EVIDENCE_SOURCE, 3),
    (0x67, CAVE_RUPEES_30, "rupees_h7", "rupees_30", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x68, CAVE_DOOR_REPAIR, "door_repair_i7", "pay_20", OPEN_BURN, EVIDENCE_SOURCE, 0),
    (0x6A, CAVE_DOOR_REPAIR, "door_repair_k7", "pay_20", OPEN_BURN, EVIDENCE_SOURCE, 0),
    (0x6B, CAVE_RUPEES_100, "rupees_100_l7", "rupees_100", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x6D, CAVE_DUNGEON_8, "dungeon_8", "dungeon_8", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x6F, CAVE_SHOP_ARROWS, "shop_p7", "shop", OPEN_OPEN, EVIDENCE_SOURCE, 3),
    (0x70, CAVE_PAID_HINT_B, "paid_hint_a8", "hint", OPEN_OPEN, EVIDENCE_SOURCE, 0),
    (0x71, CAVE_RUPEES_30, "rupees_b8", "rupees_30", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x74, CAVE_DUNGEON_3, "dungeon_3", "dungeon_3", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x75, CAVE_UNIQUE_19, "hint_f8", "hint", OPEN_OPEN, EVIDENCE_SOURCE, 0),
    (0x76, CAVE_GAMBLE, "gamble_g8", "gamble", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x77, CAVE_WOOD_SWORD, "wooden_sword", "wooden_sword", OPEN_OPEN, EVIDENCE_VERIFIED, 0),
    (0x78, CAVE_POTION, "potion_i8", "potion", OPEN_BURN, EVIDENCE_VERIFIED, 0),
    (0x79, CAVE_WARP, "warp_j8", "warp", OPEN_PUSH_ROCK, EVIDENCE_SOURCE, 0),
    (0x7B, CAVE_TAKE_ANY, "heart_l8", "heart_container", OPEN_BOMB, EVIDENCE_VERIFIED, 0),
    (0x7C, CAVE_GAMBLE, "gamble_m8", "gamble", OPEN_BOMB, EVIDENCE_SOURCE, 0),
    (0x7D, CAVE_DOOR_REPAIR, "door_repair_n8", "pay_20", OPEN_BOMB, EVIDENCE_SOURCE, 0),
)

# Not in AttrsB (cave_id 0): Armos / ladder / fairy. Still rando-relevant.
_Q1_EXTRAS: tuple[OwLocation, ...] = (
    OwLocation("bracelet_armos", 0x24, CAVE_NONE, "armos_item", "bracelet", OPEN_ARMOS, EVIDENCE_SOURCE),
    OwLocation("ladder_heart", 0x5F, CAVE_NONE, "ladder_item", "heart_container", OPEN_LADDER, EVIDENCE_SOURCE),
    OwLocation("fairy_j4", 0x39, CAVE_NONE, "fairy", "fairy", OPEN_OPEN, EVIDENCE_SOURCE),
    OwLocation("fairy_d5", 0x43, CAVE_NONE, "fairy", "fairy", OPEN_OPEN, EVIDENCE_SOURCE),
)

def _from_row(row: tuple[int, int, str, str, str, str, int]) -> OwLocation:
    screen, cave_id, name, vanilla, open_how, evidence, slots = row
    return OwLocation(
        name=name,
        screen=screen,
        cave_id=cave_id,
        kind=CAVE_KIND[cave_id],
        vanilla=vanilla,
        open=open_how,
        evidence=evidence,
        shop_slots=slots,
    )


Q1_VANILLA: tuple[OwLocation, ...] = tuple(_from_row(row) for row in _Q1_CAVES) + _Q1_EXTRAS

_BY_SCREEN: dict[int, OwLocation] = {loc.screen: loc for loc in Q1_VANILLA}
_BY_NAME: dict[str, OwLocation] = {loc.name: loc for loc in Q1_VANILLA}

# Recorder-secret screens (ROM $1EF66, 11 bytes). L7 pond 0x42 is one of them.
RECORDER_SCREENS: tuple[int, ...] = (
    0x42, 0x06, 0x29, 0x2B, 0x30, 0x3A, 0x3C, 0x58, 0x60, 0x6E, 0x72,
)
# Stepladder-enabled screens (ROM $1F20D). 0x5F is the coast heart.
LADDER_SCREENS: tuple[int, ...] = (0x17, 0x18, 0x19, 0x27, 0x4F, 0x5F)


def location(name: str) -> OwLocation:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(name) from exc


def location_at(screen: int) -> OwLocation | None:
    return _BY_SCREEN.get(int(screen))


def q1_item_locations() -> tuple[OwLocation, ...]:
    """Caves/items a rando pool can stuff (not dungeons, warps, hints, gamble)."""
    return tuple(loc for loc in Q1_VANILLA if loc.q1_item_location)


def q1_shops() -> tuple[OwLocation, ...]:
    return tuple(loc for loc in Q1_VANILLA if loc.kind == "shop")


# --- Enemy farms -----------------------------------------------------------
# Vanilla $18500 (128 spawn bytes) and $18580 bit7 packed 16 bytes (LSB =
# screen 0 of each octet). Drop groups: ZSR item-drops chart / z1r wiki.
# Overworld enemies do not respawn until you leave the screen.

_VANILLA_SPAWN = bytes.fromhex(
    "0042421fc1e6e4021f000110cece0000"
    "41e4c16542e41f1f1f1fce0000daceda"
    "21210242005adadada50cfe74eaa4900"
    "2121e4004f000008e82fe74f0a43aa09"
    "2121042f471a000050e8cdc4aa4343ab"
    "828363a269074769695a4763434383aa"
    "e48383ecaa696947474769ec4444eca8"
    "5a8362430ee74e00478d4dd0d0494809"
)
_VANILLA_GROUPED = bytes.fromhex("60002a000028044500929c8979cc2400")

DROP_A = "A"  # ~31%: rupee, heart, fairy
DROP_B = "B"  # ~41%: rupee, heart, bomb, clock
DROP_C = "C"  # ~59%: rupee, heart, rupee_5, clock
DROP_X = "X"  # no drop (armos, boulder, mixed-unknown)

DROPS_BY_GROUP: dict[str, tuple[str, ...]] = {
    DROP_A: ("rupee", "heart", "fairy"),
    DROP_B: ("rupee", "heart", "bomb", "clock"),
    DROP_C: ("rupee", "heart", "rupee_5", "clock"),
    DROP_X: (),
}

# ObjType → drop group when $18580 bit7 is clear (id is a real type).
_DROP_GROUP_BY_TYPE: dict[int, str] = {
    0x01: DROP_B,  # lynel_blue
    0x02: DROP_A,  # lynel
    0x03: DROP_B,  # moblin_blue
    0x04: DROP_A,  # moblin
    0x07: DROP_A,  # octorok
    0x08: DROP_A,  # octorok_fast
    0x09: DROP_B,  # octorok_blue
    0x0A: DROP_B,  # octorok_blue_fast
    0x0D: DROP_C,  # tektite_blue
    0x0E: DROP_A,  # tektite
    0x0F: DROP_A,  # leever_blue
    0x10: DROP_C,  # leever
    0x11: DROP_A,  # zora
    0x1A: DROP_A,  # peahat
    0x1E: DROP_X,  # armos
    0x20: DROP_X,  # boulder
    0x21: DROP_C,  # ghini
}

_PREY_NAME: dict[int, str] = {
    0x01: "lynel_blue",
    0x02: "lynel",
    0x03: "moblin_blue",
    0x04: "moblin",
    0x07: "octorok",
    0x08: "octorok_fast",
    0x09: "octorok_blue",
    0x0A: "octorok_blue_fast",
    0x0D: "tektite_blue",
    0x0E: "tektite",
    0x0F: "leever_blue",
    0x10: "leever",
    0x11: "zora",
    0x1A: "peahat",
    0x1E: "armos",
    0x20: "boulder",
    0x21: "ghini",
}

# Live restock pairs used by RupeeFarmController (leave/return to respawn).
# Direction is FROM the farm screen TO the neighbor (leave direction).
# L1 0x77→0x37 uses the previous hop screen; L2-only screens (0x59/0x49/0x4A)
# use the previous L2 hop. Skip farm_at is None (start 0x77).
_RESTOCK: dict[int, tuple[int, str]] = {
    0x78: (0x77, "LEFT"),  # I8 octorok A, L1 previous
    0x68: (0x78, "DOWN"),  # I7 octorok A, L1 previous
    0x58: (0x68, "DOWN"),  # I6 mixed, L1 previous
    0x48: (0x58, "DOWN"),  # I5 leever C, L1 previous
    0x38: (0x48, "DOWN"),  # I4 mixed, L1 previous
    0x37: (0x38, "RIGHT"),  # H4 octorok_fast A, L1 previous
    0x59: (0x58, "LEFT"),  # J6 peahat A, L2 previous
    0x49: (0x59, "DOWN"),  # J5 mixed, L2 previous
    0x4A: (0x49, "LEFT"),  # K5 tektite C, L2 previous
}


def _ines_prg(rom: bytes) -> bytes:
    return rom[INES_HEADER:] if rom[:4] == b"NES\x1a" else rom


def _grouped_bit(packed: bytes, screen: int) -> bool:
    return bool(packed[screen >> 3] & (1 << (screen & 7)))


def pack_group_bits(layout: bytes) -> bytes:
    """Pack 128 ``$18580`` bit7 flags into 16 bytes (LSB = first screen)."""
    out = bytearray(16)
    for screen, byte in enumerate(layout):
        if byte & 0x80:
            out[screen >> 3] |= 1 << (screen & 7)
    return bytes(out)


@dataclass(frozen=True)
class OwSpawn:
    """One overworld screen's enemy spawn from ``$18500`` / ``$18580``."""

    screen: int
    count: int
    monster_id: int
    grouped: bool
    prey: str
    drop_group: str

    @property
    def grid(self) -> str:
        return grid_name(self.screen)

    @property
    def drops(self) -> tuple[str, ...]:
        return DROPS_BY_GROUP.get(self.drop_group, ())

    @property
    def farmable(self) -> bool:
        return bool(self.drops)


@dataclass(frozen=True)
class FarmSpot:
    """Enemy-drop farm: kill on ``screen``, restock by scrolling out/in."""

    name: str
    screen: int
    prey: str
    drop_group: str
    drops: tuple[str, ...]
    count: int
    restock_neighbor: int | None
    restock_direction: str | None
    evidence: str

    @property
    def grid(self) -> str:
        return grid_name(self.screen)


def _spawns_from(spawn: bytes, grouped: bytes) -> tuple[OwSpawn, ...]:
    rows: list[OwSpawn] = []
    for screen, raw in enumerate(spawn):
        if raw == 0:
            continue
        count_idx = (raw >> 6) & 3
        monster_id = raw & 0x3F
        is_group = _grouped_bit(grouped, screen)
        if is_group:
            prey = f"group_{monster_id:02x}"
            drop_group = "mixed"
        else:
            prey = _PREY_NAME.get(monster_id, f"type_{monster_id:02x}")
            drop_group = _DROP_GROUP_BY_TYPE.get(monster_id, DROP_X)
        rows.append(
            OwSpawn(
                screen=screen,
                count=SPAWN_COUNTS[count_idx],
                monster_id=monster_id,
                grouped=is_group,
                prey=prey,
                drop_group=drop_group,
            )
        )
    return tuple(rows)


def q1_spawns() -> tuple[OwSpawn, ...]:
    """Vanilla first-quest overworld spawns (embedded ``$18500``)."""
    return _spawns_from(_VANILLA_SPAWN, _VANILLA_GROUPED)


def spawns_from_rom(rom: bytes) -> tuple[OwSpawn, ...]:
    """Read ``$18500`` / ``$18580`` from an iNES or headerless PRG dump."""
    data = _ines_prg(rom)
    spawn = data[ROM_OW_SPAWN : ROM_OW_SPAWN + 128]
    layout = data[ROM_OW_LAYOUT : ROM_OW_LAYOUT + 128]
    if len(spawn) != 128 or len(layout) != 128:
        raise ValueError("ROM too short for overworld spawn/layout tables")
    return _spawns_from(spawn, pack_group_bits(layout))


def _farm_from_spawn(spawn: OwSpawn) -> FarmSpot:
    restock = _RESTOCK.get(spawn.screen)
    neighbor, direction = restock if restock is not None else (None, None)
    evidence = EVIDENCE_VERIFIED if restock is not None else "rom"
    return FarmSpot(
        name=f"farm_{grid_name(spawn.screen).lower()}_{spawn.prey}",
        screen=spawn.screen,
        prey=spawn.prey,
        drop_group=spawn.drop_group,
        drops=spawn.drops if spawn.drop_group != "mixed" else ("rupee", "heart"),
        count=spawn.count,
        restock_neighbor=neighbor,
        restock_direction=direction,
        evidence=evidence,
    )


def q1_farms() -> tuple[FarmSpot, ...]:
    """Every OW screen that can drop rupees/hearts/bombs (vanilla Q1)."""
    return tuple(
        _farm_from_spawn(spawn)
        for spawn in q1_spawns()
        if spawn.farmable or spawn.drop_group == "mixed"
    )


Q1_SPAWNS: tuple[OwSpawn, ...] = q1_spawns()
Q1_FARMS: tuple[FarmSpot, ...] = q1_farms()
_FARM_BY_SCREEN: dict[int, FarmSpot] = {spot.screen: spot for spot in Q1_FARMS}


def farm_at(screen: int) -> FarmSpot | None:
    return _FARM_BY_SCREEN.get(int(screen))


_SKIP_PREY = frozenset({"lynel", "lynel_blue", "peahat", "zora"})
# Heart-farm chase is walker prey only. 0x48 leevers killed natural L2 at
# (186,93); 0x58 mixed group burned both farm attempts then 0x5C maze death.
_HEART_PREY = frozenset(
    {
        "octorok",
        "octorok_fast",
        "octorok_blue",
        "octorok_blue_fast",
        "moblin",
        "moblin_blue",
        "tektite",
        "tektite_blue",
    }
)


def restock_for(screen: int) -> tuple[int, str] | None:
    """Catalog restock pair (neighbor, leave-direction) or None."""
    spot = farm_at(screen)
    if spot is None or spot.restock_neighbor is None or not spot.restock_direction:
        return None
    return (int(spot.restock_neighbor), str(spot.restock_direction))


def worth_rupee_farm(screen: int) -> FarmSpot | None:
    """Farmable rupee screen that hop policy may divert into.

    Skips lynel / peahat / zora. Mixed-group screens still drop rupees.
    """
    spot = farm_at(screen)
    if spot is None:
        return None
    if spot.prey in _SKIP_PREY:
        return None
    if "rupee" not in spot.drops and "rupee_5" not in spot.drops:
        return None
    return spot


def worth_heart_farm(screen: int) -> FarmSpot | None:
    """Heart-farm divert: octorok / moblin / tektite only.

    Skip leevers (0x48 chase walked into the north trees) and mixed groups
    (0x58 burned both attempts). 0x4A tektites stay legal. Rupee farms may
    still use leevers.
    """
    spot = farm_at(screen)
    if spot is None:
        return None
    if spot.prey not in _HEART_PREY:
        return None
    if "heart" not in spot.drops and "fairy" not in spot.drops:
        return None
    return spot


def rupee_farms() -> tuple[FarmSpot, ...]:
    return tuple(s for s in q1_farms() if "rupee" in s.drops or "rupee_5" in s.drops)


def five_rupee_farms() -> tuple[FarmSpot, ...]:
    """Drop group C: blue tektite, red leever, ghini (best rupee/kill)."""
    return tuple(s for s in q1_farms() if s.drop_group == DROP_C)


def bomb_farms() -> tuple[FarmSpot, ...]:
    """Drop group B: blue octorok / blue moblin / blue lynel."""
    return tuple(s for s in q1_farms() if "bomb" in s.drops)


def easy_farms() -> tuple[FarmSpot, ...]:
    """Low-HP group A walkers (octorok / moblin / tektite), not lynel/peahat."""
    easy_prey = {"octorok", "octorok_fast", "moblin", "tektite"}
    return tuple(s for s in q1_farms() if s.prey in easy_prey)


def locations_from_rom(rom: bytes) -> tuple[OwLocation, ...]:
    """Q1-active cave dests from a ROM, overlaying vanilla open/name when known.

    Unknown dests keep ``vanilla="rom_cave_{id:02x}"``. Extra non-cave
    locations (bracelet, ladder heart, fairies) are appended unchanged.
    """
    out: list[OwLocation] = []
    seen: set[int] = set()
    for attrs in decode_ow_attrs(rom):
        if attrs.q1_ignore or attrs.cave_id == CAVE_NONE:
            continue
        seen.add(attrs.screen)
        known = _BY_SCREEN.get(attrs.screen)
        if known is not None and known.cave_id == attrs.cave_id:
            out.append(known)
            continue
        kind = CAVE_KIND.get(attrs.cave_id, "cave")
        vanilla = f"rom_cave_{attrs.cave_id:02x}"
        open_how = known.open if known is not None else OPEN_SECRET
        name = known.name if known is not None else f"ow_{grid_name(attrs.screen).lower()}"
        out.append(
            OwLocation(
                name=name,
                screen=attrs.screen,
                cave_id=attrs.cave_id,
                kind=kind,
                vanilla=vanilla,
                open=open_how,
                evidence="rom",
                shop_slots=3 if kind == "shop" else 0,
            )
        )
    for extra in _Q1_EXTRAS:
        if extra.screen not in seen:
            out.append(extra)
    return tuple(out)


__all__ = [
    "CAVE_KIND",
    "CaveWare",
    "OwSecret",
    "SECRET_PAYOUT",
    "SECRET_RUPEE_CAVES",
    "cave_keeper",
    "decode_cave_wares",
    "secret_payout",
    "DROPS_BY_GROUP",
    "FarmSpot",
    "INES_HEADER",
    "LADDER_SCREENS",
    "OwLocation",
    "OwScreenAttrs",
    "OwSpawn",
    "Q1_FARMS",
    "Q1_SPAWNS",
    "Q1_VANILLA",
    "RECORDER_SCREENS",
    "ROM_OW_ATTRS_B",
    "ROM_OW_ATTRS_F",
    "ROM_OW_LAYOUT",
    "ROM_OW_SPAWN",
    "bomb_farms",
    "cave_id_from_attrs_b",
    "decode_ow_attrs",
    "easy_farms",
    "farm_at",
    "five_rupee_farms",
    "grid_name",
    "location",
    "location_at",
    "locations_from_rom",
    "q1_farms",
    "q1_item_locations",
    "q1_shops",
    "q1_spawns",
    "restock_for",
    "rupee_farms",
    "spawns_from_rom",
    "worth_heart_farm",
    "worth_rupee_farm",
]
