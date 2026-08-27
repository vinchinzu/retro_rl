"""First-quest dungeon treasures vs Survival spine collection.

Wiki plan: ``docs/research/DUNGEON_WALKTHROUGHS.md`` (Zelda Dungeon 100%).
Bow is L1 (west of ``0x23``), not L6. L6 item is Magical Rod. Gohma still
needs bow + wooden arrows (OW shop). Default ``--through level2`` and later
never run ``level1-bow``; that through is a side branch enter-stop
(play ``0x22`` 1/1). ``--through level1-bow-cellar`` is the stairs hop;
``ADDR_BOW`` is still 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOOK,
    ADDR_BOOMERANG,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_LADDER,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MAGIC_KEY,
    ADDR_MAX_BOMBS,
    ADDR_RAFT,
    ADDR_RING,
    ADDR_ROD,
    ADDR_WHISTLE,
)
from zelda_i.survival_spine import SPINE_THROUGH

# required_gate: later dungeon/boss cannot proceed without it
# required_combat: wiki dungeon treasure; spine collects it for combat
# optional_hud: Compass / Map
# optional_capacity: bomb-bag old-man rooms (100R)
# optional_upgrade: replaced by a later item, or not a door gate
KIND_GATE = "required_gate"
KIND_COMBAT = "required_combat"
KIND_HUD = "optional_hud"
KIND_CAP = "optional_capacity"
KIND_UP = "optional_upgrade"

LIVE_COLLECTED = "collected"  # default spine 1/1
LIVE_SKIPPED = "skipped"  # default spine never visits
LIVE_SIDE = "side_branch"  # through exists; not on default path
LIVE_BLOCKED = "blocked"  # through exists; 3/3 red
LIVE_OPEN = "open"  # on tape leftover; not collected yet
LIVE_LATER = "not_reached"


@dataclass(frozen=True)
class DungeonTreasure:
    """One first-quest dungeon treasure from the wiki plan."""

    name: str
    dungeon: int
    kind: str
    addr: int
    ram_attr: str
    wiki_room: str
    through: str | None
    on_default_spine: bool
    live: str
    gates: str


TREASURES: tuple[DungeonTreasure, ...] = (
    DungeonTreasure(
        name="bow",
        dungeon=1,
        kind=KIND_GATE,
        addr=ADDR_BOW,
        ram_attr="bow",
        wiki_room="L1 west of 0x23; dest play 0x22 then cellar",
        through="level1-bow-cellar",
        on_default_spine=False,
        live=LIVE_SIDE,
        gates="L6 Gohma eye; L9 Silver Arrows need the bow first",
    ),
    DungeonTreasure(
        name="wooden_boomerang",
        dungeon=1,
        kind=KIND_UP,
        addr=ADDR_BOOMERANG,
        ram_attr="boomerang",
        wiki_room="L1 0x44 after Goriya clear (RoomItemId 0x1D)",
        through=None,
        on_default_spine=False,
        live=LIVE_SKIPPED,
        gates="none; L2 magical boomerang replaces it",
    ),
    DungeonTreasure(
        name="magical_boomerang",
        dungeon=2,
        kind=KIND_COMBAT,
        addr=ADDR_MAGIC_BOOMERANG,
        ram_attr="magical_boomerang",
        wiki_room="L2 0x4f (RoomItemId 0x1E)",
        through="level2",
        on_default_spine=True,
        live=LIVE_COLLECTED,
        gates="stun; spine stop magic_boomerang",
    ),
    DungeonTreasure(
        name="raft",
        dungeon=3,
        kind=KIND_GATE,
        addr=ADDR_RAFT,
        ram_attr="raft",
        wiki_room="L3 mode-9 0x0f after 0x69 stairs",
        through="level3",
        on_default_spine=True,
        live=LIVE_COLLECTED,
        gates="L4 raft-dock island 0x55→0x45",
    ),
    DungeonTreasure(
        name="stepladder",
        dungeon=4,
        kind=KIND_GATE,
        addr=ADDR_LADDER,
        ram_attr="ladder",
        wiki_room="L4 mode-9 0x60 under 0x32",
        through="level4-stepladder",
        on_default_spine=True,
        live=LIVE_COLLECTED,
        gates="L4 water; later OW ladder hearts",
    ),
    DungeonTreasure(
        name="whistle",
        dungeon=5,
        kind=KIND_GATE,
        addr=ADDR_WHISTLE,
        ram_attr="whistle",
        wiki_room="L5 cellar 0x04",
        through="level5-whistle",
        on_default_spine=True,
        live=LIVE_COLLECTED,
        gates="Digdogger shrink; L7 pond drain",
    ),
    DungeonTreasure(
        name="l5_bomb_upgrade",
        dungeon=5,
        kind=KIND_CAP,
        addr=ADDR_MAX_BOMBS,
        ram_attr="",
        wiki_room="L5 old-man 100R (8→12)",
        through=None,
        on_default_spine=False,
        live=LIVE_SKIPPED,
        gates="none; Survival count top-up never writes max_bombs",
    ),
    DungeonTreasure(
        name="magical_rod",
        dungeon=6,
        kind=KIND_COMBAT,
        addr=ADDR_ROD,
        ram_attr="rod",
        wiki_room="L6 cellar 0x75 under 0x09",
        through="level6-rod",
        on_default_spine=True,
        live=LIVE_COLLECTED,
        gates="combat; does not replace bow for Gohma",
    ),
    DungeonTreasure(
        name="red_candle",
        dungeon=7,
        kind=KIND_GATE,
        addr=ADDR_CANDLE,
        ram_attr="",
        wiki_room="L7 tip-of-nose cellar (blue→red)",
        through=None,
        on_default_spine=False,
        live=LIVE_LATER,
        gates="L8 bush if ADDR_CANDLE is still 0",
    ),
    DungeonTreasure(
        name="l7_bomb_upgrade",
        dungeon=7,
        kind=KIND_CAP,
        addr=ADDR_MAX_BOMBS,
        ram_attr="",
        wiki_room="L7 old-man 100R (12→16)",
        through=None,
        on_default_spine=False,
        live=LIVE_LATER,
        gates="none; same Survival count shortcut",
    ),
    DungeonTreasure(
        name="book_of_magic",
        dungeon=8,
        kind=KIND_UP,
        addr=ADDR_BOOK,
        ram_attr="",
        wiki_room="L8 staircase (wand flames)",
        through=None,
        on_default_spine=False,
        live=LIVE_LATER,
        gates="none; credits do not require it",
    ),
    DungeonTreasure(
        name="magical_key",
        dungeon=8,
        kind=KIND_UP,
        addr=ADDR_MAGIC_KEY,
        ram_attr="",
        wiki_room="L8 staircase",
        through=None,
        on_default_spine=False,
        live=LIVE_LATER,
        gates="none; L9 splits Magical Key vs key-farm",
    ),
    DungeonTreasure(
        name="red_ring",
        dungeon=9,
        kind=KIND_UP,
        addr=ADDR_RING,
        ram_attr="",
        wiki_room="L9 stairs after Patra/Map",
        through=None,
        on_default_spine=False,
        live=LIVE_LATER,
        gates="none; damage quartered vs base",
    ),
    DungeonTreasure(
        name="silver_arrows",
        dungeon=9,
        kind=KIND_GATE,
        addr=ADDR_ARROWS,
        ram_attr="arrows",
        wiki_room="L9 stairs (ADDR_ARROWS=2)",
        through=None,
        on_default_spine=False,
        live=LIVE_LATER,
        gates="Ganon after stun; needs L1 bow first",
    ),
)


@dataclass(frozen=True)
class OwGate:
    """Overworld buy that a later dungeon gate still needs. Not a dungeon drop."""

    name: str
    cost_rupees: int
    live: str
    gates: str
    notes: str


OW_GATES: tuple[OwGate, ...] = (
    OwGate(
        name="wooden_arrows",
        cost_rupees=80,
        live=LIVE_LATER,
        gates="Gohma with bow; leftover ~39R is short",
        notes="not candle shop 0x5E; Gathering hyp 0x6B not live",
    ),
    OwGate(
        name="bait",
        cost_rupees=60,
        live=LIVE_LATER,
        gates="L7 Hungry Goriya",
        notes="source shop 0x34 Armos top-middle; not live",
    ),
    OwGate(
        name="blue_candle",
        cost_rupees=60,
        live=LIVE_LATER,
        gates="L8 bush 0x6D unless L7 already dropped red",
        notes="live shop 0x5E; skip buy if ADDR_CANDLE≠0",
    ),
)


def treasure(name: str) -> DungeonTreasure:
    """Lookup by ``name``. Raises if the wiki catalog has no such item."""
    for item in TREASURES:
        if item.name == name:
            return item
    raise KeyError(name)


def required_gate_skips_on_default_spine() -> tuple[DungeonTreasure, ...]:
    """Required-gate dungeon items the default L6 spine has not collected."""
    return tuple(
        item
        for item in TREASURES
        if item.kind == KIND_GATE
        and item.dungeon <= 6
        and not item.on_default_spine
    )


def default_spine_collected() -> tuple[str, ...]:
    """Names collected on ``--through level6-north2c`` (bow=0 leftover)."""
    return tuple(item.name for item in TREASURES if item.on_default_spine)


def assert_through_wired(item: DungeonTreasure) -> None:
    """Side-branch and collected through-names must exist on the spine."""
    if item.through is None:
        return
    if item.through not in SPINE_THROUGH:
        raise AssertionError(f"{item.name} through {item.through!r} not in SPINE_THROUGH")
