"""Level 7 (Demon) overworld scaffold — planning only.

Gated by Whistle (L5) for pond drain and Bait/Food for hungry Goriya.
Screen ids are **source hypotheses**; not live-verified.

See ``docs/LEVEL7_ROUTE.md``.
"""

from __future__ import annotations

from typing import Any

from zelda_i.overworld import ScreenHop
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

from zelda_i.anchors import (
    SCREEN_LEVEL7_BAIT_SHOP_HYP,
    SCREEN_LEVEL7_POND_HYP,
    TF_BIT_L7 as LEVEL7_TRIFORCE_BIT,
)

SOURCE_HYPOTHESIS = True
LEVEL7 = 7
# Source / Data Crystal style: candle 1=blue, 2=red (confirm live).
CANDLE_RED_PLANNED = 2

LEVEL7_BAIT_SHOP_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x67, "UP"),
    ScreenHop(0x66, "LEFT"),
    ScreenHop(0x65, "LEFT"),
    ScreenHop(0x64, "LEFT"),
    ScreenHop(0x54, "UP"),
    ScreenHop(0x44, "UP"),
    ScreenHop(SCREEN_LEVEL7_BAIT_SHOP_HYP, "UP"),
)

# From bait shop screen to pond (source).
LEVEL7_POND_FROM_SHOP_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x44, "DOWN"),
    ScreenHop(0x54, "DOWN"),
    ScreenHop(0x53, "LEFT"),
    ScreenHop(0x52, "LEFT"),
    ScreenHop(SCREEN_LEVEL7_POND_HYP, "UP"),
)


def has_whistle(ram) -> bool:
    return bool(read_u8(ram, ADDR_WHISTLE))


def has_food(ram) -> bool:
    """Bait / Food inventory (hungry Goriya)."""
    return bool(read_u8(ram, ADDR_FOOD))


def has_red_candle(ram) -> bool:
    """Planned: candle byte == 2 means Red Candle (verify live)."""
    return read_u8(ram, ADDR_CANDLE) >= CANDLE_RED_PLANNED


def required_caps_for_entry() -> frozenset[str]:
    """Whistle required to open pond stairs (source). Food needed mid-dungeon."""
    return frozenset({"whistle"})


def required_caps_for_clear() -> frozenset[str]:
    return frozenset({"whistle", "food", "red_candle"})


def missing_entry_caps(ram) -> list[str]:
    missing: list[str] = []
    if not has_whistle(ram):
        missing.append("whistle")
    return missing


def missing_clear_caps(ram) -> list[str]:
    missing = missing_entry_caps(ram)
    if not has_food(ram):
        missing.append("food")
    return missing


def on_level7_pond_hyp(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL7_POND_HYP
    )


def on_level7_bait_shop_hyp(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL7_BAIT_SHOP_HYP
    )


def level7_dungeon_play(snap: ZeldaSnapshot) -> bool:
    return snap.level == LEVEL7 and snap.mode == PLAY_MODE


def level7_triforce_stop(snap: ZeldaSnapshot) -> bool:
    return bool(snap.triforce & LEVEL7_TRIFORCE_BIT)


def level7_entry_stop(_snap: ZeldaSnapshot) -> bool:
    """Placeholder until live entry room id exists."""
    return False


def level7_overworld_stop(_snap: ZeldaSnapshot) -> bool:
    """Placeholder pond/door stop — False until live."""
    return False


def planning_report() -> dict[str, Any]:
    return {
        "level": LEVEL7,
        "name": "The Demon",
        "status": "planning",
        "source_hypothesis": SOURCE_HYPOTHESIS,
        "required_entry_caps": sorted(required_caps_for_entry()),
        "required_clear_caps": sorted(required_caps_for_clear()),
        "triforce_bit": LEVEL7_TRIFORCE_BIT,
        "ram": {
            "whistle": hex(ADDR_WHISTLE),
            "food": hex(ADDR_FOOD),
            "candle": hex(ADDR_CANDLE),
        },
        "screens_hypothesized": {
            "bait_shop": hex(SCREEN_LEVEL7_BAIT_SHOP_HYP),
            "pond": hex(SCREEN_LEVEL7_POND_HYP),
        },
        "bait_shop_hops_from_start": [
            {"target": hex(h.target), "dir": h.direction}
            for h in LEVEL7_BAIT_SHOP_HOPS
        ],
        "pond_hops_from_shop": [
            {"target": hex(h.target), "dir": h.direction}
            for h in LEVEL7_POND_FROM_SHOP_HOPS
        ],
        "live": {
            "pond_screen": None,
            "entry_room": None,
            "boss_room": None,
        },
        "docs": "nes/zelda_i/docs/LEVEL7_ROUTE.md",
    }
