"""Level 7 (Demon) overworld approach and gated entry helpers.

Gated by Whistle (L5) for pond drain and Bait/Food for hungry Goriya.
The start-to-pond hop table is usable without either item so geometry can be
verified independently; opening the pond still requires the naturally earned
Whistle.

See ``docs/LEVEL7_ROUTE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from zelda_i.overworld.graph import ScreenHop, path_screens_from_hops
from zelda_i.overworld.path import OverworldPathController
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

# Source-only bait-shop arithmetic.  It is not used by the executable pond
# controller because live 0x67 is a sealed tree pocket (LEVEL3_ROUTE).
LEVEL7_BAIT_SHOP_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x67, "UP"),
    ScreenHop(0x66, "LEFT"),
    ScreenHop(0x65, "LEFT"),
    ScreenHop(0x64, "LEFT"),
    ScreenHop(0x54, "UP"),
    ScreenHop(0x44, "UP"),
    ScreenHop(SCREEN_LEVEL7_BAIT_SHOP_HYP, "UP"),
)

# Executable pond approach: reuse the live west-forest path through 0x55,
# join 0x64, then use the source pond suffix directly from 0x54.  Mapping the
# pond does not require the optional bait-shop detour.
LEVEL7_POND_APPROACH_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x57, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x56, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x55, "LEFT", align_y=133),
    ScreenHop(0x65, "DOWN", align_x=112),
    ScreenHop(0x64, "LEFT", align_y=141),
    ScreenHop(0x54, "UP"),
)

# From bait shop screen to pond (source).
LEVEL7_POND_FROM_SHOP_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x44, "DOWN"),
    ScreenHop(0x54, "DOWN"),
    ScreenHop(0x53, "LEFT"),
    ScreenHop(0x52, "LEFT"),
    ScreenHop(SCREEN_LEVEL7_POND_HYP, "UP"),
)

LEVEL7_POND_HOPS: tuple[ScreenHop, ...] = LEVEL7_POND_APPROACH_HOPS + (
    ScreenHop(0x53, "LEFT", align_y=141),
    # 0x53 west: central y≈141 is tree-blocked; lower gap is y≈189.
    ScreenHop(0x52, "LEFT", align_y=189),
    ScreenHop(SCREEN_LEVEL7_POND_HYP, "UP", align_x=112),
)
LEVEL7_POND_SCREENS: tuple[int, ...] = path_screens_from_hops(0x77, LEVEL7_POND_HOPS)


class Level7NavPhase(Enum):
    HOP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class OverworldToLevel7PondController(OverworldPathController):
    """Walk from the start screen to the Demon pond on screen ``0x42``.

    This controller deliberately stops at the pond.  Whistle selection, pond
    drain, and dungeon entry belong to the next route boundary so a missing
    entry capability cannot silently turn a geometry result into an entry
    claim.
    """

    phase: Level7NavPhase = Level7NavPhase.HOP
    hops: tuple[ScreenHop, ...] = LEVEL7_POND_HOPS
    require_sword: bool = True

    def end_screen(self) -> int:
        return self.hops[-1].target

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # 0x65→0x64 arrives on the east ledge at ~(232,109).  UP is blocked
        # there: descend to the open middle band, cross to the visible left
        # north gap at x≈48, then climb.  x≈120 is under the central tree isle.
        if hop.target == 0x54 and snap.screen == 0x64:
            if snap.link_x > 180 and snap.link_y < 132:
                return self._swing("DOWN", "64_east_ledge_down")
            if snap.link_x > 56:
                return self._swing("LEFT", "64_cross_to_north")
            if snap.link_x < 40:
                return self._swing("RIGHT", "64_north_ax")
            return self._swing("UP", "64_north")
        return None


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


def level7_overworld_stop(_snap: ZeldaSnapshot) -> bool:
    """Exact overworld geometry stop: controllable on the pond screen."""
    return on_level7_pond_hyp(_snap)


def planning_report() -> dict[str, Any]:
    return {
        "level": LEVEL7,
        "name": "The Demon",
        "status": "pond_controller_partial",
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
            {"target": hex(h.target), "dir": h.direction} for h in LEVEL7_BAIT_SHOP_HOPS
        ],
        "pond_hops_from_shop": [
            {"target": hex(h.target), "dir": h.direction}
            for h in LEVEL7_POND_FROM_SHOP_HOPS
        ],
        "pond_hops_from_start": [
            {"target": hex(h.target), "dir": h.direction} for h in LEVEL7_POND_HOPS
        ],
        "live": {
            "pond_screen": None,
            "entry_room": None,
            "boss_room": None,
        },
        "docs": "nes/zelda_i/docs/LEVEL7_ROUTE.md",
    }
