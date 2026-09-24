"""Overworld leftover packets and mouth-stitch contracts for Zelda I.

One leftover type. Field names match ``level8.entry.PostLevel7Handoff`` so
L7/L9 can consume the same packet. L8 already owns its copy; do not grow
``graph.py`` (530) or steal ``level7/8/9/overworld.py``.

``verified`` and ``route_eligible`` stay false until a dungeon owner measures
the full leave. Pose facts in ``MOUTH_STITCHES`` are documentation, not a
complete packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zelda_i.anchors import (
    FULL_TRIFORCE,
    SCREEN_BRACELET_ARMOS,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL4_ENTRANCE,
    SCREEN_LEVEL4_RAFT_DOCK,
    SCREEN_LEVEL5_ENTRANCE,
    SCREEN_LEVEL6_ENTRANCE,
    SCREEN_LEVEL7_BAIT_SHOP,
    SCREEN_LEVEL7_ENTRANCE,
    SCREEN_LEVEL8_BUSH,
    SCREEN_LEVEL9_ENTRANCE,
    SCREEN_MAGICAL_SWORD_GRAVE,
    TF_BIT_L1,
    TF_BIT_L2,
    TF_BIT_L3,
    TF_BIT_L4,
    TF_BIT_L5,
    TF_BIT_L6,
    TF_BIT_L7,
    TF_BIT_L8,
    TF_BITS_ALL,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_MAGIC_KEY,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

CANDLE_RED = 2
TF_BIT_BY_LEVEL: dict[int, int] = {
    1: TF_BIT_L1,
    2: TF_BIT_L2,
    3: TF_BIT_L3,
    4: TF_BIT_L4,
    5: TF_BIT_L5,
    6: TF_BIT_L6,
    7: TF_BIT_L7,
    8: TF_BIT_L8,
}
# TF after clearing that dungeon (bits 1..N).
CUMULATIVE_TF: dict[int, int] = {
    0: 0x00,
    1: 0x01,
    2: 0x03,
    3: 0x07,
    4: 0x0F,
    5: 0x1F,
    6: 0x3F,
    7: 0x7F,
    8: TF_BITS_ALL,
}

_MEASURED_FIELDS = (
    "screen",
    "link_x",
    "link_y",
    "mode",
    "triforce",
    "keys",
    "bombs",
    "rupees",
    "heart_containers",
    "selected_item",
    "whistle",
    "food",
    "rod",
    "bow",
    "arrows",
    "candle",
)


@dataclass(frozen=True)
class OverworldHandoff:
    """Settled post-fanfare leftover. Game-local fields; L8-shaped packet.

    L8 hard-codes incoming ``candle=2``; this shared type defaults ``None`` so
    an incomplete packet cannot fake Red Candle.
    """

    screen: int | None = None
    link_x: int | None = None
    link_y: int | None = None
    mode: int | None = None
    triforce: int | None = None
    keys: int | None = None
    bombs: int | None = None
    rupees: int | None = None
    heart_containers: int | None = None
    selected_item: int | None = None
    whistle: int | None = None
    food: int | None = None
    rod: int | None = None
    bow: int | None = None
    arrows: int | None = None
    candle: int | None = None
    magic_key: int | None = None
    xy_tolerance: int = 4
    evidence: str = "hypothesis"
    verified: bool = False
    route_eligible: bool = False

    def complete(self) -> bool:
        if not self.verified:
            return False
        return all(getattr(self, name) is not None for name in _MEASURED_FIELDS)

    def mismatch(self, snap: ZeldaSnapshot, ram: Any) -> str | None:
        if not self.complete():
            return "handoff_unmeasured"
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.transitioning:
            return "handoff_not_settled_overworld"
        if self.mode is not None and snap.mode != self.mode:
            return "handoff_mode_mismatch"
        if snap.screen != self.screen:
            return "handoff_screen_mismatch"
        if abs(snap.link_x - int(self.link_x)) > self.xy_tolerance:
            return "handoff_x_mismatch"
        if abs(snap.link_y - int(self.link_y)) > self.xy_tolerance:
            return "handoff_y_mismatch"
        if snap.triforce != self.triforce:
            return "handoff_triforce_mismatch"
        # Health is not a handoff field: full hearts are the Survival
        # refill's (a last-heart run arrives chipped), and the container
        # count is route history (run 9 reached L7 with 10, run 3 with 11).
        consumables = (
            ("keys", snap.keys, self.keys),
            ("bombs", snap.bombs, self.bombs),
            ("rupees", snap.rupees, self.rupees),
        )
        for label, actual, expected in consumables:
            if int(actual) < int(expected):
                return f"handoff_{label}_mismatch"
        checks = (
            ("selected_item", read_u8(ram, ADDR_SELECTED_ITEM), self.selected_item),
            ("whistle", read_u8(ram, ADDR_WHISTLE), self.whistle),
            ("food", read_u8(ram, ADDR_FOOD), self.food),
            ("rod", snap.rod, self.rod),
            ("bow", snap.bow, self.bow),
            ("arrows", snap.arrows, self.arrows),
            ("candle", read_u8(ram, ADDR_CANDLE), self.candle),
        )
        for label, actual, expected in checks:
            if int(actual) != int(expected):
                return f"handoff_{label}_mismatch"
        if self.magic_key is not None:
            if int(read_u8(ram, ADDR_MAGIC_KEY)) != int(self.magic_key):
                return "handoff_magic_key_mismatch"
        return None


UNMEASURED_HANDOFF = OverworldHandoff()


def handoff_from_ram(
    ram: Any,
    *,
    evidence: str = "hypothesis",
    verified: bool = False,
) -> OverworldHandoff:
    """Copy a settled snapshot into a leftover packet. Never sets route_eligible."""
    snap = read_snapshot(ram)
    return OverworldHandoff(
        screen=snap.screen,
        link_x=snap.link_x,
        link_y=snap.link_y,
        mode=snap.mode,
        triforce=snap.triforce,
        keys=snap.keys,
        bombs=snap.bombs,
        rupees=snap.rupees,
        heart_containers=snap.heart_containers,
        selected_item=read_u8(ram, ADDR_SELECTED_ITEM),
        whistle=read_u8(ram, ADDR_WHISTLE),
        food=read_u8(ram, ADDR_FOOD),
        rod=snap.rod,
        bow=snap.bow,
        arrows=snap.arrows,
        candle=read_u8(ram, ADDR_CANDLE),
        magic_key=read_u8(ram, ADDR_MAGIC_KEY),
        evidence=evidence,
        verified=verified,
        route_eligible=False,
    )


def enter_gate_ok(level: int, handoff: OverworldHandoff) -> bool:
    """Mouth item/TF gates. Incomplete packets fail closed."""
    if level == 7:
        return bool(handoff.whistle)
    if level == 8:
        return handoff.candle == CANDLE_RED
    if level == 9:
        return handoff.triforce == FULL_TRIFORCE
    return True


@dataclass(frozen=True)
class InlandDescendSpec:
    """Intra-screen micro: walk inland on the arrival y, then drop, then travel.

    Pattern from live 0x53 leftover (224,173): direct DOWN toward y≈189 is
    blocked. LEFT inland first, then descend into the open band, then LEFT.
    L7 owns the pond controller; this is the shared motion type only.
    """

    inland_x: int
    y_lo: int
    y_hi: int
    inland_dir: str = "LEFT"
    travel_dir: str = "LEFT"
    x_tol: int = 8


# Hypothesis numbers for the 0x53 leftover. Not a route claim.
HYP_SCREEN_53_INLAND_DESCEND = InlandDescendSpec(
    inland_x=160,
    y_lo=183,
    y_hi=195,
    inland_dir="LEFT",
    travel_dir="LEFT",
)


def inland_then_descend(snap: ZeldaSnapshot, spec: InlandDescendSpec) -> str:
    """Return the cardinal for this frame. Inland before any vertical drop."""
    if spec.inland_dir == "LEFT":
        inland = snap.link_x <= spec.inland_x + spec.x_tol
    elif spec.inland_dir == "RIGHT":
        inland = snap.link_x >= spec.inland_x - spec.x_tol
    else:
        inland = abs(snap.link_x - spec.inland_x) <= spec.x_tol
    if not inland:
        return spec.inland_dir
    if snap.link_y < spec.y_lo:
        return "DOWN"
    if snap.link_y > spec.y_hi:
        return "UP"
    return spec.travel_dir


def y_band_travel_hop(
    target: int, direction: str, spec: InlandDescendSpec
) -> ScreenHop:
    """ScreenHop for the exit after inland-then-descend. Do not use align_y."""
    return ScreenHop(
        target, direction, y_band_lo=spec.y_lo, y_band_hi=spec.y_hi
    )


def _pose(
    *,
    screen: int | None = None,
    x: int | None = None,
    y: int | None = None,
    tf: int | None = None,
    evidence: str = "hypothesis",
    **items: int | None,
) -> OverworldHandoff:
    return OverworldHandoff(
        screen=screen,
        link_x=x,
        link_y=y,
        mode=PLAY_MODE if screen is not None else None,
        triforce=tf,
        evidence=evidence,
        verified=False,
        route_eligible=False,
        **items,
    )


@dataclass(frozen=True)
class MouthStitch:
    """One dungeon-leave → next-mouth row. Status is not a route claim."""

    from_level: int
    to_level: int
    leave: OverworldHandoff
    mouth_screen: int
    enter_items: tuple[str, ...]
    status: str
    notes: str = ""


MOUTH_STITCHES: tuple[MouthStitch, ...] = (
    MouthStitch(
        1,
        2,
        _pose(screen=0x37, x=112, y=125, tf=0x01, evidence="verified"),
        0x3C,
        ("wooden_sword", "tf_0x01"),
        "verified",
        "L1 fanfare leave OW 0x37 ~(112,125) mode 5; Moon door live.",
    ),
    MouthStitch(
        2,
        3,
        _pose(screen=0x3C, x=112, y=125, tf=0x03, evidence="verified"),
        SCREEN_LEVEL3_ENTRANCE,
        ("wooden_sword",),
        "verified",
        "L2 fanfare leave OW 0x3C ~(112,125); Manji 0x74 live.",
    ),
    MouthStitch(
        3,
        4,
        _pose(screen=0x74, x=128, y=125, tf=0x07, evidence="verified"),
        SCREEN_LEVEL4_ENTRANCE,
        ("raft",),
        "verified",
        "L3 fanfare leave OW 0x74 ~(128,125) raft=1; island 0x45 via dock "
        f"0x{SCREEN_LEVEL4_RAFT_DOCK:02X}.",
    ),
    MouthStitch(
        4,
        5,
        _pose(screen=0x45, x=128, y=125, tf=0x0F, evidence="live"),
        SCREEN_LEVEL5_ENTRANCE,
        (),
        "live",
        "L4 fanfare leave OW 0x45 ~(128,125) ±4 mode 5; xy from "
        "Level4Complete settle 633f (pin TF 0x0C). Fixture-live, "
        "route_eligible=false.",
    ),
    MouthStitch(
        5,
        6,
        _pose(screen=0x0B, x=112, y=125, tf=0x1F, evidence="live", whistle=1),
        SCREEN_LEVEL6_ENTRANCE,
        (),
        "live",
        "L5 fanfare leave OW 0x0B ~(112,125) ±4 mode 5, Whistle earned. "
        "xy from Level5Complete settle 1116f (pin TF 0x1C, whistle=1 on pin, "
        "not granted). L6 mouth 0x22 verified live. Do not grant Whistle. "
        "Fixture-live, route_eligible=false.",
    ),
    MouthStitch(
        6,
        7,
        _pose(
            screen=SCREEN_LEVEL6_ENTRANCE,
            x=112,
            y=125,
            tf=0x3F,
            evidence="measured",
            keys=2,
            bombs=8,
            rupees=42,
            heart_containers=8,
            selected_item=2,
            whistle=1,
            food=0,
            rod=1,
            bow=1,
            arrows=1,
            candle=0,
        ),
        SCREEN_LEVEL7_ENTRANCE,
        ("whistle",),
        "measured leave / mouth+pond hypothesis",
        "Post-L6 fanfare leave MEASURED + verified: OW 0x22 (112,125) mode 5 "
        "TF 0x3F, keys 2 bombs 8 rupees 42, selected_item 2, Whistle 1 Food 0 "
        "Candle 0, 8 HC full (--through level6-exit 2/2). Carried as "
        "level7.entry.MEASURED_POST_L6_EXIT (verified=True). The post-L6 "
        "controller walks the bait prefix 0x22->0x25 green; the pond 0x42, "
        f"bait shop 0x{SCREEN_LEVEL7_BAIT_SHOP:02X} geometry, and drain remain "
        "hypothesis. 0x53 (224,173) LEFT-inland-before-DOWN. Food is the "
        "Hungry-Goriya gate inside, not a pond-drain gate.",
    ),
    MouthStitch(
        7,
        8,
        UNMEASURED_HANDOFF,
        SCREEN_LEVEL8_BUSH,
        ("candle2",),
        "UNMEASURED",
        "Post-L7 leftover UNMEASURED. Bush 0x6D live from 0x5D south x≈48; "
        "burn unsolved; Candle 2 comes from L7. Expected leave TF 0x7F.",
    ),
    MouthStitch(
        8,
        9,
        UNMEASURED_HANDOFF,
        SCREEN_LEVEL9_ENTRANCE,
        ("bombs", "tf_0xff", "magic_key"),
        "UNMEASURED",
        "Post-L8 leftover UNMEASURED. Spectacle Rock 0x05 bomb-rock is "
        "source/fixture-live (entry room 0x76), not natural post-L8. "
        "Old Man wants TF 0xFF.",
    ),
)

# OW shortcuts needed later. Do not grant on the stitch packet.
# Grave 0x21 / Armos 0x24 live in anchors; white-sword cave 0x0A is still hyp.
SCREEN_WHITE_SWORD_CAVE_HYP = 0x0A


__all__ = [
    "CANDLE_RED",
    "CUMULATIVE_TF",
    "HYP_SCREEN_53_INLAND_DESCEND",
    "MOUTH_STITCHES",
    "SCREEN_BRACELET_ARMOS",
    "SCREEN_MAGICAL_SWORD_GRAVE",
    "SCREEN_WHITE_SWORD_CAVE_HYP",
    "TF_BIT_BY_LEVEL",
    "TF_BITS_ALL",
    "UNMEASURED_HANDOFF",
    "InlandDescendSpec",
    "MouthStitch",
    "OverworldHandoff",
    "enter_gate_ok",
    "handoff_from_ram",
    "inland_then_descend",
    "y_band_travel_hop",
]
