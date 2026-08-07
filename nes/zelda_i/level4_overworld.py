"""Level 4 (Snake) overworld scaffold — planning only.

Gated by Raft from Level 3 (``ADDR_RAFT``). Screen ids and hop tables are
**source hypotheses** from Zelda Dungeon walkthroughs; they are not live-
verified. Controllers that claim door success must replace hypotheses after
probe.

See ``docs/LEVEL4_ROUTE.md``.
"""

from __future__ import annotations

from typing import Any

from zelda_i.overworld import ScreenHop
from zelda_i.ram import (
    ADDR_LADDER,
    ADDR_RAFT,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

# --- Source-hypothesized geometry (NOT live) ---
# Dock toward L4 island: start U, L×2, U → 0x55; raft carries N → island 0x45.
SOURCE_HYPOTHESIS = True

SCREEN_LEVEL4_DOCK_HYP = 0x55  # mainland dock (source short path)
SCREEN_LEVEL4_ISLAND_HYP = 0x45  # post-raft island / door candidate
SCREEN_RAFT_HEART_DOCK_HYP = 0x3F  # east coast heart (E×8 N×4 from start)

LEVEL4 = 4
LEVEL4_TRIFORCE_BIT = 0x08

# Placeholder hops from start (0x77) to dock — geometry TBD live.
# Do not wire into Clean NamedRoutes until verified.
LEVEL4_DOCK_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x67, "UP"),
    ScreenHop(0x66, "LEFT"),
    ScreenHop(0x65, "LEFT"),
    ScreenHop(SCREEN_LEVEL4_DOCK_HYP, "UP"),
)

# Post-raft island entry is engine-driven (walk onto dock with Raft).
# No hop table for water transit; controller must detect screen change after
# dock walk when ``has_raft``.


def has_raft(ram) -> bool:
    """True when Raft inventory flag is set (L3 item)."""
    return bool(read_u8(ram, ADDR_RAFT))


def has_ladder(ram) -> bool:
    """True when Stepladder inventory flag is set (L4 dungeon item)."""
    return bool(read_u8(ram, ADDR_LADDER))


def required_caps_for_entry() -> frozenset[str]:
    """Named capabilities required to *enter* L4 (source)."""
    return frozenset({"raft"})


def required_caps_for_clear() -> frozenset[str]:
    """Caps expected by end of clear (source planning)."""
    return frozenset({"raft", "ladder"})


def missing_entry_caps(ram) -> list[str]:
    missing: list[str] = []
    if not has_raft(ram):
        missing.append("raft")
    return missing


def on_level4_dock_hyp(snap: ZeldaSnapshot) -> bool:
    """Source hypothesis: mainland dock screen. Not a verified stop."""
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL4_DOCK_HYP
    )


def level4_dungeon_play(snap: ZeldaSnapshot) -> bool:
    """True if snapshot is play mode inside level 4 (any room)."""
    return snap.level == LEVEL4 and snap.mode == PLAY_MODE


def level4_triforce_stop(snap: ZeldaSnapshot) -> bool:
    """Inventory stop: shard 4 bit set. Not a route-success claim by itself."""
    return bool(snap.triforce & LEVEL4_TRIFORCE_BIT)


def level4_entry_stop(_snap: ZeldaSnapshot) -> bool:
    """Placeholder door/entry stop — always False until live room id exists.

    Future: level==4, mode play, screen==entry_room after mode-16 settle.
    """
    return False


def level4_overworld_stop(_snap: ZeldaSnapshot) -> bool:
    """Placeholder OW stop at island door — False until live screen id."""
    return False


def planning_report() -> dict[str, Any]:
    """Machine-readable planning summary for probes / docs."""
    return {
        "level": LEVEL4,
        "name": "The Snake",
        "status": "planning",
        "source_hypothesis": SOURCE_HYPOTHESIS,
        "required_entry_caps": sorted(required_caps_for_entry()),
        "triforce_bit": LEVEL4_TRIFORCE_BIT,
        "ram": {
            "raft": hex(ADDR_RAFT),
            "ladder": hex(ADDR_LADDER),
        },
        "screens_hypothesized": {
            "dock": hex(SCREEN_LEVEL4_DOCK_HYP),
            "island_or_door": hex(SCREEN_LEVEL4_ISLAND_HYP),
            "raft_heart_dock": hex(SCREEN_RAFT_HEART_DOCK_HYP),
        },
        "dock_hops_from_start": [
            {"target": hex(h.target), "dir": h.direction} for h in LEVEL4_DOCK_HOPS
        ],
        "live": {
            "door_screen": None,
            "entry_room": None,
            "boss_room": None,
        },
        "docs": "nes/zelda_i/docs/LEVEL4_ROUTE.md",
    }
