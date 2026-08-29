"""Level 9 (Death Mountain) overworld anchors and entry capability gates.

Gated by full Triforce (``ADDR_TRIFORCE == 0xFF``) for interior progress.
Bomb-rock OW screen can be mapped earlier.  Spectacle Rock and the entrance
room are live; the natural interior route remains future work.

See ``docs/LEVEL9_ROUTE.md``.
"""

from __future__ import annotations

from typing import Any

from zelda_i.level9.ganon import (
    ADDR_GANON_OBJ_PHASE_BASE,
    GANON_BROWN_STATE,
    OBJ_GANON,
    ROOM_GANON,
    credits_rolling,
    final_ending_screen,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_RING,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

from zelda_i.anchors import FULL_TRIFORCE, SCREEN_LEVEL9_ROCK_HYP

SOURCE_HYPOTHESIS = True
SCREEN_LEVEL9_POTION_NEAR_HYP = 0x04  # one left (source)
LEVEL9 = 9
ROOM_LEVEL9_ENTRY = 0x76
# Source / Data Crystal style values (confirm live).
RING_RED_PLANNED = 2
ARROWS_SILVER_PLANNED = 2

LEVEL9_ROCK_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT"),
    ScreenHop(0x68, "UP"),
    ScreenHop(0x58, "UP"),
    ScreenHop(0x48, "UP"),
    ScreenHop(0x38, "UP"),
    ScreenHop(0x28, "UP"),
    ScreenHop(0x27, "LEFT"),
    ScreenHop(0x17, "UP"),
    ScreenHop(0x07, "UP"),
    ScreenHop(0x06, "LEFT"),
    ScreenHop(SCREEN_LEVEL9_ROCK_HYP, "LEFT"),
)


def has_full_triforce(ram) -> bool:
    return read_u8(ram, ADDR_TRIFORCE) == FULL_TRIFORCE


def triforce_bits(ram) -> int:
    return int(read_u8(ram, ADDR_TRIFORCE))


def has_red_ring(ram) -> bool:
    return read_u8(ram, ADDR_RING) >= RING_RED_PLANNED


def has_silver_arrows(ram) -> bool:
    return read_u8(ram, ADDR_ARROWS) >= ARROWS_SILVER_PLANNED


def required_caps_for_entry() -> frozenset[str]:
    """Full TF for Old Man; bombs for rock (source)."""
    return frozenset({"full_triforce", "bombs"})


def required_caps_for_ganon() -> frozenset[str]:
    return frozenset({"full_triforce", "silver_arrows"})


def missing_entry_caps(ram, *, rock_only: bool = False) -> list[str]:
    """Caps missing for entry attempt.

    ``rock_only``: map bomb-rock OW without requiring full TF.
    """
    missing: list[str] = []
    if not rock_only and not has_full_triforce(ram):
        missing.append("full_triforce")
    # Bombs checked by caller snap.bombs if desired; not hard-fail here.
    return missing


def on_level9_rock_hyp(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL9_ROCK_HYP
    )


def level9_dungeon_play(snap: ZeldaSnapshot) -> bool:
    return snap.level == LEVEL9 and snap.mode == PLAY_MODE


def level9_entry_stop(snap: ZeldaSnapshot) -> bool:
    return level9_dungeon_play(snap) and snap.screen == ROOM_LEVEL9_ENTRY


def level9_overworld_stop(snap: ZeldaSnapshot) -> bool:
    return on_level9_rock_hyp(snap)


def level9_ending_stop(snap: ZeldaSnapshot) -> bool:
    """True once the update loop reaches rolling credits or its final page."""
    return credits_rolling(snap) or final_ending_screen(snap)


def level9_ganon_planning_notes() -> dict[str, Any]:
    return {
        "policy": "stun Ganon (sword) until brown, then Silver Arrow on B",
        "silver_arrows_ram": hex(ADDR_ARROWS),
        "silver_arrows_value_planned": ARROWS_SILVER_PLANNED,
        "object_type_id": OBJ_GANON,
        "brown_state_ram": hex(0x00AC),
        "brown_state_initial": GANON_BROWN_STATE,
        "dying_phase_base": hex(ADDR_GANON_OBJ_PHASE_BASE),
        "live_verified": True,
    }


def planning_report() -> dict[str, Any]:
    return {
        "level": LEVEL9,
        "name": "Death Mountain",
        "status": "backward_recon_live_natural_route_pending",
        "source_hypothesis": SOURCE_HYPOTHESIS,
        "required_entry_caps": sorted(required_caps_for_entry()),
        "required_ganon_caps": sorted(required_caps_for_ganon()),
        "full_triforce": FULL_TRIFORCE,
        "ram": {
            "triforce": hex(ADDR_TRIFORCE),
            "ring": hex(ADDR_RING),
            "arrows": hex(ADDR_ARROWS),
        },
        "screens_hypothesized": {
            "bomb_rock": hex(SCREEN_LEVEL9_ROCK_HYP),
            "potion_near": hex(SCREEN_LEVEL9_POTION_NEAR_HYP),
        },
        "rock_hops_from_start": [
            {"target": hex(h.target), "dir": h.direction} for h in LEVEL9_ROCK_HOPS
        ],
        "ganon": level9_ganon_planning_notes(),
        "ending_stop": "mode=0x13, updating!=0, submode=3 credits or 4 final",
        "live": {
            "rock_screen": SCREEN_LEVEL9_ROCK_HYP,
            "entry_room": ROOM_LEVEL9_ENTRY,
            "red_ring_room": None,
            "silver_arrow_room": None,
            "ganon_room": ROOM_GANON,
        },
        "docs": "nes/zelda_i/docs/LEVEL9_ROUTE.md",
    }
