"""K4 Wave branch geometry: named bands, seats, and pure predicates.

Room-prefixed constants only — never a bare ``DOOR_X`` shared across
Bubble / Single / Double hops (import-order shadow caused continuous
regression when Bubble 470 overwrote Double Chamber 920).
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import WallJumpTiming
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUBBLE,
    ROOM_DOUBLE_CHAMBER,
    ROOM_SINGLE_CHAMBER,
    ROOM_WAVE,
)

# Wave Beam collected bit on ``collected_beams`` / ``equipped_beams``.
WAVE_BEAM_MASK = 0x0001

# ---------------------------------------------------------------------------
# Wave → Double Chamber return (WR_*) — rr-pd0i / Phase B hop 8
# Continuous tip / pure Wave successor ~(171,123) pose 137; left blue door
# into Double top-right ~(18,139). Human leave clean window ~f5720–5908.
# ---------------------------------------------------------------------------
WAVE_DOOR_X = 48
WAVE_LEAVE_FRAMES = 420
WAVE_DOUBLE_SETTLE = 280

# ---------------------------------------------------------------------------
# K4.8 Bubble → Single Chamber (BSC_*)
# Live pins from post_speed_return_to_bubble_pure + human Bubble→SC (2026-08-06).
# Top settle ~(472,115); drop band x≈381; door sill ~(492,395).
# ---------------------------------------------------------------------------
BSC_TOP_Y_MAX = 200
BSC_DROP_X = (370, 400)
BSC_DROP_TARGET_X = 385
BSC_MID_Y = (220, 340)
BSC_FLOOR_Y = 360
BSC_DOOR_Y = (380, 420)
BSC_DOOR_X = 470
BSC_SINGLE_SETTLE = 320
BSC_TOP_WALK_FRAMES = 400
BSC_DROP_FRAMES = 500
BSC_NAV_TO_DOOR_FRAMES = 1200
BSC_DOOR_PUSH_FRAMES = 400

# ---------------------------------------------------------------------------
# K4.9 Single → Double Chamber (SC_*)
# Live pure (2026-08-06): top→mid y267→floor y395 → stationary missiles →
# spin-hop gap → RIGHT into Double ``0xADAD``. Upper door is red (missiles).
# ---------------------------------------------------------------------------
SC_TOP_Y = 200
SC_MID_Y = (250, 290)
SC_FLOOR_Y = (380, 420)
SC_SHOT_X = (115, 135)
SC_DOOR_X = 220
SC_DOUBLE_SETTLE = 320

# ---------------------------------------------------------------------------
# K4.10 Double Chamber → Wave Beam PLM (DC_*)
# Live (2026-08-06, rr-dbu.10): entry ~(61,139); upper hop → Kamer seat
# x∈[370,375] y≤139; blue gate open = exact human tape buttons f4650–5200.
# Super door + Wave chozo → rr-re9.
# ---------------------------------------------------------------------------
DC_WAVE_SETTLE = 280
DC_GATE_X = (360, 430)
# Hop delivery band (wide); open phase reseats tighter to human tape.
DC_GATE_SEAT_X = (365, 390)
DC_GATE_SEAT_Y_MAX = 200
# Human open seat (tape f4650 ~(378,139); pure green band 370–375 @ y≤139).
DC_GATE_OPEN_SEAT_X = (370, 375)
DC_GATE_OPEN_SEAT_Y = 139
DC_PAST_GATE_X = 480
# Prefixed so Bubble BSC_DOOR_X is never overwritten at import time.
DC_DOOR_X = 920
DC_DOOR_Y_MAX = 180
# Past-gate **missile ledge** (y≈139). Super door runway lives HERE — not the
# spike floor (~y400). Solid ledge x≈414–608; launch edge ~x600 (later =
# higher door contact for WJ).
DC_LEDGE_Y_MAX = 165
DC_RUNWAY_X = 425
DC_EDGE_X = 600
DC_MISSILE_X = 495
# High door-column classic WJ off **right** wall: away (LEFT) then LEFT+A.
# walljump_once phases: delay_into=LEFT×N, into=LEFT+A×M.
# Live pure: contact ~(923,238) → LEFT×3 + LEFT+A×6 → left spin → right to
# sill ~(929,116). Never open-loop WJ on spike floor (y≳280).
DC_WJ = WallJumpTiming(
    into="LEFT",
    flip="LEFT",
    into_frames=6,
    amid_frames=0,
    flip_frames=0,
    delay_into_frames=3,
)
DC_WJ_LEFT_FOLLOW = 8


def has_wave(state: SuperMetroidState) -> bool:
    return bool(int(state.collected_beams) & WAVE_BEAM_MASK)


def dc_on_missile_ledge(state: SuperMetroidState) -> bool:
    """True when on the past-gate missile ledge (not spike floor)."""
    return (
        state.room_id == ROOM_DOUBLE_CHAMBER
        and state.samus_y <= DC_LEDGE_Y_MAX
        and state.velocity_y == 0
    )


def dc_on_sill(state: SuperMetroidState) -> bool:
    """True when high enough at Super door for sill / door push."""
    return (
        state.room_id == ROOM_DOUBLE_CHAMBER
        and state.samus_x >= DC_DOOR_X - 20
        and state.samus_y < DC_DOOR_Y_MAX
        and state.velocity_y == 0
    )


__all__ = [
    "WAVE_BEAM_MASK",
    "WAVE_DOOR_X",
    "WAVE_LEAVE_FRAMES",
    "WAVE_DOUBLE_SETTLE",
    "ROOM_BUBBLE",
    "ROOM_SINGLE_CHAMBER",
    "ROOM_DOUBLE_CHAMBER",
    "ROOM_WAVE",
    "BSC_TOP_Y_MAX",
    "BSC_DROP_X",
    "BSC_DROP_TARGET_X",
    "BSC_MID_Y",
    "BSC_FLOOR_Y",
    "BSC_DOOR_Y",
    "BSC_DOOR_X",
    "BSC_SINGLE_SETTLE",
    "BSC_TOP_WALK_FRAMES",
    "BSC_DROP_FRAMES",
    "BSC_NAV_TO_DOOR_FRAMES",
    "BSC_DOOR_PUSH_FRAMES",
    "SC_TOP_Y",
    "SC_MID_Y",
    "SC_FLOOR_Y",
    "SC_SHOT_X",
    "SC_DOOR_X",
    "SC_DOUBLE_SETTLE",
    "DC_WAVE_SETTLE",
    "DC_GATE_X",
    "DC_GATE_SEAT_X",
    "DC_GATE_SEAT_Y_MAX",
    "DC_GATE_OPEN_SEAT_X",
    "DC_GATE_OPEN_SEAT_Y",
    "DC_PAST_GATE_X",
    "DC_DOOR_X",
    "DC_DOOR_Y_MAX",
    "DC_LEDGE_Y_MAX",
    "DC_RUNWAY_X",
    "DC_EDGE_X",
    "DC_MISSILE_X",
    "DC_WJ",
    "DC_WJ_LEFT_FOLLOW",
    "has_wave",
    "dc_on_missile_ledge",
    "dc_on_sill",
]
