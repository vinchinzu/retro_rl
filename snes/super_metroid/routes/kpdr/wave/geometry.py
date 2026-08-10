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
# Double → Single Chamber return (DTS_*) — rr-qpkd / Phase B hop 9
# Source post_wave_to_double_chamber_pure ~(984,139) Super ledge.
# Human tape f6052–6752: LEFT hop off Super column → morph mid drop →
# floor LEFT (morph tunnel ~x450–550) → gap hop → bottom-left blue door y395.
# ---------------------------------------------------------------------------
DTS_LEDGE_Y_MAX = 170
DTS_HOP_LAUNCH_X = 950  # spin hop once left of this on high ledge
DTS_MID_X = (760, 860)  # Super column mid platforms
DTS_MID_Y = (220, 360)
DTS_FLOOR_Y_MIN = 420
DTS_FLOOR_Y = (430, 470)
DTS_MORPH_TUNNEL_X = (440, 580)  # low ceiling; must morph-roll
DTS_GAP_LAUNCH_X = 190  # floor spin hop toward left door past this
DTS_DOOR_X = 40
DTS_DOOR_Y = (380, 415)
DTS_SINGLE_SETTLE = 280
DTS_LEDGE_FRAMES = 200
DTS_DROP_FRAMES = 700
DTS_FLOOR_FRAMES = 900
DTS_DOOR_FRAMES = 500

# ---------------------------------------------------------------------------
# Single → Bubble return (STB_*) — rr-u0y8 / Phase B hop 10
# Source post_double_to_single_chamber_pure ~(216,630) deep shaft.
# Human tape f6908–7496: deep LEFT → left-wall spin climbs through mid
# platforms y523 → y395 → upper → top y139 left blue door → Bubble 0xACB3
# settle ~(472–525,395) right sill (transition first-frame (19,139)).
# ---------------------------------------------------------------------------
STB_DEEP_Y_MIN = 580  # pure pin / post-door settle band
STB_MID_LOW_Y = (500, 540)  # first land platform ~y523
STB_FLOOR_Y = (380, 420)  # missile / mid door height ~y395
STB_MID_HI_Y = (250, 290)  # mid platforms ~y267
STB_UPPER_Y = (190, 230)  # upper ledge ~y213
STB_TOP_Y_MAX = 160  # top shelf y139
STB_WALL_X = 55  # left wall approach
STB_MID_LOW_LAND_X = (45, 160)
STB_FLOOR_LAND_X = (80, 160)
STB_DOOR_X = 40  # top-left blue door push
STB_DOOR_Y = (120, 160)
STB_BUBBLE_SETTLE = 280
STB_DEEP_FRAMES = 200
STB_CLIMB_FRAMES = 1800
STB_DOOR_FRAMES = 400

# ---------------------------------------------------------------------------
# Bubble → Upper Norfair Farm return (BTF_*) — rr-czg9 / Phase B hop 11
# Source post_single_to_bubble_pure ~(472,395) right mid sill (node 6).
# Human tape f7624–8952: LEFT+A climb to upper ~y155 → LEFT hop/drop morph
# shaft → mid-low y531 → morph tunnels down-right → bottom y745–905 LEFT
# through low ceiling → bottom-most left blue door (node 4) y907 → Farm
# ``0xAF72`` settle right top ~(472–523,139).
# ---------------------------------------------------------------------------
BTF_MID_Y = (380, 420)  # pin / Single-door sill ~y395
BTF_UPPER_Y = (130, 180)  # first upper land ~y155
BTF_MID_LOW_Y = (500, 560)  # mid shaft land ~y531
BTF_BOTTOM_Y_MIN = 700  # bottom morph zone
BTF_BOTTOM_FLOOR_Y = (880, 950)  # bottom floor / door height ~y907
BTF_DOOR_X = 40
BTF_DOOR_Y = (880, 940)
BTF_FARM_SETTLE = 280
BTF_CLIMB_FRAMES = 500
BTF_DROP_FRAMES = 900
BTF_BOTTOM_FRAMES = 1600
BTF_DOOR_FRAMES = 400

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
    "DTS_LEDGE_Y_MAX",
    "DTS_HOP_LAUNCH_X",
    "DTS_MID_X",
    "DTS_MID_Y",
    "DTS_FLOOR_Y_MIN",
    "DTS_FLOOR_Y",
    "DTS_MORPH_TUNNEL_X",
    "DTS_GAP_LAUNCH_X",
    "DTS_DOOR_X",
    "DTS_DOOR_Y",
    "DTS_SINGLE_SETTLE",
    "DTS_LEDGE_FRAMES",
    "DTS_DROP_FRAMES",
    "DTS_FLOOR_FRAMES",
    "DTS_DOOR_FRAMES",
    "STB_DEEP_Y_MIN",
    "STB_MID_LOW_Y",
    "STB_FLOOR_Y",
    "STB_MID_HI_Y",
    "STB_UPPER_Y",
    "STB_TOP_Y_MAX",
    "STB_WALL_X",
    "STB_MID_LOW_LAND_X",
    "STB_FLOOR_LAND_X",
    "STB_DOOR_X",
    "STB_DOOR_Y",
    "STB_BUBBLE_SETTLE",
    "STB_DEEP_FRAMES",
    "STB_CLIMB_FRAMES",
    "STB_DOOR_FRAMES",
    "BTF_MID_Y",
    "BTF_UPPER_Y",
    "BTF_MID_LOW_Y",
    "BTF_BOTTOM_Y_MIN",
    "BTF_BOTTOM_FLOOR_Y",
    "BTF_DOOR_X",
    "BTF_DOOR_Y",
    "BTF_FARM_SETTLE",
    "BTF_CLIMB_FRAMES",
    "BTF_DROP_FRAMES",
    "BTF_BOTTOM_FRAMES",
    "BTF_DOOR_FRAMES",
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
