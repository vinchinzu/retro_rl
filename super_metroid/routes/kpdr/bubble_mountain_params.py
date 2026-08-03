"""Bubble Mountain geometry and timing constants (no controller imports).

Shared by :mod:`bubble_mountain` (predicates / lower / door) and
:mod:`bubble_mountain_mid` (launch + climb loop). Keep this module free of
session/runtime imports so both sides can depend on it without cycles.
"""

from __future__ import annotations

# Bubble Mountain (0xACB3) is 2×4 screens. Entry node 3 mid-left ≈ (39–60, 634).
# Climb to top-right green Super door node 7 (block [31, 7] ≈ x496 y112).
# Wrong-door traps: left y≈624 Rising Tide, y≈368 Save, y≈112 Missiles Super;
# right y≈368 Single Chamber.

LOWER_FRAMES = 3500
MID_REPIN_FRAMES = 900
MID_FRAMES = 5500
DOOR_FRAMES = 900
TO_BAT_SETTLE_FRAMES = 320
MID_Y = 400
TOP_Y = 200
TOP_X = 300

# true_ground (1/2/9/10) — mid-air charge/reseat/land only (spin apex 25≠land).
# stand_pin (+25/26/27/28) — lip seat, right-shelf seat, mid-iso pin, approach walk.
TRUE_GROUND = frozenset({1, 2, 9, 10})
STAND_PIN = frozenset({1, 2, 9, 10, 25, 26, 27, 28})
STANDING_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38, 137, 138})

# Hard-cap x to avoid Single Chamber outer-wall height trap (~y360 at x≥400).
CAVITY_X_MAX = 395
MID_STAND_X = (77, 160)

# R5 lower-left ledge path: solid shelves (entry floor ~y651 → save fire seat).
# R16: final hop lands y395 fire solid (x~50–56), not mid-iso float (105,370)
# that falls onto the solid lip (~79,427).
FLOOR_SHELF_X = 108
LOWER_SHELVES: tuple[tuple[int, int], ...] = (
    (120, 560),
    (110, 515),
    (100, 475),
    (90, 450),
    (70, 420),
    (50, 395),
)

# R6 solid save-door lip (place-grid jumpable; mid float at y~370 is NOT).
LIP_X = (65, 100)
LIP_Y = (410, 450)

# R7/R9/R10 right-structure after height class.
HEIGHT_CLASS_Y = 280
MID_RESEAT_Y = 320
RIGHT_SHELF_X = 300
RIGHT_SHELF_Y = 390  # include lower shelf landings (~y379)
RIGHT_WJ_PERIOD = 8
RIGHT_WJ_INTO = 2
RIGHT_WJ_BOUNCE = 2
MIDHIGH_Y = 450  # engage open-loop WJ/drop while y≤450

# Lip launch timings (R6/R9/R10 proven height class min_y=260).
LIP_CHARGE = 12
LIP_SPIN = 44
LIP_EXTEND = 70

# R14/R15 save-door runway (maprando left climb / human bubble_jump_try):
# stand outside Save (~y395), max-left runway x∈[25,90], run RIGHT, spin-glide,
# consecutive walljumps. R15 pin recon from bubble_human_runway.state:
#   run21 + spin83 + L20 a4 R8 L24 a2 R14 + right-spin → Phase D (x≥300 y≤200)
#   best height class min_y≈76–84 with same double-WJ family.
# Do NOT enter Save 0xB0DD (x≲20 while grounded).
SAVE_RUNWAY_X = (25, 90)
SAVE_RUNWAY_Y = (380, 430)
SAVE_RUNWAY_FIRE_X = (25, 60)  # max-left fire window (do not walk-right to center)
# R16: left-blocker clear (missile/X) when walk-left stalls at ~x37 on pure.
SAVE_CLEAR_X_FRAMES = 12
SAVE_EDGE_LEFT_FRAMES = 40  # walk left toward fire after solid land
SAVE_RUN_FRAMES = 21
SAVE_SPIN_FRAMES = 83
SAVE_APPROACH_BA = 4  # B+A coast into wall
SAVE_APPROACH_IDLE = 2
SAVE_APPROACH_TURN = 2
# R15 double walljump (clears ceiling lip / Phase D on human runway pin).
SAVE_WJ_LEFT_A = 20  # first WJ hold LEFT+A
SAVE_WJ_AMID = 4  # A-only between flips
SAVE_WJ_RIGHT_A = 8  # first flip RIGHT+A
SAVE_WJ2_LEFT_A = 24  # second WJ LEFT+A
SAVE_WJ2_AMID = 2
SAVE_WJ2_RIGHT_A = 14  # second flip then right-spin finish
SAVE_WJ_FOLLOW = 56  # RIGHT+B+A spin after double WJ (Phase D push)

# R13 floor-reclimb: after height class, if deep (y>MIDHIGH), re-seat on the
# mid-right floor solid (~y531 x∈[270,310]) and charged-HJ + period-8 WJ.
# Place recon: (288,531) p8i2b2 → Phase C ~(302,428). Not a shelf land; climb
# from that contact is a follow-on card.
FLOOR_RECLIMB_Y = 480  # treat as "deep" — use floor runway, not lip re-seat
FLOOR_RUNWAY_X = (270, 310)
FLOOR_RUNWAY_Y = 500  # solid shelf band ~y531
FLOOR_RECLIMB_CHARGE = 12
FLOOR_RECLIMB_SPIN = 44

# Phase ladder: usable right contact is the R11 bottleneck.
PHASE_C_X_MIN = 300
PHASE_C_Y_MAX = 430
PHASE_C_Y_MIN = 200
PHASE_D_X = TOP_X
PHASE_D_Y = TOP_Y
