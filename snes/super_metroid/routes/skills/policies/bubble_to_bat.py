"""Bubble Mountain → Bat Cave hop policy (geometry + budgets + WJ timings).

Room-agnostic skills read these attributes; they must not hardcode Bubble
geometry. Product compose lives in :mod:`super_metroid.routes.kpdr.to_bat_cave`.

No controller imports — safe for all skill layers.
"""

from __future__ import annotations

from super_metroid.routes.controller_common import WallJumpTiming
from super_metroid.routes.kpdr.rooms import ROOM_BAT_CAVE, ROOM_BUBBLE

# ---------------------------------------------------------------------------
# Room identity
# ---------------------------------------------------------------------------

ROOM_ID = ROOM_BUBBLE
EXIT_ROOM_ID = ROOM_BAT_CAVE

# Bubble Mountain (0xACB3) is 2×4 screens. Entry node 3 mid-left ≈ (39–60, 634).
# Climb to top-right green Super door node 7 (block [31, 7] ≈ x496 y112).
# Wrong-door traps: left y≈624 Rising Tide, y≈368 Save, y≈112 Missiles Super;
# right y≈368 Single Chamber.

LOWER_FRAMES = 3500
MID_REPIN_FRAMES = 900
MID_FRAMES = 5500
DOOR_FRAMES = 1200  # headroom for fall re-climb on natural compose
TO_BAT_SETTLE_FRAMES = 320
# R19 Phase E: sticky right-structure WJ + Super pressure (from Phase D pin).
# Door shell ≈ x496 y112; fire when x≥DOOR_SUPER_X and y≤DOOR_SUPER_Y.
DOOR_SUPER_X = 420
DOOR_SUPER_Y = 160
DOOR_WJ_PERIOD = 10
DOOR_WJ_INTO = 3  # LEFT+A
DOOR_WJ_BOUNCE = 2  # RIGHT+A
DOOR_X_CAP = 480  # pull left if past outer wall
# Right-shelf fail mode (continuous Ceres successor can fall to ~(446,383)):
# jump LEFT off the shelf before re-approaching the climbable structure.
DOOR_OUTER_X = 400
# The door controller is reactive now; an entry crouch only lets the new
# continuous Ceres successor fall off the top structure before seeking.
DOOR_CROUCH_FRAMES = 0
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
# R17: stationary left-face X spray (no dir) then walk — avoids KB pose 138
# that LEFT+X-while-walking produces. Human pin free-walks; pure needs clear.
SAVE_STATIONARY_FACE = 6
SAVE_STATIONARY_X = 28  # R18: pure left-blocker needs longer L-angle spray
# Human-matched fire seat band (maprando left climb / bubble_human_runway).
# Integer (27,395)p2 is necessary but not sufficient — pure dumps still lack
# human run-windup/subpixel; open-loop R15 tops only on human pin state.
# Continuous Spazer (rr-cwu): seat x=32 x_sub~4k fails Phase D (min_y=161
# mx~267); pure seat x=31 x_sub~49k tops. Cap human_hi at 30 so seat_max_left
# keeps walking left of the right-edge reject.
SAVE_HUMAN_SEAT_X = (25, 30)
# Fire-seat dash: 21f proven on human pin. Max dash value without Speed is ~32f,
# but longer bare dash from x~27 walks off the short runway (probe RED).
SAVE_RUN_FRAMES = 21
SAVE_DASH_MAX_FRAMES = 32  # experiment / longer platforms only
# R18 pure natural seat: arm-pump RED (desyncs jump). Human pin isolation still
# greens with arm-pump + R15 WJ2 (24/14/56) — product follows pure.
SAVE_ARM_PUMP = False
SAVE_ARM_PUMP_PERIOD = 2  # frames per L↔R toggle when arm-pump experiment on
SAVE_CROUCH_FRAMES = 2  # crouch-jump experiment (+~8 px start); off in product
SAVE_SPIN_FRAMES = 83
SAVE_APPROACH_BA = 4  # B+A coast into wall
SAVE_APPROACH_IDLE = 2
SAVE_APPROACH_TURN = 2
# R18 double walljump — pure natural **live** seat Phase D (full lower→seat):
#   WJ1 L20 a4 R8 + WJ2 L14 a2 R6 + follow40 → top (mx200≥300, pose 84).
# Dump-only seat (shorter idle) preferred WJ2 L16/R10; product follows **live**.
# R15 human-pin isolation used WJ2 L24/R14/follow56 + arm-pump (not pure product).
# Skills: consecutive_walljumps / double_walljump (always 2 WJ).
SAVE_WJ_LEFT_A = 20  # first WJ hold LEFT+A
SAVE_WJ_AMID = 4  # A-only between flips
SAVE_WJ_RIGHT_A = 8  # first flip RIGHT+A
SAVE_WJ2_LEFT_A = 14  # R18 live pure Phase D
SAVE_WJ2_AMID = 2
SAVE_WJ2_RIGHT_A = 6  # R18 live pure Phase D
SAVE_WJ_FOLLOW = 40  # RIGHT+B+A spin after double WJ (Phase D push)
# Wall-ready seek: RIGHT+B+A only (never LEFT+A — that burns WJ1).
WJ_INTO_X = 250
WJ_LATCH_TIMEOUT = 36
WJ_APPROACH_X = (230, 290)
WJ_APPROACH_Y = (200, 340)
# R18: after right-wall WJ1, human tops via left-wall pose 84 at x~212 y~157.
# Pure dump still fails that contact (not fixed by enemy kill / 3-byte patch).
# Named groups for one-knob cards:
#   WJ1_FLIP_FOLLOW — SAVE_WJ_RIGHT_A / SAVE_WJ_AMID / SAVE_WJ_FOLLOW (+ R15_WJ1)
#   WJ2_LEFT_SEEK   — WJ2_LEFT_X / SEEK / INTO / FLIP / Y (walljump_second_left_wall)
WJ2_LEFT_X = 220  # seek band: left of this while y high after WJ1
WJ2_LEFT_Y = 200  # y band for left-wall contact (human ~157)
WJ2_LEFT_SEEK = 28
WJ2_LEFT_INTO = 8  # RIGHT+A into after seek (was hardcoded skill default)
WJ2_LEFT_FLIP = 16  # RIGHT+B+A flip after into
# Experiment: damage-boost hold during KB (Geruta/Waver) — not product.
DMG_BOOST_HOLD_FRAMES = 8

# R19: enemy-phase-aware fire (Geruta slots 4/6). Open-loop product fire tops
# only when patrol geometry matches a short clear window; wait with pure idle
# (preserves max-left seat). Period ~144f between class-A and class-B windows.
# Proven on post_bubble_fire_start_fullpure_r18: wait 89–93 (class A) and
# 233–235 (class B). Zeroing HP alone does NOT unlock Phase D.
FIRE_PHASE_MAX_WAIT = 280
# Class A: e4 deep-left + e6 mid (fullpure ~89–93, pure_seat ~108–110)
FIRE_PHASE_A_E4 = (117, 125, 270, 276)  # x0,x1,y0,y1
FIRE_PHASE_A_E6 = (190, 198, 158, 172)
# Class B: e4 deep-center + e6 lower-mid (fullpure ~233–235)
FIRE_PHASE_B_E4 = (158, 165, 272, 276)
FIRE_PHASE_B_E6 = (175, 182, 184, 190)
# Note: live isolation seat (179,113)/(146,155) also tops, but a pure_seat
# near-miss (185,105)/(140,157) does NOT — do not widen a "class C" box.
# Geruta enemy ID pointer (bank A0 header) for slots 4/6 in Bubble.
FIRE_PHASE_GERUTA_ID = 0xD63F
FIRE_PHASE_SLOTS = (4, 6)

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

# Public phase-ladder aliases (tests / probe / k4 re-exports).
BUBBLE_PHASE_C_X_MIN = PHASE_C_X_MIN
BUBBLE_PHASE_C_Y_MAX = PHASE_C_Y_MAX
BUBBLE_PHASE_C_Y_MIN = PHASE_C_Y_MIN
BUBBLE_PHASE_D_X = PHASE_D_X
BUBBLE_PHASE_D_Y = PHASE_D_Y

# Documented vertical classes (px/frame, approximate; Hi-Jump route).
HIJUMP_WALLJUMP_VY0 = 5.33
REGULAR_WALLJUMP_VY0 = 4.41
DAMAGE_BOOST_HX = 5.25  # Geruta/Waver-class horizontal KB magnitude (approx)

POSE_KNOCKBACK = frozenset({137, 138})
POSE_STAND_LEFT = frozenset({2, 10})
POSE_STAND_RIGHT = frozenset({1, 9})

# Door WJ pose family (right structure; not only latch 132).
DOOR_WJ_POSES = frozenset({81, 82, 83, 84, 132})
DOOR_FALL_Y = 220

# R15 Phase-D proven consecutive pair (human runway pin open-loop).
R15_WJ1 = WallJumpTiming(
    into="LEFT",
    flip="RIGHT",
    into_frames=SAVE_WJ_LEFT_A,
    amid_frames=SAVE_WJ_AMID,
    flip_frames=SAVE_WJ_RIGHT_A,
    delay_into_frames=0,
)
R15_WJ2 = WallJumpTiming(
    into="LEFT",
    flip="RIGHT",
    into_frames=SAVE_WJ2_LEFT_A,
    amid_frames=SAVE_WJ2_AMID,
    flip_frames=SAVE_WJ2_RIGHT_A,
    delay_into_frames=0,
)
R15_DOUBLE: tuple[WallJumpTiming, WallJumpTiming] = (R15_WJ1, R15_WJ2)

DOOR_WJ = WallJumpTiming(
    into="LEFT",
    flip="RIGHT",
    into_frames=DOOR_WJ_INTO,
    amid_frames=1,
    flip_frames=DOOR_WJ_BOUNCE + 2,
)
