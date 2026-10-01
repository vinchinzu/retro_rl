"""Measured Link walk rates. Each number names the pin that showed it.

NES Zelda walks 1 px/frame and only turns on an 8 px lattice. These rates
are from the SNES ROM. The subpixel unit is 256 per pixel (``$2B`` x,
``$2A`` y). On floor it toggles 0 and 128, which is the 1-then-2 step.
"""

from __future__ import annotations

__all__ = [
    "FLOOR_DIAGONAL_AXIS_PX",
    "FLOOR_DIAGONAL_PIN",
    "FLOOR_STRAIGHT_PIN",
    "FLOOR_STRAIGHT_STEP_PX",
    "GRASS_STEP_PX",
    "HAS_TURN_LATTICE",
    "ICE_COAST_PIN",
    "ICE_COAST_PX",
    "ICE_STRAIGHT_STEP_PX",
    "LINK_BODY",
    "IFRAMES_ON_HIT",
    "KNOCKBACK_DX",
    "KNOCKBACK_DY",
    "KNOCKBACK_FRAMES",
    "KNOCKBACK_HIT_XY",
    "KNOCKBACK_PIN",
    "KNOCKBACK_REST_XY",
    "KNOCKBACK_STATE",
    "KNOCKBACK_TIMER_ON_HIT",
    "SHALLOW_PIN",
    "SHALLOW_STRAIGHT_STEP_PX",
    "SPIN_CHARGE_COUNT",
    "SPIN_CHARGE_FRAMES",
    "SPIN_CHARGE_PIN",
    "STAIR_END_XY",
    "STAIR_PIN",
    "STAIR_SUBMODULE",
    "STAIR_TRANSITION_DX",
    "STAIR_TRANSITION_DY",
    "STAIR_TRANSITION_FRAMES",
    "SUBPIXELS_PER_PIXEL",
    "SWORD_HP_FRAME",
    "SWORD_HURT_FRAME",
    "SWORD_HURT_PIN",
    "SWORD_REACH",
    "SWORD_REACH_PIN",
    "SWORD_REACH_STEP",
    "SWORD_SAME_FRAME_TURNS",
    "SWORD_SWING_FRAMES",
    "SWORD_SWING_PIN",
    "TURN_PROBE_XY",
    "body_overlaps",
    "repeating_delta",
]

SUBPIXELS_PER_PIXEL = 256

# FighterSword hold RIGHT, 48 frames: (1, 2) repeating, +72 px. The same
# pair on HyruleCastleGrounds RIGHT and UP is 48 px in 32 frames. LEFT on
# that pin starts on the 2 px half; the phase follows the subpixel.
FLOOR_STRAIGHT_STEP_PX = (1, 2)
FLOOR_STRAIGHT_PIN = "FighterSword"

# FighterSword RIGHT+UP, first 12 frames, before the north wall dropped UP.
# Each frame was (+1, -1). Not the straight 1-then-2 pair on either axis.
FLOOR_DIAGONAL_AXIS_PX = 1
FLOOR_DIAGONAL_PIN = "FighterSword"

# FighterSword: one RIGHT leaves (2804, 2680), so x % 8 == 4. UP from
# there keeps that column, and y walks through every residue mod 8.
HAS_TURN_LATTICE = False
TURN_PROBE_XY = (2804, 2680)

# FighterSword shallow, tile (48, 15) at (2944, 2680). $114=0x09, $359=1,
# $345=0. Straight RIGHT is 1,1,1,2 (5 px / 4 frames). A 36-frame hold
# on that pattern is 45 px. Diagonal in the pool was not a clean sample.
SHALLOW_STRAIGHT_STEP_PX = (1, 1, 1, 2)
SHALLOW_PIN = "FighterSword"

# Not isolated. Indoor attribute 0x40 is absent. HyruleCastleGrounds at
# (2480, 2528) after RIGHT:64 still steps the floor pair until a wall
# (13 times -2,-1, one -1, then zeros). $351 stayed 0.
GRASS_STEP_PX = None

# Play_ice_0x0e_melted, all four body tiles ice at (7335, 376). LEFT 72
# frames reaches the pair (-2, 0), (-1, 0): the floor (1, 2) pair on the
# opposite phase, 1.5 px/frame. Release then coasts 18 px to (7237, 376).
# LEFT 24 has not reached that cap and coasts 7. Floor UP coasts 0. The
# vertical holds on this band stall before a coast, so 18 is that horizontal
# cap only. A walker releases this far short of the goal and does not tap
# the opposite direction to trim the slide.
ICE_STRAIGHT_STEP_PX = (2, 1)
ICE_COAST_PX = 18
ICE_COAST_PIN = "Play_ice_0x0e_melted"

# CastleMain room 0x61, hold UP onto attribute 0x1D. Before submodule 8
# the step is still the floor pair. Submodule 8 itself: 57 frames, dx=0,
# dy=-39, steps in {(0,-2),(0,-1),(0,0)}. Ends (760, 3137), room stays
# 0x61, next submodule 0. Not a steady px/frame.
STAIR_SUBMODULE = 8
STAIR_TRANSITION_FRAMES = 57
STAIR_TRANSITION_DX = 0
STAIR_TRANSITION_DY = -39
STAIR_END_XY = (760, 3137)
STAIR_PIN = "CastleMain"

# CastleB1Guard. Hold DOWN into the guard, then release the pad.
# The frame before the HP drop is (1273, 3696), $46=19, $31F=0, $5D=0.
# Hit frame: (1274, 3695), hp 0x18→0x14, $46=18, $31F=57, $5D=2.
# 18 frames later, buttons up: (1300, 3689), $46=0, $31F=39, $5D=0.
# The vector from the hit pose is (+26, -6). I-frames outlast recoil.
# A held-DOWN run reached $31F=0 57 frames after the same hit.
KNOCKBACK_TIMER_ON_HIT = 18
KNOCKBACK_FRAMES = 18
KNOCKBACK_DX = 26
KNOCKBACK_DY = -6
KNOCKBACK_HIT_XY = (1274, 3695)
KNOCKBACK_REST_XY = (1300, 3689)
KNOCKBACK_STATE = 2
IFRAMES_ON_HIT = 57
KNOCKBACK_PIN = "CastleB1Guard"

# FighterSword, one B after two idle frames. $50 and $3A bit 7 stay set on
# frames 1-13 and clear on frame 14, with $3C and $354. $5D stays 0.
# $3C across frames 1-13 is 0,1,2,3,4,5,5,5,5,6,7,8,8.
SWORD_SWING_FRAMES = 13
SWORD_SWING_PIN = "FighterSword"

# FighterSword dir 0x04. RIGHT alone becomes 0x06 and x 2803→2804 on that
# frame. B+RIGHT stays 0x04 at (2803, 2680). RIGHT held through the swing
# turns on frame 14, the frame $50 clears.
SWORD_SAME_FRAME_TURNS = False

# CastleB1Guard. RIGHT:14, idle 157-160, DOWN, then B. Link hp stays 24.
# Slot 0 type 0x41: $0EF0 becomes 0x9D on swing frame 4 ($3C first hits 3)
# and the soldier is shoved. HP 6→4 on frame 10. Frames 1-3 do not set it.
SWORD_HURT_FRAME = 4
SWORD_HP_FRAME = 10
SWORD_HURT_PIN = "CastleB1Guard"

# FighterSword, B held from a settled pose. $79 is 47 after 60 B frames and
# the release leaves $5D=0. $79 is 48 after 61 B frames and the release sets
# $5D=3, $3C=144, $354=15.
SPIN_CHARGE_FRAMES = 61
SPIN_CHARGE_COUNT = 48
SPIN_CHARGE_PIN = "FighterSword"

# CastleB1Guard what-if. Slot 0, a blue soldier, is re-pinned onto the
# cell each frame with velocity 0. dx, dy are sprite origin minus Link
# at the B press, step 2. A cell hits when $0EF0 becomes nonzero during
# the 13-frame swing. Facings are one frame from the pin: DOWN 0x02,
# LEFT 0x04, RIGHT 0x06, UP 0x00. Some arc cells connect after frame 4.
SWORD_REACH_STEP = 2
SWORD_REACH_PIN = "CastleB1Guard"
SWORD_REACH: dict[str, frozenset[tuple[int, int]]] = {
    "DOWN": frozenset((
        (12, 12), (14, 12), (12, 14), (12, 16), (12, 18), (-10, 20), (-8, 20),
        (-6, 20), (-4, 20), (-2, 20), (0, 20), (2, 20), (4, 20), (6, 20), (8, 20),
        (10, 20), (12, 20), (14, 20), (-10, 22), (-8, 22), (-6, 22), (-4, 22),
        (-2, 22), (0, 22), (2, 22), (4, 22), (6, 22), (8, 22), (10, 22), (12, 22),
        (14, 22), (-10, 24), (-8, 24), (-6, 24), (-4, 24), (-2, 24), (0, 24), (2, 24),
        (4, 24), (6, 24), (8, 24), (10, 24), (12, 24), (14, 24), (-10, 26), (-8, 26),
        (-6, 26), (-4, 26), (-2, 26), (0, 26), (2, 26), (4, 26), (6, 26), (8, 26),
        (10, 26), (-10, 28), (-8, 28), (-6, 28), (-4, 28), (-2, 28), (0, 28), (2, 28),
        (4, 28), (6, 28), (8, 28), (10, 28), (12, 28), (-10, 30), (-8, 30), (-6, 30),
        (-4, 30), (-2, 30), (0, 30), (2, 30), (4, 30), (6, 30), (8, 30), (10, 30),
        (12, 30), (14, 30), (16, 30), (-10, 32), (-8, 32), (-6, 32), (-4, 32),
        (-2, 32), (0, 32), (2, 32), (4, 32), (6, 32), (8, 32), (10, 32), (12, 32),
        (14, 32), (16, 32), (-10, 34), (-8, 34), (-6, 34), (-4, 34), (-2, 34), (0, 34),
        (2, 34), (4, 34), (6, 34), (8, 34), (10, 34), (12, 34), (14, 34), (-10, 36),
        (-8, 36), (-6, 36), (-4, 36), (-2, 36), (0, 36), (2, 36), (4, 36), (6, 36),
        (8, 36), (10, 36), (12, 36), (14, 36), (-10, 38), (-8, 38), (-6, 38), (-4, 38),
        (-2, 38), (0, 38), (2, 38), (4, 38), (6, 38), (8, 38), (10, 38), (12, 38),
        (14, 38),
    )),
    "LEFT": frozenset((
        (-24, -8), (-22, -8), (-20, -8), (-18, -8), (-16, -8), (-14, -8), (-12, -8),
        (-10, -8), (-8, -8), (-6, -8), (-4, -8), (-30, -6), (-28, -6), (-26, -6),
        (-24, -6), (-22, -6), (-20, -6), (-18, -6), (-16, -6), (-14, -6), (-12, -6),
        (-22, -4), (-20, -4), (-18, -4), (-16, -4), (-14, -4), (-12, -4), (-22, -2),
        (-20, -2), (-18, -2), (-16, -2), (-14, -2), (-12, -2), (-22, 0), (-20, 0),
        (-18, 0), (-16, 0), (-14, 0), (-12, 0), (-22, 2), (-20, 2), (-18, 2), (-16, 2),
        (-14, 2), (-12, 2), (-22, 4), (-20, 4), (-18, 4), (-16, 4), (-14, 4), (-12, 4),
        (-22, 6), (-20, 6), (-18, 6), (-16, 6), (-14, 6), (-12, 6), (-24, 8), (-22, 8),
        (-20, 8), (-18, 8), (-16, 8), (-14, 8), (-12, 8), (-28, 10), (-26, 10),
        (-24, 10), (-22, 10), (-20, 10), (-18, 10), (-16, 10), (-14, 10), (-12, 10),
        (-28, 12), (-26, 12), (-24, 12), (-22, 12), (-20, 12), (-18, 12), (-16, 12),
        (-14, 12), (-12, 12), (-28, 14), (-26, 14), (-24, 14), (-22, 14), (-20, 14),
        (-18, 14), (-16, 14), (-14, 14), (-12, 14), (-30, 16), (-28, 16), (-26, 16),
        (-24, 16), (-22, 16), (-20, 16), (-18, 16), (-16, 16), (-14, 16), (-12, 16),
        (-30, 18), (-28, 18), (-26, 18), (-24, 18), (-22, 18), (-20, 18), (-18, 18),
        (-16, 18), (-14, 18), (-12, 18), (-30, 20), (-28, 20), (-26, 20), (-24, 20),
        (-22, 20), (-20, 20), (-18, 20), (-16, 20), (-14, 20), (-12, 20), (-10, 20),
        (-8, 20), (-6, 20), (-4, 20), (-2, 20), (-30, 22), (-28, 22), (-26, 22),
        (-24, 22), (-22, 22), (-20, 22), (-18, 22), (-16, 22), (-14, 22), (-12, 22),
        (-10, 22), (-8, 22), (-6, 22), (-4, 22), (-22, 24), (-20, 24), (-18, 24),
        (-16, 24), (-14, 24), (-12, 24), (-10, 24),
    )),
    "RIGHT": frozenset((
        (4, -8), (6, -8), (8, -8), (10, -8), (12, -8), (14, -8), (16, -8), (18, -8),
        (20, -8), (22, -8), (24, -8), (12, -6), (14, -6), (16, -6), (18, -6), (20, -6),
        (22, -6), (24, -6), (26, -6), (28, -6), (30, -6), (12, -4), (14, -4), (16, -4),
        (18, -4), (20, -4), (22, -4), (24, -4), (24, -2), (24, 0), (24, 2), (24, 4),
        (24, 6), (24, 8), (26, 8), (12, 10), (14, 10), (16, 10), (18, 10), (20, 10),
        (22, 10), (24, 10), (26, 10), (28, 10), (30, 10), (32, 10), (12, 12), (14, 12),
        (16, 12), (18, 12), (20, 12), (22, 12), (24, 12), (26, 12), (28, 12), (30, 12),
        (32, 12), (12, 14), (14, 14), (16, 14), (18, 14), (20, 14), (22, 14), (24, 14),
        (26, 14), (28, 14), (30, 14), (32, 14), (12, 16), (14, 16), (16, 16), (18, 16),
        (20, 16), (22, 16), (24, 16), (26, 16), (28, 16), (30, 16), (32, 16), (12, 18),
        (14, 18), (16, 18), (18, 18), (20, 18), (22, 18), (24, 18), (26, 18), (28, 18),
        (30, 18), (32, 18), (4, 20), (6, 20), (8, 20), (10, 20), (12, 20), (14, 20),
        (16, 20), (18, 20), (20, 20), (22, 20), (24, 20), (26, 20), (28, 20), (30, 20),
        (32, 20), (4, 22), (6, 22), (8, 22), (10, 22), (12, 22), (14, 22), (16, 22),
        (18, 22), (20, 22), (22, 22), (24, 22), (26, 22), (28, 22), (30, 22), (10, 24),
        (12, 24), (14, 24), (16, 24), (18, 24), (20, 24), (22, 24), (24, 24),
    )),
    "UP": frozenset((
        (-12, -24), (-10, -24), (-8, -24), (-6, -24), (-4, -24), (-2, -24), (0, -24),
        (-12, -22), (-10, -22), (-8, -22), (-6, -22), (-4, -22), (-2, -22), (0, -22),
        (-12, -20), (-10, -20), (-8, -20), (-6, -20), (-4, -20), (-2, -20), (0, -20),
        (2, -20), (-12, -18), (-10, -18), (-8, -18), (-6, -18), (-4, -18), (-2, -18),
        (0, -18), (2, -18), (4, -18), (6, -18), (-18, -16), (-16, -16), (-14, -16),
        (-12, -16), (-10, -16), (-8, -16), (-6, -16), (-4, -16), (-2, -16), (0, -16),
        (2, -16), (4, -16), (6, -16), (-18, -14), (-16, -14), (-14, -14), (-12, -14),
        (-10, -14), (-8, -14), (-6, -14), (-4, -14), (-2, -14), (0, -14), (2, -14),
        (4, -14), (6, -14), (-18, -12), (-16, -12), (-14, -12), (-12, -12), (-10, -12),
        (-8, -12), (-6, -12), (-4, -12), (-2, -12), (0, -12), (2, -12), (4, -12),
        (6, -12), (-18, -10), (-16, -10), (-14, -10), (-12, -10), (-10, -10),
        (-8, -10), (-6, -10), (-4, -10), (-2, -10), (0, -10), (2, -10), (4, -10),
        (6, -10), (-18, -8), (-16, -8), (-14, -8), (-12, -8), (-10, -8), (-8, -8),
        (-6, -8), (-4, -8), (-2, -8), (0, -8), (2, -8), (4, -8), (6, -8), (8, -8),
        (10, -8), (12, -8), (-18, -6), (-16, -6), (-14, -6), (-12, -6), (12, -6),
        (-18, -4), (-16, -4), (-14, -4), (-12, -4), (12, -4), (-18, -2), (-16, -2),
        (-14, -2), (-12, -2), (-18, 0), (-16, 0), (-14, 0), (-12, 0), (-18, 2),
        (-16, 2), (-14, 2), (-12, 2), (-18, 4), (-16, 4), (-14, 4), (-12, 4), (-18, 6),
        (-16, 6), (-14, 6), (-12, 6), (-18, 8), (-16, 8), (-14, 8), (-12, 8), (12, 10),
    )),
}


# Inclusive pixel offsets from (link_x, link_y). CastleMain layer 0, mask
# $01F8, room origin (512, 3072), wall byte 0x04.
# LEFT stops (688, 3496): the pixel to the west is wall, link_x is floor.
# RIGHT stops (832, 3496): link_x+16 is wall, link_x+15 is floor.
# DOWN off the door stops (700, 3496): link_y+24 is wall, link_y+23 is floor.
# UP stops (656, 3176): link_y..link_y+7 sit in the wall; link_y+8..+23 are
# floor, and one more UP does not move. The coordinate itself is not a
# collision pixel.
LINK_BODY = (0, 8, 15, 23)


def body_overlaps(x: int, y: int, blocked) -> bool:
    """True when any ``LINK_BODY`` pixel is blocked. ``blocked(px, py)``."""
    x0, y0, x1, y1 = LINK_BODY
    for py in range(y + y0, y + y1 + 1):
        for px in range(x + x0, x + x1 + 1):
            if blocked(px, py):
                return True
    return False


def repeating_delta(step: tuple[int, ...], frames: int, phase: int = 0) -> int:
    """Signed px over ``frames`` of a repeating step, starting at ``phase``."""
    cycle = len(step)
    if cycle == 0:
        raise ValueError("step is empty")
    if frames < 0:
        raise ValueError(f"frames must be >= 0, got {frames}")
    return sum(step[(phase + i) % cycle] for i in range(frames))
