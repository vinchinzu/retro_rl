"""Pins the walk rates measured on named pins. No ROM."""

from __future__ import annotations

from alttp.walk_physics import (
    FLOOR_DIAGONAL_AXIS_PX,
    FLOOR_STRAIGHT_STEP_PX,
    GRASS_STEP_PX,
    HAS_TURN_LATTICE,
    LINK_BODY,
    IFRAMES_ON_HIT,
    KNOCKBACK_DX,
    KNOCKBACK_DY,
    KNOCKBACK_FRAMES,
    KNOCKBACK_HIT_XY,
    KNOCKBACK_REST_XY,
    KNOCKBACK_TIMER_ON_HIT,
    SHALLOW_STRAIGHT_STEP_PX,
    SPIN_CHARGE_COUNT,
    SPIN_CHARGE_FRAMES,
    STAIR_END_XY,
    STAIR_SUBMODULE,
    STAIR_TRANSITION_DX,
    STAIR_TRANSITION_DY,
    STAIR_TRANSITION_FRAMES,
    SUBPIXELS_PER_PIXEL,
    SWORD_HP_FRAME,
    SWORD_HURT_FRAME,
    SWORD_REACH,
    SWORD_REACH_STEP,
    SWORD_SAME_FRAME_TURNS,
    SWORD_SWING_FRAMES,
    TURN_PROBE_XY,
    body_overlaps,
    repeating_delta,
)


def test_floor_diagonal_shallow_and_no_lattice() -> None:
    assert SUBPIXELS_PER_PIXEL == 256
    assert FLOOR_STRAIGHT_STEP_PX == (1, 2)
    # FighterSword RIGHT, 48 frames.
    assert repeating_delta(FLOOR_STRAIGHT_STEP_PX, 48) == 72
    # HyruleCastleGrounds RIGHT and UP, 32 frames.
    assert repeating_delta(FLOOR_STRAIGHT_STEP_PX, 32) == 48
    assert FLOOR_DIAGONAL_AXIS_PX == 1
    assert FLOOR_DIAGONAL_AXIS_PX * 12 == 12
    assert HAS_TURN_LATTICE is False
    assert TURN_PROBE_XY[0] % 8 == 4
    assert SHALLOW_STRAIGHT_STEP_PX == (1, 1, 1, 2)
    # FighterSword shallow RIGHT, 36 frames on that cycle.
    assert repeating_delta(SHALLOW_STRAIGHT_STEP_PX, 36) == 45
    assert GRASS_STEP_PX is None


def test_stairs_and_knockback() -> None:
    assert STAIR_SUBMODULE == 8
    assert STAIR_TRANSITION_FRAMES == 57
    assert (STAIR_TRANSITION_DX, STAIR_TRANSITION_DY) == (0, -39)
    assert STAIR_END_XY == (760, 3137)
    assert KNOCKBACK_TIMER_ON_HIT == 18
    assert KNOCKBACK_FRAMES == 18
    assert IFRAMES_ON_HIT == 57
    # Released-pad end still has 57 - 18 frames of i-frames left.
    assert IFRAMES_ON_HIT - KNOCKBACK_FRAMES == 39
    assert (
        KNOCKBACK_REST_XY[0] - KNOCKBACK_HIT_XY[0],
        KNOCKBACK_REST_XY[1] - KNOCKBACK_HIT_XY[1],
    ) == (KNOCKBACK_DX, KNOCKBACK_DY)


def test_sword_timing_on_the_measured_pins() -> None:
    # FighterSword tap clears on frame 14. Hurt is inside that window.
    assert SWORD_SWING_FRAMES == 13
    assert SWORD_SAME_FRAME_TURNS is False
    assert SWORD_HURT_FRAME == 4
    assert SWORD_HP_FRAME == 10
    assert SWORD_HURT_FRAME < SWORD_HP_FRAME <= SWORD_SWING_FRAMES
    # Hold 60 leaves $79 at 47. Hold 61 reaches 48 and the release spins.
    assert SPIN_CHARGE_FRAMES == 61
    assert SPIN_CHARGE_COUNT == 48
    assert SPIN_CHARGE_COUNT == SPIN_CHARGE_FRAMES - 13


def test_sword_reach_table() -> None:
    # CastleB1Guard, still soldier, step 2. Sums lock the whole cell set.
    assert SWORD_REACH_STEP == 2
    assert set(SWORD_REACH) == {"DOWN", "LEFT", "RIGHT", "UP"}
    assert (0, 24) in SWORD_REACH["DOWN"]
    assert (0, 8) not in SWORD_REACH["DOWN"]
    assert (0, 40) not in SWORD_REACH["DOWN"]
    assert (-16, 0) in SWORD_REACH["LEFT"]
    assert (-32, 0) not in SWORD_REACH["LEFT"]
    assert (24, 0) in SWORD_REACH["RIGHT"]
    assert (34, 0) not in SWORD_REACH["RIGHT"]
    assert (0, -16) in SWORD_REACH["UP"]
    assert (0, -26) not in SWORD_REACH["UP"]
    sums = {
        "DOWN": 20062,
        "LEFT": -1036,
        "RIGHT": 14642,
        "UP": -10044,
    }
    counts = {"DOWN": 134, "LEFT": 147, "RIGHT": 127, "UP": 135}
    for facing, cells in SWORD_REACH.items():
        assert len(cells) == counts[facing]
        assert sum(dx * 3 + dy * 5 for dx, dy in cells) == sums[facing]
        assert all(dx % SWORD_REACH_STEP == 0 and dy % SWORD_REACH_STEP == 0 for dx, dy in cells)


def test_link_body_blocks_the_probe_wall() -> None:
    # CastleMain stops. Wall pixels are the 0x04 cells the press could not enter.
    assert LINK_BODY == (0, 8, 15, 23)
    probes = (
        # west column x 680..687, body rows at the stop
        ((688, 3496), (-1, 0), lambda px, py: 680 <= px <= 687 and 3504 <= py <= 3519),
        # east column starts at x 848
        ((832, 3496), (1, 0), lambda px, py: 848 <= px <= 855 and 3504 <= py <= 3519),
        # south row starts at y 3520
        ((700, 3496), (0, 1), lambda px, py: 3520 <= py <= 3527 and 700 <= px <= 715),
        # north row is y 3176..3183, above the body
        ((656, 3176), (0, -1), lambda px, py: 3176 <= py <= 3183 and 656 <= px <= 671),
    )
    for (x, y), (dx, dy), wall in probes:
        assert body_overlaps(x, y, wall) is False
        assert body_overlaps(x + dx, y + dy, wall) is True
