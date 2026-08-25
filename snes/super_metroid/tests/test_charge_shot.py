"""Unit tests for the shared charge-release beam primitive (no emulator)."""

from __future__ import annotations

import numpy as np

from super_metroid.ram import FACING_LEFT, FACING_RIGHT
from super_metroid.routes.skills.charge_shot import (
    ADDR_BEAM_CHARGE,
    CHARGE_FULL,
    FIRE_RANGE_PX,
    JUMP_LEAD,
    MOVEMENT_TURNING,
    aim_shot_buttons,
    beam_charge_counter,
    in_shot_seat,
    is_turning,
    position_then_charge_action,
)


def test_is_turning_is_movement_14() -> None:
    assert MOVEMENT_TURNING == 14
    assert is_turning(14)
    assert not is_turning(0)
    assert not is_turning(9)


def test_beam_charge_counter_reads_0cd0() -> None:
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_BEAM_CHARGE] = 60
    assert beam_charge_counter(ram) == 60
    ram[ADDR_BEAM_CHARGE] = 0x78
    ram[ADDR_BEAM_CHARGE + 1] = 0
    assert beam_charge_counter(ram) == 120
    assert beam_charge_counter(None) == 0
    assert beam_charge_counter(np.zeros(8, dtype=np.uint8)) == 0


def test_aim_shot_buttons_r_is_diagonal_not_up_left() -> None:
    diag = aim_shot_buttons(-80, -80, fire=True, include_face=True)
    assert "R" in diag
    assert "X" in diag
    assert "LEFT" in diag
    assert "UP" not in diag
    under = aim_shot_buttons(0, -80, fire=True)
    assert under == ("UP", "X")
    flat = aim_shot_buttons(-30, -10, fire=True)
    assert flat == ("X",)
    down = aim_shot_buttons(40, 50, fire=True, include_face=True)
    assert "L" in down
    assert "R" not in down


def test_in_shot_seat_range_and_clamp() -> None:
    assert in_shot_seat(672, 187, 638, 168)
    assert not in_shot_seat(879, 187, 638, 168)
    assert in_shot_seat(
        672, 187, 500, 168, fire_range_px=20, approach_x_min=672
    )
    assert not in_shot_seat(879, 187, 638, 168, approach_x_min=672)


def test_position_then_charge_walks_then_releases() -> None:
    walk = position_then_charge_action(
        879, 187, FACING_LEFT, 638, 168, charge=0
    )
    assert walk[0] == "LEFT"
    assert "B" in walk
    assert "X" in walk
    turn = position_then_charge_action(
        672, 187, FACING_RIGHT, 638, 168, charge=0
    )
    assert turn == ("LEFT",)
    turning = position_then_charge_action(
        672, 187, FACING_LEFT, 638, 168, movement_type=14, charge=40
    )
    assert turning == ("LEFT",)
    assert "X" not in turning
    charge = position_then_charge_action(
        672, 187, FACING_LEFT, 638, 168, charge=0
    )
    assert "X" in charge
    assert "LEFT" not in charge
    grounded_full = position_then_charge_action(
        672, 187, FACING_LEFT, 638, 168, charge=CHARGE_FULL, velocity_y=0
    )
    assert "X" in grounded_full
    assert "A" in grounded_full
    release = position_then_charge_action(
        672, 187, FACING_LEFT, 638, 168, charge=CHARGE_FULL, velocity_y=2
    )
    assert "X" not in release
    assert "A" in release
    jump_lead = position_then_charge_action(
        672, 187, FACING_LEFT, 638, 168, charge=CHARGE_FULL - JUMP_LEAD
    )
    assert "X" in jump_lead
    assert "A" in jump_lead


def test_clamp_fires_instead_of_walking_through_robot() -> None:
    """Seat east of the Workrobot: fire, do not walk into 624."""
    names = position_then_charge_action(
        672,
        187,
        FACING_LEFT,
        638,
        168,
        charge=0,
        approach_x_min=672,
        fire_range_px=FIRE_RANGE_PX,
    )
    assert "X" in names
    assert "LEFT" not in names
    still_east = position_then_charge_action(
        800,
        187,
        FACING_LEFT,
        638,
        168,
        charge=0,
        approach_x_min=672,
    )
    assert still_east[0] == "LEFT"
    assert "B" in still_east
