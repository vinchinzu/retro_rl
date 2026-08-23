"""Unit tests for Ceres elevator shaft climb actions (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.ceres.elev_escape import (
    CeresShaftClimb,
    _ceres_at_checkpoint,
    _ceres_elev_leaving,
    _ceres_elev_ship_band,
    _ceres_elev_top_seat,
    climb_ceres_shaft_action,
    ship_pad_action,
)
from super_metroid.routes.kpdr.ceres.geometry import (
    CERES_ELEV_HOPS,
    _CERES_ELEV_BOTTOM_Y,
    _CERES_ELEV_LEDGE_Y,
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_TOP_X,
    _CERES_ELEV_TOP_Y,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR
from super_metroid.routes.skills.geometry import FACE_LEFT_POSES, FACE_RIGHT_POSES


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_CERES_ELEVATOR,
        "samus_x": 60,
        "samus_y": _CERES_ELEV_LEDGE_Y,
        "pose": 10,
        "game_state": 8,
        "velocity_y": 0,
        "momentum_x": 0,
        "samus_x_sub": 0,
        "speed_flag": 0,
        "health": 19,
        "timer_type": 6,
        "facing": FACING_LEFT,
    }
    values.update(overrides)
    if "facing" not in overrides:
        pose = int(values["pose"])
        if pose in FACE_RIGHT_POSES:
            values["facing"] = FACING_RIGHT
        elif pose in FACE_LEFT_POSES:
            values["facing"] = FACING_LEFT
    return replace(base, **values)


def test_turn_pose_does_not_spin() -> None:
    climb = CeresShaftClimb()
    act = climb.action(_state(pose=15, samus_x=60, momentum_x=1))
    assert "A" not in act


def test_cold_ledge_pumps_before_jump() -> None:
    climb = CeresShaftClimb()
    act = climb.action(_state(pose=1, samus_x=60, momentum_x=0))
    assert "A" not in act
    assert act[0] == "RIGHT"
    assert "B" in act


def test_takeoff_waits_for_momentum_and_facing() -> None:
    hop = CERES_ELEV_HOPS[0]
    cold = _state(pose=10, samus_x=80, momentum_x=0)
    assert not hop.ready(cold)
    facing_left = _state(pose=10, samus_x=80, momentum_x=2)
    assert not hop.ready(facing_left)
    ready = _state(pose=9, samus_x=80, momentum_x=2, samus_x_sub=100)
    assert hop.ready(ready)


def test_windowed_takeoff_spins() -> None:
    climb = CeresShaftClimb()
    act = climb.action(_state(pose=9, samus_x=80, momentum_x=2))
    assert act == ("RIGHT", "B", "A")


def test_running_approach_arm_pumps() -> None:
    climb = CeresShaftClimb()
    act = climb.action(_state(pose=9, samus_x=50, momentum_x=2, speed_flag=1))
    assert act[0] == "RIGHT"
    assert "B" in act
    assert "A" not in act
    assert "L" in act or "R" in act


def test_center_landing_pose_settles() -> None:
    climb = CeresShaftClimb(last_ground_y=475)
    first = climb.action(_state(samus_x=130, samus_y=475, pose=166))
    assert "A" not in first
    assert first[0] in {"LEFT", "RIGHT"}


def test_left_pit_hops_right() -> None:
    action = climb_ceres_shaft_action(
        _state(samus_x=45, samus_y=_CERES_ELEV_BOTTOM_Y + 20, pose=2)
    )
    assert action == ("RIGHT", "A")


def test_knockback_idles_for_runup() -> None:
    climb = CeresShaftClimb(side="LEFT")
    action = climb.action(_state(samus_x=40, pose=137), knockback=True)
    assert action == ()
    assert climb.side == "RIGHT"


def test_latch_releases_a_then_flips() -> None:
    climb = CeresShaftClimb(side="LEFT", release_frames=2)
    first = climb.action(_state(pose=POSE_WALL_LATCH, samus_x=45, samus_y=400))
    assert first == ()
    assert climb.releasing
    second = climb.action(_state(pose=25, samus_x=45, samus_y=400, velocity_y=-4))
    assert second == ()
    third = climb.action(_state(pose=25, samus_x=45, samus_y=400, velocity_y=-4))
    assert third[0] == "RIGHT"
    assert "A" in third


def test_airborne_spin_holds_jump() -> None:
    climb = CeresShaftClimb(side="RIGHT", last_ground_y=571)
    air = _state(samus_x=120, samus_y=500, pose=25, velocity_y=-5)
    assert climb.action(air) == ("RIGHT", "B", "A")
    assert climb.action(air) == ("RIGHT", "B", "A")


def test_release_a_over_center_on_descent() -> None:
    climb = CeresShaftClimb(last_ground_y=571)
    land = _state(samus_x=130, samus_y=475, pose=25, velocity_y=2)
    assert climb.action(land) == ()


def test_countdown_leave_idles() -> None:
    leaving = _state(game_state=32, samus_y=75)
    assert climb_ceres_shaft_action(leaving) == ()


def test_uncrouch_before_jump() -> None:
    assert climb_ceres_shaft_action(_state(pose=39, samus_y=400)) == ("UP",)


def test_inbound_door_transition_is_not_leaving() -> None:
    inbound = _state(game_state=11, samus_y=139, samus_x=40)
    assert not _ceres_elev_leaving(inbound)
    assert climb_ceres_shaft_action(inbound) == ()
    fade = _state(game_state=9, samus_y=139)
    assert not _ceres_elev_leaving(fade)


def test_ceres_success_is_leaving() -> None:
    assert _ceres_elev_leaving(_state(game_state=32, samus_y=75))
    assert _ceres_elev_leaving(_state(game_state=8, room_id=0x91F8))


def test_ship_band_is_not_leave() -> None:
    pad = _state(samus_y=75, samus_x=_CERES_ELEV_SHIP_X, pose=2)
    assert _ceres_elev_ship_band(pad)
    assert not _ceres_elev_leaving(pad)


def test_ship_pad_walks_through_target_x() -> None:
    assert ship_pad_action(_state(samus_x=_CERES_ELEV_SHIP_X + 20)) == ("LEFT",)
    assert ship_pad_action(_state(samus_x=_CERES_ELEV_SHIP_X - 20)) == ("RIGHT",)
    assert ship_pad_action(_state(samus_x=_CERES_ELEV_SHIP_X)) == ()


def test_top_seat_is_s10_not_raw_y() -> None:
    mid = _state(samus_x=80, samus_y=_CERES_ELEV_TOP_Y + 20, pose=9)
    assert not _ceres_elev_top_seat(mid)
    seat = _state(
        samus_x=_CERES_ELEV_TOP_X,
        samus_y=_CERES_ELEV_TOP_Y,
        pose=137,
    )
    assert _ceres_elev_top_seat(seat)


def test_checkpoint_requires_ground_or_knockback_pose() -> None:
    airborne_apex = _state(samus_y=475, pose=25, velocity_y=0)
    assert not _ceres_at_checkpoint(airborne_apex, 475)
    debris_knockback = _state(samus_y=475, pose=137, velocity_y=0)
    assert _ceres_at_checkpoint(debris_knockback, 475)
