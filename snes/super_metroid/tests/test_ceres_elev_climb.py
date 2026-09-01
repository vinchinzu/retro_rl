"""Unit tests for Ceres elevator shaft climb actions (no emulator)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import super_metroid.routes.kpdr.ceres.elev_escape as elev_escape
from super_metroid.paths import GAME_DIR
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.ceres.elev_escape import (
    CERES_ELEV_BENCH_FRAMES,
    _CERES_475_TO_363_AWAY,
    _CERES_475_TO_363_INTO,
    _ceres_any_wall_latch,
    _ceres_elev_entry_action,
    _ceres_elev_leaving,
    _ceres_elev_ship_band,
    _ceres_elev_top_seat,
    _ceres_fast_entry_window,
    _ceres_planted_at,
    ship_pad_action,
)
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_TOP_X,
    _CERES_ELEV_TOP_Y,
)
from super_metroid.routes.kpdr.ceres.spine import tas_hop_clock
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR
from super_metroid.routes.skills.geometry import FACE_LEFT_POSES, FACE_RIGHT_POSES

_TAS_CERES_HOPS = GAME_DIR / "tas" / "bodies" / "sniq_100_ceres_lsnes_hops.json"


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_CERES_ELEVATOR,
        "samus_x": 60,
        "samus_y": 475,
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


def test_inbound_door_transition_is_not_leaving() -> None:
    inbound = _state(game_state=11, samus_y=139, samus_x=40)
    assert not _ceres_elev_leaving(inbound)
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


def test_fast_entry_requires_preserved_spin_phase() -> None:
    fast = _state(
        samus_x=216,
        samus_y=628,
        pose=26,
        velocity_y=4,
        vertical_direction=1,
        momentum_x=2,
        invincibility_timer=5,
    )
    assert _ceres_fast_entry_window(fast)
    assert _ceres_fast_entry_window(replace(fast, samus_y=632, pose=25))
    assert not _ceres_fast_entry_window(replace(fast, samus_y=651, pose=10))
    assert not _ceres_fast_entry_window(replace(fast, samus_y=620))


def test_elev_entry_keeps_fast_spin_window() -> None:
    fast = _state(
        samus_x=216,
        samus_y=628,
        pose=26,
        velocity_y=4,
        vertical_direction=1,
        momentum_x=2,
        invincibility_timer=5,
    )
    assert _ceres_elev_entry_action(fast) is None
    floor = _state(samus_x=216, samus_y=651, pose=10)
    assert _ceres_elev_entry_action(floor) is None
    door = _state(samus_x=216, samus_y=139, pose=26, game_state=11)
    assert _ceres_elev_entry_action(door) == ()
    stale = _state(samus_x=216, samus_y=139, pose=26, game_state=8)
    assert _ceres_elev_entry_action(stale) == ()


def test_475_to_363_is_left_walljump() -> None:
    """TAS plants 475 at x=163 facing left; RIGHT runup dumps the well."""
    assert _CERES_475_TO_363_INTO == "LEFT"
    assert _CERES_475_TO_363_AWAY == "RIGHT"
    plant = _state(samus_x=163, samus_y=475, pose=167, velocity_y=0)
    assert _ceres_planted_at(plant, 475)


def test_left_wall_latch_is_contact() -> None:
    assert _ceres_any_wall_latch(_state(pose=131, samus_x=155, samus_y=404))
    assert _ceres_any_wall_latch(_state(pose=POSE_WALL_LATCH, samus_x=211, samus_y=600))
    assert not _ceres_any_wall_latch(_state(pose=25, samus_x=155, samus_y=404))


def test_elev_to_landing_is_tas_wj_speed() -> None:
    """Red until Sniq TAS WJ lands at 2246f. Do not edit the hops JSON."""
    raw = json.loads(_TAS_CERES_HOPS.read_text())
    assert raw["schema"] == "sm_tas_ceres_hops_v2"
    assert raw["source"] == "lsnes_oracle"
    hops = {h["name"]: h for h in raw["hops"]}
    elev = hops["elev_to_landing"]
    tas_frames = tas_hop_clock(elev)["frames"]
    assert tas_frames == 2246
    wj = raw["elev_wj"]
    assert wj["plant_to_plant_frames"] == 31
    assert wj["left_latch_131"]["pose"] == 131
    fast = wj["fast_entry"]
    assert fast["pose"] == 25
    assert (fast["x"], fast["y"]) == (216, 632)
    assert not hasattr(elev_escape, "CeresShaftClimb")
    assert not hasattr(elev_escape, "_ceres_checkpoint_shaft")
    assert _CERES_475_TO_363_INTO == "LEFT"
    assert _CERES_475_TO_363_AWAY == "RIGHT"
    assert CERES_ELEV_BENCH_FRAMES == tas_frames
