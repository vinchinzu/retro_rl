"""Unit tests for Ceres elevator shaft climb actions (no emulator).

Occupancy tests lock half-tile shaft faces and named ledge faces so col 13
cannot silently become a full 16px solid. They read the editor export and
``rom.sfc`` bytes; they do not boot stable-retro.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import super_metroid.routes.kpdr.ceres.magnet as magnet
from super_metroid.generalist.solid import editor_rooms_dir
from super_metroid.paths import GAME_DIR, INTEGRATION_DIR
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.ceres.magnet import (
    CERES_ELEV_BENCH_FRAMES,
    CERES_ELEV_MAX_FRAMES,
    _ceres_elev_entry_action,
    _ceres_elev_grounded,
    _ceres_elev_leaving,
    _ceres_elev_ship_band,
    _ceres_fast_entry_window,
    _ceres_planted_at,
    ship_pad_action,
)
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_ELEV_171_LAUNCH_X,
    _CERES_ELEV_267_LAUNCH_X,
    _CERES_ELEV_363_LAUNCH_X,
    _CERES_ELEV_475_LAUNCH_X,
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_TOP_X,
    _CERES_ELEV_TOP_Y,
    _CERES_ELEV_WJ_RELEASE_FRAMES,
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


def test_elev_leave_is_gs32_or_landing_room() -> None:
    """Leave detector only. Reaching landing is not a TAS pass."""
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


def test_top_rung_is_planted_not_a_right_wall_seat() -> None:
    """The climb lands 171 on its west end; the old s10 east seat is gone."""
    assert _ceres_planted_at(_state(samus_x=66, samus_y=_CERES_ELEV_TOP_Y, pose=167),
                             _CERES_ELEV_TOP_Y)
    assert not hasattr(magnet, "_ceres_elev_top_seat")
    assert _CERES_ELEV_TOP_X == 211  # right wall the entry wall jump kicks off


def test_fast_entry_requires_preserved_spin_phase() -> None:
    """Momentum floor is 1: the door leave halves to 1.375 before this band."""
    fast = _state(
        samus_x=216,
        samus_y=628,
        pose=26,
        velocity_y=4,
        vertical_direction=1,
        momentum_x=1,
        invincibility_timer=5,
    )
    assert _ceres_fast_entry_window(fast)
    assert _ceres_fast_entry_window(replace(fast, samus_y=633, momentum_x=1))
    assert _ceres_fast_entry_window(replace(fast, samus_y=632, pose=25))
    assert not _ceres_fast_entry_window(replace(fast, samus_y=651, pose=26))
    assert not _ceres_fast_entry_window(replace(fast, samus_y=651, pose=10))
    assert not _ceres_fast_entry_window(replace(fast, samus_y=620))
    assert not _ceres_fast_entry_window(replace(fast, velocity_y=-1))
    assert not _ceres_fast_entry_window(replace(fast, momentum_x=0))
    assert not _ceres_fast_entry_window(replace(fast, invincibility_timer=0))


def test_elev_entry_keeps_fast_spin_window() -> None:
    fast = _state(
        samus_x=216,
        samus_y=628,
        pose=26,
        velocity_y=4,
        vertical_direction=1,
        momentum_x=1,
        invincibility_timer=5,
    )
    assert _ceres_elev_entry_action(fast) is None
    floor = _state(samus_x=216, samus_y=651, pose=10)
    assert _ceres_elev_entry_action(floor) is None
    door = _state(samus_x=216, samus_y=139, pose=26, game_state=11)
    assert _ceres_elev_entry_action(door) == ()
    stale = _state(samus_x=216, samus_y=139, pose=26, game_state=8)
    assert _ceres_elev_entry_action(stale) == ()


def test_475_plant_is_a_land_pose() -> None:
    """TAS plants 475 at x=163; the wall-jump climb plants it at x=156."""
    assert _ceres_planted_at(_state(samus_x=163, samus_y=475, pose=167), 475)
    assert _ceres_planted_at(_state(samus_x=156, samus_y=475, pose=167), 475)
    assert not _ceres_planted_at(_state(samus_x=156, samus_y=475, pose=25), 475)


def test_entry_wall_jump_release_is_two_frames() -> None:
    """One release frame reads as a jump cut here and never latches 132."""
    assert _CERES_ELEV_WJ_RELEASE_FRAMES == 2
    assert POSE_WALL_LATCH == 132


def test_ledge_launch_x_sit_inside_their_measured_bands() -> None:
    """Each rung's launch x is the middle of its band, not an edge."""
    for launch_x, (low, high) in (
        (_CERES_ELEV_475_LAUNCH_X, (130, 144)),
        (_CERES_ELEV_363_LAUNCH_X, (185, 211)),
        (_CERES_ELEV_267_LAUNCH_X, (132, 156)),
        (_CERES_ELEV_171_LAUNCH_X, (41, 55)),
    ):
        assert low < launch_x < high
        assert min(launch_x - low, high - launch_x) >= 4


def test_ledge_grounded_is_movement_type_not_pose() -> None:
    """Ledge hops end on movement type 0/1; a spin (type 3) is still airborne."""
    assert _ceres_elev_grounded(_state(movement_type=0, velocity_y=0))
    assert _ceres_elev_grounded(_state(movement_type=1, velocity_y=0))
    assert not _ceres_elev_grounded(_state(movement_type=3, velocity_y=2))
    assert not _ceres_elev_grounded(_state(movement_type=20, velocity_y=0))


def test_elev_to_landing_is_tas_wj_speed() -> None:
    """TAS target is 2246f. Do not edit the hops JSON. No checkpoint recover.

    MAX_FRAMES is a hang cap, not the pass. BENCH is the TAS target.
    """
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
    assert not hasattr(magnet, "CeresShaftClimb")
    assert not hasattr(magnet, "_ceres_checkpoint_shaft")
    assert not hasattr(magnet, "_ceres_seat_ledge")
    assert CERES_ELEV_BENCH_FRAMES == tas_frames
    # 3349f was the y571 checkpoint recover. Do not grow the hang cap to it.
    assert CERES_ELEV_MAX_FRAMES < 3349


# --- occupancy (editor JSON + ROM bytes, no emulator) ---

_SLOPE_TABLE_PC = 0xA0B2B  # unheadered LoROM $94:8B2B
_SHAPE1_HALF = bytes([16] * 8 + [0] * 8)
_ROM_SFC = INTEGRATION_DIR / "rom.sfc"


def _df45_editor() -> dict:
    root = editor_rooms_dir()
    if root is None:
        pytest.skip("snes_editor navigation export missing")
    path = Path(root) / "room_DF45.json"
    if not path.is_file():
        pytest.skip("room_DF45.json missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("roomIdHex") != "0xDF45":
        pytest.skip(f"unexpected room {payload.get('roomIdHex')}")
    return payload


def _allmap():
    from super_metroid.scratch.ceres_elev_wj import allmap as module

    try:
        module.occupancy()
    except SystemExit as exc:
        pytest.skip(str(exc))
    return module


def test_shape1_half_tile_is_not_full_solid() -> None:
    """BTS 0x01 is the right 8px; treating the tile as clip-8 would fail this."""
    from super_metroid.scratch.ceres_elev_wj.allmap import (
        CLIP_SLOPE,
        CLIP_SOLID,
        is_solid_px,
    )

    assert not is_solid_px(CLIP_SLOPE, 0x01, 7, 0)
    assert is_solid_px(CLIP_SLOPE, 0x01, 8, 0)
    assert is_solid_px(CLIP_SLOPE, 0x41, 7, 0)
    assert not is_solid_px(CLIP_SLOPE, 0x41, 8, 0)
    assert is_solid_px(CLIP_SOLID, 0, 7, 0)
    assert is_solid_px(CLIP_SOLID, 0, 8, 0)


def test_named_faces_are_half_tile_or_ledge_edges() -> None:
    """Live-pin faces. A full-solid col 13 makes x=215 solid and fails."""
    am = _allmap()
    assert am.is_solid(216, 530) and not am.is_solid(215, 530)
    assert am.is_solid(39, 530) and not am.is_solid(40, 530)
    assert am.is_solid(160, 390) and not am.is_solid(159, 390)
    assert am.is_solid(156, 496) and not am.is_solid(156, 495)
    assert am.is_solid(189, 384) and not am.is_solid(189, 383)
    assert am.is_solid(96, 500) and not am.is_solid(95, 500)
    assert am.is_solid(159, 500) and not am.is_solid(160, 500)
    assert am.wj_band(160, "right") == [147, 155]
    assert am.wj_band(96, "right") == [83, 91]
    assert am.wj_band(160, "left") == [165, 173]


def test_editor_clip_bts_at_named_tiles() -> None:
    payload = _df45_editor()
    coll, bts = payload["collision"], payload["bts"]
    assert (coll[33][13], bts[33][13]) == (1, 0x01)  # shaft right, (216, 530)
    assert (coll[33][2], bts[33][2]) == (1, 0x41)    # shaft left, (39, 530)
    assert (coll[24][10], bts[24][10]) == (8, 0x00)  # 363 left face (160, 390)
    assert (coll[24][9], bts[24][9]) == (0, 0x00)    # air west of 363 face
    assert (coll[31][9], bts[31][9]) == (8, 0x00)    # 475 floor (156, 496)
    assert (coll[24][11], bts[24][11]) == (8, 0x00)  # 363 floor (189, 384)
    assert (coll[31][6], bts[31][6]) == (8, 0x00)    # 475 box (96, 500)


@pytest.mark.skipif(not _ROM_SFC.is_file(), reason="rom.sfc missing")
def test_rom_shape1_slope_table_is_eight_solid_eight_air() -> None:
    blob = _ROM_SFC.read_bytes()
    assert blob[_SLOPE_TABLE_PC + 16 : _SLOPE_TABLE_PC + 32] == _SHAPE1_HALF
