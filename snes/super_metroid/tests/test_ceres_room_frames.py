"""Ceres outbound room-frame table + morph graph prefix (no emulator)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import numpy as np

from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GS_ORDINARY, parse_state
from super_metroid.routes.kpdr.ceres.geometry import _CERES_SCI_DOOR_Y
from super_metroid.routes.kpdr.ceres.scientist import (
    CeresScientistCross,
    scientist_on_entry_ledge,
)
from super_metroid.routes.kpdr.ceres.spine import (
    CERES_DOOR_EDGES,
    CERES_MILESTONES,
    CERES_SCIENTIST_MAX_FRAMES,
    DEFAULT_TAS_CLOCK,
    TAS_CLOCK_ROOM_FLIP,
    TAS_CLOCK_SETTLED_GS8,
    ceres_hops_vs_tas,
    load_tas_ceres_hops,
    tas_hop_clock,
)
from super_metroid.routes.kpdr.early_spine import MORPH_DOOR_EDGES, MORPH_MILESTONES
from super_metroid.routes.kpdr.ceres.magnet import (
    CeresFallingEscapeTrack,
    CeresMagnetEscapeTrack,
    ceres_falling_escape_action,
    ceres_magnet_escape_action,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_SCIENTIST,
)
from super_metroid.routes.kpdr.ceres.outbound import (
    CeresFlatEscape,
    play_ceres_flat_to_scientist,
)
from super_metroid.takeoff import shoulder_pump_button


def test_scientist_lip_runs_without_jump() -> None:
    st = replace(
        parse_state(np.zeros(0x2000, dtype=np.uint8), frame=0),
        room_id=ROOM_CERES_SCIENTIST,
        game_state=GS_ORDINARY,
        samus_x=39,
        samus_y=_CERES_SCI_DOOR_Y,
        pose=17,
        facing=FACING_RIGHT,
        momentum_x=2,
        speed_flag=1,
        samus_x_sub=100,
    )
    assert scientist_on_entry_ledge(st)
    act = CeresScientistCross().action(st)
    assert "RIGHT" in act
    assert "B" in act
    assert "A" not in act


def test_scientist_max_frames() -> None:
    assert CERES_SCIENTIST_MAX_FRAMES == 400


def test_morph_door_edges_prefix_is_ceres() -> None:
    n = len(CERES_DOOR_EDGES)
    assert MORPH_DOOR_EDGES[:n] == CERES_DOOR_EDGES
    assert all(left is right for left, right in zip(MORPH_DOOR_EDGES, CERES_DOOR_EDGES))


def test_morph_milestones_prefix_is_ceres() -> None:
    n = len(CERES_MILESTONES)
    assert MORPH_MILESTONES[:n] == CERES_MILESTONES
    assert all(left is right for left, right in zip(MORPH_MILESTONES, CERES_MILESTONES))


def test_tas_ceres_hops_table_has_last_room() -> None:
    hops = load_tas_ceres_hops()
    assert hops[-1]["name"] == "elev_to_landing"
    assert hops[-1]["from"] == "0xDF45"
    assert hops[-1]["to"] == "0x91F8"
    assert hops[-1]["frames"] == 2246


def test_tas_ceres_hops_expose_both_clocks() -> None:
    hops = {h["name"]: h for h in load_tas_ceres_hops()}
    magnet = hops["magnet_to_scientist"]
    assert magnet["frames"] == 344
    flip = tas_hop_clock(magnet, TAS_CLOCK_ROOM_FLIP)
    gs8 = tas_hop_clock(magnet, TAS_CLOCK_SETTLED_GS8)
    assert flip["frames"] == 344
    assert flip["source_enter"] == 9114
    assert gs8["frames"] == 345
    assert gs8["source_gs8"] == 9232
    assert gs8["dest_gs8"] == 9577
    assert gs8["dwell_frames"] == 183
    assert gs8["transition_frames"] == 162
    assert DEFAULT_TAS_CLOCK == TAS_CLOCK_SETTLED_GS8
    elev = tas_hop_clock(hops["elev_to_falling"], TAS_CLOCK_SETTLED_GS8)
    assert elev["source_gs8"] == 8639
    assert elev["frames"] == 311
    missing = tas_hop_clock({"from": "0xE0B5", "to": "0xE06B", "frames": 1842})
    assert missing["frames"] == 1842
    assert missing["fallback"] == "room_flip_frames"
    rev = tas_hop_clock(hops["flat_escape"], TAS_CLOCK_SETTLED_GS8)
    assert rev["frames"] == 259
    assert rev["dwell_frames"] == 97
    assert rev["transition_frames"] == 162
    sci_rev = tas_hop_clock(hops["scientist_escape"], TAS_CLOCK_SETTLED_GS8)
    assert sci_rev["frames"] == 263
    assert sci_rev["dwell_frames"] == 101


def test_hops_vs_tas_measures_up_to_last_room() -> None:
    tas = [
        {"from": "0xDF8D", "to": "0xDF45", "frames": 281, "name": "falling_to_elev"},
        {"from": "0xDF45", "to": "0x91F8", "frames": 2246, "name": "elev_to_landing"},
    ]
    visits = [
        {
            "room_id": 0xDF8D,
            "dest_room_id": 0xDF45,
            "room_id_hex": "0xDF8D",
            "dest_room_id_hex": "0xDF45",
            "room_frames": 442,
            "entry_frame": 16014,
            "exit_frame": 16456,
        }
    ]
    out = ceres_hops_vs_tas(
        visits,
        landing_frame=19697,
        first_control_frame=10860,
        tas_hops=tas,
    )
    assert out["up_to_last_room"]["frames"] == 16456 - 10860
    assert out["last_room"]["frames"] == 19697 - 16456
    last = out["hops"][-1]
    assert last["name"] == "elev_to_landing"
    assert last["synthesized"] is True
    assert last["delta_frames"] == (19697 - 16456) - 2246
    assert last["clock"] == TAS_CLOCK_SETTLED_GS8


def test_hops_vs_tas_labels_room_flip_mismatch() -> None:
    hops = load_tas_ceres_hops()
    visits = [
        {
            "room_id": 0xDFD7,
            "dest_room_id": 0xE021,
            "room_id_hex": "0xDFD7",
            "dest_room_id_hex": "0xE021",
            "room_frames": 390,
            "dwell_frames": 228,
            "transition_frames": 162,
            "entry_frame": 0,
            "exit_frame": 390,
        }
    ]
    gs8 = ceres_hops_vs_tas(visits, tas_hops=hops)
    row = gs8["hops"][0]
    assert gs8["clock"] == TAS_CLOCK_SETTLED_GS8
    assert row["tas_frames"] == 345
    assert row["delta_frames"] == 45
    assert row["delta_dwell_frames"] == 45
    assert row["delta_transition_frames"] == 0
    assert "clock_mismatch" not in row
    flip = ceres_hops_vs_tas(visits, tas_hops=hops, clock=TAS_CLOCK_ROOM_FLIP)
    assert flip["hops"][0]["tas_frames"] == 344
    assert flip["hops"][0]["clock_mismatch"] is True


def _flat_state(**overrides):
    base = parse_state(np.zeros(0x2000, dtype=np.uint8), frame=0)
    values = {
        "room_id": ROOM_CERES_FLAT,
        "game_state": GS_ORDINARY,
        "samus_x": 472,
        "samus_y": 139,
        "pose": 18,
        "facing": FACING_RIGHT,
        "momentum_x": 2,
        "speed_flag": 1,
        "samus_x_sub": 100,
    }
    values.update(overrides)
    return replace(base, **values)


def test_flat_escape_never_jumps() -> None:
    st = _flat_state()
    act = CeresFlatEscape().action(st)
    assert act == ("LEFT", "B", shoulder_pump_button(0))
    assert "A" not in act


def test_flat_escape_door_settle_does_not_jump() -> None:
    st = _flat_state(game_state=11)
    act = CeresFlatEscape().action(st)
    assert act == ("LEFT",)
    assert "A" not in act


def test_flat_escape_is_noop_when_already_in_scientist() -> None:
    session = Mock()
    session.state = _flat_state(
        room_id=ROOM_CERES_SCIENTIST, samus_x=472, game_state=GS_ORDINARY
    )
    play_ceres_flat_to_scientist(session)
    session.step.assert_not_called()


def test_flat_escape_is_noop_when_already_in_magnet() -> None:
    session = Mock()
    session.state = _flat_state(room_id=ROOM_CERES_MAGNET, samus_x=216)
    play_ceres_flat_to_scientist(session)
    session.step.assert_not_called()


def test_flat_door_transition_is_not_done() -> None:
    from super_metroid.routes.kpdr.ceres.outbound import _flat_escape_past

    st = _flat_state(room_id=ROOM_CERES_SCIENTIST, game_state=11, samus_x=20)
    assert not _flat_escape_past(st)


def _magnet_state(**overrides):
    base = parse_state(np.zeros(0x2000, dtype=np.uint8), frame=0)
    values = {
        "room_id": ROOM_CERES_MAGNET,
        "game_state": GS_ORDINARY,
        "samus_x": 216,
        "samus_y": 395,
        "pose": 18,
        "facing": FACING_LEFT,
        "momentum_x": 2,
        "speed_flag": 1,
        "samus_x_sub": 100,
        "knockback_timer": 0,
        "invincibility_timer": 0,
        "vertical_direction": 0,
        "velocity_y": 0,
    }
    values.update(overrides)
    return replace(base, **values)


def test_magnet_escape_slope_taps_l_not_r() -> None:
    seat = _magnet_state(samus_x=120, samus_y=347)
    names, track = ceres_magnet_escape_action(
        seat, CeresMagnetEscapeTrack(phase="slope")
    )
    assert names == ("LEFT", "B")
    assert "R" not in names
    assert "A" not in names
    assert track.phase == "slope"
    still, track = ceres_magnet_escape_action(seat, track)
    assert still == ("LEFT", "B", "L")
    assert "R" not in still


def test_magnet_escape_does_not_force_pump_while_stopped() -> None:
    st = _magnet_state(samus_x=120, samus_y=347, momentum_x=0, speed_flag=0)
    names, track = ceres_magnet_escape_action(st, CeresMagnetEscapeTrack(phase="slope"))
    assert names == ("LEFT", "B")
    assert "L" not in names
    assert track.phase == "slope"


def test_magnet_escape_jumps_347_tas_window() -> None:
    st = _magnet_state(samus_x=74, samus_y=347)
    names, track = ceres_magnet_escape_action(st, CeresMagnetEscapeTrack(phase="slope"))
    assert names == ("LEFT", "B", "A")
    assert track.phase == "shelf_hop"


def test_magnet_escape_does_not_jump_under_shelf() -> None:
    st = _magnet_state(samus_x=88, samus_y=347)
    names, track = ceres_magnet_escape_action(st, CeresMagnetEscapeTrack(phase="slope"))
    assert "A" not in names
    assert track.phase == "slope"


def test_magnet_escape_west_of_window_still_jumps() -> None:
    st = _magnet_state(samus_x=50, samus_y=347)
    names, track = ceres_magnet_escape_action(st, CeresMagnetEscapeTrack(phase="slope"))
    assert "A" in names
    assert track.phase == "shelf_hop"


def test_magnet_escape_267_jumps_right_into_steam() -> None:
    st = _magnet_state(
        samus_x=118, samus_y=267, facing=FACING_RIGHT, pose=9
    )
    names, track = ceres_magnet_escape_action(
        st, CeresMagnetEscapeTrack(phase="shelf")
    )
    assert names == ("RIGHT", "B", "A")
    assert track.phase == "steam_hop"


def test_magnet_escape_267_waits_for_hidden_jet() -> None:
    st = _magnet_state(
        samus_x=118, samus_y=267, facing=FACING_RIGHT, pose=9
    )
    names, track = ceres_magnet_escape_action(
        st, CeresMagnetEscapeTrack(phase="shelf", steam_shown=False)
    )
    assert "A" not in names
    assert track.phase == "shelf"


def test_magnet_escape_hops_139_magnet_stop() -> None:
    st = _magnet_state(samus_x=45, samus_y=139, pose=138)
    names, track = ceres_magnet_escape_action(
        st, CeresMagnetEscapeTrack(phase="exit")
    )
    assert "A" in names
    assert track.phase == "exit"


def test_magnet_escape_holds_left_through_fade() -> None:
    st = _magnet_state(game_state=11, samus_x=20, samus_y=139)
    names, track = ceres_magnet_escape_action(st, CeresMagnetEscapeTrack(phase="exit"))
    assert names == ("LEFT",)
    assert "A" not in names


def test_magnet_escape_done_in_falling() -> None:
    st = _magnet_state(
        room_id=ROOM_CERES_FALLING, samus_x=472, samus_y=139, pose=18
    )
    names, track = ceres_magnet_escape_action(st, CeresMagnetEscapeTrack(phase="exit"))
    assert names == ()
    assert track.phase == "done"


def _falling_state(**overrides):
    values = {
        "room_id": ROOM_CERES_FALLING,
        "samus_x": 472,
        "samus_y": 139,
        "pose": 18,
    }
    values.update(overrides)
    return _magnet_state(**values)


def test_falling_hops_187_onto_171() -> None:
    st = _falling_state(samus_x=347, samus_y=187, momentum_x=2, speed_flag=1)
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="run_off")
    )
    assert "A" in names
    assert track.phase == "floor_hop"


def test_falling_keeps_left_until_tas_tile() -> None:
    st = _falling_state(
        samus_x=320, samus_y=171, pose=10, momentum_x=2, speed_flag=1
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="shelf")
    )
    assert "LEFT" in names
    assert "A" not in names
    assert track.phase == "shelf"


def test_falling_turns_right_before_tile() -> None:
    st = _falling_state(
        samus_x=311, samus_y=171, pose=10, momentum_x=2, speed_flag=1
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="shelf")
    )
    assert names == ("RIGHT",)
    assert "A" not in names
    assert track.phase == "dboost"


def test_falling_does_not_right_on_tile_contact() -> None:
    st = _falling_state(
        samus_x=294, samus_y=171, pose=10, momentum_x=2, speed_flag=1
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="shelf")
    )
    assert "LEFT" in names
    assert "RIGHT" not in names
    assert "A" in names
    assert track.phase == "dboost"


def test_falling_dboost_does_not_right_on_steam() -> None:
    """RIGHT on the 294 contact frame while facing left is p84."""
    st = _falling_state(
        samus_x=294,
        samus_y=171,
        pose=10,
        facing=FACING_LEFT,
        momentum_x=2,
        speed_flag=1,
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="dboost", held=3)
    )
    assert "RIGHT" not in names
    assert "X" not in names
    assert track.phase == "dboost"


def test_falling_dboost_falls_into_jet_facing_right() -> None:
    """Stay facing right into the 294 jet. No X (p47 on this pin)."""
    st = _falling_state(
        samus_x=308,
        samus_y=171,
        pose=25,
        facing=FACING_RIGHT,
        momentum_x=2,
        speed_flag=1,
        vertical_direction=0,
        velocity_y=0,
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="dboost", held=3)
    )
    assert names == ("B", "A")
    assert "X" not in names
    assert "RIGHT" not in names
    assert track.phase == "dboost"


def test_falling_dboost_holds_left_on_pose_80() -> None:
    st = _falling_state(
        samus_x=220,
        samus_y=157,
        pose=80,
        vertical_direction=1,
        velocity_y=4,
        movement_type=25,
        knockback_timer=0,
        invincibility_timer=90,
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="dboost", contacted=True, held=8)
    )
    assert names == ("LEFT", "B", "A")
    assert track.boosted is True


def test_falling_door_jumps_for_pose_25() -> None:
    st = _falling_state(samus_x=37, samus_y=139, pose=10, momentum_x=2)
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="exit")
    )
    assert "A" in names
    assert "LEFT" in names


def test_falling_done_in_elev() -> None:
    st = _falling_state(
        room_id=ROOM_CERES_ELEVATOR,
        samus_x=216,
        samus_y=632,
        pose=25,
        vertical_direction=1,
        velocity_y=4,
        invincibility_timer=36,
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="exit")
    )
    assert names == ()
    assert track.phase == "done"
