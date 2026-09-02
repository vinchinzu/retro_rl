"""Ceres outbound room-frame table + morph graph prefix (no emulator)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import numpy as np

from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GS_ORDINARY, parse_state
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_FALLING_DOOR_JUMP_X,
    _CERES_FALLING_DOOR_SHUTTER_FRAMES,
    _CERES_FIRST_DOOR_FADE,
    _CERES_SCI_DOOR_Y,
)
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
    elev_to_landing_within_tas,
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
    CERES_FIRST_TAS_PAD,
    CeresFirstMoonfallTrack,
    CeresFlatEscape,
    ceres_first_moonfall_action,
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


def _last_room_visits() -> list[dict]:
    return [
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


_LAST_ROOM_TAS = [
    {"from": "0xDF8D", "to": "0xDF45", "frames": 281, "name": "falling_to_elev"},
    {"from": "0xDF45", "to": "0x91F8", "frames": 2246, "name": "elev_to_landing"},
]


def test_hops_vs_tas_slow_last_room_is_a_miss() -> None:
    """landing_frame synthesizes the hop. A +995 delta is not a TAS pass."""
    out = ceres_hops_vs_tas(
        _last_room_visits(),
        landing_frame=19697,
        first_control_frame=10860,
        tas_hops=_LAST_ROOM_TAS,
    )
    assert out["up_to_last_room"]["frames"] == 16456 - 10860
    assert out["last_room"]["frames"] == 19697 - 16456
    last = out["hops"][-1]
    assert last["name"] == "elev_to_landing"
    assert last["synthesized"] is True
    assert last["delta_frames"] == 995
    assert last["delta_frames"] > 0
    assert last["clock"] == TAS_CLOCK_SETTLED_GS8
    assert not elev_to_landing_within_tas(out["hops"])


def test_hops_vs_tas_last_room_reports_zero_delta_when_aligned() -> None:
    landing_frame = 16456 + 2246
    out = ceres_hops_vs_tas(
        _last_room_visits(),
        landing_frame=landing_frame,
        first_control_frame=10860,
        tas_hops=_LAST_ROOM_TAS,
    )
    last = out["hops"][-1]
    assert last["delta_frames"] == 0
    assert elev_to_landing_within_tas(out["hops"])
    assert not elev_to_landing_within_tas([])
    assert not elev_to_landing_within_tas(
        [{"name": "elev_to_landing", "delta_frames": None}]
    )


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


def test_falling_door_crouches_out_the_shutter() -> None:
    """$E23F is shut for the first frames; walking it is pose-138 knockback."""
    st = _falling_state(
        samus_x=50,
        samus_y=144,
        pose=10,
        momentum_x=2,
        invincibility_timer=36,
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="exit")
    )
    assert names == ("DOWN",)
    assert track.phase == "exit"


def test_falling_door_runs_left_once_the_shutter_is_up() -> None:
    st = _falling_state(
        samus_x=50,
        samus_y=139,
        pose=10,
        momentum_x=0,
        invincibility_timer=22,
    )
    names, track = ceres_falling_escape_action(
        st,
        CeresFallingEscapeTrack(
            phase="exit", held=_CERES_FALLING_DOOR_SHUTTER_FRAMES
        ),
    )
    assert "A" not in names
    assert "LEFT" in names


def test_falling_door_still_running_east_of_takeoff() -> None:
    st = _falling_state(
        samus_x=39,
        samus_y=139,
        pose=10,
        momentum_x=1,
        invincibility_timer=14,
    )
    names, track = ceres_falling_escape_action(
        st,
        CeresFallingEscapeTrack(
            phase="exit", held=_CERES_FALLING_DOOR_SHUTTER_FRAMES
        ),
    )
    assert "A" not in names
    assert "LEFT" in names


def test_falling_door_jumps_at_the_takeoff_x() -> None:
    """x=33 mx=1 leaves on the 4th air frame at y≈121 (elev 633)."""
    st = _falling_state(
        samus_x=_CERES_FALLING_DOOR_JUMP_X,
        samus_y=139,
        pose=10,
        momentum_x=1,
        invincibility_timer=11,
    )
    names, track = ceres_falling_escape_action(
        st,
        CeresFallingEscapeTrack(
            phase="exit", held=_CERES_FALLING_DOOR_SHUTTER_FRAMES
        ),
    )
    assert names == ("LEFT", "B", "A")


def test_falling_door_holds_a_through_the_ascent() -> None:
    """momentum_x halves on air frame 2, so height is the only lever left."""
    st = _falling_state(
        samus_x=23,
        samus_y=130,
        pose=26,
        momentum_x=1,
        invincibility_timer=8,
        vertical_direction=1,
        velocity_y=4,
    )
    names, track = ceres_falling_escape_action(
        st,
        CeresFallingEscapeTrack(
            phase="exit", held=_CERES_FALLING_DOOR_SHUTTER_FRAMES + 4
        ),
    )
    assert names == ("LEFT", "B", "A")


def test_falling_does_not_jump_standing() -> None:
    st = _falling_state(
        samus_x=37, samus_y=139, pose=1, momentum_x=0, speed_flag=0
    )
    names, track = ceres_falling_escape_action(
        st,
        CeresFallingEscapeTrack(
            phase="exit", held=_CERES_FALLING_DOOR_SHUTTER_FRAMES
        ),
    )
    assert "A" not in names
    assert track.phase == "exit"


def test_falling_does_not_jump_without_iframes() -> None:
    st = _falling_state(
        samus_x=_CERES_FALLING_DOOR_JUMP_X,
        samus_y=139,
        pose=10,
        momentum_x=2,
        invincibility_timer=0,
    )
    names, track = ceres_falling_escape_action(
        st,
        CeresFallingEscapeTrack(
            phase="exit", held=_CERES_FALLING_DOOR_SHUTTER_FRAMES
        ),
    )
    assert "A" not in names


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


def test_falling_low_elev_entry_is_not_done() -> None:
    """(216, 642) p25 mx=0 is the live miss. Do not hold A into y=651."""
    st = _falling_state(
        room_id=ROOM_CERES_ELEVATOR,
        samus_x=216,
        samus_y=642,
        pose=25,
        vertical_direction=1,
        velocity_y=4,
        momentum_x=0,
        speed_flag=0,
        invincibility_timer=0,
        game_state=GS_ORDINARY,
    )
    names, track = ceres_falling_escape_action(
        st, CeresFallingEscapeTrack(phase="exit")
    )
    assert names == ()
    assert track.phase != "done"


def _elev_floor(**overrides):
    values = {
        "room_id": ROOM_CERES_ELEVATOR,
        "game_state": GS_ORDINARY,
        "samus_x": 205,
        "samus_y": 651,
        "pose": 17,
        "facing": FACING_RIGHT,
        "movement_type": 1,
        "momentum_x": 2,
        "speed_flag": 1,
        "samus_x_sub": 45056,
    }
    values.update(overrides)
    return replace(parse_state(np.zeros(0x2000, dtype=np.uint8), frame=0), **values)


def test_ceres_first_pad_ends_with_tas_8787_l() -> None:
    assert len(CERES_FIRST_TAS_PAD) == 150
    assert CERES_FIRST_TAS_PAD[141] == ("B", "RIGHT", "L")
    assert CERES_FIRST_TAS_PAD[142] == ("B", "RIGHT", "L")
    assert CERES_FIRST_TAS_PAD[143] == ("B", "RIGHT")
    assert CERES_FIRST_TAS_PAD[148] == ("B", "RIGHT", "L")
    assert CERES_FIRST_TAS_PAD[149] == ("B", "RIGHT")


def test_ceres_first_keeps_l_at_203() -> None:
    st = _elev_floor(samus_x=203, samus_x_sub=4096)
    names, track = ceres_first_moonfall_action(
        st, CeresFirstMoonfallTrack("fall", held=141)
    )
    assert names == ("B", "RIGHT", "L")
    assert track.invert_l is False
    assert track.held == 142


def test_ceres_first_skips_extra_l_at_pose_17_x205() -> None:
    """TAS 8782 is B+RIGHT at (206, p17). Extra L inverts the 228 pulse."""
    st = _elev_floor(samus_x=205)
    names, track = ceres_first_moonfall_action(
        st, CeresFirstMoonfallTrack("fall", held=142)
    )
    assert names == ("B", "RIGHT")
    assert "L" not in names
    assert track.invert_l is True
    assert track.held == 144
    later = replace(st, samus_x=209)
    names, track = ceres_first_moonfall_action(later, track)
    assert names == ("B", "RIGHT", "L")
    assert track.held == 145


def test_ceres_first_l_at_228_after_skip() -> None:
    st = _elev_floor(samus_x=228, samus_x_sub=8192)
    names, track = ceres_first_moonfall_action(
        st, CeresFirstMoonfallTrack("fall", held=148, invert_l=True)
    )
    assert names == ("B", "RIGHT", "L")
    assert track.held == 149
    at_233 = replace(st, samus_x=233, pose=9, samus_x_sub=0)
    names, track = ceres_first_moonfall_action(at_233, track)
    assert names == ("B", "RIGHT")
    assert "R" not in names
    assert "L" not in names


def test_ceres_first_last_floor_l_on_pose_17() -> None:
    st = _elev_floor(samus_x=234, pose=17, samus_x_sub=0)
    names, track = ceres_first_moonfall_action(
        st, CeresFirstMoonfallTrack("fall", held=149, invert_l=True)
    )
    assert names == ("B", "RIGHT", "L")
    assert "R" not in names


def test_ceres_first_does_not_l_every_p17_after_214() -> None:
    st = _elev_floor(samus_x=218)
    names, track = ceres_first_moonfall_action(
        st, CeresFirstMoonfallTrack("fall", held=145, invert_l=True)
    )
    assert names == ("B", "RIGHT")
    assert "L" not in names


def test_ceres_first_fade_b_right_then_idle_then_last_r() -> None:
    door = _elev_floor(game_state=9, samus_x=237, pose=17)
    names, track = ceres_first_moonfall_action(
        door, CeresFirstMoonfallTrack("land", held=4)
    )
    assert names == ("RIGHT", "B")
    assert track.phase == "exit"
    fade = replace(
        door,
        game_state=11,
        room_id=ROOM_CERES_FALLING,
        samus_x=39,
        samus_y=139,
    )
    names, track = ceres_first_moonfall_action(
        fade, CeresFirstMoonfallTrack("exit", held=1)
    )
    assert names == ()
    names, track = ceres_first_moonfall_action(
        fade, CeresFirstMoonfallTrack("exit", held=_CERES_FIRST_DOOR_FADE - 1)
    )
    assert names == ("RIGHT", "B", "R")
