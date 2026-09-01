"""Ceres outbound room-frame table + morph graph prefix (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.ram import FACING_RIGHT, GS_ORDINARY, parse_state
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
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_SCIENTIST,
)
from super_metroid.routes.kpdr.ceres.outbound import (
    CeresFlatEscape,
    play_ceres_flat_to_scientist,
)
from super_metroid.takeoff import shoulder_pump_button
from unittest.mock import Mock


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


def test_flat_escape_arm_pump_alternates() -> None:
    cross = CeresFlatEscape()
    st = _flat_state(samus_x=300)
    first = cross.action(st)
    second = cross.action(st)
    assert first == ("LEFT", "B", "L")
    assert second == ("LEFT", "B", "L")
    third = cross.action(st)
    fourth = cross.action(st)
    assert third == ("LEFT", "B", "R")
    assert fourth == ("LEFT", "B", "R")


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
