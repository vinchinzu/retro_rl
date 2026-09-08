"""Pin the L8 ``RoomHopSpec`` rows: geometry is row data, not code.

Phase 2.3 folded eight near-identical L8 interior gate controllers into rows
of ``zelda_i.dungeon.door_hop.RoomHopController``.  Each row's origin, door,
budget, fail rooms and reason strings were measured live; this table is the
regression oracle for that conversion.  Behaviour of the shared engine itself
is covered by the per-room tests (west/south/east/stairs/cellar/passage/
triforce); this file only asserts that nothing drifted in the rows.
"""

from __future__ import annotations

import pytest

from zelda_i.dungeon.door_hop import RoomHopController, RoomHopSpec
from zelda_i.level8.cellar import CELLAR_RETURN_GATE, CELLAR_RETURN_MAX_FRAMES
from zelda_i.level8.passage import CELLAR_CROSS_MAX_FRAMES, PASSAGE_2F_GATE
from zelda_i.level8.path import (
    EAST_3E_GATE,
    LEVEL8_PATH_GATES,
    SOUTH_1E_GATE,
    SOUTH_2E_GATE,
    WEST_1F_GATE,
    make_east_3e_controller,
    make_south_1e_controller,
    make_south_2e_controller,
    make_west_1f_controller,
)
from zelda_i.level8.stairs import STAIRS_3F_GATE, make_stairs_3f_controller
from zelda_i.level8.triforce import (
    NORTH_3C_GATE,
    NORTH_3C_MAX_FRAMES,
    RAM_CLAIM,
    TF_ROOM_HYP,
    make_north_3c_controller,
)
from zelda_i.level8.cellar import make_magic_key_cellar_return_controller
from zelda_i.level8.passage import make_passage_2f_controller

ALL_GATES: tuple[RoomHopSpec, ...] = (
    WEST_1F_GATE,
    SOUTH_1E_GATE,
    SOUTH_2E_GATE,
    EAST_3E_GATE,
    STAIRS_3F_GATE,
    NORTH_3C_GATE,
    CELLAR_RETURN_GATE,
    PASSAGE_2F_GATE,
)

# (spec_id, origin, door, done_reason, max_frames)
EXPECTED = (
    ("level8_west_1f", 0x1F, "LEFT", "left_0x1f_west", 4000),
    ("level8_south_1e", 0x1E, "DOWN", "left_0x1e_south", 4000),
    ("level8_south_2e", 0x2E, "DOWN", "left_0x2e_south", 4000),
    ("level8_east_3e", 0x3E, "RIGHT", "left_0x3e_east", 4000),
    ("level8_stairs_3f", 0x3F, "STAIRS", "left_0x3f_stairs", 4000),
    ("level8_north_3c", 0x3C, "UP", "left_0x3c_north", NORTH_3C_MAX_FRAMES),
    (
        "level8_magic_key_cellar_return",
        0x0F,
        "STAIRS",
        "left_0x0f_stairs",
        CELLAR_RETURN_MAX_FRAMES,
    ),
    ("level8_passage_2f", 0x2F, "STAIRS", "left_0x2f_stairs", CELLAR_CROSS_MAX_FRAMES),
)


def test_gate_rows_are_the_measured_table() -> None:
    """Row identity: origin/door/budget per gate, in suffix order."""
    got = tuple(
        (g.spec_id, g.origin, g.door, g.done_reason, g.max_frames) for g in ALL_GATES
    )
    assert got == EXPECTED


@pytest.mark.parametrize("gate", ALL_GATES, ids=[g.spec_id for g in ALL_GATES])
def test_every_gate_is_level8_and_samples_every_12_frames(gate: RoomHopSpec) -> None:
    assert gate.level == 8
    assert gate.sample_period == 12
    assert (gate.step is None) != (gate.policy_fn is None)


def test_path_gate_table_is_the_four_clone_rooms() -> None:
    assert LEVEL8_PATH_GATES == (
        WEST_1F_GATE,
        SOUTH_1E_GATE,
        SOUTH_2E_GATE,
        EAST_3E_GATE,
    )
    for gate in LEVEL8_PATH_GATES:
        # Cellar 0x0F (also mode-9) and Gleeok 0x3C are the two fail rooms.
        assert [(f.rooms, f.note, f.on_passage) for f in gate.fails] == [
            ((0x0F,), "cellar_0x{screen:02x}", True),
            ((0x3C,), "gleeok_0x3c", False),
        ]
        assert gate.scroll_button == gate.door
        assert gate.settle_button == gate.door
        assert gate.blocked_rooms == frozenset({0x0F, 0x3C})


def test_stairs_gate_accepts_a_mode9_arrival() -> None:
    """The 0x3F walk-on lands in mode-9; play-only arrival would miss it."""
    assert STAIRS_3F_GATE.require_play_arrival is False
    assert STAIRS_3F_GATE.passage_arrival is True
    assert STAIRS_3F_GATE.arrive_note == "m{mode}_0x{screen:02x}_{x}_{y}"
    assert STAIRS_3F_GATE.unexpected_note == "unexpected_0x{screen:02x}"
    assert STAIRS_3F_GATE.unexpected_play_only is False
    assert STAIRS_3F_GATE.passage_hold_reason == "cellar_hold"
    assert STAIRS_3F_GATE.settle_reason == "dest_settle"
    assert STAIRS_3F_GATE.scroll_button is None


def test_north_3c_gate_fails_south_and_both_cellars() -> None:
    assert [(f.rooms, f.note, f.on_passage) for f in NORTH_3C_GATE.fails] == [
        ((0x0F, 0x2F), "cellar_0x{screen:02x}", True),
        ((0x4C,), "south_0x4c", False),
    ]
    assert dict(NORTH_3C_GATE.report_extra) == {
        "tf_room_hyp": TF_ROOM_HYP,
        "assumed_0x2c": False,
        "policy": RAM_CLAIM,
    }


def test_cellar_cross_rows_keep_their_novel_policies() -> None:
    """0x0F / 0x2F walk in mode-9; their policy stays a row callable."""
    assert CELLAR_RETURN_GATE.policy_fn is not None
    assert CELLAR_RETURN_GATE.arrive_any is True
    assert CELLAR_RETURN_GATE.fails == ()
    assert CELLAR_RETURN_GATE.unexpected_note == ""
    assert PASSAGE_2F_GATE.policy_fn is not None
    assert [(f.rooms, f.note, f.play_only) for f in PASSAGE_2F_GATE.fails] == [
        ((0x3C,), "gleeok_0x3c", False),
        ((0x3F,), "returned_source_0x3f", True),
    ]


FACTORIES = (
    (make_west_1f_controller, WEST_1F_GATE, 0x1E),
    (make_south_1e_controller, SOUTH_1E_GATE, 0x2E),
    (make_south_2e_controller, SOUTH_2E_GATE, 0x3E),
    (make_east_3e_controller, EAST_3E_GATE, 0x3F),
    (make_stairs_3f_controller, STAIRS_3F_GATE, 0x2F),
    (make_north_3c_controller, NORTH_3C_GATE, 0x2C),
    (make_magic_key_cellar_return_controller, CELLAR_RETURN_GATE, 0x1F),
    (make_passage_2f_controller, PASSAGE_2F_GATE, 0x4C),
)


@pytest.mark.parametrize(
    "factory,gate,dest", FACTORIES, ids=[g.spec_id for _, g, _ in FACTORIES]
)
def test_factory_binds_its_row_and_measured_dest(factory, gate, dest) -> None:
    ctl = factory()
    assert isinstance(ctl, RoomHopController)
    assert ctl.spec is gate
    assert ctl.dest == dest
    assert ctl.spec_id == gate.spec_id
    assert ctl.stage_id == gate.spec_id
    assert ctl.max_frames == gate.max_frames
    assert ctl.require_level == 8
    assert ctl.done_reason == gate.done_reason
    assert ctl.report()["door"] == gate.door
    assert ctl.report()["route_eligible"] is False
    assert ctl.report()["writes"] == 0
