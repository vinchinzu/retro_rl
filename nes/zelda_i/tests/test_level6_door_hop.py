"""Parametrized L6 DoorHopSpec table: dest RAM, two occupancy smokes."""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.door_hop import (
    DoorHopController,
    DoorHopSpec,
    door_band_goal,
    door_hop_success,
)
from zelda_i.level6.door_hop import (
    EAST39_SPEC,
    L6_DOOR_HOPS,
    NORTH2C_SPEC,
    SOUTH18_SPEC,
    SOUTH1D_SPEC,
    WEST19_SPEC,
    WEST2D_SPEC,
)
from zelda_i.level6.occupancy import l6_play_dest_success, record_l6_walk
from zelda_i.level6.overworld import LEVEL6
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

DEST_SPECS = (
    WEST19_SPEC,
    SOUTH18_SPEC,
    SOUTH1D_SPEC,
    WEST2D_SPEC,
    NORTH2C_SPEC,
)
WRONG_NEIGHBOR = 0x3A


def _ids(spec: DoorHopSpec) -> str:
    return spec.spec_id


_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 6,
    "x": 120,
    "y": 141,
    "triforce": 0x1F,
    "rod": 1,
}


def _ram(*, screen: int, **fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, screen=screen, **fields)


def _snap(*, screen: int, **fields: int):
    return read_snapshot(_ram(screen=screen, **fields))


@pytest.mark.parametrize("spec", DEST_SPECS, ids=_ids)
def test_door_hop_dest_room_success(spec: DoorHopSpec) -> None:
    dest_room = spec.dest_room
    assert dest_room is not None
    dest = _snap(
        screen=dest_room, mode=PLAY_MODE, level=6, triforce=0x1F, rod=1,
    )
    assert door_hop_success(spec, dest)
    still = _snap(
        screen=spec.room, mode=PLAY_MODE, level=6, triforce=0x1F, rod=1,
    )
    assert not door_hop_success(spec, still)


@pytest.mark.parametrize("spec", DEST_SPECS, ids=_ids)
def test_door_hop_wrong_neighbor_fails(spec: DoorHopSpec) -> None:
    neighbor = (
        spec.fail_backtrack if spec.fail_backtrack is not None else WRONG_NEIGHBOR
    )
    snap = _snap(
        screen=neighbor, mode=PLAY_MODE, level=6, triforce=0x1F, rod=1,
    )
    assert not door_hop_success(spec, snap)
    assert neighbor != spec.dest_room
    assert neighbor != spec.room


def test_south1d_leftover_not_up_then_down_at_goal() -> None:
    leftover = _snap(screen=SOUTH1D_SPEC.room, x=96, y=157)
    first = DoorHopController(SOUTH1D_SPEC).step(leftover)
    assert list(first.action) != list(nes_action("UP"))
    gx, gy = SOUTH1D_SPEC.goal
    hold = DoorHopController(SOUTH1D_SPEC).step(
        _snap(screen=SOUTH1D_SPEC.room, x=gx, y=gy)
    )
    assert (gx, gy) == (120, 189)
    assert list(hold.action) == list(nes_action("DOWN"))


def test_west2d_align_y_then_left() -> None:
    """North leftover holds DOWN; waist LEFT; SW pocket (32,189) holds UP."""
    leftover = _snap(screen=WEST2D_SPEC.room, x=120, y=77)
    first = DoorHopController(WEST2D_SPEC).step(leftover)
    assert list(first.action) == list(nes_action("DOWN"))
    assert list(first.action) != list(nes_action("LEFT"))
    west = DoorHopController(WEST2D_SPEC).step(
        _snap(screen=WEST2D_SPEC.room, x=120, y=141)
    )
    assert list(west.action) == list(nes_action("LEFT"))
    door = DoorHopController(WEST2D_SPEC).step(
        _snap(screen=WEST2D_SPEC.room, x=32, y=141)
    )
    assert list(door.action) == list(nes_action("LEFT"))
    # Occupancy boxed here on the power-on tape; cardinal UP re-acquires y=141.
    pocket = DoorHopController(WEST2D_SPEC).step(
        _snap(screen=WEST2D_SPEC.room, x=32, y=189)
    )
    assert list(pocket.action) == list(nes_action("UP"))
    assert WEST2D_SPEC.cardinal_hold is True
    assert WEST2D_SPEC.align == "y"


def test_north2c_align_x_then_up() -> None:
    """East leftover holds LEFT; column UP; waist leftover (71,141) holds RIGHT."""
    leftover = _snap(screen=NORTH2C_SPEC.room, x=224, y=141)
    first = DoorHopController(NORTH2C_SPEC).step(leftover)
    assert list(first.action) == list(nes_action("LEFT"))
    assert list(first.action) != list(nes_action("UP"))
    column = DoorHopController(NORTH2C_SPEC).step(
        _snap(screen=NORTH2C_SPEC.room, x=120, y=141)
    )
    assert list(column.action) == list(nes_action("UP"))
    door = DoorHopController(NORTH2C_SPEC).step(
        _snap(screen=NORTH2C_SPEC.room, x=120, y=93)
    )
    assert list(door.action) == list(nes_action("UP"))
    # Occupancy south_open_halt boxed here on the power-on tape.
    shuffled = DoorHopController(NORTH2C_SPEC).step(
        _snap(screen=NORTH2C_SPEC.room, x=71, y=141)
    )
    assert list(shuffled.action) == list(nes_action("RIGHT"))
    assert NORTH2C_SPEC.cardinal_hold is True
    assert NORTH2C_SPEC.align == "x"


def test_east39_north_band_leftover_drops_to_waist_then_right() -> None:
    """Power-on leftover (95,109) holds DOWN to y=141, not RIGHT into the wall."""
    assert EAST39_SPEC.clip_buttons == ("DOWN",)
    leftover = _snap(screen=EAST39_SPEC.room, x=95, y=109)
    first = DoorHopController(EAST39_SPEC).step(leftover)
    assert list(first.action) == list(nes_action("DOWN"))
    assert list(first.action) != list(nes_action("RIGHT"))
    # Once on the waist the cardinal hold carries RIGHT toward the door.
    waist = DoorHopController(EAST39_SPEC).step(
        _snap(screen=EAST39_SPEC.room, x=120, y=141)
    )
    assert list(waist.action) == list(nes_action("RIGHT"))
    door = DoorHopController(EAST39_SPEC).step(
        _snap(screen=EAST39_SPEC.room, x=208, y=141)
    )
    assert list(door.action) == list(nes_action("RIGHT"))


@pytest.mark.parametrize("spec", L6_DOOR_HOPS, ids=_ids)
def test_every_row_binds_the_l6_occupancy_engine(spec: DoorHopSpec) -> None:
    """The engine is shared; each L6 row injects L6's level and predicates.

    Losing an injection would silently grade a hop against the generic
    play-dest rule (no rod / TF 0x1F check) and stop failing on an OW leave.
    """
    assert spec.level == LEVEL6
    assert spec.success_fn is l6_play_dest_success
    assert spec.record_fn is record_l6_walk
    # Shared band defaults still match the measured L6 door geometry.
    assert (spec.south_band_y, spec.north_band_y) == (181, 109)


def test_up_hop_off_column_leftover_binds_door_column() -> None:
    """Off-column leftover uses the door column, not leftover x (UP into a wall at x=208)."""
    assert door_band_goal("UP", (208, 157), (120, 109)) == (120, 109)
    assert door_band_goal("UP", (118, 157), (120, 109)) == (118, 109)
    assert door_band_goal("DOWN", (96, 157), (120, 189)) == (120, 181)
    assert door_band_goal("RIGHT", (120, 143), (208, 141)) == (208, 143)
    assert door_band_goal("LEFT", (120, 77), (32, 141)) == (32, 141)
    spec = DoorHopSpec(
        spec_id="up_leftover_door_column",
        room=0x2C,
        goal=(120, 109),
        hold_dir="UP",
        policy="leftover-relative north mouth",
    )
    leftover = _snap(screen=spec.room, x=208, y=157)
    ctl = DoorHopController(spec)
    first = ctl.step(leftover)
    assert ctl.goal == (120, 109)
    dest = ctl._path_dest((208, 157))
    assert dest[0] == 120
    assert dest != (208, 157)
    assert list(first.action) != list(nes_idle_action())
    assert list(first.action) == list(nes_action("LEFT"))
