"""Parametrized L6 DoorHopSpec table: dest RAM, two occupancy smokes."""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.nes import nes_action
from zelda_i.level6.door_hop import (
    DoorHopSpec,
    Level6DoorHopController,
    NORTH2C_SPEC,
    SOUTH18_SPEC,
    SOUTH1D_SPEC,
    WEST19_SPEC,
    WEST2D_SPEC,
    door_hop_success,
)
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)

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


def _ram(
    *,
    screen: int,
    x: int = 120,
    y: int = 141,
    mode: int = PLAY_MODE,
    level: int = 6,
    triforce: int = 0x1F,
    rod: int = 1,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_TRIFORCE] = triforce
    ram[ADDR_ROD] = rod
    return ram


def _snap(
    *,
    screen: int,
    x: int = 120,
    y: int = 141,
    mode: int = PLAY_MODE,
    level: int = 6,
    triforce: int = 0x1F,
    rod: int = 1,
):
    return read_snapshot(
        _ram(
            screen=screen, x=x, y=y, mode=mode, level=level,
            triforce=triforce, rod=rod,
        )
    )


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
    first = Level6DoorHopController(SOUTH1D_SPEC).step(leftover)
    assert list(first.action) != list(nes_action("UP"))
    gx, gy = SOUTH1D_SPEC.goal
    hold = Level6DoorHopController(SOUTH1D_SPEC).step(
        _snap(screen=SOUTH1D_SPEC.room, x=gx, y=gy)
    )
    assert (gx, gy) == (120, 189)
    assert list(hold.action) == list(nes_action("DOWN"))


def test_west2d_align_y_then_left() -> None:
    leftover = _snap(screen=WEST2D_SPEC.room, x=120, y=77)
    first = Level6DoorHopController(WEST2D_SPEC).step(leftover)
    assert list(first.action) != list(nes_action("LEFT"))
    west = Level6DoorHopController(WEST2D_SPEC).step(
        _snap(screen=WEST2D_SPEC.room, x=120, y=141)
    )
    assert list(west.action) == list(nes_action("LEFT"))
