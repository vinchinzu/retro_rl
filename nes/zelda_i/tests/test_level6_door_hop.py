"""Parametrized L6 DoorHopSpec table: dest RAM, two occupancy smokes."""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.nes import nes_action
from zelda_i.level6.door_hop import (
    DoorHopSpec,
    EAST39_SPEC,
    INLAND29_SPEC,
    Level6DoorHopController,
    NORTH2C_SPEC,
    SOUTH18_SPEC,
    SOUTH1D_SPEC,
    SOUTH29_SPEC,
    WEST19_SPEC,
    WEST2D_SPEC,
    door_hop_success,
    inland29_success,
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
    """North leftover holds DOWN; waist LEFT; SW pocket (32,189) holds UP."""
    leftover = _snap(screen=WEST2D_SPEC.room, x=120, y=77)
    first = Level6DoorHopController(WEST2D_SPEC).step(leftover)
    assert list(first.action) == list(nes_action("DOWN"))
    assert list(first.action) != list(nes_action("LEFT"))
    west = Level6DoorHopController(WEST2D_SPEC).step(
        _snap(screen=WEST2D_SPEC.room, x=120, y=141)
    )
    assert list(west.action) == list(nes_action("LEFT"))
    door = Level6DoorHopController(WEST2D_SPEC).step(
        _snap(screen=WEST2D_SPEC.room, x=32, y=141)
    )
    assert list(door.action) == list(nes_action("LEFT"))
    # Occupancy boxed here on the power-on tape; cardinal UP re-acquires y=141.
    pocket = Level6DoorHopController(WEST2D_SPEC).step(
        _snap(screen=WEST2D_SPEC.room, x=32, y=189)
    )
    assert list(pocket.action) == list(nes_action("UP"))
    assert WEST2D_SPEC.cardinal_hold is True
    assert WEST2D_SPEC.align == "y"


def test_north2c_align_x_then_up() -> None:
    """East leftover holds LEFT; column UP; waist leftover (71,141) holds RIGHT."""
    leftover = _snap(screen=NORTH2C_SPEC.room, x=224, y=141)
    first = Level6DoorHopController(NORTH2C_SPEC).step(leftover)
    assert list(first.action) == list(nes_action("LEFT"))
    assert list(first.action) != list(nes_action("UP"))
    column = Level6DoorHopController(NORTH2C_SPEC).step(
        _snap(screen=NORTH2C_SPEC.room, x=120, y=141)
    )
    assert list(column.action) == list(nes_action("UP"))
    door = Level6DoorHopController(NORTH2C_SPEC).step(
        _snap(screen=NORTH2C_SPEC.room, x=120, y=93)
    )
    assert list(door.action) == list(nes_action("UP"))
    # Occupancy south_open_halt boxed here on the power-on tape.
    shuffled = Level6DoorHopController(NORTH2C_SPEC).step(
        _snap(screen=NORTH2C_SPEC.room, x=71, y=141)
    )
    assert list(shuffled.action) == list(nes_action("RIGHT"))
    assert NORTH2C_SPEC.cardinal_hold is True
    assert NORTH2C_SPEC.align == "x"


def test_south29_live_leftover_goes_down() -> None:
    """Waist leftover (120,141) occupancies DOWN. Not RIGHT+DOWN clip."""
    leftover = _snap(screen=SOUTH29_SPEC.room, x=120, y=141)
    first = Level6DoorHopController(SOUTH29_SPEC).step(leftover)
    assert list(first.action) == list(nes_action("DOWN"))
    assert list(first.action) != list(nes_action("RIGHT", "DOWN"))
    assert SOUTH29_SPEC.clip_buttons is None
    door = Level6DoorHopController(SOUTH29_SPEC).step(
        _snap(screen=SOUTH29_SPEC.room, x=120, y=189)
    )
    assert list(door.action) == list(nes_action("DOWN"))
    trap = Level6DoorHopController(SOUTH29_SPEC).step(
        _snap(screen=SOUTH29_SPEC.room, x=63, y=133)
    )
    assert list(trap.action) != list(nes_action("UP"))
    assert list(trap.action) != list(nes_action("LEFT", "UP"))
    assert trap.reason != "south_clip"


def test_east39_north_band_leftover_drops_to_waist_then_right() -> None:
    """Power-on leftover (95,109) holds DOWN to y=141, not RIGHT into the wall."""
    assert EAST39_SPEC.clip_buttons == ("DOWN",)
    leftover = _snap(screen=EAST39_SPEC.room, x=95, y=109)
    first = Level6DoorHopController(EAST39_SPEC).step(leftover)
    assert list(first.action) == list(nes_action("DOWN"))
    assert list(first.action) != list(nes_action("RIGHT"))
    # Once on the waist the cardinal hold carries RIGHT toward the door.
    waist = Level6DoorHopController(EAST39_SPEC).step(
        _snap(screen=EAST39_SPEC.room, x=120, y=141)
    )
    assert list(waist.action) == list(nes_action("RIGHT"))
    door = Level6DoorHopController(EAST39_SPEC).step(
        _snap(screen=EAST39_SPEC.room, x=208, y=141)
    )
    assert list(door.action) == list(nes_action("RIGHT"))


def test_inland29_south_mouth_clips_left_up() -> None:
    leftover = _snap(screen=INLAND29_SPEC.room, x=120, y=205)
    first = Level6DoorHopController(INLAND29_SPEC).step(leftover)
    assert first.reason == "inland_clip"
    assert list(first.action) == list(nes_action("LEFT", "UP"))
    assert list(first.action) != list(nes_action("UP"))
    door = Level6DoorHopController(INLAND29_SPEC).step(
        _snap(screen=INLAND29_SPEC.room, x=120, y=93)
    )
    assert list(door.action) == list(nes_action("UP"))
    dest = _snap(screen=0x19, x=120, y=205)
    assert inland29_success(dest)
    assert not inland29_success(leftover)
    back = _snap(screen=0x39, x=120, y=93)
    assert not inland29_success(back)
