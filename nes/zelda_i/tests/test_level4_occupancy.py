"""L4 occupancy seeds (no emulator). Dest cells stay open; solids stay blocked."""

from __future__ import annotations

from zelda_i.level4_dungeon import LADDER_60_PICKUP_XY, MAP_21_PICKUP_XY, RIGHT_20_STAND
from zelda_i.level4_occupancy import room_20_grid, room_21_grid, room_60_grid


def test_dest_cells_not_overblocked() -> None:
    assert room_60_grid().passable(*LADDER_60_PICKUP_XY)
    assert room_20_grid().passable(*RIGHT_20_STAND)
    assert room_21_grid().passable(*MAP_21_PICKUP_XY)


def test_documented_solids_are_blocked() -> None:
    assert not room_60_grid().passable(49, 133)
    assert not room_20_grid().passable(160, 150)
    assert not room_21_grid().passable(48, 140)
