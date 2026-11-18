"""Fewest-turn tile BFS on synthetic grids. No ROM."""

from __future__ import annotations

import numpy as np

from alttp.dungeon.nav import door_line, find_path, free_chest_stand, room_origin
from alttp.room_map import KnownDoor, RoomMap, RoomMapPoint
from alttp.walk_physics import LINK_BODY


def _turns(path: list[tuple[int, int]]) -> int:
    steps = [
        (x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:])
    ]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def test_open_floor_path_has_one_turn() -> None:
    tiles = np.zeros((16, 16), dtype=np.uint8)
    path = find_path(tiles, (0, 0), (24, 24), LINK_BODY)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (24, 24)
    assert _turns(path) == 1
    assert all(abs(x1 - x0) + abs(y1 - y0) == 8 for (x0, y0), (x1, y1) in zip(path, path[1:]))


def test_pit_deep_water_and_spike_block_unless_allowed() -> None:
    bypass = np.zeros((16, 16), dtype=np.uint8)
    bypass[1:3, 2:6] = 0x20
    path = find_path(bypass, (0, 0), (48, 0), LINK_BODY)
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (48, 0)
    for x, y in path:
        cols = range(x // 8, (x + 15) // 8 + 1)
        rows = range((y + 8) // 8, (y + 23) // 8 + 1)
        assert all(int(bypass[row, col]) != 0x20 for row in rows for col in cols)

    sealed = np.full((16, 16), 0x20, dtype=np.uint8)
    sealed[1:3, 0:2] = 0
    sealed[1:3, 6:8] = 0
    assert find_path(sealed, (0, 0), (48, 0), LINK_BODY) is None
    water = sealed.copy()
    water[water == 0x20] = 0x08
    spike = sealed.copy()
    spike[spike == 0x20] = 0x44
    assert find_path(water, (0, 0), (48, 0), LINK_BODY) is None
    assert find_path(spike, (0, 0), (48, 0), LINK_BODY) is None
    assert find_path(sealed, (0, 0), (48, 0), LINK_BODY, allow=("pit",)) is not None
    assert find_path(water, (0, 0), (48, 0), LINK_BODY, allow=("deep_water",)) is not None
    assert find_path(spike, (0, 0), (48, 0), LINK_BODY, allow=("spike",)) is not None


def test_raised_peg_blocks_and_down_peg_does_not() -> None:
    """0x67 is a wall for the current grid. 0x66 is floor. No peg write."""
    raised = np.full((16, 16), 0x67, dtype=np.uint8)
    raised[1:3, 0:2] = 0x00
    raised[1:3, 6:8] = 0x00
    assert find_path(raised, (0, 0), (48, 0), LINK_BODY) is None
    down = raised.copy()
    down[down == 0x67] = 0x66
    assert find_path(down, (0, 0), (48, 0), LINK_BODY) is not None


def test_free_chest_stand_is_south_of_the_graphic() -> None:
    """16x16 chest. The stand is the left column, body on the floor under it."""
    tiles = np.zeros((64, 64), dtype=np.uint8)
    tiles[48:50, 47:49] = 0x58
    stand = free_chest_stand(tiles, (392, 404), 47, 48)
    assert stand == (376, 392)
    body_cols = range(stand[0] // 8, (stand[0] + 15) // 8 + 1)
    body_rows = range((stand[1] + 8) // 8, (stand[1] + 23) // 8 + 1)
    assert all(int(tiles[row, col]) == 0 for row in body_rows for col in body_cols)


def test_free_chest_stand_skips_when_the_south_floor_is_walled() -> None:
    tiles = np.zeros((64, 64), dtype=np.uint8)
    tiles[48:50, 47:49] = 0x58
    tiles[50:52, 45:51] = 0x27
    assert free_chest_stand(tiles, (392, 404), 47, 48) is None


def test_room_origin_is_the_512px_corner() -> None:
    assert room_origin(760, 3520) == (512, 3072)
    assert room_origin(700, 3326) == (512, 3072)


def test_type_02_mouth_uses_the_measured_centre_and_tolerance() -> None:
    """CastleMain mouths. Tolerance is the ±15 capture window, not a guess."""
    west = np.zeros((64, 64), dtype=np.uint8)
    west[32:34, 4:8] = 0x85
    line = door_line(west, 0x0F8A, 2, 0x02, (512, 3072))
    assert line is not None
    assert line.axis == "y" and line.centre == 3320 and line.tolerance == 15
    assert abs(3305 - line.centre) <= line.tolerance
    assert abs(3335 - line.centre) <= line.tolerance
    assert abs(3304 - line.centre) > line.tolerance
    assert abs(3336 - line.centre) > line.tolerance

    south = np.zeros((64, 64), dtype=np.uint8)
    south[56:60, 31:33] = 0x8E
    line = door_line(south, 0x1BBC, 1, 0x0102, (512, 3072))
    assert line is not None
    assert line.axis == "x" and line.centre == 760 and line.tolerance == 15
    assert abs(745 - line.centre) <= line.tolerance
    assert abs(775 - line.centre) <= line.tolerance
    assert abs(744 - line.centre) > line.tolerance
    assert abs(776 - line.centre) > line.tolerance

    unmeasured = west.copy()
    unmeasured[16:18, 1:5] = 0x81
    assert door_line(unmeasured, 0x0784, 2, 0x00, (512, 3072)) is None


def test_missing_approach_is_not_a_hand_point() -> None:
    loaded = KnownDoor.from_dict(
        {"label": "west", "direction": "LEFT", "toRoom": 0x60, "path": ["hall"]}
    )
    assert loaded.approach_xy is None
    room = RoomMap(
        room_base_id=0x61,
        name="fixture",
        points=(RoomMapPoint("hall", 720, 3320),),
        doors=(loaded,),
    )
    assert room.waypoints_for_door(loaded) == ((720, 3320, "hall", 12),)
