"""A walk that starts at the shipping bin may leave by the x=11 lane; no other walk may.

With the west ring grown and the y=31 fence regrown, (11,26)-(11,28) is the only
way out of the bin stand, yet they sit in FARM_NO_GO_TILES. Opening them for
every farm path broke the hoe-stand navigation (40 CROP_ESTABLISH failures in a
power-on year), so only a walk that starts on the lane or the stand gets them.
"""
from retro_harness import WorldState

from harvest.maps.farm_pond import SHIP_BIN_LANE_TILES, SHIP_BIN_STAND_TILES
from harvest.maps.map_config import Waypoint
from harvest.planner.tasks.navigation import MultiMapNavTask
from tests.day_plan_test_helpers import make_navigation_ram, set_player_pos


def _nav_from(tile, *, allow=frozenset()):
    ram = make_navigation_ram(current_tile=tile, blocked_tile=(63, 63), blocked_id=0xA1)
    set_player_pos(ram, tile[0] * 16 + 8, tile[1] * 16 + 8)
    task = MultiMapNavTask(waypoints=[Waypoint(0x00, (136, 424), radius=8)], allow_tiles=allow)
    task.reset(WorldState(frame=0, ram=ram, info={}))
    return task


def test_a_walk_from_the_bin_stand_may_use_the_lane():
    for tile in sorted(SHIP_BIN_STAND_TILES):
        assert SHIP_BIN_LANE_TILES <= _nav_from(tile)._pathfinder.allow_tiles, tile


def test_a_walk_from_anywhere_else_keeps_the_lane_closed():
    for tile in ((13, 20), (12, 29), (14, 30), (11, 25), (11, 31)):
        assert not (SHIP_BIN_LANE_TILES & _nav_from(tile)._pathfinder.allow_tiles), tile


def test_the_routes_own_allowed_tiles_are_kept():
    extra = frozenset({(30, 30)})
    assert extra <= _nav_from((13, 20), allow=extra)._pathfinder.allow_tiles
    assert extra <= _nav_from((11, 29), allow=extra)._pathfinder.allow_tiles
