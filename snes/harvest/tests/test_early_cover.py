"""Year 3 Summer cover: the ragged stands and the sown rings' bands start the day they are free, not on D20.

r45 (998, the best of the r41..r47 batch) counted 2,545 tiles on Summer D9 and the same 2,545 on D21: "no stands" for twelve
mornings, because the full stands were done and the 5x5 of every crop site stayed reserved until the final fill. The last nine
mornings then took 405 tiles (2,951, one over); rdet, r42, r46 and r47 missed by 1-2 points. Each test goes red without the
early path, and every one runs with the switch off to show Year 1-2 and the baseline do not move.
"""
from __future__ import annotations

import pytest

from harvest.maps import farm_pond
from harvest.planner import field_cover as fc
from harvest.planner import run_strategy as rs
from tests.test_field_cover import W, index, morning, open_field

BOX = (2, 3, 62, 60)
CORN = 0x3A


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score_herd")
    monkeypatch.setenv(fc.EARLY_COVER_ENV, "1")
    monkeypatch.delenv(rs.FIELD_COVER_ENV, raising=False)


def _site_tiles(site, year=2):
    return farm_pond.plot_crop_tiles(site, year)


def _field_with(sown):
    """An open field whose listed Summer sites stand in corn; the rest are bare."""
    tiles = open_field(box=BOX)
    for site in sown:
        for x, y in _site_tiles(site):
            tiles[y * W + x] = CORN
    return tiles


SITES = farm_pond.active_plot_centers(2, 1)


def _box(site, r):
    return {(site[0] + dx, site[1] + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)}


def test_a_sown_site_gives_up_its_band_but_keeps_its_ring_block():
    site = SITES[0]
    freed = fc.released_cells(_field_with([site]), 2, 1, index(2, 1, 5))
    assert freed and freed <= _box(site, 2) - _box(site, 1)
    assert not freed & _box(site, 1)                                    # the eight crops and the centre stay reserved
    assert len(freed) > 8


def test_a_band_cell_an_unsown_site_still_needs_stays_reserved(monkeypatch):
    """Two sites at pitch 4 share the lane between them: while the east one waits for its bag the lane stays reserved."""
    west, east = (20, 30), (24, 30)
    monkeypatch.setattr(farm_pond, "active_plot_centers", lambda year=0, season=None: (west, east))
    tiles = _field_with([west])
    freed = fc.released_cells(tiles, 2, 1, index(2, 1, 5))
    lane = _box(west, 2) & _box(east, 2)
    assert lane and not freed & lane                                    # x=22: the east site's own band
    assert freed == (_box(west, 2) - _box(west, 1)) - lane
    both = fc.released_cells(_field_with([west, east]), 2, 1, index(2, 1, 5))
    assert lane <= both                                                 # both sown: the shared lane is nobody's any more


def test_nothing_is_released_before_sowing_after_the_final_fill_or_outside_year_three_summer(monkeypatch):
    assert fc.released_cells(_field_with([]), 2, 1, index(2, 1, 5)) == frozenset()          # nothing sown yet
    assert fc.released_cells(_field_with(SITES), 2, 1, index(2, 1, 25)) == frozenset()       # the final fill releases everything
    assert fc.released_cells(_field_with(SITES), 2, 0, index(2, 0, 25)) == frozenset()       # Spring
    assert fc.released_cells(_field_with(SITES), 1, 1, index(1, 1, 5)) == frozenset()        # Year 2
    monkeypatch.setenv(fc.EARLY_COVER_ENV, "0")
    assert fc.released_cells(_field_with(SITES), 2, 1, index(2, 1, 5)) == frozenset()        # switch off


def test_the_bin_stand_is_never_released():
    from harvest.maps.farm_pond import SHIP_BIN_STAND_TILES

    assert not fc.released_cells(_field_with(SITES), 2, 1, index(2, 1, 5)) & set(SHIP_BIN_STAND_TILES)


def test_reserved_cells_shrink_by_exactly_the_released_band():
    tiles = _field_with(SITES)
    freed = fc.released_cells(tiles, 2, 1, index(2, 1, 5))
    base = fc.reserved_cells(2, index(2, 1, 5))
    assert freed and freed <= base and len(base - freed) < len(base)


def _ragged_field():
    """Every full stand is covered; one two-square strip is left, which only a ragged stand takes."""
    tiles = [0xA0] * (W * W)
    for y in range(20, 26):
        for x in range(2, 8):
            tiles[y * W + x] = 0x70
    tiles[20 * W + 2] = 0x01
    tiles[21 * W + 2] = 0x01
    for y in range(DOOR_Y := fc.DOOR_STAND[1], 20):
        tiles[y * W + fc.DOOR_STAND[0]] = 0x01
    tiles[20 * W + fc.DOOR_STAND[0]] = 0x01
    for x in range(2, fc.DOOR_STAND[0] + 1):
        tiles[20 * W + x] = 0x01 if x in (2,) or x >= 8 else tiles[20 * W + x]
    return tiles


def test_when_no_full_stand_is_left_the_ragged_stands_start_before_the_final_fill():
    tiles = _ragged_field()
    day10 = morning(year=2, season=1, day=10)
    early = fc.cover_errand(day10, spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert early.ragged and early.bags > 0 and not early.final_fill
    assert early.buy == early.bags                                       # a ragged day buys the final-fill load, not 24
    assert early.plan.stands and all(c < 6 for c in [early.plan.cells])  # one- and two-square stands only


def test_a_day_with_full_stands_left_still_sows_those_first(monkeypatch):
    tiles = open_field(box=BOX)
    e = fc.cover_errand(morning(year=2, season=1, day=10), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert not e.ragged and e.buy == fc.FINAL_FILL_TARGET_BAGS and len(e.plan.stands) > e.buy      # whole stands, the big load


def test_the_switch_off_and_other_years_keep_the_no_stands_morning(monkeypatch):
    tiles = _ragged_field()
    monkeypatch.setenv(fc.EARLY_COVER_ENV, "0")
    off = fc.cover_errand(morning(year=2, season=1, day=10), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert off.reason == "no stands" and not off.ragged
    monkeypatch.setenv(fc.EARLY_COVER_ENV, "1")
    year_two = fc.cover_errand(morning(year=1, season=1, day=10), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert year_two.reason == "no stands"


def test_the_phases_carry_the_released_cells_and_the_planter_frees_them():
    from harvest.tasks.grass_planter import GrassPlantTask

    tiles = _field_with(SITES[:6])
    e = fc.cover_errand(morning(year=2, season=1, day=5), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert e.released
    phase = fc.cover_field_phase(e)
    assert phase.params["released"] == [list(c) for c in e.released] and phase.params["ragged"] == e.ragged
    free = set(e.released)
    from harvest.maps.farm_pond import crop_working_tiles

    task = GrassPlantTask(no_go_rects=[], release_cells=tuple(free))
    task._init_no_go()
    assert not task._no_go_tiles & free
    assert task._no_go_tiles == set(crop_working_tiles()) - free            # the rest of the reservation stands


def test_year_three_summer_loads_the_big_stock_while_the_field_is_still_open(monkeypatch):
    """r45's D4-D8 loads were 24 bags a day; the harvest days after D13 leave no clock for the shop."""
    tiles = open_field(box=BOX)
    summer = fc.cover_errand(morning(year=2, season=1, day=5), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert not summer.ragged and summer.buy == fc.FINAL_FILL_TARGET_BAGS
    spring = fc.cover_errand(morning(year=2, season=0, day=5), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert spring.buy == fc.TARGET_STOCK_BAGS                                    # Spring keeps the modest load
    monkeypatch.setenv(fc.EARLY_COVER_ENV, "0")
    off = fc.cover_errand(morning(year=2, season=1, day=5), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert off.buy == fc.TARGET_STOCK_BAGS
    year_two = fc.cover_errand(morning(year=1, season=1, day=5), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert year_two.buy == fc.TARGET_STOCK_BAGS
