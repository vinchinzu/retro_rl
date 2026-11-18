"""Terrain ledger: the ROM's 2,950-tile rule, hoeing, and the grass-bag plan (no ROM, no pins)."""

from __future__ import annotations

import pytest

from harvest.core.harvest_state import projected_farm_development_percent
from harvest.maps.farm_final_layout import (
    GRASS_BAG_TILES,
    MIN_COUNTED_TILES,
    TERRAIN_CLASSES,
    disjoint_full_windows,
    plan_grass_bags,
    terrain_class,
    terrain_ledger,
)
from harvest.core.tile_catalog import MAP_TILE_COUNT, MAP_WIDTH

STRUCTURE = 0xA8
UNTILLED, DRIED, WEED, STONE, TILLED, WET, GRASS, SEED, CROP, FENCE = (
    0x01, 0x02, 0x03, 0x04, 0x07, 0x08, 0x70, 0x1D, 0x3A, 0x05,
)


def _farm(rect=(2, 2, 61, 51), fill=UNTILLED) -> list[int]:
    """64x64 farm that is structure everywhere except ``rect`` (x0, y0, x1, y1 inclusive)."""
    tiles = [STRUCTURE] * MAP_TILE_COUNT
    x0, y0, x1, y1 = rect
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            tiles[y * MAP_WIDTH + x] = fill
    return tiles


def _set(tiles: list[int], cells, value: int) -> None:
    for x, y in cells:
        tiles[y * MAP_WIDTH + x] = value


def test_class_names_follow_the_rom_count_not_the_bot_names():
    # 0x06 is "rock" in the bot catalog, but the ROM counts it as broken fence.
    assert terrain_class(0x06) == "fence"
    assert terrain_class(0x05) == "fence"
    assert terrain_class(SEED) == "fall_seed"
    assert terrain_class(0x7A) == "grass" and terrain_class(0x85) == "grass"
    assert terrain_class(CROP) == "crop" and terrain_class(0x55) == "crop"
    assert terrain_class(TILLED) == terrain_class(WET) == "tilled"
    assert terrain_class(UNTILLED) == "untilled" and terrain_class(DRIED) == "dried"
    assert terrain_class(WEED) == "weed" and terrain_class(STONE) == "stone"
    assert terrain_class(0x09) == "heavy_debris" and terrain_class(0x0D) == "heavy_debris"
    assert terrain_class(0xA0) == "structure"
    assert {terrain_class(i) for i in range(0xA0)} <= set(TERRAIN_CLASSES)


@pytest.mark.parametrize(
    ("counted", "dev_byte", "points"),
    [(2950, 100, 50), (2949, 99, 49), (2921, 99, 49), (2920, 98, 49), (2891, 98, 49), (2890, 97, 48), (3200, 100, 50)],
)
def test_fifty_points_need_exactly_2950_counted_tiles(counted, dev_byte, points):
    tiles = _farm(rect=(0, 0, 63, 60), fill=UNTILLED)
    cells = [(x, y) for y in range(61) for x in range(64)][:counted]
    _set(tiles, cells, GRASS)
    ledger = terrain_ledger(tiles)
    assert ledger.counted == counted
    assert (ledger.dev_byte, ledger.points) == (dev_byte, points)
    assert projected_farm_development_percent(counted) == dev_byte
    assert ledger.to_go == max(0, MIN_COUNTED_TILES - counted)


def test_the_field_needs_99_4_percent_of_the_clear_pin_pool():
    # 2,968 cells on the FarmClear pin; 2,974 on the Fall mask.
    for pool, required, slack in ((2968, 99.39, 18), (2974, 99.19, 24)):
        rows, extra = divmod(pool, 60)
        tiles = _farm(rect=(2, 2, 61, 1 + rows))
        _set(tiles, [(2 + i, 2 + rows) for i in range(extra)], UNTILLED)
        ledger = terrain_ledger(tiles)
        assert ledger.pool == pool
        assert ledger.required_pct == pytest.approx(required, abs=0.01)
        assert ledger.slack == slack


def test_worked_share_counts_hoed_planted_and_crops_but_not_bare_ground():
    tiles = _farm(rect=(2, 2, 11, 11))  # 100 cells
    _set(tiles, [(2 + i, 2) for i in range(10)], TILLED)   # 10 hoed
    _set(tiles, [(2 + i, 3) for i in range(10)], GRASS)    # 10 grass
    _set(tiles, [(2 + i, 4) for i in range(10)], CROP)     # 10 crops
    _set(tiles, [(2, 5), (3, 5)], FENCE)                   # 2 fences
    _set(tiles, [(2 + i, 6) for i in range(5)], WEED)
    _set(tiles, [(2 + i, 7) for i in range(5)], STONE)
    _set(tiles, [(2 + i, 8) for i in range(5)], DRIED)
    ledger = terrain_ledger(tiles)
    assert ledger.pool == 100
    assert ledger.classes["tilled"] == 10 and ledger.classes["grass"] == 10
    assert ledger.worked == 10 + 10 + 10 + 2
    assert ledger.worked_pct == 32.0
    assert ledger.counted == 10 + 10 + 2  # hoed soil is worked but does not count
    # untilled 100-37 are bare; weed, stone, dried, untilled all need a hoe swing.
    assert ledger.hoe_swings_left == 100 - 32
    assert ledger.clear_left == 10


def test_full_field_plan_is_328_bags_at_best_and_costs_more_with_real_edges():
    tiles = _farm()  # 60 x 50 = 3,000 untilled cells
    ledger = terrain_ledger(tiles)
    assert ledger.pool == 3000 and ledger.to_go == MIN_COUNTED_TILES
    assert ledger.bags_min == 328  # ceil(2950 / 9)
    # 20 x 16 nine-cell blocks cover 2,880; the two-row strip left gives six cells a bag.
    assert ledger.bags_plan == 320 + 12
    assert ledger.bag_cost_g == 332 * 500
    assert ledger.hoe_swings_left == 3000
    assert ledger.stamina_left == 2 * 3000 + 332


def test_plan_is_not_executable_until_every_planned_square_is_hoed():
    tiles = _farm(rect=(2, 2, 10, 10))  # 81 cells, 9 blocks
    ledger = terrain_ledger(tiles, need=81)
    assert ledger.bags_plan == 9 and ledger.plan_unhoed == 81 and ledger.plan_ready_bags == 0
    _set(tiles, [(x, y) for y in range(2, 5) for x in range(2, 5)], TILLED)
    one_block = terrain_ledger(tiles, need=81)
    assert one_block.plan_ready_bags == 1 and one_block.plan_unhoed == 72
    assert one_block.ready_blocks == 1
    _set(tiles, [(x, y) for y in range(2, 11) for x in range(2, 11)], WET)
    ready = terrain_ledger(tiles, need=81)
    assert ready.plan_unhoed == 0 and ready.plan_ready_bags == ready.bags_plan == 9
    assert ready.ready_blocks == 9
    assert ready.hoe_swings_left == 0  # hoeing is finished; only the seed is left


def test_a_square_left_bare_inside_a_bag_is_a_missing_tile_not_a_missing_bag():
    tiles = _farm(rect=(2, 2, 10, 10))
    _set(tiles, [(x, y) for y in range(2, 11) for x in range(2, 11)], GRASS)
    _set(tiles, [(5, 5)], UNTILLED)  # a skipped square
    ledger = terrain_ledger(tiles, need=81)
    assert ledger.counted == 80 and ledger.to_go == 1
    assert ledger.bags_plan == 1 and ledger.plan_unhoed == 1
    assert ledger.full_grass_blocks < 9


def test_full_grass_blocks_count_disjoint_nine_tile_blocks():
    tiles = _farm(rect=(2, 2, 13, 4))  # 12 x 3
    _set(tiles, [(x, y) for y in range(2, 5) for x in range(2, 8)], GRASS)  # two blocks
    _set(tiles, [(x, y) for y in range(2, 5) for x in range(8, 14)], SEED)  # two more, fall seed
    ledger = terrain_ledger(tiles)
    assert ledger.full_grass_blocks == 4
    mask = [False] * MAP_TILE_COUNT
    for y in range(3):
        for x in range(4):  # 4 x 3: one block and a spare column
            mask[y * MAP_WIDTH + x] = True
    assert disjoint_full_windows(mask) == 1


def test_fence_already_counts_and_never_asks_for_a_bag():
    tiles = _farm(rect=(2, 2, 4, 4))
    _set(tiles, [(3, 3)], FENCE)
    ledger = terrain_ledger(tiles, need=9)
    assert ledger.counted == 1 and ledger.to_go == 8
    assert ledger.classes["fence"] == 1


def test_blocked_cells_stay_in_the_pool_but_never_take_a_bag():
    tiles = _farm(rect=(2, 2, 4, 4))
    _set(tiles, [(2, 2)], 0xA0)  # the Fall-mask cells read 0xA0 on the clear pin
    cover = tuple((x, y) for y in range(2, 5) for x in range(2, 5))
    ledger = terrain_ledger(tiles, cover=cover, need=8)
    assert ledger.pool == 9 and ledger.classes["blocked"] == 1
    assert ledger.to_go == 8 and ledger.bags_plan == 1
    assert plan_grass_bags({3 * MAP_WIDTH + 3}, 1) == [(3, 3, 1)]


def test_planted_grass_counter_is_reported_next_to_the_map():
    tiles = _farm(rect=(2, 2, 4, 4), fill=GRASS)
    ledger = terrain_ledger(tiles, planted_grass=9, need=9)
    assert ledger.planted_grass == 9 and ledger.counted == 9 and ledger.points == 0
    assert "planted_grass" in ledger.as_dict()
    assert "counted 9/9" in ledger.line()


def test_bag_tiles_match_the_rom_pattern():
    assert GRASS_BAG_TILES == 9  # PreCheckToolSuccessSquare steps $096B through nine squares


def test_scorecard_prints_the_terrain_table_from_the_day_journal():
    from harvest.tools import final_score

    tiles = _farm(rect=(2, 2, 10, 10))
    morning = terrain_ledger(tiles, planted_grass=0, need=81).as_dict()
    _set(tiles, [(x, y) for y in range(2, 11) for x in range(2, 11)], GRASS)
    done = terrain_ledger(tiles, planted_grass=81, need=81).as_dict()
    journal = [
        {"plan_season": 0, "plan_day": 1, "development_census": {"ledger": morning}},
        {"plan_season": 0, "plan_day": 2, "development_census": {"ledger": morning}},
        {"plan_season": 1, "plan_day": 1, "development_census": {"ledger": done}},
    ]
    report = {"start": {"year": 0}, "day_journal": journal}
    rows = final_score.terrain_rows(report)
    assert [row["label"] for row in rows] == ["Y1 Sp D1", "Y1 Su D1"]
    text = "\n".join(final_score.render_terrain(report))
    assert "Farm development (terrain worked)" in text
    assert "| Y1 Su D1 | 100.0% | 81 | 1 |" in text  # 81 * 10 // 295 = 2, one point: 2,950 is the bar
    assert final_score.render_terrain({"day_journal": [{"plan_season": 0, "plan_day": 1}]}) == []
