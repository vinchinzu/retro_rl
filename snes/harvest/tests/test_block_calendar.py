"""Block calendar: notch-block geometry, the Spring/Summer sowing gates, the forecast and the farm layout rows."""

from __future__ import annotations

import pytest

from harvest.paths import GAME_DIR
from harvest.planner import block_calendar as bc


def test_a_notch_block_plants_its_centre_and_leaves_one_side_cell_bare():
    east = bc.block_cells((8, 35), "east")
    west = bc.block_cells((8, 35), "west")
    assert len(east) == len(west) == 8
    assert (8, 35) in east and (9, 35) not in east       # xxx/xxo/xxx
    assert (8, 35) in west and (7, 35) not in west       # xxx/oxx/xxx
    ring = bc.block_cells((13, 28), "ring")
    assert (13, 28) not in ring and len(ring) == 8


def test_lattice_pairs_face_their_notches_across_one_shared_lane():
    blocks = dict(bc.lattice_blocks(yard=()))
    for (x, y), notch in blocks.items():
        if notch != "east":
            continue
        partner = blocks.get((x + 4, y))
        assert partner == "west"
        ne, nw = bc.notch_cell((x, y), "east"), bc.notch_cell((x + 4, y), "west")
        assert nw[0] - ne[0] == 2 and ne[1] == nw[1]     # notch, lane, notch on one row


def test_every_crop_of_an_open_field_block_has_a_stand():
    blocks = bc.plan_blocks(bc.open_field_walkable, core=14, late=0)
    near = [b for b in blocks if b.role == "core" and b.notch != "ring" and b.center != (19, 28)]
    assert near and all(len(b.live_crops) == 8 for b in near)
    # The notch is a stand: the centre crop is picked from it.
    b = next(b for b in blocks if b.center == (8, 35))
    centre = next(c for c in b.crops if c.tile == (8, 35))
    assert centre.stand == bc.notch_cell((8, 35), b.notch)


def test_core_blocks_last_potato_ripens_on_summer_d1_or_d2_and_takes_corn_that_day():
    blocks = bc.plan_blocks(bc.open_field_walkable, core=12, late=2)
    core = [b for b in blocks if b.role == "core"]
    late = [b for b in blocks if b.role == "late"]
    assert {b.potato_ripe_summer_day for b in core} == {1, 2}
    assert all(b.spring_last_sow + bc.POTATO_NIGHTS - bc.SEASON_DAYS == b.summer_plant_day for b in core)
    assert all(b.summer_crop == "corn" for b in core)
    # The late blocks are the farthest live ones and finish their potatoes in Summer.
    assert max(b.rank for b in core) < min(b.rank for b in late)
    assert {b.potato_ripe_summer_day for b in late} == {5, 6}


def test_sowing_gate_follows_the_calendar():
    core = bc.BlockPlan((8, 35), "west", role="core", spring_last_sow=25, summer_crop="corn", summer_plant_day=1)
    late = bc.BlockPlan((4, 43), "east", role="late", spring_last_sow=30, summer_crop="tomato", summer_plant_day=6)
    assert bc.sow_crop(core, 0, 25) == "potato" and bc.sow_crop(core, 0, 26) is None
    assert bc.sow_crop(late, 0, 30) == "potato"
    assert bc.sow_crop(core, 1, 1) == "corn" and bc.sow_crop(core, 1, 19) is None
    assert bc.sow_crop(late, 1, 5) is None and bc.sow_crop(late, 1, 6) == "tomato"


def test_shop_days_follow_the_power_on_week():
    assert bc.weekday(1, 1) == 2                    # Summer D1 is a Wednesday
    assert not bc.shop_open(0, 27) and not bc.shop_open(0, 28)   # Spring D27/D28: the weekend
    assert not bc.shop_open(0, bc.FLOWER_FESTIVAL_DAY)


def test_forecast_never_ships_past_the_clock_and_waters_within_the_cap():
    blocks = bc.plan_blocks(bc.open_field_walkable, core=12, late=2)
    fc = bc.forecast(blocks)
    assert all(d["ship_frames"] <= bc.SHIP_DAY_TICKS for d in fc["days"])
    assert all(d["water_tiles"] <= bc.MAX_WATER_TILES for d in fc["days"])
    summer = [d for d in fc["days"] if d["season"] == 1]
    assert not any(d["shipped"].get("corn") for d in summer if d["day"] < 13)   # 12 nights from D1
    assert fc["totals"]["money_end"] > fc["totals"]["money_spring_end"] > 0


def test_farm_layout_rows_are_the_calendar_plan():
    from harvest.maps.farm_pond import BLOCK_PLAN_ROWS

    if not (GAME_DIR / "LN_d2_perfect_poweron.state").is_file():
        pytest.skip("the Day-3 pin is local-only")
    blocks = bc.plan_blocks(bc.pin_walkable(), core=12, late=2)
    rows = tuple((b.center, b.notch, b.role, b.spring_last_sow, b.summer_crop, b.summer_plant_day)
                 for b in blocks if b.role != "off")
    assert rows == BLOCK_PLAN_ROWS


@pytest.mark.skipif(__import__("harvest.maps.farm_pond", fromlist=["x"]).ring_layout_name() != "blocks",
                    reason="layout rows apply under HARVEST_RING_LAYOUT=blocks")
def test_blocks_layout_counts_the_notch_shape_and_gates_by_calendar():
    from harvest.maps import farm_pond

    tiles = farm_pond.plot_crop_tiles((8, 35))
    assert (8, 35) in tiles and (7, 35) not in tiles
    assert farm_pond.plot_layout_name((12, 35)) == "eight_east_notch"
    assert farm_pond.plot_layout_name((13, 28)) == "eight_tile_ring"
    assert farm_pond.calendar_bag((8, 35), 0, 26) == "potato" and farm_pond.calendar_bag((8, 35), 0, 27) is None
    assert farm_pond.calendar_bag((12, 43), 0, 29) == "potato"


@pytest.mark.skipif(__import__("harvest.maps.farm_pond", fromlist=["x"]).ring_layout_name() != "blocks",
                    reason="layout rows apply under HARVEST_RING_LAYOUT=blocks")
def test_year_two_adds_six_blocks_that_year_one_never_sows(monkeypatch):
    from harvest.maps import farm_pond

    monkeypatch.setenv("HARVEST_Y2_BLOCKS", "1")
    assert farm_pond.active_plot_centers(1) != farm_pond.POCKET_PLANT_CENTERS
    monkeypatch.delenv("HARVEST_Y2_BLOCKS")
    assert farm_pond.active_plot_centers(1) == farm_pond.POCKET_PLANT_CENTERS   # off by default
    monkeypatch.setenv("HARVEST_Y2_BLOCKS", "1")

    assert len(farm_pond.POCKET_PLANT_CENTERS) == len(farm_pond.BLOCK_PLAN_ROWS)
    assert farm_pond.active_plot_centers(0) == farm_pond.POCKET_PLANT_CENTERS
    assert len(farm_pond.active_plot_centers(1)) == len(farm_pond.BLOCK_PLAN_ROWS) + 6
    for center in farm_pond.Y2_ONLY_CENTERS:
        assert farm_pond.calendar_bag(center, 0, 1, 0) is None          # Year 1: not in the calendar
        assert farm_pond.calendar_bag(center, 0, 1, 1) == "potato"      # Year 2 Spring D1: open
        assert farm_pond.calendar_bag(center, 0, 27, 1) is None          # past its last potato sow
        assert farm_pond.calendar_bag(center, 1, 2, 1) == "corn"         # Summer bag from D1/D2
    assert farm_pond.calendar_bag((8, 35), 0, 1, 0) == "potato"          # the first fourteen are unchanged
