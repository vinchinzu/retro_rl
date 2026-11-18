"""Whole-field cover (`harvest.planner.field_cover`): the cash floor, the east-first stand plan, the load, the hoe pass."""

from __future__ import annotations

import numpy as np
import pytest

from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_MAP
from harvest.planner import field_cover as fc
from harvest.planner import livestock_econ as econ
from harvest.planner import run_strategy as rs

W = 64
DOOR = fc.DOOR_STAND


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(rs.ENV_VAR, raising=False)
    monkeypatch.delenv(rs.FIELD_COVER_ENV, raising=False)


def ram(**fields):
    result = np.zeros(0x20000, dtype=np.uint8)
    for name, value in fields.items():
        spec = field_spec(name)
        value = spec.to_storage(value)
        for shift in range(spec.width):
            result[spec.address + shift] = (value >> (8 * shift)) & 255
    return result


def open_field(fill: int = 0x01, box=(2, 20, 40, 50)) -> list[int]:
    """A farm array: structure (0xA0) everywhere but an open box, with the door stand and a walkway cut in."""
    tiles = [0xA0] * (W * W)
    x0, y0, x1, y1 = box
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            tiles[y * W + x] = fill
    for y in range(DOOR[1], y0 + 1):
        tiles[y * W + DOOR[0]] = 0x01
    return tiles


def grassed(tiles: list[int], count: int) -> list[int]:
    """``tiles`` with the first ``count`` open squares (raster order, far from the door's column) already grass."""
    out = list(tiles)
    done = 0
    for y in range(40, 51):
        for x in range(2, 41):
            if done < count and out[y * W + x] == 0x01:
                out[y * W + x] = 0x79
                done += 1
    return out


def put(data: np.ndarray, tiles: list[int]) -> np.ndarray:
    data[ADDR_MAP:ADDR_MAP + W * W] = np.array(tiles, dtype=np.uint8)
    return data


def index(year: int, season: int, day: int) -> int:
    return fc.day_index(year, season, day)


BIG = (2, 3, 62, 60)      # the real farm's open box: ~380 whole stands, more than one rucksack stack


# ── the window and the money ─────────────────────────────────────────

def test_calendar_indexes_match_the_ledger():
    assert index(0, 0, 1) == 0
    assert index(2, 1, 30) == fc.TALLY_INDEX == 299
    assert econ.calendar_ymd(fc.TALLY_INDEX) == (3, 1, 30)
    assert econ.calendar_ymd(fc.START_INDEX) == (2, 0, 1)
    assert econ.calendar_ymd(fc.DEADLINE_INDEX) == (3, 1, 29)


def test_sowing_windows_are_spring_and_summer_of_years_two_and_three():
    assert not fc.sowing_open(index(0, 1, 10))           # Year 1: gold compounds in crops
    assert fc.sowing_open(index(1, 0, 1)) and fc.sowing_open(index(1, 1, 30))
    assert not fc.sowing_open(index(1, 2, 5))            # Fall seed writes 0x1D and Winter D1 wipes it
    assert not fc.sowing_open(index(1, 3, 5))
    assert fc.sowing_open(index(2, 1, 29)) and not fc.sowing_open(index(2, 1, 30))
    assert fc.sow_days(fc.START_INDEX, fc.DEADLINE_INDEX) == 119


def test_there_is_no_schedule_line_or_daily_quota():
    """Owner, 2026-11-03: "straight line nonsense .. do the entire thing"."""
    for gone in ("scheduled_tiles", "SCHEDULE_FRONT", "LAST_WEEK_INDEX", "MAX_BAGS_PER_ERRAND", "BATCH_BAGS", "URGENT_BEHIND"):
        assert not hasattr(fc, gone), gone


def test_cash_floor_protects_the_money_term_and_falls_away_early():
    first = fc.cash_floor_g(fc.START_INDEX)                                      # 179 days of income still to come
    assert fc.RESERVE_G <= first == fc.MONEY_TERM_G + fc.MARGIN_G - fc.INCOME_G_PER_DAY * 179 < 25_000
    assert fc.cash_floor_g(fc.START_INDEX - 200) == fc.RESERVE_G                 # far enough out, only the reserve
    assert fc.cash_floor_g(fc.TALLY_INDEX) == fc.MONEY_TERM_G + fc.MARGIN_G      # nothing left to earn
    floors = [fc.cash_floor_g(i) for i in range(fc.START_INDEX, fc.TALLY_INDEX + 1)]
    assert floors == sorted(floors) and min(floors) >= fc.RESERVE_G


# ── the stands ───────────────────────────────────────────────────────

def test_stands_are_disjoint_east_side_first_and_walkable_from_the_door():
    tiles = open_field()
    plan = fc.plan_cover_stands(tiles)
    assert len(plan.stands) > 100 and plan.cells <= 9 * len(plan.stands)
    assert plan.cells <= plan.pool
    whole = fc.plan_cover_stands(tiles, min_cells=1)                # with ragged stands allowed, the stands claim every square of the pool
    assert whole.cells == whole.pool > plan.cells
    bands = [x // fc.BAND_WIDTH for x, _y in plan.stands]
    assert bands == sorted(bands, reverse=True)                  # east to west, never back
    assert plan.stands[0][0] // fc.BAND_WIDTH == max(bands)
    assert max(x for x, _y in plan.stands[:5]) >= max(x for x, _y in plan.stands) - fc.BAND_WIDTH
    assert plan.route[0] == DOOR and plan.route[-1] == plan.stands[0]
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(plan.route, plan.route[1:]))


def test_the_sweep_zigzags_so_each_hop_stays_short():
    plan = fc.plan_cover_stands(open_field())
    hops = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(plan.stands, plan.stands[1:])]
    assert sorted(hops)[int(len(hops) * 0.9)] <= 2 * fc.BAND_WIDTH and sum(hops) / len(hops) <= fc.BAND_WIDTH + 2


def test_cover_bfs_detours_around_the_south_wells_walkable_looking_rim():
    tiles = open_field(box=(2, 20, 40, 60))
    for y, row in enumerate(((0xA3, 0xA1, 0xA2), (0xA1, 0xC8, 0xA1), (0xA8, 0xA8, 0xA8)), 47):
        for x, tile in enumerate(row, 36):
            tiles[y * W + x] = tile
    route = fc.route_to(fc.reachable(tiles), (36, 55))
    assert route and route[-1] == (36, 55)
    assert not any(36 <= x <= 38 and 47 <= y <= 49 for x, y in route)


def test_a_limit_caps_the_stand_list_from_the_east():
    plan = fc.plan_cover_stands(open_field(), 5)
    full = fc.plan_cover_stands(open_field())
    assert plan.stands == full.stands[:5]


def test_reserved_crop_footprints_are_neither_stood_on_nor_counted():
    tiles = open_field()
    reserved = {(x, y) for x in range(2, 20) for y in range(20, 36)}
    plan = fc.plan_cover_stands(tiles, reserved=reserved)
    assert plan.stands
    for cx, cy in plan.stands:
        assert (cx, cy) not in reserved
    # The reserve leaves the pool, and the stands cover only what is left of it.
    free = fc.plan_cover_stands(tiles)
    assert plan.pool < free.pool and plan.cells <= plan.pool


def test_the_bin_lane_stays_bare_even_when_the_crop_footprints_are_released():
    from harvest.maps.farm_pond import SHIP_BIN_STAND_TILES

    final = fc.reserved_cells(2, index(2, 1, 25))
    assert final == frozenset(SHIP_BIN_STAND_TILES)
    assert fc.reserved_cells(2, index(2, 1, 5)) > final             # crops still ripening: footprints and lane stay out
    plan = fc.plan_cover_stands(open_field(box=(2, 20, 40, 50)), reserved=final, min_cells=1)
    for cx, cy in plan.stands:
        assert not any((cx + dx, cy + dy) in final for dx in (-1, 0, 1) for dy in (-1, 0, 1)) or plan.cells < 9 * len(plan.stands)


def test_counted_squares_are_not_planned_again_and_ragged_stands_need_the_final_fill():
    tiles = open_field(box=(2, 20, 7, 25))                  # a 6x6 patch: four whole stands
    for x in range(2, 5):
        for y in range(20, 23):
            tiles[y * W + x] = 0x70                          # one stand already grass
    assert len(fc.plan_cover_stands(tiles, 10).stands) == 3
    odd = open_field(box=(2, 20, 8, 25))                    # 7 wide: a one-column strip of 6 squares is left over
    assert len(fc.plan_cover_stands(odd, 10).stands) == 4    # the strip's 3-square stands are under the 6-square floor
    assert len(fc.plan_cover_stands(odd, 10, min_cells=1).stands) == 6   # the final fill takes them


def test_debris_that_needs_an_axe_or_hammer_is_left_alone():
    tiles = open_field()
    for x in range(10, 13):
        for y in range(30, 33):
            tiles[y * W + x] = 0x09                          # a stump
    plan = fc.plan_cover_stands(tiles, 500)
    assert plan.pool < 39 * 31 - 9                          # the nine stump squares are not in the pool
    for cx, cy in plan.stands:
        assert tiles[cy * W + cx] in (0x01, 0x02, 0x03, 0x04, 0x07, 0x08)


def test_the_hoe_pass_lists_only_stands_that_still_hold_bare_ground():
    bare = fc.plan_cover_stands(open_field())
    assert bare.unhoed == bare.stands and bare.unhoed_route == bare.route
    hoed = fc.plan_cover_stands(open_field(fill=0x07))
    assert hoed.stands == bare.stands and hoed.unhoed == () and hoed.unhoed_route == ()
    tiles = open_field(fill=0x07)
    cx, cy = bare.stands[3]
    tiles[(cy - 1) * W + cx - 1] = 0x01                          # four bare squares in one stand
    tiles[(cy - 1) * W + cx] = 0x01
    tiles[cy * W + cx - 1] = 0x02
    tiles[cy * W + cx] = 0x03
    one = fc.plan_cover_stands(tiles)
    assert one.unhoed == (bare.stands[3],)
    tiles[(cy - 1) * W + cx] = 0x07
    tiles[cy * W + cx] = 0x07                                    # two left: under MIN_TILL_CELLS
    assert fc.plan_cover_stands(tiles).unhoed == ()
    assert fc.plan_cover_stands(tiles, min_cells=1).unhoed == (bare.stands[3],)     # the final fill takes any


def test_waypoints_follow_the_route_and_end_on_the_first_stand():
    tiles = open_field()
    plan = fc.plan_cover_stands(tiles, 3)
    px = fc.waypoint_px(plan.route, hop=6)
    assert px[-1] == (plan.stands[0][0] * 16 + 8, plan.stands[0][1] * 16 + 8)
    assert fc.waypoint_px(()) == []


# ── the errand ───────────────────────────────────────────────────────

def morning(year=1, season=0, day=10, **over):
    return ram(**({"year": year, "season": season, "day": day, "tilemap": 0} | over))


def test_the_cash_path_never_builds_the_field(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "cash")
    e = fc.cover_errand(morning(), spendable_g=500_000, stock=0, shop_open=True, tiles=open_field())
    assert e.bags == 0 and e.reason == "strategy"


def test_surplus_buys_a_modest_load_and_counts_stock_already_held(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score")
    tiles = open_field()
    floor = fc.cash_floor_g(index(1, 0, 10))
    e = fc.cover_errand(morning(), spendable_g=floor + 70 * 500, stock=0, shop_open=True, tiles=tiles)
    # The whole field is wanted, but one purchase does not create a 200-bag backlog.
    assert e.behind == fc.MIN_COUNTED_TILES - fc.durable_counted(tiles)
    assert e.buy == e.bags == fc.TARGET_STOCK_BAGS and len(e.plan.stands) > 70 and not e.final_fill
    plenty = fc.cover_errand(morning(), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert plenty.buy == plenty.bags == fc.TARGET_STOCK_BAGS
    top_up = fc.cover_errand(morning(), spendable_g=floor + 70 * 500, stock=10, shop_open=True, tiles=tiles)
    assert top_up.buy == fc.TARGET_STOCK_BAGS - 10 and top_up.bags == fc.TARGET_STOCK_BAGS


def test_even_a_rich_late_farm_never_buys_a_stack_or_tops_up_a_backlog(monkeypatch):
    """ROM probe: 300 bags were charged and 255 held, so 45 bags (22,500G) vanished."""
    monkeypatch.setenv(rs.ENV_VAR, "score")
    tiles = open_field(box=BIG)
    assert len(fc.plan_cover_stands(tiles).stands) > fc.STACK_CAP
    e = fc.cover_errand(morning(year=2, season=1, day=29), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert e.buy == e.bags == fc.FINAL_FILL_TARGET_BAGS < fc.STACK_CAP == 255
    again = fc.cover_errand(morning(), spendable_g=9e9, stock=100, shop_open=True, tiles=tiles)
    assert again.buy == 0 and again.bags == 100


def test_a_small_load_waits_for_a_real_one_unless_it_finishes_the_field(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score")
    tiles = open_field()
    floor = fc.cash_floor_g(index(1, 0, 10))
    few = fc.cover_errand(morning(), spendable_g=floor + (fc.MIN_TRIP_BAGS - 1) * 500, stock=0, shop_open=True, tiles=tiles)
    assert few.buy == 0 and few.bags == 0 and few.reason == "no gold above the floor"
    enough = fc.cover_errand(morning(), spendable_g=floor + fc.MIN_TRIP_BAGS * 500, stock=0, shop_open=True, tiles=tiles)
    assert enough.buy == fc.MIN_TRIP_BAGS
    patch = open_field(box=(2, 20, 7, 25))                              # four stands: the last of the field
    last = fc.cover_errand(morning(), spendable_g=floor + 4 * 500, stock=0, shop_open=True, tiles=patch)
    assert last.buy == last.bags == 4


def test_gold_under_the_floor_buys_nothing_but_stock_in_hand_is_still_worked(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score")
    tiles = open_field()
    late = morning(year=2, season=1, day=15)                    # floor ~ 80,000G: 99,840 + 10,000 - 500 * 14 days
    poor = fc.cover_errand(late, spendable_g=50_000, stock=0, shop_open=True, tiles=tiles)
    assert poor.bags == 0 and poor.reason == "no gold above the floor"
    held = fc.cover_errand(late, spendable_g=50_000, stock=4, shop_open=True, tiles=tiles)
    assert held.bags == 4 and held.buy == 0
    closed = fc.cover_errand(late, spendable_g=500_000, stock=0, shop_open=False, tiles=tiles)
    assert closed.bags == 0 and closed.reason == "shop closed"


def test_outside_the_sowing_windows_and_when_complete_it_does_nothing(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score")
    tiles = open_field()
    assert fc.cover_errand(morning(year=0, season=1), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles).reason == "window"
    assert fc.cover_errand(morning(year=1, season=2), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles).reason == "window"
    done = open_field(fill=0x79, box=BIG)                           # every square already counted: 3,538 grass
    assert fc.durable_counted(done) >= fc.MIN_COUNTED_TILES
    assert fc.cover_errand(morning(), spendable_g=9e9, stock=0, shop_open=True, tiles=done).reason == "complete"
    assert fc.cover_errand(morning(), spendable_g=0, stock=7, shop_open=False, tiles=done).reason != "complete"   # paid for


def test_the_final_fill_releases_the_crop_footprints_but_not_the_lane(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score")
    tiles = open_field()
    e = fc.cover_errand(morning(year=2, season=1, day=25), spendable_g=900_000, stock=0, shop_open=True, tiles=tiles)
    assert e.final_fill and e.behind == fc.MIN_COUNTED_TILES - fc.durable_counted(tiles) and e.bags > 0
    early = fc.cover_errand(morning(year=2, season=1, day=5), spendable_g=900_000, stock=0, shop_open=True, tiles=tiles)
    assert not early.final_fill


def test_env_forces_the_field_on_or_off(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score")
    monkeypatch.setenv(rs.FIELD_COVER_ENV, "0")
    assert fc.cover_errand(morning(), spendable_g=9e9, stock=0, shop_open=True, tiles=open_field()).reason == "strategy"
    monkeypatch.setenv(rs.ENV_VAR, "cash")
    monkeypatch.setenv(rs.FIELD_COVER_ENV, "1")
    assert fc.cover_errand(morning(), spendable_g=9e9, stock=0, shop_open=True, tiles=open_field()).bags > 0


def test_presets_and_the_cover_alias():
    assert not rs.CASH.field_cover
    assert rs.SCORE.field_cover and rs.SCORE_HERD.field_cover and rs.PERFECT.field_cover
    assert rs.get("cover") is rs.SCORE
    assert rs.SCORE_HERD.pasture_max_tiles >= 400 and rs.PERFECT.pasture_max_tiles >= 400


def test_busy_mornings_keep_their_harvest_even_when_rich_or_in_the_last_week(monkeypatch):
    """A cover purchase never gains urgency from the approaching ending; paid bags can still be sown."""
    monkeypatch.setenv(rs.ENV_VAR, "score")
    tiles = open_field(box=BIG)
    floor = fc.cash_floor_g(index(1, 0, 10))
    mid = floor + 12 * 500
    light = fc.cover_errand(morning(), spendable_g=mid, stock=0, shop_open=True, tiles=tiles, ripe=fc.LIGHT_RIPE_TILES)
    assert light.buy == 12
    busy = fc.cover_errand(morning(), spendable_g=mid, stock=0, shop_open=True, tiles=tiles, ripe=fc.LIGHT_RIPE_TILES + 1)
    assert busy.buy == 0 and busy.bags == 0 and busy.reason.startswith("busy morning")
    big = fc.cover_errand(morning(), spendable_g=900_000, stock=0, shop_open=True, tiles=tiles, ripe=200)
    assert big.buy == 0
    # Bags already in hand are still sown tonight, trip or no trip.
    held = fc.cover_errand(morning(), spendable_g=mid, stock=12, shop_open=True, tiles=tiles, ripe=200)
    assert held.buy == 0 and held.bags == 12
    # The last four mornings follow the same harvest rule.
    last = fc.cover_errand(morning(year=2, season=1, day=26), spendable_g=900_000, stock=0, shop_open=True, tiles=tiles, ripe=200)
    assert last.buy == 0 and last.reason.startswith("busy morning")


# ── the pasture follows the herd it plans ────────────────────────────

def test_a_bigger_herd_plans_its_pasture_past_the_default_cap(monkeypatch):
    base = dict(year=1, season=0, day=5, weekday=2, num_cows=2, num_chickens=2, planted_grass=0)
    monkeypatch.setenv(rs.ENV_VAR, "cash")
    cash = econ.pasture_target_tiles(ram(**base))
    assert cash == econ.GRASS_PER_ANIMAL * (4 + 2)                  # unchanged: 16 per animal now owned, plus two
    monkeypatch.setenv(rs.ENV_VAR, "score_herd")
    herd = econ.pasture_target_tiles(ram(**base))
    # Size the pasture for all twelve planned cows before buying them.
    assert herd == econ.GRASS_PER_ANIMAL * (12 + 2 + 2) > econ.PASTURE_MAX_TILES
    year_one = econ.pasture_target_tiles(ram(**(base | dict(year=0))))
    assert year_one == econ.GRASS_PER_ANIMAL * (4 + 2)              # Year 1 cash is scarce: sized for what it owns


# ── the phases ───────────────────────────────────────────────────────

RICH = dict(weekday=2, money=150_000, num_cows=2, num_chickens=2, planted_grass=500, stored_grass=100)


def test_the_morning_buys_the_cover_bags_and_the_evening_sows_and_hoes_the_field(monkeypatch):
    from harvest.planner.day_phase_catalog import pasture_day_phases
    from harvest.planner.field_cover import field_cover_phases

    monkeypatch.setenv(rs.ENV_VAR, "score")
    data = put(morning(grass_seeds=0, **RICH), open_field())
    phases = pasture_day_phases(data)
    buy = next(p for p in phases if p.phase == "BUY_GRASS_SEEDS")
    assert buy.params["target_stock"] >= fc.MIN_TRIP_BAGS
    assert not [p for p in phases if p.phase in ("COVER_FIELD", "COVER_TILL")]      # the work waits for the evening
    assert field_cover_phases(data) == []                                            # no stock yet: nothing to sow
    stocked = put(morning(grass_seeds=6, **RICH), open_field())
    cover = field_cover_phases(stocked)
    assert [p.phase for p in cover] == ["COVER_FIELD", "COVER_TILL"]
    sow, hoe = cover
    everything = fc.plan_cover_stands(open_field(), reserved=fc.reserved_cells(1, index(1, 0, 10))).stands
    assert sow.params["bag_limit"] == 6 and len(sow.params["stands"]) == len(everything) > 6      # bags cap the sowing, not the list
    assert [tuple(s) for s in sow.params["stands"]] == list(everything) and sow.params["route_px"]
    assert hoe.params["till_only"] and [tuple(s) for s in hoe.params["stands"]] == list(everything)
    assert not sow.params["final_fill"]
    monkeypatch.setenv(rs.ENV_VAR, "cash")
    assert field_cover_phases(stocked) == []
    assert not [p for p in pasture_day_phases(data) if p.params.get("target_stock", 0) > 6]


def test_a_hoed_field_needs_only_the_sowing(monkeypatch):
    from harvest.planner.field_cover import field_cover_phases

    monkeypatch.setenv(rs.ENV_VAR, "score")
    hoed = put(morning(grass_seeds=6, **RICH), open_field(fill=0x07))
    assert [p.phase for p in field_cover_phases(hoed)] == ["COVER_FIELD"]


def test_till_ahead_can_be_disabled_without_limiting_paid_sowing(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score")
    data = put(morning(grass_seeds=6, **RICH), open_field())
    old = fc.field_cover_phases(data)
    monkeypatch.setenv("HARVEST_COVER_TILL_AHEAD", "0")
    new = fc.field_cover_phases(data)
    assert [p.phase for p in new] == ["COVER_FIELD"]
    assert new[0] == old[0]


def test_the_planner_builds_the_cover_tasks_with_the_chosen_stands(monkeypatch):
    from harvest.planner.field_cover import field_cover_phases
    from harvest.planner.day_phase_registry import TaskBuildContext, _build_grass
    from harvest.tasks.grass_planter import GrassPlantTask
    from retro_harness import WorldState

    monkeypatch.setenv(rs.ENV_VAR, "score")
    data = put(morning(grass_seeds=4, input_lock=1, **RICH), open_field())
    sow, hoe = field_cover_phases(data)
    world = WorldState(frame=0, ram=data, info={})
    planter = _build_grass(TaskBuildContext(), sow, world).tasks[-1]
    assert isinstance(planter, GrassPlantTask)
    assert planter.stands == [tuple(s) for s in sow.params["stands"]] and planter.bag_limit == 4
    assert planter.target_tiles == 0 and planter.bounds == (0, 0, 63, 63) and not planter.till_only
    planter.reset(world)
    assert planter._chunk_order == [(cx - 1, cy - 1) for cx, cy in planter.stands]
    tilling = _build_grass(TaskBuildContext(), hoe, world)
    hoer = tilling.tasks[-1]
    assert hoer.till_only and hoer.till_limit == 0 and hoer.bag_limit == 0 and hoer.max_failures >= 400
    assert hoer.stands == [tuple(s) for s in hoe.params["stands"]] and tilling.keep_last_reason
    hoer.reset(world)
    assert hoer._chunk_order == [(cx - 1, cy - 1) for cx, cy in hoer.stands]


# ── stamina ──────────────────────────────────────────────────────────

def test_cover_soaks_as_often_as_the_bar_runs_dry_while_it_makes_progress():
    from harvest.planner.day_phase_stamina import COVER_IDLE_PASSES, refresh_marks, spa_refresh_allowed

    idle = "stamina_low cleared=0 tilled=0 planted=0"
    busy = "stamina_low cleared=0 tilled=9 planted=8"
    for phase in ("COVER_FIELD", "COVER_TILL"):
        assert spa_refresh_allowed(phase, 0, set())
        assert spa_refresh_allowed(phase, 4, set())                        # a fully watered day's four do not count against it
        assert spa_refresh_allowed(phase, 63, {phase}, busy)
        assert spa_refresh_allowed(phase, 200, {phase}, "stamina_low cleared=0 tilled=45 planted=0")
        # A soak that failed on a blocked gate leaves the next pass idle: forgiven twice, then the phase gives up.
        done = refresh_marks(phase, set())
        assert done == {phase}
        for n in range(COVER_IDLE_PASSES):
            assert spa_refresh_allowed(phase, 5 + n, done, idle)
            done |= refresh_marks(phase, done, idle)
        assert not spa_refresh_allowed(phase, 9, done, idle)               # a wedged planter stops
        assert spa_refresh_allowed(phase, 9, done, busy)                   # progress always earns another soak
        # A restart with an empty bar (the last soak failed, so no pass ran and there is no reason) is idle too, or a blocked gate loops for ever.
        stuck = refresh_marks(phase, set())
        for n in range(COVER_IDLE_PASSES):
            assert spa_refresh_allowed(phase, 20 + n, stuck, "")
            stuck |= refresh_marks(phase, stuck, "")
        assert not spa_refresh_allowed(phase, 30, stuck, "")
        assert refresh_marks(phase, set(), "") == {phase}                  # the first soak of a phase is never idle
    assert not spa_refresh_allowed("PLANT_PASTURE", 2, set())                    # every other phase keeps its two


def test_the_cover_planter_stops_with_stamina_low_instead_of_swinging_a_dead_hoe():
    from harvest.tasks.grass_planter import GrassPlantTask
    from retro_harness import TaskStatus, WorldState

    def world(stamina):
        data = put(morning(input_lock=1, grass_seeds=3, max_stamina=100, stamina=stamina), open_field())
        return WorldState(frame=0, ram=data, info={})

    spent = world(0)
    for task in (
        GrassPlantTask(bounds=(0, 0, 63, 63), no_go_rects=[], stands=[(5, 23)], bag_limit=1),
        GrassPlantTask(bounds=(0, 0, 63, 63), no_go_rects=[], stands=[(5, 23)], till_only=True),
    ):
        task.reset(spent)
        done = task.step(spent)
        assert done.status == TaskStatus.SUCCESS and done.reason.startswith("stamina_low")
        rested = world(80)
        task.reset(rested)
        assert task.step(rested).status == TaskStatus.RUNNING
    # The pasture planter is untouched: no stand list, no stamina gate.
    pasture = GrassPlantTask(bounds=(13, 33, 27, 48), no_go_rects=[], target_tiles=48, bag_limit=2)
    pasture.reset(spent)
    assert not pasture.step(spent).reason.startswith("stamina_low")


def test_a_restart_walks_to_the_first_stand_with_work_left_not_to_the_first_of_the_list(monkeypatch):
    """ROM probe: restarts after a soak walked to the done first stand; every far stand then looked unreachable (162 skips, 94 of 253 bags sown)."""
    from harvest.planner.day_phase_registry import TaskBuildContext, _build_grass
    from harvest.tasks import grass_planter as gp
    from retro_harness import WorldState

    monkeypatch.setenv(rs.ENV_VAR, "score")
    gp._DEAD_STANDS.clear()
    tiles = open_field()
    plan = fc.plan_cover_stands(tiles, reserved=fc.reserved_cells(1, index(1, 0, 10)))
    first, second, third = plan.stands[:3]
    assert fc.first_open_stand(tiles, plan.stands) == first
    done = list(tiles)
    for cx, cy in (first,):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                done[(cy + dy) * W + cx + dx] = 0x70                       # the first stand is grass
    assert fc.first_open_stand(done, plan.stands) == second
    assert fc.first_open_stand(done, plan.stands, skip=[second]) == third      # a dead stand is not the way in
    hoed = [0x07 if t == 0x01 else t for t in done]
    assert fc.first_open_stand(hoed, plan.stands, till=True) is None         # nothing bare is left to hoe
    assert fc.first_open_stand(hoed, plan.stands) == second                  # but hoed soil still wants its seed
    data = put(morning(grass_seeds=3, input_lock=1, **RICH), done)
    px = fc.restart_route_px(data, plan.stands)
    assert px and px[-1] == (second[0] * 16 + 8, second[1] * 16 + 8)
    spec = fc.field_cover_phases(data)[0]
    nav = _build_grass(TaskBuildContext(), spec, WorldState(frame=0, ram=data, info={})).tasks[1]
    assert (second[0] * 16 + 8, second[1] * 16 + 8) in [wp.target_px for wp in nav.waypoints]
    assert (first[0] * 16 + 8, first[1] * 16 + 8) not in [wp.target_px for wp in nav.waypoints]


def test_a_stand_that_times_out_is_dead_for_the_day_and_skipped_after_every_soak():
    """ROM probe: one unreachable stand cost nine 1,200-step tile timeouts, again after every soak: 400 timeouts, 480k frames, nothing planted."""
    from harvest.tasks import grass_planter as gp
    from retro_harness import TaskStatus, WorldState

    data = put(morning(input_lock=1, grass_seeds=3, max_stamina=100, stamina=80), open_field())
    world = WorldState(frame=0, ram=data, info={})
    stands = [(40, 24), (37, 24), (34, 24)]
    task = gp.GrassPlantTask(bounds=(0, 0, 63, 63), no_go_rects=[], stands=stands, bag_limit=3)
    gp._DEAD_STANDS.clear()
    task.reset(world)
    task._chunk_origin = (39, 23)
    task._chunk_index = 0
    task._target_tile = (40, 24)
    task._steps_on_target = task.max_steps_per_target + 1
    task._total_steps = 5
    task.step(world)
    today = task._today(data)
    assert gp.dead_stands(today) == {(39, 23)}
    assert task._target_tile is None
    again = gp.GrassPlantTask(bounds=(0, 0, 63, 63), no_go_rects=[], stands=stands, bag_limit=3)
    again.reset(world)                                                       # the restart after a soak
    again._handle_scan(data)
    assert again._chunk_origin != (39, 23)                                   # stand 0 is skipped without a single swing, retry pass included
    assert gp.dead_stands((today[0], today[1], today[2] + 1)) == set()       # tomorrow it is tried again, once
    assert gp.dead_stands(today) == set()                                    # (and yesterday's memory is gone)
    gp._DEAD_STANDS.clear()


def test_the_hoe_pass_walks_the_planners_stands_not_the_whole_grid():
    from harvest.tasks.grass_planter import GrassPlantTask
    from retro_harness import WorldState

    data = put(morning(input_lock=1, grass_seeds=0, max_stamina=100, stamina=80), open_field())
    world = WorldState(frame=0, ram=data, info={})
    ordered = [(40, 24), (37, 24), (37, 27)]
    cover = GrassPlantTask(bounds=(0, 0, 63, 63), no_go_rects=[], stands=ordered, till_only=True)
    cover.reset(world)
    assert cover._chunk_order == [(39, 23), (36, 23), (36, 26)]
    grid = GrassPlantTask(bounds=(0, 0, 8, 8), no_go_rects=[], till_only=True, till_limit=4)
    grid.reset(world)
    assert len(grid._chunk_order) == 9                                      # the old tiling is untouched


def test_the_sequence_hands_the_planters_stamina_reason_to_the_day_plan():
    from harvest.planner.day_phase_catalog import PhaseSpec
    from harvest.planner.day_phase_registry import TaskBuildContext, _build_grass
    from harvest.tasks.primitives import TaskSequence
    from retro_harness import Task, TaskResult, TaskStatus, WorldState

    class Done(Task):
        def __init__(self, reason):
            self.name, self.reason = "done", reason

        def reset(self, world): ...
        def can_start(self, world): return True
        def step(self, world): return TaskResult(status=TaskStatus.SUCCESS, reason=self.reason)

    world = WorldState(frame=0, ram=ram(), info={})
    kept = TaskSequence(tasks=[Done("stamina_low tilled=3")], keep_last_reason=True)
    kept.reset(world)
    assert kept.step(world).reason == "stamina_low tilled=3"
    plain = TaskSequence(tasks=[Done("stamina_low tilled=3")])
    plain.reset(world)
    assert plain.step(world).reason == "sequence complete"
    spec = PhaseSpec("COVER_FIELD", "grass", {"stands": [[5, 23]], "bag_limit": 1})
    assert _build_grass(TaskBuildContext(), spec, world).keep_last_reason
    assert not _build_grass(TaskBuildContext(), PhaseSpec("PLANT_PASTURE", "grass"), world).keep_last_reason


def test_the_evening_sows_every_bag_in_hand_so_no_stack_is_carried_overnight(monkeypatch):
    """ROM A/B: ~11 leftover bags carried to the morning broke CROP_ESTABLISH (0x02 not in carry pair)."""
    monkeypatch.setenv(rs.ENV_VAR, "score")
    tiles = open_field()
    night = fc.cover_errand(morning(), spendable_g=0, stock=90, shop_open=False, tiles=tiles)
    assert night.bags == 90 and night.buy == 0 and len(night.plan.stands) > 90
    over = fc.cover_errand(morning(), spendable_g=0, stock=500, shop_open=False, tiles=tiles)
    assert over.bags == len(over.plan.stands)                                # never more bags than stands: the rest is stock


def test_the_grass_buy_timeout_follows_the_bags(monkeypatch):
    from harvest.planner.day_phase_catalog import pasture_day_phases

    monkeypatch.setenv(rs.ENV_VAR, "score")
    data = put(morning(grass_seeds=0, **RICH), open_field(box=BIG))
    buy = next(p for p in pasture_day_phases(data) if p.phase == "BUY_GRASS_SEEDS")
    assert buy.params["target_stock"] == fc.TARGET_STOCK_BAGS
    assert buy.params["timeout"] >= 4_000 + fc.BAG_BUY_FRAMES * buy.params["target_stock"] > 14_000


@pytest.mark.rom
def test_late_cover_route_reaches_the_unfinished_field_past_the_south_well(monkeypatch):
    """The 851 ending retried this same well-crossing route for weeks without sowing."""
    from harvest.paths import GAME_DIR
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE
    from harvest.planner.day_phase_types import DayPlannerPolicy
    from harvest.planner.tasks.multi_nav import MultiMapNavTask
    from harvest.maps.map_config import Waypoint
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import make_action
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

    pin = "PR_ending12_20261111_s1_Y3_Spring_D9"
    if not (GAME_DIR / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is not present")
    monkeypatch.setenv(rs.ENV_VAR, "score_herd")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.cover_well_route"))
    try:
        env.reset()
        tiles = fc.farm_tiles(env.get_ram())
        plan = fc.plan_cover_stands(tiles, reserved=fc.reserved_cells(2, index(2, 0, 9)))
        assert plan.stands
        exit_task = DayPlanTask(phase_sequence=[EXIT_TO_FARM_PHASE],
                               policy=DayPlannerPolicy(include_end_day=False))
        walk = MultiMapNavTask(waypoints=[Waypoint(0, px, radius=8) for px in fc.waypoint_px(plan.route)],
                               timeout=9000, auto_slice=True)
        frame = 0
        for task in (exit_task, walk):
            task.reset(WorldState(frame, env.get_ram(), {}, None))
            for _ in range(12_000):
                result = task.step(WorldState(frame, env.get_ram(), {}, None))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(result.action.action if result.action else make_action())
                frame += 1
            assert result.status == TaskStatus.SUCCESS, result.reason
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize("pin,year,day", [
    ("PR_e2es4_Y2_Spring_D3", 1, 3),
    ("PR_ending12_20261111_s1_Y3_Spring_D9", 2, 9),
])
def test_year_two_small_cover_load_is_bought_and_sown_through_the_day_plan(monkeypatch, pin, year, day):
    """Natural Y2 and late Y3 farms: native payment and durable cover without edited RAM.

    This tests the morning pasture errand and the evening sowing component; the whole-field hoe pass and the
    intervening crop work remain outside this bounded receipt.
    """
    import json
    from harvest.core.ram_catalog import read_ram_value
    from harvest.core.scene import peddler_decline_action
    from harvest.paths import GAME_DIR, PROJECT_DIR
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE, pasture_day_phases
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from harvest.planner.tasks.home_recover import sanitize_action
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import make_action
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

    if not (GAME_DIR / f"{pin}.state").exists():
        pytest.skip(f"local pin {pin} is not present")
    monkeypatch.setenv(rs.ENV_VAR, "score")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.cover_load"))
    frames = 0

    def play(phases, limit):
        nonlocal frames
        from harvest.planner.day_phase_types import DayPlannerPolicy

        task = DayPlanTask(phase_sequence=phases, policy=DayPlannerPolicy(include_end_day=False))
        task.reset(WorldState(frame=frames, ram=env.get_ram(), info={}, obs=None))
        for _ in range(limit):
            # These two guards belong to MultiDayPlannerTask, outside the day-plan component under test.
            declined = peddler_decline_action(env.get_ram(), frames)
            if declined is not None:
                env.step(declined)
                frames += 1
                continue
            result = task.step(WorldState(frame=frames, ram=env.get_ram(), info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                assert result.status == TaskStatus.SUCCESS, result.reason
                return
            env.step(sanitize_action(result.action.action if result.action else make_action(), env.get_ram()))
            frames += 1
        pytest.fail(f"cover component exceeded {limit} frames: {result.reason}")

    try:
        env.reset()
        before = {k: int(read_ram_value(env.get_ram(), k)) for k in ("money", "grass_seeds")}
        initial_tiles = fc.durable_counted(fc.farm_tiles(env.get_ram()))
        phases = pasture_day_phases(env.get_ram())
        buy = next(p for p in phases if p.phase == "BUY_GRASS_SEEDS")
        assert 8 <= buy.params["target_stock"] <= fc.TARGET_STOCK_BAGS + 2
        play([EXIT_TO_FARM_PHASE, buy], 40_000)
        purchased = int(read_ram_value(env.get_ram(), "grass_seeds")) - before["grass_seeds"]
        paid = before["money"] - int(read_ram_value(env.get_ram(), "money"))
        assert purchased == buy.params["target_stock"] - before["grass_seeds"]
        assert paid == purchased * 500
        assert int(read_ram_value(env.get_ram(), "money")) >= fc.cash_floor_g(fc.day_index(year, 0, day))
        # The real planner reaches cover after the day's work, well after the shop return has settled.
        play([fc.PhaseSpec("WAIT_FOR_COVER", "wait_until_time",
                           {"target_hour": 18, "timeout": 20_000})], 25_000)
        # The sowing phase itself is unbounded by the purchase target: it works every paid bag in hand.
        evening = fc.field_cover_phases(env.get_ram())
        assert evening, {k: int(read_ram_value(env.get_ram(), k)) for k in ("hour", "tilemap", "grass_seeds")}
        sow = evening[0]
        play([sow], 250_000)
        remaining = int(read_ram_value(env.get_ram(), "grass_seeds"))
        gained = fc.durable_counted(fc.farm_tiles(env.get_ram())) - initial_tiles
        assert remaining == 0 and gained >= 8 * fc.MIN_STAND_CELLS
        assert tuple(int(read_ram_value(env.get_ram(), k)) for k in ("year", "season", "day")) == (year, 0, day)
        audit = env.audit()
        assert audit.ram_writes == audit.mid_run_loads == 0
        receipt = dict(pin=pin, bought=purchased, paid_g=paid, remaining=remaining,
                       durable_tiles_gained=gained, frames=frames, ram_writes=audit.ram_writes,
                       mid_run_loads=audit.mid_run_loads, scope="morning buy + cover sowing components")
        out = PROJECT_DIR / f"recordings/cover_policy_20261112/{pin}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(receipt)
    finally:
        env.close()


def _healthy_hay_cut_cover_morning(hour=8, hay=46):
    from tests.cow_test_helpers import set_cow

    data = put(morning(year=2, season=1, day=22, weekday=5, hour=hour, money=462_290,
                       num_cows=12, num_chickens=11, planted_grass=1919,
                       stored_grass=hay, grass_seeds=0), open_field(box=BIG))
    for y in range(36, 39):
        for x in range(14, 23):
            data[ADDR_MAP + y * W + x] = 0x79
    for slot in range(12):
        set_cow(data, slot, happiness=255)
    return data


def test_healthy_hay_cut_still_buys_a_surplus_funded_cover_load(monkeypatch):
    from harvest.planner.day_phase_catalog import pasture_day_phases

    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    phases = pasture_day_phases(_healthy_hay_cut_cover_morning())
    names = [p.phase for p in phases]
    assert names == ['BUY_GRASS_SEEDS', 'CUT_FODDER']
    assert phases[0].params['target_stock'] == fc.FINAL_FILL_TARGET_BAGS      # Y3 Summer D22 is a final-fill day
    assert phases[0].params['latest_start_hour'] == 15


def test_healthy_hay_cut_allows_an_early_afternoon_cover_trip_but_keeps_the_deadline(monkeypatch):
    from harvest.planner.day_phase_catalog import pasture_day_phases

    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    early = [p.phase for p in pasture_day_phases(_healthy_hay_cut_cover_morning(hour=14), afternoon=True)]
    assert early == ['BUY_GRASS_SEEDS', 'CUT_FODDER']
    late = [p.phase for p in pasture_day_phases(_healthy_hay_cut_cover_morning(hour=15), afternoon=True)]
    assert late == ['CUT_FODDER']
    empty = [p.phase for p in pasture_day_phases(_healthy_hay_cut_cover_morning(hay=0))]
    assert empty == ['CUT_FODDER', 'BUY_GRASS_SEEDS']


def test_healthy_hay_cut_keeps_harvest_before_the_cover_buy_and_cut(monkeypatch):
    from harvest.planner.day_plan_phases import build_outdoor_day_phases_from_ram

    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    data = _healthy_hay_cut_cover_morning()
    data[ADDR_MAP + 29 * W + 13] = 0x52
    names = [p.phase for p in build_outdoor_day_phases_from_ram(data)]
    assert names.index('HARVEST_ROUTE') < names.index('BUY_GRASS_SEEDS') < names.index('CUT_FODDER')


def test_cover_buy_rechecks_its_start_deadline_after_the_harvest(monkeypatch):
    from types import SimpleNamespace
    from harvest.planner.day_phase_catalog import pasture_day_phases
    from harvest.planner.day_phase_registry import _build_shop_buy
    from retro_harness import WorldState, TaskStatus

    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    buy = next(p for p in pasture_day_phases(_healthy_hay_cut_cover_morning()) if p.phase == 'BUY_GRASS_SEEDS')
    world = WorldState(0, _healthy_hay_cut_cover_morning(hour=15), {}, None)
    task = _build_shop_buy(SimpleNamespace(social_visits=set()), buy, world)
    task.reset(world)
    result = task.step(world)
    assert result.status == TaskStatus.SUCCESS and 'no_work:shop start deadline' in result.reason


@pytest.mark.rom
def test_native_twelve_cow_hay_cut_buys_and_sows_cover_on_the_same_day(monkeypatch):
    """Natural D22: full animal prefix, then the grass components; crop work is outside this receipt."""
    import json
    from harvest.core.ram_catalog import read_ram_value
    from harvest.paths import GAME_DIR, PROJECT_DIR
    from harvest.planner.day_phase_catalog import pasture_day_phases
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    from harvest.planner.day_phase_types import DayPlannerPolicy
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import make_action
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

    pin = 'PR_cover_d20_d25_20261112_s0_Y3_Summer_D22'
    if not (GAME_DIR / f'{pin}.state').exists():
        pytest.skip(f'missing natural pin {pin}')
    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all('harvest.cover_cut'))
    frame = 0

    def play(phases, limit=120000):
        nonlocal frame
        task = DayPlanTask(phase_sequence=list(phases), policy=DayPlannerPolicy(include_end_day=False))
        task.reset(WorldState(frame, env.get_ram(), {}, None))
        for _ in range(limit):
            result = task.step(WorldState(frame, env.get_ram(), {}, None))
            if result.status != TaskStatus.RUNNING:
                assert result.status == TaskStatus.SUCCESS, (frame, result.reason)
                return task.phase_results
            env.step(result.action.action if result.action else make_action())
            frame += 1
        pytest.fail(f'cover component timed out: {task.phase_text}')

    try:
        env.reset()
        initial = fc.durable_counted(fc.farm_tiles(env.get_ram()))
        before_money = int(read_ram_value(env.get_ram(), 'money'))
        before_shipments = rs.shipped_totals(env.get_ram())
        morning = MultiDayPlannerTask()._build_day_task(WorldState(0, env.get_ram(), {}, None))
        at = next(i for i, phase in enumerate(morning.phases) if phase.phase == 'DYNAMIC_OUTDOOR_PLAN')
        play(morning.phases[:at])
        assert int(read_ram_value(env.get_ram(), 'fed_cows_n')) == 12
        hay_after_feeding = int(read_ram_value(env.get_ram(), 'stored_grass'))
        phases = pasture_day_phases(env.get_ram())
        assert [p.phase for p in phases] == ['BUY_GRASS_SEEDS', 'CUT_FODDER']
        rows = play(phases)
        bought = int(read_ram_value(env.get_ram(), 'grass_seeds'))
        assert bought == fc.TARGET_STOCK_BAGS
        assert int(read_ram_value(env.get_ram(), 'money')) == before_money - bought * 500
        assert int(read_ram_value(env.get_ram(), 'stored_grass')) > hay_after_feeding
        assert any(r['phase'] == 'CUT_FODDER' and r['status'] == 'success' for r in rows)
        evening = fc.field_cover_phases(env.get_ram())
        assert evening and evening[0].phase == 'COVER_FIELD'
        play(evening[:1], limit=300000)
        after = fc.durable_counted(fc.farm_tiles(env.get_ram()))
        assert after - initial >= bought  # final fill includes partial edge stands, down to one tile per bag
        assert int(read_ram_value(env.get_ram(), 'grass_seeds')) == 0
        assert rs.shipped_totals(env.get_ram()) == before_shipments
        assert int(read_ram_value(env.get_ram(), 'day')) == 22
        audit = env.audit()
        assert audit.ram_writes == audit.mid_run_loads == 0
        receipt = dict(pin=pin, frames=frame, bought=bought, paid_g=bought * 500,
                       durable_before=initial, durable_after=after, gained=after-initial,
                       counted_after=fc.counted(fc.farm_tiles(env.get_ram())),
                       hay_after_feeding=hay_after_feeding, hay_final=int(read_ram_value(env.get_ram(), 'stored_grass')),
                       ram_writes=audit.ram_writes, mid_run_loads=audit.mid_run_loads,
                       scope='full animal prefix + cover buy/cut/sow components; no crop work')
        out = PROJECT_DIR / 'recordings/cover_policy_20261112/hay_cut_and_cover.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2)+'\n')
        print(receipt)
    finally:
        env.close()


@pytest.mark.rom
def test_last_allowed_cover_shop_hour_reaches_the_counter_before_closing(monkeypatch):
    """A native bin departure at ROM14:14 (14:56) validates the <15 start guard."""
    import json
    from harvest.core.ram_catalog import read_ram_value
    from harvest.paths import PROJECT_DIR
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    from harvest.planner.day_phase_types import DayPlannerPolicy
    from harvest.planner.tasks.inventory_time import WaitUntilTimeTask
    from harvest.planner.tasks.navigation import NavTask
    from harvest.tasks.buy_seeds import BuyShelfItemTask
    from harvest.tasks.nav import Point
    from tests.test_crop_quota_policy import _rom_env, _run_native

    monkeypatch.setenv(rs.ENV_VAR, 'score_herd')
    pin = 'PR_cover_d20_d25_20261112_s0_Y3_Summer_D22'
    env = _rom_env(pin, 'harvest.late_cover_shop')
    frames = 0
    try:
        frames += _run_native(env, DayPlanTask(phase_sequence=[EXIT_TO_FARM_PHASE],
                                               policy=DayPlannerPolicy(include_end_day=False)))
        frames += _run_native(env, NavTask(target_px=Point(185, 492), radius=4, soft_radius=4))
        frames += _run_native(env, WaitUntilTimeTask(target_hour=14, target_minute=14, timeout=15000))
        fields = ('hour', 'minute', 'tilemap', 'player_x', 'player_y', 'money', 'grass_seeds')
        before = {k: int(read_ram_value(env.get_ram(), k)) for k in fields}
        frames += _run_native(env, BuyShelfItemTask(stock_field='grass_seeds', bags=24, timeout=30000))
        after = {k: int(read_ram_value(env.get_ram(), k)) for k in fields}
        assert after['grass_seeds'] - before['grass_seeds'] == 24
        assert before['money'] - after['money'] == 12000
        assert after['tilemap'] == 0
        audit = env.audit()
        assert audit.ram_writes == audit.mid_run_loads == 0
        receipt = dict(pin=pin, frames=frames, before=before, after=after, bought=24, paid_g=12000,
                       ram_writes=audit.ram_writes, mid_run_loads=audit.mid_run_loads,
                       entry_steps=['load natural D22 morning once', 'native house exit', 'native walk to bin',
                                    'native outdoor wait until ROM14:14'],
                       scope='native bin departure at ROM14:14, grass purchase and farm return')
        out = PROJECT_DIR / 'recordings/cover_policy_20261112/bin_to_shop_1456.json'
        out.write_text(json.dumps(receipt, indent=2)+'\n')
        print(receipt)
    finally:
        env.close()


def test_final_fill_days_load_more_bags_than_a_building_day(monkeypatch):
    """`y3h`: the last nine mornings sowed ragged one- and two-cell stands, so 24 bags counted ~35 tiles a day and
    342 tiles were still short at the tally. The evening hoes ~110 tiles whatever the bag count; bags are the limit."""
    monkeypatch.setenv(rs.ENV_VAR, "score")
    monkeypatch.setenv(fc.EARLY_COVER_ENV, "0")        # the default loads 60 bags all Year 3 Summer (tests/test_early_cover.py)
    tiles = open_field(box=BIG)
    building = fc.cover_errand(morning(year=2, season=1, day=10), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert not building.final_fill and building.buy == fc.TARGET_STOCK_BAGS
    final = fc.cover_errand(morning(year=2, season=1, day=21), spendable_g=9e9, stock=0, shop_open=True, tiles=tiles)
    assert final.final_fill and final.buy == final.bags == fc.FINAL_FILL_TARGET_BAGS > fc.TARGET_STOCK_BAGS
    top_up = fc.cover_errand(morning(year=2, season=1, day=21), spendable_g=9e9, stock=30, shop_open=True, tiles=tiles)
    assert top_up.buy == fc.FINAL_FILL_TARGET_BAGS - 30 and top_up.bags == fc.FINAL_FILL_TARGET_BAGS
    poor = fc.cover_errand(morning(year=2, season=1, day=21), spendable_g=fc.cash_floor_g(index(2, 1, 21)) + 40 * 500,
                           stock=0, shop_open=True, tiles=tiles)
    assert poor.buy == 40
