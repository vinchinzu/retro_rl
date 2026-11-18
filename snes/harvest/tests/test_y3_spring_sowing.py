"""Year 3 Spring is the last window for potatoes and turnips: sites, bags, the weekend lookahead, the court yield.

``fixD`` (fresh power-on, 977) shipped potato 386 and turnip 390 of 496. Its Y3 Spring journal showed twelve sites
only, one crop bought per trip for half of them, no bag bought on the Friday before a shut weekend, and a D2 court
that took the seed window. Each test below goes red on the code before those changes.
"""
from __future__ import annotations

import pytest

from harvest.maps import farm_pond
from harvest.planner import run_strategy as rs
from harvest.planner import social_calendar as social
from tests.layout_markers import legacy_rings_only
from tests.test_run_strategy import set_field, set_tile, world
from tests.test_social_calendar import weekday, write

FRIDAY = 5
FIELD = (2, 3, 62, 60)


@pytest.fixture(autouse=True)
def score_herd_with_court(monkeypatch):
    monkeypatch.setenv(rs.ENV_VAR, "score_herd")
    monkeypatch.setenv(social.ENV_VAR, "1")
    monkeypatch.delenv(farm_pond.Y3_EXTRA_SITES_ENV, raising=False)
    monkeypatch.delenv(farm_pond.Y3_SUMMER_EXTRA_ENV, raising=False)
    monkeypatch.delenv(farm_pond.SOON_RIPE_ENV, raising=False)
    monkeypatch.delenv(rs.Y3_ALL_BAGS_ENV, raising=False)


def _fill(ram, center, tile_id, year=2):
    for x, y in farm_pond.plot_crop_tiles(center, year):
        set_tile(ram, x, y, tile_id)


@legacy_rings_only
def test_year_three_summer_keeps_only_the_asked_extra_sites(monkeypatch):
    """``HARVEST_Y3_SUMMER_EXTRA_SITES=N`` keeps N extras through Summer (corn and tomatoes); Spring and Years 1-2 never change."""
    assert len(farm_pond.active_plot_centers(2, 1)) == 12                      # default: the old twelve
    monkeypatch.setenv(farm_pond.Y3_SUMMER_EXTRA_ENV, "4")
    assert len(farm_pond.active_plot_centers(2, 1)) == 16
    assert len(farm_pond.active_plot_centers(2, 0)) == 24                      # Spring already has all of them
    assert len(farm_pond.active_plot_centers(1, 1)) == 12                      # Year 2 is untouched
    monkeypatch.setenv(farm_pond.Y3_SUMMER_EXTRA_ENV, "99")
    assert len(farm_pond.active_plot_centers(2, 1)) == 24                      # capped at the twelve extras
    monkeypatch.setenv(farm_pond.Y3_EXTRA_SITES_ENV, "2")
    assert len(farm_pond.active_plot_centers(2, 1)) == 14                      # never more than Year 3 sows at all


@legacy_rings_only
def test_year_three_sows_eight_more_sites_than_year_two(monkeypatch):
    assert [len(farm_pond.active_plot_centers(year)) for year in (0, 1, 2)] == [9, 12, 24]
    assert len(farm_pond.active_plot_centers(2, 0)) == 24 and len(farm_pond.active_plot_centers(2, 1)) == 12   # Summer: twelve
    monkeypatch.setenv(farm_pond.Y3_EXTRA_SITES_ENV, "0")
    assert len(farm_pond.active_plot_centers(2)) == 12
    monkeypatch.setenv(rs.ENV_VAR, "cash")
    assert len(farm_pond.active_plot_centers(2)) == 9


@legacy_rings_only
def test_the_extra_sites_are_clear_of_each_other_and_the_lanes():
    sites = farm_pond.active_plot_centers(2)
    assert len(set(sites)) == len(sites)
    left, top, right, bottom = FIELD
    for i, (cx, cy) in enumerate(sites):
        assert left + 2 <= cx <= right - 2 and top + 2 <= cy <= bottom - 2, (cx, cy)
        for other in sites[i + 1:]:
            assert max(abs(cx - other[0]), abs(cy - other[1])) >= 4, ((cx, cy), other)   # rings never touch
    for center in farm_pond.SCORE_Y3_EXTRA_CENTERS:
        cells = {(center[0] + dx, center[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        assert not cells & farm_pond.FARM_NO_GO_TILES, center
        assert not cells & set(farm_pond.SHIP_BIN_LANE_TILES), center
        assert all(x != 6 and y not in (33, 37) for x, y in cells), center        # the hard-coded walk lanes
    from harvest.planner.field_cover import reserved_cells

    reserved = reserved_cells(2, 0)                       # the whole-field cover leaves every site's 5x5 alone
    for cx, cy in sites:
        assert {(cx + dx, cy + dy) for dx in range(-2, 3) for dy in range(-2, 3)} <= reserved, (cx, cy)


def test_year_three_summer_holds_back_rings_for_the_crop_that_has_none_standing():
    """y3m Summer D1: 17 corn bags for 17 empty sites left tomatoes (26 short of 496) no ring, so none ever grew."""
    w = world(season=1, day=1, shipped={"corn": 281, "tomato": 470})
    set_field(w.ram, "year", 2)
    corn = rs.seed_bags_for_quota(w.ram, "corn", 17, 0)
    assert 0 < corn < 17
    for x in range(10, 14):                                    # the corn bags are sown: corn stands, tomatoes do not
        for y in range(40, 52):
            set_tile(w.ram, x, y, 0x3A)
    tomato = rs.seed_bags_for_quota(w.ram, "tomato", 17 - corn, 0)
    assert tomato >= 1 and corn + tomato <= 17
    set_field(w.ram, "year", 1)
    assert rs.seed_bags_for_quota(w.ram, "tomato", 3, 0) == 2      # Year 2 keeps its proportional split


def test_year_three_spring_buys_a_bag_for_every_empty_site_of_the_crop_it_picks():
    """fixD D3 bought 7 turnip bags for 12 empty sites: one crop a trip, so the other five waited for D6/D7."""
    w = world(season=0, day=3)
    set_field(w.ram, "year", 2)
    assert rs.seed_bags_for_quota(w.ram, "turnip", 12, 0) == 12
    assert rs.seed_bags_for_quota(w.ram, "turnip", 12, 5) == 7            # stock is still taken off
    set_field(w.ram, "year", 1)
    assert rs.seed_bags_for_quota(w.ram, "turnip", 12, 0) == 6            # Year 2 keeps its proportional split


def test_year_three_spring_bags_never_pass_the_crops_remaining_demand():
    w = world(season=0, day=3, shipped={"turnip": 480})
    set_field(w.ram, "year", 2)
    assert rs.seed_bags_for_quota(w.ram, "turnip", 12, 0) == 2            # 16 crops left = two rings


def test_year_three_reserves_one_spare_ring_against_tiles_that_never_ship():
    """Preview y3k_a stopped buying with 497 turnips committed and shipped 494: a sown tile can fail to ripen."""
    def morning(year):
        w = world(season=0, day=22, shipped={"turnip": 440})
        set_field(w.ram, "year", year)
        for x in range(10, 18):                                 # 56 standing turnip tiles = 496 committed
            for y in range(40, 47):
                set_tile(w.ram, x, y, 0x66)
        return w.ram

    year_two, year_three = morning(1), morning(2)
    assert rs.spring_committed_totals(year_two)["turnip"] == 440 + 56
    assert rs.spring_committed_totals(year_three)["turnip"] == 440 + 56 - rs.Y3_STANDING_SLACK
    assert rs.Y3_STANDING_SLACK == 24                                   # three spare rings
    assert rs.seed_bags_for_quota(year_three, "turnip", 5, 0) == 3      # the spare rings are still bought
    assert rs.seed_bags_for_quota(year_two, "turnip", 5, 0) == 0


def _ring_morning(*, day, weekday_, tile, year=2):
    ram = weekday(season=0, day=day, hour=6, weekday=weekday_)
    write(ram, "year", year)
    for center in farm_pond.active_plot_centers(year):
        _fill(ram, center, 0x54 if center != (13, 28) else tile, year)
    return ram


@legacy_rings_only
def test_a_ring_ripening_over_a_shut_weekend_gets_its_bag_on_the_friday():
    """fixD Y3 Spring D24 (Friday): six turnip rings ripened on D25, the shop was shut D25-D26 and D27 is past the last sowing."""
    friday = _ring_morning(day=24, weekday_=FRIDAY, tile=0x6A)               # two watered nights from ripe (0x6E)
    assert farm_pond.pocket_plant_targets(friday, replant_ripe=True) == ((13, 28),)
    assert farm_pond.pocket_plant_targets(friday) == ()                       # nothing is empty yet
    wednesday = _ring_morning(day=24, weekday_=3, tile=0x6A)
    assert farm_pond.pocket_plant_targets(wednesday, replant_ripe=True) == ()  # the shop opens tomorrow
    young = _ring_morning(day=24, weekday_=FRIDAY, tile=0x64)                # four nights away: ripe after Monday
    assert farm_pond.pocket_plant_targets(young, replant_ripe=True) == ()


@legacy_rings_only
def test_the_weekend_lookahead_stops_where_a_bag_can_no_longer_ripen_and_outside_year_three(monkeypatch):
    late = _ring_morning(day=26, weekday_=FRIDAY, tile=0x6A)                  # D27 and D28 are past SPRING_LAST_SOW
    assert farm_pond.pocket_plant_targets(late, replant_ripe=True) == ()
    year_two = _ring_morning(day=24, weekday_=FRIDAY, tile=0x6A, year=1)
    assert farm_pond.pocket_plant_targets(year_two, replant_ripe=True) == ()
    friday = _ring_morning(day=24, weekday_=FRIDAY, tile=0x6A)
    monkeypatch.setenv(farm_pond.SOON_RIPE_ENV, "0")
    assert farm_pond.pocket_plant_targets(friday, replant_ripe=True) == ()


def test_the_eve_of_a_festival_counts_as_a_shut_day():
    ram = weekday(season=0, day=22, hour=6, weekday=3)                          # Spring D23 is the Flower Festival
    write(ram, "year", 2)
    assert farm_pond.shop_closed_days_ahead(ram) == 1
    write(ram, "weekday", FRIDAY)
    write(ram, "day", 24)
    assert farm_pond.shop_closed_days_ahead(ram) == 2


def test_court_yields_to_the_opening_seed_trip_in_year_three_only():
    """fixD Y3 Spring D2: Ellen's court ran 4,189 frames first and the seed window shut at 13:11; twelve sites sat bare a day."""
    d2 = weekday(season=0, day=2, hour=6, weekday=4)
    write(d2, "year", 2)
    assert social.town_court_phase(d2) is not None                           # she would be courted on any other weekday
    assert social.spring_sowing_trip_first(d2)
    assert social.social_phase(d2) is None
    for quiet in (dict(year=1), dict(day=4), dict(weekday=6)):
        other = weekday(season=0, day=quiet.get("day", 2), hour=6, weekday=quiet.get("weekday", 4))
        write(other, "year", quiet.get("year", 2))
        assert not social.spring_sowing_trip_first(other), quiet
    # Once the sites hold crops the court keeps its day.
    for center in farm_pond.active_plot_centers(2):
        _fill(d2, center, 0x54)
    assert not social.spring_sowing_trip_first(d2)


@pytest.mark.rom
@legacy_rings_only
def test_every_year_three_extra_site_takes_a_full_bag_from_the_natural_morning(monkeypatch):
    """ROM: from the fixD Y3 Spring D2 pin, buy turnip bags at the shop and sow each extra site's eight tiles."""
    from harvest.core.tile_catalog import get_tile_at
    from harvest.planner.tasks.inventory_exit import ExitToFarmTask
    from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask, EnsureCropSeedsTask
    from harvest.tasks.buy_seeds import BuyShelfItemTask
    from harvest.tasks.skills import farm_pocket_plant_skill
    from tests.test_crop_quota_policy import _rom_env, _run_native

    env = _rom_env("PR_e2efixD_20261114_Y3_Spring_D2", "harvest.y3_extra_sites")
    try:
        sites = farm_pond.SCORE_Y3_EXTRA_CENTERS
        frames = _run_native(env, ExitToFarmTask(), limit=20000)          # the morning pin starts in the house
        frames += _run_native(env, BuyShelfItemTask(stock_field="turnip_seeds", bags=len(sites)), limit=40000)
        frames += _run_native(env, EnsureCarryToolTask(tool_id=2, exit_when_done=True))
        frames += _run_native(env, EnsureCropSeedsTask(seed_type="turnip"))
        for center in sites:
            for attempt in range(3):            # the farm dog can sit on a stand; the planner retries a failed site too
                try:
                    frames += _run_native(env, farm_pocket_plant_skill(seed_type="turnip", center=center,
                                                                       ram=env.get_ram(), include_water=False))
                    break
                except AssertionError as error:
                    if attempt == 2:
                        pytest.fail(f"site {center} did not take its bag: {error}")
            planted = [t for t in farm_pond.plot_crop_tiles(center, 2) if 0x62 <= get_tile_at(env.get_ram(), *t) <= 0x6F]
            assert len(planted) == 8, (center, planted)
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
        print("Y3_EXTRA_SITES_PLANTED", frames, env.audit())
    finally:
        env.close()


def test_year_three_spring_court_goes_before_the_harvest(monkeypatch):
    """y3k_b: the harvest ran to the 17:00 bin deadline, the court started among closed doors three weekdays running."""
    from harvest.planner import power_berry_calendar
    from harvest.planner.day_plan import build_day_phases_from_ram

    monkeypatch.setattr(power_berry_calendar, "due_source", lambda ram, shops_closed=False: None)

    def names(year, season=0):
        ram = weekday(season=season, day=10, hour=6, weekday=2)
        write(ram, "year", year)
        set_tile(ram, 13, 28, 0x60)                                  # a ripe potato the harvest will take
        return [p.phase for p in build_day_phases_from_ram(ram)]

    y3 = names(2)
    assert "SOCIAL_TOWN" in y3 and "HARVEST_ROUTE" not in y3[:y3.index("SOCIAL_TOWN")]
    y2 = names(1)
    assert "SOCIAL_TOWN" in y2 and y2.index("HARVEST_ROUTE") < y2.index("SOCIAL_TOWN")      # earlier years keep their order
    monkeypatch.setenv(social.COURT_FIRST_ENV, "0")
    y3_off = names(2)
    assert y3_off.index("HARVEST_ROUTE") < y3_off.index("SOCIAL_TOWN")


def test_the_last_window_courts_a_girl_who_ends_under_the_496_floor():
    """y3m: Nina ended at 495 (cap 500, need 5 < one +8 visit), so the 31st heart point was lost with no wrap anywhere near."""
    ram = weekday(season=1, day=10, hour=6, weekday=2)                           # Year 3 Summer weekday: Nina's season
    write(ram, "year", 2)
    for girl in ("maria", "ann", "nina", "ellen", "eve"):
        write(ram, f"{girl}_hearts", 500)
    assert social.town_court_phase(ram) is None
    write(ram, "nina_hearts", 495)
    phase = social.town_court_phase(ram)
    assert phase is not None and phase.params["girl"] == "nina" and phase.params["loops"] == 1
    assert phase.params["cap"] >= 495 + social.TOWN_COURT_GAIN                  # the visit may land at 503
    write(ram, "year", 1)
    assert social.town_court_phase(ram) is None                                   # earlier years keep their caps


def test_final_fill_mornings_tolerate_more_ripe_crops_than_building_days():
    """y3p_s2: 46-96 ripe tiles every morning of D14-D22 kept the shop trip off until D25, leaving 69 tiles of field."""
    from harvest.planner import field_cover as fc
    from tests.test_field_cover import BIG, index, morning, open_field

    tiles = open_field(box=BIG)
    mid = fc.cash_floor_g(index(2, 1, 20)) + 60 * 500
    building = fc.cover_errand(morning(year=2, season=1, day=15), spendable_g=mid, stock=0, shop_open=True, tiles=tiles, ripe=60)
    assert building.buy == 0 and building.reason.startswith("busy morning")
    final = fc.cover_errand(morning(year=2, season=1, day=21), spendable_g=mid, stock=0, shop_open=True, tiles=tiles, ripe=60)
    assert final.buy > 0 and final.final_fill
    too_many = fc.cover_errand(morning(year=2, season=1, day=21), spendable_g=mid, stock=0, shop_open=True, tiles=tiles,
                               ripe=fc.FINAL_FILL_LIGHT_RIPE_TILES + 1)
    assert too_many.buy == 0
