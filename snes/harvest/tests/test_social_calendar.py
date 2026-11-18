"""Relationship calendar: the Sunday church gate, heart caps and the stops the church task takes."""
from __future__ import annotations

import pytest

from harvest.core.ram_catalog import field_spec, ram_index
from harvest.planner import social_calendar as social
from tests.day_plan_test_helpers import make_date_world


def write(ram, name, amount):
    spec = field_spec(name)
    index = ram_index(ram, spec.address, live_offset=spec.live_offset)
    stored = spec.to_storage(amount, raw=True)
    for byte in range(spec.width):
        ram[index + byte] = stored >> (8 * byte) & 255


def write_u16(ram, address, amount):
    index = ram_index(ram, address)
    ram[index], ram[index + 1] = amount & 255, amount >> 8 & 255


def sunday(*, year=0, season=2, day=3, hour=8, married=0, hearts=8):
    world = make_date_world(0, season=season, day=day, hour=hour)
    ram = world.ram
    write(ram, "year", year)
    write(ram, "weekday", 0)
    write(ram, "marriage_flags", married)
    write_u16(ram, social.EVENT_68, 0x00BF)   # first church visit done (power-on Fall D3 pin)
    for girl in ("maria", "ann", "nina", "ellen", "eve"):
        write(ram, f"{girl}_hearts", hearts)
    return ram


@pytest.fixture(autouse=True)
def church_on(monkeypatch):
    monkeypatch.setenv(social.ENV_VAR, "1")


def test_opening_day_two_keeps_the_forage_and_seed_route_ahead_of_courting():
    ram = sunday(season=0, day=2)
    write(ram, "weekday", 2)
    assert social.social_phase(ram) is None
    write(ram, "day", 3)
    assert social.social_phase(ram).phase == "SOCIAL_TOWN"


def test_unmarried_girls_stop_below_the_200_heart_event():
    ram = sunday()
    phase = social.church_phase(ram)
    assert phase.phase == "SOCIAL_CHURCH" and phase.kind == "church_court"
    assert phase.params["targets"] == {"maria": 199, "ann": 199}
    assert phase.params["max_entries"] == social.CHURCH_MAX_ENTRIES   # (199 - 8) / 4 = 48 entries wanted
    write(ram, "maria_hearts", 196)
    write(ram, "ann_hearts", 196)
    assert social.church_phase(ram) is None                            # one more +4 would cross 199


def test_after_the_wedding_the_cap_keeps_room_for_the_festivals_still_ahead():
    ram = sunday(day=17, married=0x10)
    # Two Flowers, two New Years, two Eggs, and at most one Harvest berry visit: 500 - 30 - 20 - 20 - 8.
    assert social.church_phase(ram).params["targets"] == {"maria": 422, "ann": 422}
    # Eve is the wife, so Star Night plays her scene (festival_escape) and pays Ellen nothing.
    assert social.heart_cap(ram, "ellen") == 422
    ann_wife = sunday(day=17, married=0x02)
    assert social.heart_cap(ann_wife, "ellen") == 422 - 2 * social.STAR_NIGHT_GAIN
    late = sunday(year=2, season=1, day=7, married=0x10)
    assert social.heart_cap(late, "maria") == social.SOCIAL_TARGET     # nothing left to land on its own
    assert social.church_phase(late).params["max_entries"] == social.CHURCH_MAX_ENTRIES


def test_the_wife_and_a_silenced_girl_are_not_church_targets():
    ram = sunday(day=17, married=0x02)                                 # married to Ann
    assert set(social.church_phase(ram).params["targets"]) == {"maria"}
    write_u16(ram, social.ROMANCE_6A, 0x1000)                          # Maria's 200-heart event already played
    assert social.church_phase(ram) is None


@pytest.mark.parametrize("change", ["weekday", "rain", "snow", "storm_eve", "festival_eve", "late", "first_visit",
                                    "no_actors"])
def test_the_church_is_skipped_when_its_question_script_does_not_load(change):
    ram = sunday(day=17, married=0x10)
    assert social.church_phase(ram) is not None
    if change == "weekday":
        write(ram, "weekday", 3)
    elif change == "rain":
        write_u16(ram, social.WEATHER, 0x0002)
    elif change == "snow":
        write_u16(ram, social.WEATHER, 0x0008)
    elif change == "storm_eve":
        write(ram, "weather_tomorrow", 3)
    elif change == "festival_eve":
        write(ram, "weather_tomorrow", 6)
    elif change == "late":
        write(ram, "hour", social.CHURCH_LATEST_START_HOUR + 1)
    elif change == "first_visit":
        write_u16(ram, social.EVENT_68, 0x00BE)
    elif change == "no_actors":
        write_u16(ram, social.UPGRADE_64, 0x0020)
    assert social.church_phase(ram) is None


def test_a_festival_sunday_is_skipped():
    ram = sunday(season=2, day=20)                                     # Egg Festival
    assert social.church_phase(ram) is None


def test_entries_cover_the_neediest_girl_only():
    ram = sunday(day=17, married=0x10)
    write(ram, "maria_hearts", 414)
    write(ram, "ann_hearts", 410)
    phase = social.church_phase(ram)
    assert phase.params["max_entries"] == 3                            # Ann: (422 - 410) / 4
    assert phase.params["targets"] == {"maria": 422, "ann": 422}


def test_the_switch_follows_the_strategy(monkeypatch):
    from harvest.planner import run_strategy

    monkeypatch.delenv(social.ENV_VAR)
    monkeypatch.setattr(run_strategy, "active", lambda: run_strategy.get("cash"))
    assert not social.enabled() and social.church_phase(sunday()) is None
    monkeypatch.setattr(run_strategy, "active", lambda: run_strategy.get("score"))
    assert social.enabled()
    monkeypatch.setenv(social.ENV_VAR, "0")
    assert not social.enabled()


def test_upcoming_gain_counts_only_dates_before_the_ending():
    assert social.upcoming_gain("maria", 2, 0, 24) == 0                # Y3 Flower Festival already past
    assert social.upcoming_gain("maria", 2, 0, 22) == social.FLOWER_FESTIVAL_GAIN
    both = social.FLOWER_FESTIVAL_GAIN + social.NEW_YEAR_GAIN                  # Y3 Spring D1 and D23
    assert social.upcoming_gain("ellen", 1, 3, 23) == social.STAR_NIGHT_GAIN + both
    assert social.upcoming_gain("ellen", 1, 3, 23, wife="eve") == both
    assert social.upcoming_gain("maria", 2, 0, 1) == social.FLOWER_FESTIVAL_GAIN    # today's New Year is not ahead
    assert social.upcoming_gain("eve", 0, 2, 1, wife="eve") == 0


def test_harvest_questions_are_reserved_once_and_only_while_the_trip_is_enabled(monkeypatch):
    from harvest.planner.power_berry_calendar import HARVEST_FESTIVAL_ENV

    ram = sunday(day=17, married=0x10)
    monkeypatch.setenv(HARVEST_FESTIVAL_ENV, "0")
    before = social.heart_cap(ram, "ann")
    monkeypatch.setenv(HARVEST_FESTIVAL_ENV, "1")
    assert social.heart_cap(ram, "ann") == before - social.HARVEST_QUESTION_GAIN
    # Two remaining dates still reserve just one successful berry visit.
    assert social.upcoming_gain("ann", 0, 2, 1, wife="eve") == 500 - before + social.HARVEST_QUESTION_GAIN
    assert social.heart_cap(sunday(year=2, season=1, married=0x10), "ann") == 500


def test_church_stops_respect_caps_held_gifts_and_the_wife():
    from harvest.tasks.relationships import church_talk_wanted

    ram = sunday(day=17, married=0x10)
    targets = {"maria": 470, "ann": 470, "ellen": 470}
    assert church_talk_wanted(ram, targets) == ["ann", "maria"]        # Ellen's Sunday talk pays nothing
    write(ram, "held_item", 0x14)                                      # an egg: +8 for Ellen, +5 for the others
    assert church_talk_wanted(ram, targets) == ["ann", "ellen", "maria"]
    write(ram, "held_item", 0)
    write(ram, "ann_hearts", 470)
    assert church_talk_wanted(ram, targets, dry={"maria"}) == []   # Ann at her cap, Maria dropped
    assert church_talk_wanted(sunday(day=17, married=0x01), targets) == ["ann"]


def test_the_sunday_plan_puts_the_church_before_the_dynamic_outdoor_plan(monkeypatch):
    from harvest.planner.day_plan import build_day_phases_from_ram

    ram = sunday(day=17, married=0x10)
    names = [phase.phase for phase in build_day_phases_from_ram(ram)]
    assert "SOCIAL_CHURCH" in names
    assert names[names.index("SOCIAL_CHURCH") + 1:] == ["EXIT_TO_FARM", "DYNAMIC_OUTDOOR_PLAN"]
    monkeypatch.setenv(social.ENV_VAR, "0")
    assert "SOCIAL_CHURCH" not in [phase.phase for phase in build_day_phases_from_ram(ram)]


def star_morning(*, married=0x10, day=24, cursor=0):
    world = make_date_world(0x16, season=3, day=day, hour=6)
    ram = world.ram
    write(ram, "marriage_flags", married)
    write(ram, "dialog_text_id", 0x02B6)
    write(ram, "input_lock", 2)
    ram[0x018F] = cursor
    return ram


def same(result, expected):
    import numpy as np

    return np.array_equal(np.asarray(result.action.action), np.asarray(expected))


def test_star_night_menu_waits_turns_the_page_then_steers_to_the_wife():
    from harvest.planner.tasks.festival_escape import star_night_menu_press
    from harvest.tasks.nav import make_action

    state = {}
    ram = star_morning()
    idle = make_action()
    assert same(star_night_menu_press(ram, state, 1000), idle)                       # text still typing
    assert same(star_night_menu_press(ram, state, 1300), make_action(a=True))        # one A: the page
    assert same(star_night_menu_press(ram, state, 1400), idle)                       # menu not live yet
    assert same(star_night_menu_press(ram, state, 1504), make_action(down=True))
    ram[0x018F] = 2
    assert same(star_night_menu_press(ram, state, 1512), make_action(a=True))        # Eve's scene, index 2


def test_star_night_menu_leaves_other_cases_to_the_default_press():
    from harvest.planner.tasks.festival_escape import star_night_menu_press

    assert star_night_menu_press(star_morning(day=23), {}, 0) is None
    assert star_night_menu_press(star_morning(married=0x02), {}, 0) is None        # Ann's scene has no stands
    assert star_night_menu_press(star_morning(married=0), {}, 0) is None
    ram = star_morning()
    write(ram, "input_lock", 1)                                                    # the stale text after the choice
    assert star_night_menu_press(ram, {}, 0) is None
    state = {}
    ram = star_morning()
    for step in range(1000, 3000):
        if star_night_menu_press(ram, state, step) is None:                        # the cursor never moves
            break
    assert state.get("gave_up")


def test_the_eve_scene_map_is_handed_to_the_scene_task_only_that_night():
    from harvest.planner.tasks.festival_escape import star_night_scene_map

    from harvest.core.tile_catalog import ADDR_TILEMAP

    ram = star_morning()
    write(ram, "hour", 18)
    ram[ADDR_TILEMAP] = 0x1E
    assert star_night_scene_map(ram)
    write(ram, "day", 23)
    assert not star_night_scene_map(ram)                                            # an ordinary bar night


def weekday(*, season=2, day=18, hour=8, married=0x10, hearts=8, weekday=4):
    ram = sunday(season=season, day=day, hour=hour, married=married, hearts=hearts)
    write(ram, "weekday", weekday)
    return ram


def test_fall_weekdays_court_ellen_and_winter_weekdays_court_nina():
    fall = social.town_court_phase(weekday(season=2))
    assert fall.kind == "town_court" and fall.params["girl"] == "ellen"
    assert fall.params["loops"] == social.TOWN_COURT_MAX_LOOPS[2]
    assert fall.params["cap"] == 422                          # also reserve one Harvest Festival's questions
    winter = social.town_court_phase(weekday(season=3, day=18))
    assert winter.params["girl"] == "nina"
    assert social.social_phase(weekday(season=3, day=18)).phase == "SOCIAL_TOWN"


def test_crop_seasons_court_only_a_girl_far_behind_and_briefly():
    ram = weekday(season=0, day=8)
    assert social.town_court_phase(ram).params["loops"] == social.TOWN_COURT_MAX_LOOPS[0]
    write(ram, "ellen_hearts", social.heart_cap(ram, "ellen") - social.TOWN_COURT_BEHIND + 1)
    assert social.town_court_phase(ram) is None
    last = weekday(season=0, day=26)
    write(last, "year", 2)                                      # Year 3 Spring: her last window, top up any shortfall
    write(last, "ellen_hearts", social.SOCIAL_TARGET - 10)
    assert social.town_court_phase(last).params["loops"] == 2


@pytest.mark.parametrize("change", ["saturday", "sunday", "rain", "snow", "late", "festival", "festival_eve", "winter_d10",
                                    "capped", "unmarried_cap"])
def test_the_town_visit_is_skipped_when_her_question_cannot_pay(change):
    ram = weekday()
    assert social.town_court_phase(ram) is not None
    if change == "saturday":
        write(ram, "weekday", 6)
    elif change == "sunday":
        write(ram, "weekday", 0)
    elif change == "rain":
        write_u16(ram, social.WEATHER, 0x0002)
    elif change == "snow":
        write_u16(ram, social.WEATHER, 0x0008)
    elif change == "late":
        write(ram, "hour", social.TOWN_COURT_LATEST_START_HOUR + 1)
    elif change == "festival":
        write(ram, "day", 12)                                  # Harvest Festival
    elif change == "festival_eve":
        write(ram, "weather_tomorrow", 7)                      # Y1 Fall D11: 8 talks, nothing paid
    elif change == "winter_d10":
        ram = weekday(season=3, day=10)
    elif change == "capped":
        write(ram, "ellen_hearts", 435)
    elif change == "unmarried_cap":
        ram = weekday(married=0, hearts=192)
    assert social.town_court_phase(ram) is None


def test_the_wife_is_never_a_town_target():
    assert social.town_court_phase(weekday(season=2, married=0x08)) is None   # married to Ellen


def test_a_berry_trip_is_followed_by_the_walk_home_before_the_courting_stop(monkeypatch):
    from harvest.planner import power_berry_calendar
    from harvest.planner.day_plan import build_day_phases_from_ram

    monkeypatch.setattr(power_berry_calendar, "due_source", lambda ram, shops_closed=False: "fish")
    names = [phase.phase for phase in build_day_phases_from_ram(weekday())]
    at = names.index("SOCIAL_TOWN")
    assert names[at - 1] == "EXIT_TO_FARM" and names.index("COLLECT_POWER_BERRY_FISH") < at


def test_church_stops_before_a_question_would_pass_the_pre_wedding_cap():
    from harvest.tasks.relationships import church_talk_wanted

    ram = sunday(hearts=196)                                   # unmarried: cap 199
    assert church_talk_wanted(ram, {"maria": 199, "ann": 199}) == []
    write(ram, "maria_hearts", 195)
    assert church_talk_wanted(ram, {"maria": 199, "ann": 199}) == ["maria"]   # 195 + 4 = 199


def test_a_one_question_church_trip_waits_for_the_final_year():
    ram = sunday(day=17, married=0x10, hearts=424)              # cap 430: one +4 left
    assert social.church_phase(ram) is None
    write(ram, "year", 2)
    write(ram, "season", 1)                                     # Year 3 Summer: cap 500, a real top-up
    assert social.church_phase(ram).params["max_entries"] == 19
    write(ram, "maria_hearts", 496)
    write(ram, "ann_hearts", 496)
    assert social.church_phase(ram).params["max_entries"] == 1  # final year: even one entry is worth it


def _ripe_potato_morning(*, season=1, day=3, weekday_=5, ripe=True):
    from tests.test_run_strategy import set_tile

    ram = weekday(season=season, day=day, hour=6, weekday=weekday_)
    if ripe:
        set_tile(ram, 13, 28, 0x60)  # a ripe potato tile: the morning harvest will free its ring
    return ram


def test_courting_yields_to_the_rings_corn_trip_while_ripe_crops_free_them(monkeypatch):
    """compose5 (Y1 Summer D3): the 13 last potatoes were harvested by mid-morning, then Nina's court took the day's one
    town trip until 16:01, the shop closed at 17:00 and the weekend followed, so the rings never got corn: no Summer income."""
    from harvest.planner import run_strategy
    from harvest.planner.day_plan import build_day_phases_from_ram

    monkeypatch.setenv(run_strategy.ENV_VAR, "score")
    ram = _ripe_potato_morning()
    assert social.town_court_phase(ram) is not None                    # she would be courted on any other weekday
    assert social.ring_corn_trip_first(ram)
    assert "SOCIAL_TOWN" not in [p.phase for p in build_day_phases_from_ram(ram)]
    for quiet in (dict(weekday_=6), dict(day=4), dict(ripe=False), dict(season=2, day=3)):
        other = _ripe_potato_morning(**quiet)
        assert not social.ring_corn_trip_first(other), quiet
    monkeypatch.setenv(run_strategy.ENV_VAR, "cash")
    assert not social.ring_corn_trip_first(_ripe_potato_morning())      # only the score family caps shipping and courts
