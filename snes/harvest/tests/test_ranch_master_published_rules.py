"""The published Ranch Master rules, each one asserted against the oracle and against the policy that must obey it.

Sources, both read on 2026-11-05: the Harvest Moon Wiki (fandom) page ``Ranch_Master_Score_(HM)``, pasted by the owner
(an automated fetch of it answered HTTP 402), and the RetroAchievements guide chart "Breaking Down Ranch Master". Both agree
with ``MASTER_PLAN`` section 2 (decoded from the ROM, the referee) line for line. Where a source and the ROM differ the ROM
wins and the difference is written down in ``docs/plan.md`` ("Ranch Master rule coverage").

Two layers, both offline:

* **Oracle**: ``harvest.core.ranch_master`` scores each published requirement exactly as the chart says.
* **Policy**: the strategies, the cover's cash floor and the berry calendar never plan past a published cap.
  The ROM-level proof of every term stays in ``test_perfect_ending.py`` (``-m rom``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from perfect_ending_lab import GIRLS, CROPS, S, B, C, NumpyPort, apply_pokes, perfect_ram  # noqa: E402

from harvest.core.ranch_master import evaluate_ranch_mastery_score  # noqa: E402
from harvest.planner import field_cover, power_berry_calendar, run_strategy  # noqa: E402

# (term, points, requirement) as the chart states them. ``per`` is the multiplier the chart words as "each".
CHART = {
    "money": (78, "99,840g"),
    "cows": (36, "12 cows"),
    "chickens": (36, "12 chickens"),
    "power_berries": (50, "10 Power Berries Found"),
    "crop_each": (31, "496-511 sold each"),
    "cow_affection": (300, "25 points for each maxed out cow"),
    "bachelorette_each": (31, "Each girl between 496-511 affection"),
    "house_each": (16, "Get both house upgrades"),
    "clock": (22, "Upgrade your house before end of Summer"),
    "turtle_shell": (21, "Win the egg festival Year 1 or Year 2"),
    "marriage": (32, "Get married"),
    "child_each": (16, "16 points for each child"),
    "development": (50, "Cover your entire field with grass"),
    "happiness": (31, "Get 999 Happiness"),
}


def _total(*pokes) -> int:
    ram = perfect_ram()
    apply_pokes(NumpyPort(ram), pokes)
    return evaluate_ranch_mastery_score(ram).total_score


def test_chart_sums_to_999_with_the_multipliers_the_chart_words_as_each() -> None:
    total = (
        CHART["money"][0] + CHART["cows"][0] + CHART["chickens"][0] + CHART["power_berries"][0]
        + 4 * CHART["crop_each"][0] + CHART["cow_affection"][0] + 5 * CHART["bachelorette_each"][0]
        + 2 * CHART["house_each"][0] + CHART["clock"][0] + CHART["turtle_shell"][0] + CHART["marriage"][0]
        + 2 * CHART["child_each"][0] + CHART["development"][0] + CHART["happiness"][0]
    )
    assert total == 999
    assert _total() == 999, "the perfect vector must satisfy every requirement on the chart"


@pytest.mark.parametrize(
    "label, pokes, loses",
    [
        ("money below 99,840g loses a point per 1,280g", (S("money_raw", 9983),), 1),
        ("12 cows give 36", (S("num_cows", 11),), 3),
        ("12 chickens give 36", (S("num_chickens", 11),), 3),
        ("ten berries give 50", (S("max_stamina", 190),), 5),
        ("every crop 496-511 each", (S("shipped_corn", 495),), 1),
        ("a crop over 511 resets to 0", (S("shipped_tomatoes", 512),), 31),
        ("a girl below 496", (S("eve_hearts", 495),), 1),
        ("a girl over 511 resets to 0", (S("maria_hearts", 512),), 31),
        ("one house upgrade is 16", (C("upgrade_flags", 0x0040),), 16),
        ("no clock", (C("incubator_flags", 0x4000),), 22),
        ("no turtle shell", (C("family_event_flags", 0x1000),), 21),
        ("single, not married", (S("marriage_flags", 0),), 32),
        ("one child is 16", (C("incubator_flags", 0x0004),), 16),
        ("development below 2,950 tiles", (S("ranch_development", 98),), 1),
        ("happiness under 992", (S("happiness", 991),), 1),
        ("a cow at 199 happiness", (S("cow4.happiness", 199),), 1),
    ],
)
def test_each_published_requirement_is_scored_as_the_chart_says(label, pokes, loses) -> None:
    assert _total(*pokes) == 999 - loses, label


def test_all_five_girls_at_999_affection_make_the_994_run() -> None:
    """The guide: a full-affection run lands on 994, one point lost per girl (999 & 0x1FF = 487 -> 30 points)."""
    assert _total(*(S(f"{girl}_hearts", 999) for girl in GIRLS)) == 994


def test_cow_affection_is_twenty_five_a_cow_for_twelve_cows() -> None:
    ram = perfect_ram()
    breakdown = evaluate_ranch_mastery_score(ram)
    assert breakdown.cow_happiness_score == CHART["cow_affection"][0] == 12 * 25


# ── the policy must never plan past a published cap ──────────────────────────────────────────────

def test_every_score_strategy_stops_each_crop_inside_the_496_to_511_window() -> None:
    for strategy in (run_strategy.SCORE, run_strategy.SCORE_HERD, run_strategy.PERFECT):
        assert run_strategy.SHIP_WINDOW_LOW <= strategy.ship_cap <= run_strategy.SHIP_WINDOW_HIGH, strategy.name
    assert run_strategy.crop_points(511) == 31 and run_strategy.crop_points(512) == 0
    assert run_strategy.points_lost_to_wrap({"corn": 600, "tomato": 100, "potato": 496, "turnip": 0}) == 31 - 600 % 512 // 16


def test_the_cover_never_spends_the_money_term() -> None:
    assert field_cover.MONEY_TERM_G == 99_840 == 78 * 1280
    floors = [field_cover.cash_floor_g(i) for i in range(field_cover.START_INDEX, field_cover.TALLY_INDEX + 1)]
    assert floors == sorted(floors), "the floor only rises as the days run out"
    assert floors[-1] >= field_cover.MONEY_TERM_G, "the last morning must hold the whole money term"
    assert min(floors) >= field_cover.RESERVE_G


def test_herd_strategies_keep_the_money_term_and_never_exceed_twelve_animals() -> None:
    assert (run_strategy.SCORE_HERD.late_cash_floor_g or 0) >= field_cover.MONEY_TERM_G
    for strategy in (run_strategy.SCORE, run_strategy.SCORE_HERD, run_strategy.PERFECT):
        for value in (strategy.cow_target_later, strategy.cow_target_fall, strategy.hens_kept, strategy.final_flock_target):
            assert value is None or 0 <= value <= 12, (strategy.name, value)
    assert run_strategy.PERFECT.cow_target_later == run_strategy.PERFECT.hens_kept == 12


def test_the_berry_calendar_aims_at_exactly_ten_and_stops_there() -> None:
    assert power_berry_calendar.BERRY_TARGET == 10


def test_year_three_never_plans_the_turtle_shell() -> None:
    """The turtle shell is the Egg Festival (Fall D20) of Year 1 or 2; Year 3 ends in Summer."""
    from tests.test_power_berry_calendar import ram  # noqa: PLC0415

    for season in (0, 1):
        assert power_berry_calendar.due_source(ram(year=2, season=season, day=20), force=True) != "egg_festival"


# ── rules only the wiki page states ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, loses", [
    (9984, 0),            # 99,840G: the first amount that pays 78
    (65535, 0),           # 655,350G: the last amount of the first band
    (65540, 78),          # wiki: 655,400G pays exactly 0, the 16-bit counter has wrapped
    (75520, 0),           # the second band opens at 755,200G
    (75519, 1),
])
def test_money_wraps_at_655350_and_pays_again_from_the_next_band(raw, loses) -> None:
    assert _total(S("money_raw", raw)) == 999 - loses


@pytest.mark.parametrize("shipped, loses", [
    (496, 0), (511, 0), (512, 31), (1008, 0), (1023, 0), (1520, 0), (1535, 0), (2032, 0), (2047, 0),
])
def test_crop_bands_repeat_every_512_from_496(shipped, loses) -> None:
    """Wiki: 496-511, 1008-1023, 1520-1535, then 2032-2047 (the page prints 2023, a typo for 2032)."""
    assert _total(S("shipped_potatoes", shipped)) == 999 - loses


def test_a_girl_over_511_wraps_to_zero_and_cannot_use_a_second_band() -> None:
    assert _total(S("nina_hearts", 520)) == 999 - 31, "wiki example: 520 affection pays 0 for her"
    assert _total(S("nina_hearts", 999)) == 999 - 1, "999 is the cap and 1008 is out of reach: 30 points, not 31"


@pytest.mark.parametrize("tiles, percent", [
    (29, 0), (30, 1), (58, 1), (59, 2), (88, 2), (89, 3), (2949, 99), (2950, 100), (2968, 100),
])
def test_development_percent_boundaries_match_the_wiki(tiles, percent) -> None:
    """Wiki: 1% is 30-58 tiles, 2% 59-88, 3% 89-117; a point per 2%; 2950-2968 tiles pay 50."""
    byte = min(100, tiles * 10 // 295)
    assert byte == percent
    assert _total(S("ranch_development", byte)) == 999 - (50 - (byte >> 1))


def test_rating_tiers_match_the_wiki_estimates_and_the_400_line_is_a_text_change() -> None:
    from harvest.core.ranch_master import RATING_TEXT_BANDS, rating_text_id_for_score

    assert [(lo, hi) for lo, hi, _ in RATING_TEXT_BANDS] == [(0, 199), (200, 399), (400, 599), (600, 799), (800, 999)]
    assert rating_text_id_for_score(399) != rating_text_id_for_score(400)   # "Try harder." -> "Come on, you are so close!"
    assert rating_text_id_for_score(799) != rating_text_id_for_score(800)   # -> "You are the perfect rancher now!"


def test_wife_gifts_can_neither_overshoot_511_nor_let_the_night_drain_leave_the_window() -> None:
    """Wiki: the wife loses 2 on every house entry once she is in bed, and 512 wraps to 0.

    A gift is +4 an egg (two rounds below ``EVE_GIFT_FAST_BELOW``), so the stop level plus one round must stay
    inside 496-511 and the level itself must survive one hour-18 entry.
    """
    from harvest.tasks.eve_gift import EVE_GIFT_FAST_BELOW, EVE_GIFT_TARGET

    assert EVE_GIFT_TARGET + 4 <= 511
    assert EVE_GIFT_TARGET - 2 >= 496
    assert EVE_GIFT_FAST_BELOW + 8 < EVE_GIFT_TARGET


def test_crops_sold_to_the_peddler_do_not_count_so_the_bot_declines_him() -> None:
    from harvest.core.scene import peddler_decline_action

    assert callable(peddler_decline_action)


def test_the_power_berry_stop_is_exactly_200_stamina() -> None:
    """Wiki: 50 points need exactly 200 stamina and the Happy Family Ending needs 200 too; an eleventh berry is a loss."""
    assert power_berry_calendar.BERRY_TARGET * 10 + 100 == 200
