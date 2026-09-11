"""Livestock purchase pre-calc (`harvest.planner.livestock_econ`).

The module is a stub by design: the decision *mechanism* is real and tested
here, the *inputs* are not measured yet. These tests pin both halves — that
an unpriced animal yields "unknown, here is what to measure" and never a
fabricated answer, and that a fully priced one produces the buy day the cash
flow supports.
"""

from dataclasses import replace

from harvest.planner.livestock_econ import (
    CHICKEN,
    COW,
    LivestockSpec,
    grape_g_per_frame,
    plan_purchase,
)
from harvest.planner.spring_opt import CostModel, Ring, optimize_spring


def _plan(**kw):
    rings = [Ring(name=f"R{i+1}", order=i) for i in range(4)]
    return optimize_spring(rings=rings, crop="potato", beam_width=120, **kw)


def test_chicken_is_unpriced_and_says_so_instead_of_guessing():
    verdict = plan_purchase(_plan(), CHICKEN)
    assert verdict.blocked_on == (
        "purchase_cost_g", "products_per_day", "feed_cost_g_per_day", "chore_frames_per_day",
    )
    assert verdict.worth_it is None, "unknown must not collapse to a no"
    assert verdict.buy_day is None and verdict.payback_days is None
    for field in verdict.blocked_on:
        assert field in verdict.reason


def test_the_egg_price_is_the_one_input_we_do_have():
    assert CHICKEN.product_price_g == 50
    assert "purchase_cost_g" in CHICKEN.missing_inputs()
    assert "product_price_g" not in CHICKEN.missing_inputs()
    assert COW.product_price_g == 150


def test_a_fully_priced_animal_gets_bought_the_first_day_the_ledger_can_pay():
    plan = _plan()
    spec = LivestockSpec(name="test_bird", purchase_cost_g=1_000, product_price_g=50,
                         products_per_day=1.0, feed_cost_g_per_day=0,
                         chore_frames_per_day=0)
    buy_day = plan.earliest_day_affording(1_000)
    # Pin the horizon: whether a 20-day payback *fits* depends on how rich
    # the plan is, which moves with every CostModel recalibration. The buy
    # day and the payback arithmetic are what this test is about.
    verdict = plan_purchase(plan, spec, horizon_end_day=buy_day + 30)
    assert not verdict.blocked_on
    assert verdict.buy_day == buy_day
    assert verdict.daily_net_g == 50
    assert verdict.payback_days == 20
    assert verdict.worth_it is True


def test_chore_frames_are_charged_at_the_best_alternative_use_of_a_frame():
    """A chicken competes with the grape run for the same evening frames."""
    plan = _plan()
    cost = CostModel()
    free = LivestockSpec(name="free", purchase_cost_g=1_000, product_price_g=50,
                         products_per_day=1.0, feed_cost_g_per_day=0,
                         chore_frames_per_day=0)
    busy = replace(free, name="busy", chore_frames_per_day=2_000)
    assert grape_g_per_frame(cost) > 0
    assert plan_purchase(plan, busy, cost=cost).daily_net_g < plan_purchase(plan, free, cost=cost).daily_net_g
    # 2000 f/day of chores costs more than an egg is worth -> a real "no".
    verdict = plan_purchase(plan, busy, cost=cost)
    assert verdict.worth_it is False and not verdict.blocked_on


def test_an_animal_the_plan_can_never_afford_is_a_no_not_a_crash():
    plan = _plan()
    spec = LivestockSpec(name="gold_goose", purchase_cost_g=10_000_000,
                         product_price_g=50, products_per_day=1.0,
                         feed_cost_g_per_day=0, chore_frames_per_day=0)
    verdict = plan_purchase(plan, spec)
    assert verdict.worth_it is False
    assert verdict.buy_day is None
    assert str(plan.cashflow.peak_balance_g) in verdict.reason


def test_a_purchase_too_late_to_pay_back_inside_the_horizon_is_refused():
    plan = _plan()
    spec = LivestockSpec(name="slow", purchase_cost_g=1_000, product_price_g=50,
                         products_per_day=1.0, feed_cost_g_per_day=0,
                         chore_frames_per_day=0)
    buy_day = plan.earliest_day_affording(1_000)
    # Just short of the days needed to earn the 1000 G back at 50 G/day.
    assert plan_purchase(plan, spec, horizon_end_day=buy_day + 19).worth_it is False
    assert plan_purchase(plan, spec, horizon_end_day=buy_day + 21).worth_it is True
