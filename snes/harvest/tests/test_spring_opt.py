"""Spring-1 shipping optimiser (`harvest.planner.spring_opt`)."""

from harvest.planner.spring_opt import (
    CROPS,
    Calendar,
    CostModel,
    POTATO,
    Ring,
    RingState,
    optimize_spring,
)


def _rings(n):
    return [Ring(name=f"R{i+1}", order=i) for i in range(n)]


class CalendarTests:
    pass


def test_spring_d1_is_weekday_1_so_sundays_are_7_14_21_28():
    cal = Calendar()
    assert cal.weekday(3) == 3  # measured: Y1_D3_Morning
    assert [d for d in range(1, 31) if cal.is_sunday(d)] == [7, 14, 21, 28]
    for d in (7, 14, 21, 28):
        assert not cal.shop_open(d)


def test_potato_growth_matches_rom_six_waterings():
    r = RingState(crop="potato", waterings=5)
    assert not r.mature(CROPS)
    assert RingState(crop="potato", waterings=6).mature(CROPS)
    assert POTATO.ring_gross_g == 8 * 80


def test_more_rings_never_lowers_the_plan_value_up_to_saturation():
    prev = 0
    results = []
    for n in range(1, 6):
        plan = optimize_spring(rings=_rings(n), crop="potato", beam_width=120)
        results.append(plan.final_wallet)
    # monotone non-decreasing, and the 4-ring plan clears the current
    # reactive-campaign baseline ($1540) by a wide margin.
    assert results == sorted(results)
    assert results[3] > 4_000


def test_plan_replants_each_ring_more_than_once():
    plan = optimize_spring(rings=_rings(2), crop="potato", beam_width=120)
    establishes = sum(len(d.established) for d in plan.days)
    assert establishes >= 4  # 2 rings, at least 2 cycles each


def test_grape_runs_are_bootstrap_only_not_every_day():
    plan = optimize_spring(rings=_rings(4), crop="potato", beam_width=160)
    grape_days = [d.day for d in plan.days if d.grapes]
    assert grape_days, "expected some early grape income"
    assert max(grape_days) <= 12
    assert all(d.grapes <= 2 for d in plan.days)


def test_rain_day_advances_all_planted_rings_for_free():
    dry = optimize_spring(rings=_rings(3), crop="potato", beam_width=120)
    wet = optimize_spring(
        rings=_rings(3), crop="potato", beam_width=120,
        calendar=Calendar(rain_days=tuple(range(8, 28))),
    )
    assert wet.final_wallet > dry.final_wallet


def test_blocked_days_block_field_work_and_shop():
    cal = Calendar(blocked_days=(10, 11, 12))
    plan = optimize_spring(rings=_rings(3), crop="potato", calendar=cal, beam_width=120)
    for d in plan.days:
        if d.day in (10, 11, 12):
            assert not d.established and not d.harvested and not d.bags_bought


def test_cost_model_is_overridable_for_calibration():
    cheap = CostModel(water_ring_f=800, water_ring_marginal_f=400)
    plan = optimize_spring(rings=_rings(5), crop="potato", cost=cheap, beam_width=120)
    # cheaper watering -> more rings stay serviceable -> higher value
    base = optimize_spring(rings=_rings(5), crop="potato", beam_width=120)
    assert plan.final_wallet >= base.final_wallet
