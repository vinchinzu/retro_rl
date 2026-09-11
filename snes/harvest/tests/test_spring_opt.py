"""Spring shipping optimiser (`harvest.planner.spring_opt`)."""

from harvest.planner.spring_opt import (
    CROPS,
    Calendar,
    CostModel,
    POTATO,
    Ring,
    RingState,
    SEASON_FALL,
    SEASON_SPRING,
    SEASON_SUMMER,
    SEASON_WINTER,
    optimize_spring,
)


def _rings(n):
    return [Ring(name=f"R{i+1}", order=i) for i in range(n)]


def test_spring_d1_is_weekday_1_so_sundays_are_7_14_21_28():
    cal = Calendar()
    assert cal.weekday(3) == 3  # measured: Y1_D3_Morning
    assert [d for d in range(1, 31) if cal.is_sunday(d)] == [7, 14, 21, 28]
    for d in (7, 14, 21, 28):
        assert not cal.shop_open(d)


def test_season_of_day_matches_rom_convention():
    cal = Calendar()
    assert cal.season_of(1) == SEASON_SPRING
    assert cal.season_of(30) == SEASON_SPRING
    assert cal.season_of(31) == SEASON_SUMMER
    assert cal.season_of(60) == SEASON_SUMMER
    assert cal.season_of(61) == SEASON_FALL
    assert cal.season_of(90) == SEASON_FALL
    assert cal.season_of(91) == SEASON_WINTER


def test_potato_growth_matches_rom_six_waterings():
    r = RingState(crop="potato", waterings=5)
    assert not r.mature(CROPS)
    assert RingState(crop="potato", waterings=6).mature(CROPS)
    assert POTATO.sell_price_g == 80
    assert POTATO.days_to_first_harvest == 6


def test_more_rings_never_lowers_the_plan_value_up_to_saturation():
    # A wider beam than most tests use on purpose: monotonicity is a property
    # of the *optimum*, and a fixed-width beam is not guaranteed to preserve
    # it — at width 120 the 5-ring search loses to the 4-ring one purely to
    # pruning, because more rings means more distinct states competing for
    # the same slots. 200 is enough here; if this ever goes red, check the
    # beam before assuming the model broke.
    results = []
    for n in range(1, 6):
        plan = optimize_spring(rings=_rings(n), crop="potato", beam_width=200)
        results.append(plan.final_wallet)
    assert results == sorted(results)
    # ... and the 4-ring plan clears the reactive campaign baseline ($1540)
    # by a wide margin even under the measured runtime caps (1 bag/day,
    # 1 grape/day — see CostModel.max_bags_per_day).
    assert results[3] > 4_000


def test_plan_replants_each_ring_more_than_once():
    plan = optimize_spring(rings=_rings(2), crop="potato", beam_width=120)
    establishes = sum(len(d.established) for d in plan.days)
    assert establishes >= 4  # 2 rings, at least 2 cycles each


def test_grapes_are_an_option_not_a_daily_requirement():
    plan = optimize_spring(rings=_rings(4), crop="potato", beam_width=160)
    grape_days = [d.day for d in plan.days if d.grapes]
    assert grape_days, "expected some grape income"
    assert all(d.grapes <= 2 for d in plan.days)
    harvest_days = [d for d in plan.days if d.harvested]
    assert harvest_days, "expected some harvest days in a 4-ring plan"
    # Harvest mornings may skip or cut the grape run; do not require 2.


def test_rain_day_advances_all_planted_rings_for_free():
    # Rain only helps once watering is budget-constrained: many rings, tight
    # evening. With a huge evening every ring is watered every day anyway.
    tight = CostModel(evening_frames=14_000)
    dry = optimize_spring(rings=_rings(10), crop="potato", cost=tight, beam_width=120)
    wet = optimize_spring(
        rings=_rings(10), crop="potato", cost=tight, beam_width=120,
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


# ── Corrected season model (2026-09-10) ───────────────────────────────────


def test_evening_frames_derives_from_wake_sleep_hours_and_minute_rate():
    cost = CostModel()  # defaults: wake=6, sleep=20, minute_frames=15
    assert cost.evening_frames == 14 * 60 * 15
    explicit = CostModel(evening_frames=14_000)
    assert explicit.evening_frames == 14_000  # override still honored
    longer_day = CostModel(sleep_hour=22)
    assert longer_day.evening_frames == 16 * 60 * 15
    assert longer_day.evening_frames > cost.evening_frames


def test_default_day_length_matches_the_measured_campaign_day():
    """run12 day-change deltas: median 12859 f (n=9, range 11535-16199).

    The old 18:00 default budgeted 10800 f and under-counted the real day by
    ~27 %, which made every ring look more expensive than it is.
    """
    assert abs(CostModel().evening_frames - 12_859) < 1_000


def test_the_seed_bag_cap_is_modelled_and_the_grape_cap_is_not():
    """One bag/day is a live bot limit; one grape/day no longer is.

    Every BUY_SEEDS in every log still reads `bought potato_seeds 0->1`
    (four stacked limits, see docs/tasks/rr-20w-run13-defects.md), so the
    bag cap stays modelled. The grape cap was a `shop_bail_hour` plumbing
    bug, fixed in the working tree and proven by run13 shipping 2/2 on 7 of
    8 successful trips — so the model now allows 2.
    """
    plan = optimize_spring(rings=_rings(6), crop="potato", beam_width=200)
    assert all(d.bags_bought <= 1 for d in plan.days)
    assert CostModel().grape_max_per_day == 2
    assert any(d.grapes == 2 for d in plan.days), "2-grape days must be reachable"


def test_grape_income_is_discounted_by_measured_reliability():
    """run13: 15 berry phase starts, 8 produced grapes, 7 failed outright.

    A failed trip still burns the frames, so the success rate must scale
    income only — never the frame cost. A plan that assumes a phase which
    works 40 % of the time always works is the error this term exists to
    stop.
    """
    flaky = CostModel()                      # grape_success_rate = 0.4 (measured)
    fixed = CostModel(grape_success_rate=1.0)  # a hypothetically repaired bot
    assert flaky.grape_success_rate == 0.4

    a = optimize_spring(rings=_rings(4), crop="potato", cost=flaky, beam_width=160)
    b = optimize_spring(rings=_rings(4), crop="potato", cost=fixed, beam_width=160)
    assert b.cashflow.berry_income_g > a.cashflow.berry_income_g

    # Income discounted, frames not: a 2-grape day still books full value*rate.
    for d in a.days:
        if d.grapes:
            assert d.berry_income_g == int(d.grapes * flaky.grape_value_g * 0.4)


def test_growth_continues_into_summer_not_wiped():
    """A ring planted late in Spring keeps growing (INC) past Summer D1."""
    cost = CostModel(evening_frames=100_000)  # generous, isolate the season rule
    cal = Calendar(first_day=27, horizon_end=40)
    plan = optimize_spring(rings=_rings(1), crop="potato", calendar=cal, cost=cost,
                            start_bags=1, beam_width=60)
    # Planted ~D27 (spring), potato needs 6 waterings -> matures ~D33-34,
    # which is *into* Summer (day 31+). It must still mature and ship.
    assert any(d.established for d in plan.days), "expected the ring to be planted"
    assert plan.shipped_g >= 8 * POTATO.sell_price_g, (
        "a ring planted late in spring must still mature and ship in summer"
    )


def test_a_day28_planting_is_now_valued_not_refused():
    """Old model refused to plant after ~D24; the real cutoff is Fall D1."""
    cost = CostModel(evening_frames=100_000)
    cal = Calendar(first_day=28, horizon_end=45, checkpoint_day=44)
    plan = optimize_spring(rings=_rings(1), crop="potato", calendar=cal, cost=cost,
                            start_bags=1, beam_width=40)
    assert any(d.established for d in plan.days), "D28 planting must not be refused"


def test_decay_starts_in_fall_for_immature_watered_crops():
    from harvest.planner.spring_opt import _resolve_day, State

    cost = CostModel(evening_frames=100_000)
    cal = Calendar(horizon_end=65)
    crop = "potato"
    ring_names = ("R1",)
    # Day 60 is the last summer day; watering still INCs. Fall D1 is day 61.
    st = State(day=60, wallet=1000, bags=0, rings=(RingState(crop=crop, waterings=2, established=True),))
    nxt = _resolve_day(st, cal, CROPS, cost, crop, grapes=0, buy_bags=0, ring_names=ring_names)
    assert cal.season_of(60) == SEASON_SUMMER
    assert nxt.rings[0].waterings == 3, "summer still INCs a watered crop"
    st_fall = State(day=61, wallet=1000, bags=0, rings=(RingState(crop=crop, waterings=2, established=True),))
    nxt_fall = _resolve_day(st_fall, cal, CROPS, cost, crop, grapes=0, buy_bags=0, ring_names=ring_names)
    assert nxt_fall.rings[0].waterings < 2, "an immature watered crop must decay (DEC) in fall"


def test_winter_d1_wipes_all_rings():
    from harvest.planner.spring_opt import _resolve_day, State

    cost = CostModel(evening_frames=100_000)
    cal = Calendar(horizon_end=100)
    crop = "potato"
    ring_names = ("R1", "R2")
    st = State(
        day=91,  # Winter D1 (absolute day = 3*30 + 1)
        wallet=1000, bags=0,
        rings=(
            RingState(crop=crop, waterings=6, established=True),  # mature
            RingState(crop=crop, waterings=2, established=True),  # immature
        ),
    )
    assert cal.is_winter_wipe_day(91)
    nxt = _resolve_day(st, cal, CROPS, cost, crop, grapes=0, buy_bags=0, ring_names=ring_names)
    for r in nxt.rings:
        assert r.crop is None and r.waterings == 0 and not r.established


def test_rain_gives_a_free_growth_stage_without_watering():
    from harvest.planner.spring_opt import _resolve_day, State

    # Zero evening budget: nothing can be watered manually, but rain still
    # advances the crop.
    cost = CostModel(evening_frames=0, home_sleep_f=0)
    cal = Calendar(rain_days=(5,), horizon_end=10)
    crop = "potato"
    st = State(day=5, wallet=1000, bags=0,
               rings=(RingState(crop=crop, waterings=1, established=True),))
    nxt = _resolve_day(st, cal, CROPS, cost, crop, grapes=0, buy_bags=0, ring_names=("R1",))
    assert nxt.rings[0].waterings == 2
    assert not nxt.log[-1].watered  # no frames spent watering -- rain did it


def test_shipping_credit_posts_the_following_morning_not_same_day():
    from harvest.planner.spring_opt import _resolve_day, State

    cost = CostModel(evening_frames=100_000)
    cal = Calendar(horizon_end=10)
    crop = "potato"
    st = State(day=5, wallet=100, bags=0,
               rings=(RingState(crop=crop, waterings=6, established=True),))  # mature
    day5 = _resolve_day(st, cal, CROPS, cost, crop, grapes=0, buy_bags=0, ring_names=("R1",))
    gross = 8 * POTATO.sell_price_g
    # Harvested today: gold is queued, NOT in the same-day wallet.
    assert day5.pending_ship == gross
    assert day5.wallet == 100, "harvest gold must not be spendable the same day"
    assert day5.log[-1].wallet_end == 100
    assert day5.log[-1].pending_ship_end == gross
    # Next morning it posts.
    day6 = _resolve_day(day5, cal, CROPS, cost, crop, grapes=0, buy_bags=0, ring_names=("R1",))
    assert day6.log[-1].pending_ship_posted == gross
    assert day6.wallet >= 100 + gross


def test_summer_d1_checkpoint_and_horizon_end_are_reported_separately():
    cost = CostModel(evening_frames=100_000)
    cal = Calendar(horizon_end=60, checkpoint_day=30)
    plan = optimize_spring(rings=_rings(3), crop="potato", calendar=cal, cost=cost,
                            start_bags=2, beam_width=120)
    assert plan.summer_d1_wallet is not None
    # horizon runs to summer D30; a real optimiser should keep earning past
    # spring D30, so the horizon-end figure should be >= the summer D1 one.
    assert plan.horizon_end_wallet + plan.horizon_end_standing_value >= plan.summer_d1_wallet


# ── Sowing season gate (2026-09-10) ───────────────────────────────────────


def test_potato_cannot_be_sown_in_summer_even_with_bags_and_frames():
    """ROM: sowing a spring crop out of season burns the bag for a dead tile.

    The old model happily "planted potatoes" on Summer days and booked the
    revenue, which is why the through-summer numbers used to be inflated.
    """
    cost = CostModel(evening_frames=100_000)
    cal = Calendar(first_day=31, horizon_end=50, checkpoint_day=45)  # Summer D1..D20
    plan = optimize_spring(rings=_rings(3), crop="potato", calendar=cal, cost=cost,
                           start_bags=3, start_wallet=5_000, beam_width=60)
    assert all(cal.season_of(d.day) == SEASON_SUMMER for d in plan.days)
    assert not any(d.established for d in plan.days), "potato must not sow in summer"
    assert plan.shipped_g == 0


def test_a_spring_sown_ring_still_matures_and_ships_in_summer():
    """The sowing gate must not be confused with the growth rule."""
    cost = CostModel(evening_frames=100_000)
    cal = Calendar(first_day=28, horizon_end=40, checkpoint_day=39)
    plan = optimize_spring(rings=_rings(1), crop="potato", calendar=cal, cost=cost,
                           start_bags=1, beam_width=40)
    sown = [d.day for d in plan.days if d.established]
    assert sown and all(d <= 30 for d in sown), "sowing happens in spring"
    assert plan.shipped_g >= 8 * POTATO.sell_price_g, "and it matures in summer"


def test_default_horizon_is_spring_only_because_summer_is_unmeasured():
    from harvest.planner.spring_opt import DAYS_PER_SEASON

    assert Calendar().horizon_end == DAYS_PER_SEASON
    plan = optimize_spring(rings=_rings(2), crop="potato", beam_width=60)
    assert all(d.day <= DAYS_PER_SEASON for d in plan.days)


# ── Cash-flow ledger ──────────────────────────────────────────────────────


def test_every_ledger_line_balances():
    plan = optimize_spring(rings=_rings(4), crop="potato", beam_width=120)
    for a in plan.days:
        assert a.wallet_end == a.wallet_start + a.total_in_g - a.total_out_g, f"D{a.day}"
    # and each day opens where the previous one closed
    for prev, cur in zip(plan.days, plan.days[1:]):
        assert cur.wallet_start == prev.wallet_end


def test_cashflow_totals_account_for_every_gold_piece_ever_earned():
    plan = optimize_spring(rings=_rings(4), crop="potato", beam_width=120)
    cf = plan.cashflow
    # Gross shipped = everything that posted + whatever is still in the bin.
    assert cf.total_in_g + cf.unposted_at_end_g == plan.shipped_g
    assert cf.crop_income_g == sum(d.crop_income_g for d in plan.days)
    assert cf.berry_income_g == sum(d.berry_income_g for d in plan.days)
    assert cf.seed_spend_g == sum(d.seed_spend_g for d in plan.days)
    assert cf.livestock_spend_g == 0, "livestock is a stub; nothing may spend on it yet"
    assert cf.min_balance_g == min(d.wallet_end for d in plan.days)


def test_ledger_renders_a_row_per_day_and_names_the_binding_constraint():
    plan = optimize_spring(rings=_rings(4), crop="potato", beam_width=120)
    text = plan.ledger()
    for a in plan.days:
        assert f"\n{a.day:>3} " in "\n" + text
    assert "cash-blocked" in text and "frame-blocked" in text


def test_min_cash_reserve_is_never_spent_through():
    reserve = 1_500
    plan = optimize_spring(rings=_rings(4), crop="potato", beam_width=160,
                           min_cash_reserve=reserve)
    for a in plan.days:
        if a.seed_spend_g:
            after_buy = a.wallet_start + a.crop_income_g - a.seed_spend_g
            assert after_buy >= reserve, f"D{a.day} spent through the reserve"


def test_cash_blocked_fires_when_capital_really_is_the_constraint():
    # Broke, no bags, no grape income allowed, and an empty ring with a whole
    # evening free: the only thing missing is 200 G.
    plan = optimize_spring(rings=_rings(1), crop="potato",
                           cost=CostModel(evening_frames=100_000), beam_width=40,
                           start_wallet=0, start_bags=0, allow_grapes_through_day=0)
    assert plan.cashflow.cash_blocked_days, "expected cash to be named as the blocker"
    assert not plan.cashflow.frame_blocked_days
    assert plan.shipped_g == 0


def test_frame_blocked_fires_when_a_bag_is_in_hand_but_the_evening_is_full():
    tight = CostModel(evening_frames=4_000, home_sleep_f=0)
    plan = optimize_spring(rings=_rings(4), crop="potato", cost=tight, beam_width=60,
                           start_wallet=5_000, start_bags=4)
    assert plan.cashflow.frame_blocked_days, "expected frames to be named as the blocker"


def test_earliest_day_affording_reads_realized_cash_not_the_shipping_bin():
    plan = optimize_spring(rings=_rings(4), crop="potato", beam_width=120)
    target = plan.cashflow.peak_balance_g
    day = plan.earliest_day_affording(target)
    assert day is not None
    row = next(d for d in plan.days if d.day == day)
    assert row.wallet_end >= target
    assert all(d.wallet_end < target for d in plan.days if d.day < day)
    assert plan.earliest_day_affording(target + 1) is None
    assert plan.earliest_day_affording(target, keep_reserve_g=1) is None
