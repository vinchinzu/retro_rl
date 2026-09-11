"""Print / dump the optimised Spring shipping plan.

    uv run python -m harvest.scripts.spring_plan --rings 2 --crop potato
    uv run python -m harvest.scripts.spring_plan --rings 4 --out recordings/spring_plan.json
    uv run python -m harvest.scripts.spring_plan --rings 4 --ledger
    uv run python -m harvest.scripts.spring_plan --rings 4 --ledger --chicken
    uv run python -m harvest.scripts.spring_plan --rings 4 --through-summer
    uv run python -m harvest.scripts.spring_plan --sweep
    uv run python -m harvest.scripts.spring_plan --sensitivity
    uv run python -m harvest.scripts.spring_plan --calibrate logs/spring_d3_30/run11_grapefix.log \\
        logs/spring_d3_30/run10_headNOwip.log logs/spring_d3_30/run9_baseline_wip.log
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvest.planner.spring_opt import (
    DAYS_PER_SEASON,
    Calendar,
    CostModel,
    Ring,
    optimize_spring,
)

SUMMER_CAVEAT = (
    "NOTE: --through-summer is a projection, not a measurement. Nothing past "
    "Spring D30 has been run on the ROM by this project: summer frame costs are "
    "unmeasured, corn/tomato have no nav-proven ring sites, and hurricanes "
    "(summer-only, ~1/30 per night, 25% per-tile wipe) are not modelled. "
    "Potato/turnip cannot be sown in summer at all, so these days are a "
    "wind-down of spring plantings. See docs/SPRING_ECONOMY.md §13."
)


def _calendar(args: argparse.Namespace) -> Calendar:
    horizon = 2 * DAYS_PER_SEASON if args.through_summer else DAYS_PER_SEASON
    return Calendar(
        horizon_end=horizon,
        rain_days=tuple(args.rain_days),
        blocked_days=tuple(args.blocked_days),
    )


def _print_chicken(plan, cost: CostModel) -> None:
    from harvest.planner.livestock_econ import CHICKEN, plan_purchase

    verdict = plan_purchase(plan, CHICKEN, cost=cost)
    print("\n=== chicken (stub) ===")
    print(f"  {verdict.reason}")
    if verdict.blocked_on:
        print("  measure these, then this answers itself:")
        for field in verdict.blocked_on:
            print(f"    - {field}")
        # The half of the question the ledger *can* already answer.
        for price in (1000, 1500, 2000):
            day = plan.earliest_day_affording(price)
            when = f"D{day}" if day else "never in this plan"
            print(f"  if a chicken costs {price} G, the plan first affords it {when}")


def _rings(n: int) -> list[Ring]:
    return [Ring(name=f"R{i+1}", order=i) for i in range(n)]


def _rain_days(every: int, first_day: int, horizon_end: int) -> tuple[int, ...]:
    if every <= 0:
        return ()
    return tuple(range(first_day, horizon_end + 1, every))


def _print_sweep(args: argparse.Namespace) -> None:
    if args.through_summer:
        print(SUMMER_CAVEAT + "\n")
    print(
        f"{'rings':>5} {'crop':>7} {'summerD1 G':>10} {'horizonEnd G':>12} "
        f"{'standing G':>10} {'shipped G':>10}"
    )
    for crop in ("potato", "turnip"):
        for n in range(2, 21, 2):
            cal = _calendar(args)
            cost = CostModel(sleep_hour=args.sleep_hour)
            plan = optimize_spring(
                rings=_rings(n), crop=crop, calendar=cal, cost=cost,
                start_wallet=args.start_wallet, start_bags=args.start_bags,
                beam_width=args.beam, allow_grapes_through_day=args.grapes_through,
            )
            s_d1 = "n/a" if plan.summer_d1_wallet is None else plan.summer_d1_wallet
            print(
                f"{n:>5} {crop:>7} {s_d1:>10} {plan.horizon_end_wallet:>12} "
                f"{plan.horizon_end_standing_value:>10} {plan.shipped_g:>10}"
            )


def _print_sensitivity(args: argparse.Namespace) -> None:
    n = args.rings
    print(f"=== sensitivity @ rings={n}, crop={args.crop} ===\n")

    horizon = _calendar(args).horizon_end
    print("-- rain frequency (every Nth day rains, 0 = none) --")
    for every in (0, 7, 4, 2):
        cal = Calendar(horizon_end=horizon, rain_days=_rain_days(every, 3, horizon))
        plan = optimize_spring(rings=_rings(n), crop=args.crop, calendar=cal, beam_width=args.beam)
        print(f"  rain every {every or 'never':>5} -> summerD1={plan.summer_d1_wallet} "
              f"horizonEnd={plan.horizon_end_wallet} shipped={plan.shipped_g}")

    print("\n-- potato vs turnip --")
    for crop in ("potato", "turnip"):
        plan = optimize_spring(rings=_rings(n), crop=crop, calendar=_calendar(args),
                               beam_width=args.beam)
        print(f"  {crop:>7} -> summerD1={plan.summer_d1_wallet} "
              f"horizonEnd={plan.horizon_end_wallet} shipped={plan.shipped_g}")

    print("\n-- grape-run cutoff day (allow_grapes_through_day) --")
    for cutoff in (0, 4, 6, 10, 30):
        plan = optimize_spring(
            rings=_rings(n), crop=args.crop, calendar=_calendar(args), beam_width=args.beam,
            allow_grapes_through_day=cutoff,
        )
        print(f"  cutoff D{cutoff:>3} -> summerD1={plan.summer_d1_wallet} "
              f"horizonEnd={plan.horizon_end_wallet} shipped={plan.shipped_g}")

    print("\n-- grape price (GUESS at 150 G; berries dominate the income ledger) --")
    for value in (0, 75, 150, 225):
        cost = CostModel(grape_value_g=value)
        plan = optimize_spring(rings=_rings(n), crop=args.crop, calendar=_calendar(args),
                               cost=cost, beam_width=args.beam)
        cf = plan.cashflow
        print(f"  grape {value:>3} G -> summerD1={plan.summer_d1_wallet} "
              f"crop_in={cf.crop_income_g} berry_in={cf.berry_income_g}")

    print("\n-- sleep hour (policy: how late the day runs before sleeping) --")
    for sleep_hour in (18, 20, 22):
        cost = CostModel(sleep_hour=sleep_hour)
        plan = optimize_spring(rings=_rings(n), crop=args.crop, calendar=_calendar(args),
                               cost=cost, beam_width=args.beam)
        print(f"  sleep {sleep_hour:>2}:00 (budget={cost.evening_frames}f) -> "
              f"summerD1={plan.summer_d1_wallet} horizonEnd={plan.horizon_end_wallet} "
              f"shipped={plan.shipped_g}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rings", type=int, default=2)
    p.add_argument("--crop", default="potato", choices=["potato", "turnip"])
    p.add_argument("--start-wallet", type=int, default=250)
    p.add_argument("--start-bags", type=int, default=0)
    p.add_argument("--rain-days", type=int, nargs="*", default=[])
    p.add_argument("--blocked-days", type=int, nargs="*", default=[])
    p.add_argument("--grapes-through", type=int, default=30)
    p.add_argument("--sleep-hour", type=int, default=18, help="policy day-end hour (no forced bedtime; raise freely)")
    p.add_argument("--beam", type=int, default=200)
    p.add_argument("--sweep", action="store_true", help="compare 2..20 rings, both crops")
    p.add_argument("--sensitivity", action="store_true",
                   help="rain frequency / crop / grape-cutoff / sleep-hour sensitivity at --rings")
    p.add_argument("--calibrate", nargs="+", metavar="LOG",
                   help="parse campaign log(s) and print measured-vs-default CostModel fields")
    p.add_argument("--ledger", action="store_true",
                   help="print the day-by-day cash in / cash out table")
    p.add_argument("--through-summer", action="store_true",
                   help="project past Spring D30 to Summer D30 (UNMEASURED — see the printed note)")
    p.add_argument("--reserve", type=int, default=0, metavar="G",
                   help="cash floor the plan may never spend below (e.g. saving for livestock)")
    p.add_argument("--chicken", action="store_true",
                   help="run the livestock_econ purchase pre-calc (stub: reports what is unmeasured)")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if args.calibrate:
        from harvest.planner.spring_calibration import build_report
        print(build_report(args.calibrate))
        return

    if args.sweep:
        _print_sweep(args)
        return

    if args.sensitivity:
        _print_sensitivity(args)
        return

    cal = _calendar(args)
    cost = CostModel(sleep_hour=args.sleep_hour)

    plan = optimize_spring(
        rings=_rings(args.rings), crop=args.crop, calendar=cal, cost=cost,
        start_wallet=args.start_wallet, start_bags=args.start_bags,
        beam_width=args.beam, allow_grapes_through_day=args.grapes_through,
        min_cash_reserve=args.reserve,
    )
    if args.through_summer:
        print(SUMMER_CAVEAT + "\n")
    print(plan.table())
    if args.ledger:
        print()
        print(plan.ledger())
    if args.chicken:
        _print_chicken(plan, cost)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan.to_dict(), indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
