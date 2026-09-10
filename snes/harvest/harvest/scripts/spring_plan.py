"""Print / dump the optimised Spring-1 shipping plan.

    uv run python -m harvest.scripts.spring_plan --rings 2 --crop potato
    uv run python -m harvest.scripts.spring_plan --rings 4 --out recordings/spring_plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvest.planner.spring_opt import (
    Calendar,
    CostModel,
    Ring,
    optimize_spring,
)


def _rings(n: int) -> list[Ring]:
    return [Ring(name=f"R{i+1}", order=i) for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rings", type=int, default=2)
    p.add_argument("--crop", default="potato", choices=["potato", "turnip"])
    p.add_argument("--start-wallet", type=int, default=250)
    p.add_argument("--start-bags", type=int, default=0)
    p.add_argument("--rain-days", type=int, nargs="*", default=[])
    p.add_argument("--blocked-days", type=int, nargs="*", default=[])
    p.add_argument("--grapes-through", type=int, default=8)
    p.add_argument("--beam", type=int, default=200)
    p.add_argument("--sweep", action="store_true", help="compare 1..6 rings, both crops")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    cal = Calendar(
        rain_days=tuple(args.rain_days),
        blocked_days=tuple(args.blocked_days),
    )
    cost = CostModel()

    if args.sweep:
        print(f"{'rings':>5} {'crop':>7} {'final G':>9} {'shipped G':>10}")
        for crop in ("potato", "turnip"):
            for n in range(1, 7):
                plan = optimize_spring(
                    rings=_rings(n), crop=crop, calendar=cal, cost=cost,
                    start_wallet=args.start_wallet, start_bags=args.start_bags,
                    beam_width=args.beam, allow_grapes_through_day=args.grapes_through,
                )
                print(f"{n:>5} {crop:>7} {plan.final_wallet:>9} {plan.shipped_g:>10}")
        return

    plan = optimize_spring(
        rings=_rings(args.rings), crop=args.crop, calendar=cal, cost=cost,
        start_wallet=args.start_wallet, start_bags=args.start_bags,
        beam_width=args.beam, allow_grapes_through_day=args.grapes_through,
    )
    print(plan.table())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan.to_dict(), indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
