"""Livestock (chicken, cow) purchase economics — **STUB, not yet priced**.

The optimiser needs one answer about a chicken: *which day should we buy it,
and is it worth buying at all inside the horizon we care about?* That is a
cash-flow question, so the machinery lives next to the cash-flow ledger
(:class:`harvest.planner.spring_opt.SpringPlan`) rather than next to the
runtime coop tasks (`harvest/tasks/coop_task.py`, which already knows how to
feed, collect and ship — see `docs/livestock_purchase_plan.md`).
Measurement to-do list: `docs/SPRING_ECONOMY.md` §12.

**Everything here except the egg ship price is unmeasured.** The decision
mechanism below is complete and tested; it simply refuses to answer while any
input is ``None``, naming what is missing, rather than inventing a number.
:func:`plan_purchase` returns a :class:`PurchaseVerdict` whose ``blocked_on``
tuple *is* the measurement to-do list. Fill the constants in and the answer
falls out — no other code has to change.

## What is known

- **Egg ship price = 50 G.** `Items_Price_Table` (`bank_81.asm:3060`) reads
  as 16-bit LE values scaled ×10, and the run
  ``12, 10, 8, 6, 5, 15, 25, 35`` sits exactly where corn/tomato/potato/turnip
  (120/100/80/60, already pinned in ``docs/SPRING_ECONOMY.md`` §3.1) anchor
  the block — so the next four are egg 50 and milk S/M/L 150/250/350, which
  matches the external guides the doc previously logged as UNVERIFIED-BY-ROM.
  This is index inference off a matched contiguous block, not a decoded item
  ID table: call it ROM-corroborated, not ROM-decoded (§3.2).

## What is missing (each is a ``blocked_on`` value)

- ``purchase_cost_g`` — not in `Items_Price_Table` (that table is ship
  prices). The buy path is the Animal Shop map; start at
  `ReplaceTilesAnimalShop` (`bank_81.asm:4844`) / `MapAnimalShop`
  (`src/maps/Maps_Graphics.asm:830`) and follow its purchase handler.
- ``products_per_day`` — laying cadence for a fed, happy adult, and whether
  rain/being left outdoors interrupts it.
- ``feed_cost_g_per_day`` — bought feed vs. free fodder from cut grass; these
  are different economies and the second is a frame cost, not a gold cost.
- ``chore_frames_per_day`` — `CoopChoresTask` has never been frame-measured.
  This is the term that actually decides the question: a chicken competes
  with the grape run for the same scarce evening frames, and at the current
  model's rates the grape leg earns ~0.04 G/frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from harvest.planner.spring_opt import CostModel, SpringPlan


@dataclass(frozen=True)
class LivestockSpec:
    """One animal's steady-state economics. ``None`` means "not measured"."""

    name: str
    purchase_cost_g: Optional[int] = None
    product_price_g: Optional[int] = None        # gold per product shipped
    products_per_day: Optional[float] = None     # products a fed adult yields per day
    feed_cost_g_per_day: Optional[int] = None    # gold, if feed is bought rather than cut
    chore_frames_per_day: Optional[int] = None   # feed + collect + ship, per day, per animal
    ready_after_days: int = 0                    # delay from purchase to first product
    source: str = ""

    def missing_inputs(self) -> Tuple[str, ...]:
        return tuple(
            field for field in (
                "purchase_cost_g", "product_price_g", "products_per_day",
                "feed_cost_g_per_day", "chore_frames_per_day",
            )
            if getattr(self, field) is None
        )


CHICKEN = LivestockSpec(
    name="chicken",
    product_price_g=50,                          # ROM-corroborated, see module docstring
    ready_after_days=0,
    source="egg price: Items_Price_Table bank_81.asm:3060 (block-index inference)",
)

COW = LivestockSpec(
    name="cow",
    product_price_g=150,                         # milk S, same block; the S/M/L tier rule is unmeasured
    source="milk price: Items_Price_Table bank_81.asm:3060 (block-index inference)",
)


@dataclass(frozen=True)
class PurchaseVerdict:
    """Answer to "buy this animal, and when?" — or why we cannot say yet.

    ``blocked_on`` non-empty means every other field is ``None`` and the
    verdict is "unknown", never "no". A verdict of ``worth_it=False`` with an
    empty ``blocked_on`` is a real, priced no.
    """

    spec: LivestockSpec
    blocked_on: Tuple[str, ...] = ()
    buy_day: Optional[int] = None                # earliest day the ledger can pay for it
    daily_net_g: Optional[float] = None          # product income minus feed minus frame opportunity cost
    payback_days: Optional[int] = None
    profit_by_horizon_g: Optional[int] = None
    worth_it: Optional[bool] = None
    reason: str = ""


def grape_g_per_frame(cost: CostModel) -> float:
    """The benchmark alternative use of an evening frame.

    Coop chores are not free — they displace whatever else the frame budget
    was buying. In the current model the marginal earner is the grape leg, so
    its rate is the opportunity cost a chicken has to beat.
    """
    return cost.grape_value_g / cost.grape_first_f


def plan_purchase(
    plan: SpringPlan,
    spec: LivestockSpec = CHICKEN,
    *,
    cost: Optional[CostModel] = None,
    horizon_end_day: Optional[int] = None,
    keep_reserve_g: int = 0,
) -> PurchaseVerdict:
    """When (if ever) to buy ``spec``, given a plan's day-by-day cash flow.

    Earlier is strictly better whenever the animal nets positive gold per
    day, so the optimal purchase day is simply the first day the ledger can
    pay for it — which is why this reads ``plan.earliest_day_affording``
    rather than re-searching. The horizon test then asks whether that day
    leaves enough days to pay the purchase back.
    """
    missing = spec.missing_inputs()
    if missing:
        return PurchaseVerdict(
            spec=spec,
            blocked_on=missing,
            reason=f"{spec.name}: unmeasured inputs {', '.join(missing)} — see module docstring",
        )

    cost = cost or CostModel()
    horizon = horizon_end_day if horizon_end_day is not None else plan.horizon_end_day
    frame_cost_g = spec.chore_frames_per_day * grape_g_per_frame(cost)
    daily_net = spec.products_per_day * spec.product_price_g - spec.feed_cost_g_per_day - frame_cost_g

    if daily_net <= 0:
        return PurchaseVerdict(
            spec=spec, daily_net_g=daily_net, worth_it=False,
            reason=(f"{spec.name} nets {daily_net:.1f} G/day after feed and "
                    f"{spec.chore_frames_per_day} f/day of displaced chores"),
        )

    buy_day = plan.earliest_day_affording(spec.purchase_cost_g, keep_reserve_g=keep_reserve_g)
    if buy_day is None:
        return PurchaseVerdict(
            spec=spec, daily_net_g=daily_net, worth_it=False,
            reason=(f"plan never holds {spec.purchase_cost_g + keep_reserve_g} G — "
                    f"peak balance is {plan.cashflow.peak_balance_g} G"),
        )

    payback_days = math.ceil(spec.purchase_cost_g / daily_net)
    earning_days = max(0, horizon - buy_day - spec.ready_after_days)
    profit = int(daily_net * earning_days) - spec.purchase_cost_g
    worth_it = profit > 0
    return PurchaseVerdict(
        spec=spec,
        buy_day=buy_day,
        daily_net_g=daily_net,
        payback_days=payback_days,
        profit_by_horizon_g=profit,
        worth_it=worth_it,
        reason=(f"buy D{buy_day}, {daily_net:.1f} G/day net, pays back in "
                f"{payback_days} d, {profit:+d} G by D{horizon}"),
    )


__all__ = ["LivestockSpec", "PurchaseVerdict", "CHICKEN", "COW",
           "grape_g_per_frame", "plan_purchase"]
