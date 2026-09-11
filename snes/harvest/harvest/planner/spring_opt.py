"""Spring-1 shipping optimiser.

Forward beam search over a farm horizon starting at Spring D3 (given the ROM
crop model: :mod:`harvest.planner.crop_planner` prices + `NightlyFarmTilesCheck`
growth), a per-ring frame-cost table, and the calendar (Sundays / blocked
days / optional rain scenario). Returns the day-by-day action plan that
maximises gold, reported at two checkpoints (see ``SpringPlan``).

Model and evidence: ``docs/SPRING_ECONOMY.md``, ``docs/tasks/rr-20w-idle-day.md``.

This module is pure Python — no emulator, no numpy-heavy work.
``optimize_spring`` is a pure search that reports wallets on ``SpringPlan``.

## Season model (corrected 2026-09-10; see docs/tasks/ for the disassembly read)

Seasons are 0=Spring, 1=Summer, 2=Fall, 3=Winter, 30 days each. Contrary to
the previous version of this module, **spring crops do NOT die at Summer
D1**: `NightlyFarmTilesCheck` (bank 82 @82A811) INCs a watered tile's stage
in seasons 0 *and* 1, DECs it (dies) only in season 2 (Fall), and freezes it
in season 3 (Winter). The only wholesale farm wipe is `MonthlyFarmTilesCheck`
on Winter D1. Mature crops (0x60/0x61) are stable and immune to both the
overnight stall-on-dry rule and (per the same "0x60 never changes again"
fact) Fall decay — only *immature* watered crops decay in Fall.

## Day-budget model (corrected 2026-09-10; see docs/tasks/rr-20w-idle-day.md)

There is no forced bedtime — sleeping at 18:00 is a policy choice, not a
game limit — but the frame budget is now expressed in **in-game minutes**,
not a free-floating frame constant: `frames = minute_frames * minutes`,
`minute_frames = 15` measured from real day-length deltas. `evening_frames`
derives from `(sleep_hour - wake_hour) * 60 * minute_frames` unless
overridden, so raising the day length is a legible policy knob (`sleep_hour`)
rather than an opaque frame budget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from harvest.planner.crop_planner import (
    CROP_SPECS,
    SEASON_FALL,
    SEASON_LENGTH,
    SEASON_SPRING,
    SEASON_SUMMER,
    SEASON_WINTER,
    CropSpec,
)

# ── ROM crop model ────────────────────────────────────────────────────────

DAYS_PER_SEASON = SEASON_LENGTH

# Growth (INC) is legal in Spring and Summer; Fall DECs immature watered
# crops; Winter freezes everything (and MonthlyFarmTilesCheck wipes the farm
# on Winter D1 specifically, handled in Calendar.is_winter_wipe_day).
GROWTH_SEASONS = (SEASON_SPRING, SEASON_SUMMER)
DECAY_SEASON = SEASON_FALL

# Establish sows 8 outer tiles, not the 9th bag slot.
RING_TILES = 8


def _grow_waterings(spec: CropSpec) -> int:
    return spec.days_to_first_harvest


def _ring_gross_g(spec: CropSpec) -> int:
    return RING_TILES * spec.sell_price_g


def _plantable(spec: CropSpec, season: int) -> bool:
    """Can this crop be *sown* in ``season`` and produce a real growing tile?

    ROM (`bank_82_toolused_subrutines.asm:742-936`): potato/turnip only sow a
    live tile when ``season == 0`` (Spring); corn/tomato only in Summer.
    Sowing out of season silently drops a dead placeholder (`$00EB`, never
    touched by `NightlyFarmTilesCheck`) **and still burns the bag charge**.
    Growth (INC) is a separate rule — a spring-sown ring keeps maturing right
    through Summer (see the module docstring) — so this gate is about the
    sowing act only, never about whether standing crops survive.
    """
    return season in spec.seasons


POTATO = CROP_SPECS["potato"]
TURNIP = CROP_SPECS["turnip"]
CROPS = CROP_SPECS


# ── Frame-cost model ──────────────────────────────────────────────────────

# Measured (docs/tasks/rr-20w-idle-day.md, run11_grapefix.log day-change
# frame deltas / a nominal 06:00-18:00 = 720 in-game-minute day): the clock
# runs at a constant rate independent of what the bot does.
MINUTE_FRAMES = 15


@dataclass(frozen=True)
class CostModel:
    """Per-action frame costs.

    Every field below is tagged MEASURED (log-calibrated, with a sample
    count) or GUESS (still speculative — see ``docs/tasks/rr-20w-calibration.md``
    for the full measured-vs-default table and method). None of the
    MEASURED numbers are frame-exact: they come from linear interpolation
    between the run logs' `[RUN] f=<frame>` heartbeats (roughly every 2000
    frames), so treat them as "the right order of magnitude, calibrated
    against real campaign data" rather than cycle-accurate.

    There is **no forced bedtime** (see the module docstring):
    ``evening_frames`` derives from ``(sleep_hour - wake_hour) * 60 *
    minute_frames`` unless overridden, and a crop advances exactly one
    stage per sleep however much work the day held. Ring count is capped
    by the Fall D1 maturity cutoff, seed-capital timing, and mature rings
    piling up faster than a day's budget clears them.
    """

    minute_frames: int = MINUTE_FRAMES               # MEASURED (rr-20w-idle-day.md)
    wake_hour: int = 6                                # POLICY/measured campaign start
    sleep_hour: int = 20                              # MEASURED n=9 (run12 day-change deltas, median 12859 f
                                                      # = 14.3 h from a 06:00 wake). Still a policy knob — no
                                                      # game limit — but 18:00 under-budgeted the real day ~27%.
    evening_frames: Optional[int] = None              # None => derived from wake/sleep hours
    home_sleep_f: int = 2_550                         # MEASURED n=19 median (run13; 354-5299)
    spa_refill_f: int = 2_400                         # GUESS — no HOT_SPRING_STAMINA phase seen in calibration logs
    tool_uses_per_spa: int = 40                       # GUESS — same

    # 2-grape run measured at 6194 f (06:00->13:12) PRE-fix (rr-20w.3.1);
    # every post-fix run (run11_grapefix.log, D3-D7) stops after 1 grape
    # ("mountain grape 1/2 shipped; stopped early (shop window)") — a
    # runtime shop-window cutoff, not evidence the 2nd grape is impossible.
    # grape_first_f below is the MEASURED single-grape trip; grape_marginal_f
    # is still the old GUESS since no clean post-fix 2-grape sample exists.
    grape_first_f: int = 3_350                        # MEASURED n=6 median (run12 1-grape trips; 2733-7200)
    # No longer a guess. run13 (2-grape trips, grape fix live) measures the
    # whole MOUNTAIN_BERRY phase at 6826 f median (n=8, 6192-7913) against
    # run12's 3350 f 1-grape trips -> the 2nd grape costs ~3475 f, near
    # DOUBLE the old guess. The second grape is NOT cheap: at 150 G for
    # 3475 f it earns 0.043 G/f, essentially identical to the first
    # (0.045 G/f). There is no marginal-trip bargain here.
    grape_marginal_f: int = 3_475                     # MEASURED n=8 (run13 composite minus run12 first-leg)
    # MEASURED = 1. run12 shipped "mountain grape 1/2 ... stopped early (shop
    # window)" on 4 of 6 berry days and 2/2 on only 2, then lost one day
    # outright to `pick: farm_to_path: pixel_stuck` and another to a 15:03
    # BERRY_RUN_WINDOW cutoff — D7/D8/D9 earned no grape income at all.
    # Planning 2/day is planning on a bug fix; same caveat as max_bags_per_day.
    grape_max_per_day: int = 2                        # MEASURED: run13 ships 2/2 on 7 of 8 successful trips
    grape_value_g: int = 150                          # MEASURED (run12: "shipping_money=150->300" for 2 grapes)

    shop_roundtrip_f: int = 2_480                     # MEASURED n=5 median (run13; 2089-2573)

    # RUNTIME cap, not a ROM cap. Every BUY_SEEDS in every log reads
    # "bought potato_seeds 0->1": the establish pipeline is single-ring by
    # construction (`WorldProbe.pocket_has_plant_capacity` resolves one ring
    # target — docs/tasks/rr-20w-idle-day.md §1). Until that is fixed, a plan
    # that buys 3 bags in a day is not executable. Raise it to price the fix.
    max_bags_per_day: int = 1

    # Establish: measured *pure hoe+plant loop* (excludes the separately-
    # logged NAV_CROP walk) is tiny (133-364 f, n=3) but NAV_CROP itself is
    # highly variable (61-5000 f) when nav soft-arrives/stalls. Lowered from
    # 3400/2100 but kept a buffer above the clean-path sum rather than
    # adopting the optimistic number outright — see calibration doc.
    establish_first_f: int = 3_000                    # MEASURED (partial) n=3, buffered for nav-tail risk
    establish_replant_f: int = 1_800                  # MEASURED (partial) n=3, buffered for nav-tail risk

    # Water: only ever observed as a *composite* (2 rings + 1 mid-refill,
    # 15-16 tiles) at 1946-3653 f (median 2752, n=4) — logs never isolate a
    # single ring or a refill-free pass, so the 3 terms below are scaled
    # uniformly from the old composite (1800+1100+2800=5700) by the measured
    # ratio (2752/5700 ~= 0.48). This is evidence-driven but underdetermined;
    # see calibration doc.
    water_ring_f: int = 900                           # MEASURED (composite-derived) n=4
    water_ring_marginal_f: int = 550                  # MEASURED (composite-derived) n=4
    refill_f: int = 1_350                             # MEASURED (composite-derived) n=4

    # Harvest: one clean sample, a 7-tile ring pick+ship loop (nav excluded,
    # logged separately) at 1208 f = ~172.6 f/tile. Scaled to an 8-tile ring
    # (1381 f) for harvest_ring_f; harvest_ring_marginal_f scaled by the same
    # ratio vs. the old default (945). Single-sample evidence — flagged low
    # confidence.
    harvest_ring_f: int = 1_650                       # MEASURED n=5 median (run13; 463-3551)
    harvest_ring_marginal_f: int = 1_120              # MEASURED n=5, extrapolated by the old 1400/950 ratio

    # Reliability, not a cap. The grape phase is the least reliable code path
    # in the bot AND the largest income line in this model, and until run13
    # nothing here expressed that. run13 (D3-D22): 15 MOUNTAIN_BERRY phase
    # starts, 8 produced grapes, 7 failed outright across seven distinct
    # failure modes; a further 5 days never started the phase at all because
    # BERRY_RUN_WINDOW had already been starved. 15 grapes shipped over 19
    # days = ~118 G/day against the 300 G/day a perfect 2-grape run implies.
    # See docs/tasks/rr-20w-run13-defects.md.
    #
    # A failed run still burns the full frame cost, so this scales income
    # only — never frames. Set it to 1.0 to model a fixed bot.
    grape_success_rate: float = 0.4                   # MEASURED n=15 (run13)

    can_capacity: int = 20                            # ROM fact (docs/SPRING_ECONOMY.md), not a frame cost

    def __post_init__(self) -> None:
        if self.evening_frames is None:
            object.__setattr__(
                self,
                "evening_frames",
                (self.sleep_hour - self.wake_hour) * 60 * self.minute_frames,
            )


@dataclass(frozen=True)
class Ring:
    """An executable plot. ``order`` is the establish / water sequence index."""

    name: str
    order: int


# ── Calendar ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Calendar:
    """Which days exist, what happens on them, and where the search stops.

    ``horizon_end`` defaults to **Spring D30** — the last day this model is
    grounded in measured campaign data. Summer is fully modelled (growth
    continues, sowing does not — see ``_plantable``) and the search will run
    through it if asked, but nothing past Spring D30 has been observed on the
    ROM by this project yet: no measured summer frame costs, no corn/tomato
    nav-proven ring sites, and an un-modelled ~1/30-per-night hurricane that
    wipes 25 % of tiles. Pass ``horizon_end=2 * DAYS_PER_SEASON`` (or
    ``spring_plan --through-summer``) to project into it, and read the result
    as a projection. See ``docs/SPRING_ECONOMY.md`` §11.
    """

    first_day: int = 3                    # absolute day index; Spring D1 = 1
    horizon_end: int = DAYS_PER_SEASON    # search runs through this absolute day (default: end of Spring)
    checkpoint_day: int = DAYS_PER_SEASON  # end-of-this-day's next-morning post = the "Summer D1" headline number
    spring_d1_weekday: int = 1            # measured: Y1_D3_Morning weekday == 3
    sunday_weekday: int = 0
    blocked_days: Tuple[int, ...] = ()    # festivals: no field work (absolute days)
    rain_days: Tuple[int, ...] = ()       # scenario: free watering that night (absolute days)
    # Last absolute day a *new* watering can still advance growth (INC).
    # ROM: NightlyFarmTilesCheck DECs starting season 2 (Fall) — Fall D1 is
    # absolute day 2*DAYS_PER_SEASON + 1, so the last legal growth day is
    # 2*DAYS_PER_SEASON (end of Summer), independent of horizon_end.
    grow_deadline: int = 2 * DAYS_PER_SEASON

    def season_of(self, day: int) -> int:
        return min((day - 1) // DAYS_PER_SEASON, SEASON_WINTER)

    def day_in_season(self, day: int) -> int:
        return (day - 1) % DAYS_PER_SEASON + 1

    def is_winter_wipe_day(self, day: int) -> bool:
        """MonthlyFarmTilesCheck: the one wholesale farm wipe, Winter D1 only."""
        return self.season_of(day) == SEASON_WINTER and self.day_in_season(day) == 1

    def weekday(self, day: int) -> int:
        return (self.spring_d1_weekday + day - 1) % 7

    def is_sunday(self, day: int) -> bool:
        return self.weekday(day) == self.sunday_weekday

    def shop_open(self, day: int) -> bool:
        return not self.is_sunday(day) and day not in self.blocked_days

    def is_rain(self, day: int) -> bool:
        return day in self.rain_days

    def field_work(self, day: int) -> bool:
        return day not in self.blocked_days

    def can_still_mature(self, day: int, crop: CropSpec) -> bool:
        """True if a crop planted today can finish growing before Fall D1.

        Best case (watered every single night starting the planting night),
        the last watering that completes maturity falls on
        ``day + days_to_first_harvest - 1``.
        """
        return day + _grow_waterings(crop) - 1 <= self.grow_deadline


# ── Search state ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RingState:
    crop: Optional[str] = None
    waterings: int = 0          # growth stage (spring/summer) or remaining stage (fall decay)
    established: bool = False   # ring has been hoed at least once (0x08 stays)

    @property
    def planted(self) -> bool:
        return self.crop is not None

    def mature(self, crops: Dict[str, CropSpec]) -> bool:
        return self.planted and self.waterings >= _grow_waterings(crops[self.crop])


@dataclass(frozen=True)
class DayAction:
    """One day's plan row, doubling as a **cash-flow ledger line**.

    The ledger identity holds exactly, every day (asserted in the tests)::

        wallet_end == wallet_start + total_in_g - total_out_g

    The two income lines differ in *timing*, which is the whole point:
    ``crop_income_g`` is yesterday's harvest crossing this morning's
    NightReset ``AddMoney`` (a one-day lag), while ``berry_income_g`` is
    same-day grape cash and so is the only income that can fund a same-day
    seed run. ``pending_ship_end`` is today's harvest — value, not yet
    cash. See ``docs/SPRING_ECONOMY.md`` §11.
    """

    day: int
    grapes: int = 0
    bags_bought: int = 0
    established: Tuple[str, ...] = ()
    harvested: Tuple[str, ...] = ()
    watered: Tuple[str, ...] = ()
    refills: int = 0
    frames_used: int = 0
    wallet_end: int = 0            # realized cash-in-hand at end of day (excludes today's own harvest)
    pending_ship_end: int = 0      # today's harvest, posts at tomorrow morning's NightReset
    pending_ship_posted: int = 0   # yesterday's harvest, posted this morning (already folded into wallet_end)
    note: str = ""

    # ── cash-flow ledger ──
    wallet_start: int = 0          # cash at wake, before this morning's shipping credit
    berry_income_g: int = 0        # grape/forage sold today (same-day cash, no lag)
    seed_spend_g: int = 0          # seed bags bought today
    livestock_spend_g: int = 0     # chicken/cow purchase — always 0 until livestock_econ is priced
    cash_blocked: bool = False     # an idle ring had the frames to be sown today, but no bag and no cash for one
    frame_blocked: bool = False    # an idle ring had a bag waiting for it, but the evening ran out first

    @property
    def crop_income_g(self) -> int:
        """Shipping-bin credit posted this morning (= yesterday's harvest)."""
        return self.pending_ship_posted

    @property
    def total_in_g(self) -> int:
        return self.crop_income_g + self.berry_income_g

    @property
    def total_out_g(self) -> int:
        return self.seed_spend_g + self.livestock_spend_g

    @property
    def net_g(self) -> int:
        return self.total_in_g - self.total_out_g


@dataclass(frozen=True)
class State:
    day: int
    wallet: int
    bags: int
    rings: Tuple[RingState, ...]
    pending_ship: int = 0                 # queued gold, posts at the next morning's NightReset
    shipped_g: int = 0                    # cumulative gross ever earned (harvest + grapes), timing-independent
    log: Tuple[DayAction, ...] = ()

    def key(self) -> Tuple:
        """Dominance key for beam de-duplication — deliberately money-free.

        Two states on the same day with the same bags and the same farm are
        interchangeable *except* for how much money they hold, and money is
        monotone here: every action is a choice bounded by the wallet, never
        forced by it, so a richer state can replay any poorer state's plan
        and end at least as well off. Keeping wallet in the key would fill
        the beam with wallet-variants of one farm configuration and crowd
        out genuinely different farms — which is exactly how a *cheaper*
        cost model used to score below a dearer one.
        """
        return (self.day, self.bags,
                tuple((r.crop, r.waterings, r.established) for r in self.rings))


@dataclass(frozen=True)
class CashFlow:
    """Whole-plan cash-flow summary — the roll-up of the per-day ledger.

    ``min_balance_g`` is the tightest the wallet ever gets. It is the number
    that matters for a lump-sum purchase (a chicken): a plan that ends rich
    but dips to 40 G on D12 cannot also be carrying a livestock reserve on
    D12. ``cash_blocked_days`` lists the days an idle ring went unsown purely
    for want of 200 G — the direct, addressable cost of a tight ledger.
    """

    crop_income_g: int
    berry_income_g: int
    seed_spend_g: int
    livestock_spend_g: int
    min_balance_g: int
    min_balance_day: int
    peak_balance_g: int
    cash_blocked_days: Tuple[int, ...]
    frame_blocked_days: Tuple[int, ...]
    unposted_at_end_g: int

    @property
    def total_in_g(self) -> int:
        return self.crop_income_g + self.berry_income_g

    @property
    def total_out_g(self) -> int:
        return self.seed_spend_g + self.livestock_spend_g

    @property
    def net_g(self) -> int:
        return self.total_in_g - self.total_out_g


@dataclass(frozen=True)
class SpringPlan:
    """Optimiser output, reported at two checkpoints (do not conflate them).

    - ``summer_d1_wallet``: realized cash-in-hand the morning of Summer D1
      (``calendar.checkpoint_day`` + 1) — the traditional "spring money"
      headline number. ``None`` if the plan's log never reaches that day
      (e.g. a short horizon or a late ``first_day``).
    - ``horizon_end_wallet``: realized cash-in-hand (wallet + any posted
      pending_ship) at the end of the search horizon.
    - ``horizon_end_standing_value``: gross value of mature-but-unharvested
      rings at horizon end, reported *separately* — it is real value, but
      it has not been picked, shipped, or credited.
    - ``final_wallet`` (property): ``horizon_end_wallet +
      horizon_end_standing_value``, kept as a single combined convenience
      figure for callers that don't need the breakdown (this preserves the
      pre-2026-09-10 field name/shape).
    """

    summer_d1_wallet: Optional[int]
    horizon_end_wallet: int
    horizon_end_standing_value: int
    shipped_g: int
    days: Tuple[DayAction, ...]
    rings: Tuple[str, ...]
    crop: str
    horizon_end_day: int
    checkpoint_day: int

    @property
    def final_wallet(self) -> int:
        return self.horizon_end_wallet + self.horizon_end_standing_value

    def table(self) -> str:
        head = (f"{'Day':>3} {'Grp':>3} {'Buy':>3} {'Estab':>14} {'Harv':>14} "
                f"{'Water':>18} {'Rfl':>3} {'Frames':>7} {'Wallet':>7} {'Pend':>6}")
        lines = [head, "-" * len(head)]
        for a in self.days:
            lines.append(
                f"{a.day:>3} {a.grapes:>3} {a.bags_bought:>3} "
                f"{','.join(a.established) or '-':>14} "
                f"{','.join(a.harvested) or '-':>14} "
                f"{','.join(a.watered) or '-':>18} "
                f"{a.refills:>3} {a.frames_used:>7} {a.wallet_end:>7} {a.pending_ship_end:>6}"
                + (f"   {a.note}" if a.note else "")
            )
        lines.append("-" * len(head))
        s_d1 = "n/a" if self.summer_d1_wallet is None else str(self.summer_d1_wallet)
        lines.append(
            f"summer D1 wallet {s_d1} G   horizon-end (D{self.horizon_end_day}) wallet "
            f"{self.horizon_end_wallet} G   standing (unharvested mature) "
            f"{self.horizon_end_standing_value} G   gross shipped {self.shipped_g} G"
        )
        return "\n".join(lines)

    @property
    def cashflow(self) -> CashFlow:
        days = self.days
        balances = [(a.wallet_end, a.day) for a in days] or [(0, self.checkpoint_day)]
        low = min(balances)
        return CashFlow(
            crop_income_g=sum(a.crop_income_g for a in days),
            berry_income_g=sum(a.berry_income_g for a in days),
            seed_spend_g=sum(a.seed_spend_g for a in days),
            livestock_spend_g=sum(a.livestock_spend_g for a in days),
            min_balance_g=low[0],
            min_balance_day=low[1],
            peak_balance_g=max(b for b, _ in balances),
            cash_blocked_days=tuple(a.day for a in days if a.cash_blocked),
            frame_blocked_days=tuple(a.day for a in days if a.frame_blocked),
            unposted_at_end_g=days[-1].pending_ship_end if days else 0,
        )

    def earliest_day_affording(self, amount_g: int, *, keep_reserve_g: int = 0) -> Optional[int]:
        """First day the plan ends with ``amount_g`` in hand above a reserve.

        The pre-calc hook a lump-sum purchase needs (see
        :mod:`harvest.planner.livestock_econ`). Uses ``wallet_end`` — realized
        cash, not counting the same day's un-posted shipping bin — because
        that is what a shop till will actually accept.
        """
        for a in self.days:
            if a.wallet_end >= amount_g + keep_reserve_g:
                return a.day
        return None

    def ledger(self) -> str:
        """Day-by-day cash in / cash out — the input-output table.

        Columns are sources, not totals: ``crop`` is the shipping-bin credit
        that posted this morning (yesterday's harvest), ``berry`` is same-day
        grape cash, ``seeds``/``stock`` are the outflows. ``bin`` is today's
        harvest sitting in the shipping bin — value earned but not yet cash,
        which is why it is kept out of the balance column.
        """
        head = (f"{'Day':>3} {'Sn':>2} {'Open':>7} | {'crop':>6} {'berry':>6} {'IN':>6} | "
                f"{'seeds':>6} {'stock':>6} {'OUT':>6} | {'net':>6} {'Balance':>8} {'bin':>6}  note")
        lines = [head, "-" * len(head)]
        season_tag = {SEASON_SPRING: "Sp", SEASON_SUMMER: "Su",
                      SEASON_FALL: "Fa", SEASON_WINTER: "Wi"}
        for a in self.days:
            sn = season_tag.get(min((a.day - 1) // DAYS_PER_SEASON, SEASON_WINTER), "??")
            lines.append(
                f"{a.day:>3} {sn:>2} {a.wallet_start:>7} | "
                f"{a.crop_income_g or '':>6} {a.berry_income_g or '':>6} {a.total_in_g or '':>6} | "
                f"{a.seed_spend_g or '':>6} {a.livestock_spend_g or '':>6} {a.total_out_g or '':>6} | "
                f"{a.net_g:>6} {a.wallet_end:>8} {a.pending_ship_end or '':>6}  {a.note}"
            )
        cf = self.cashflow
        lines.append("-" * len(head))
        lines.append(
            f"{'TOTAL':>14} | {cf.crop_income_g:>6} {cf.berry_income_g:>6} {cf.total_in_g:>6} | "
            f"{cf.seed_spend_g:>6} {cf.livestock_spend_g:>6} {cf.total_out_g:>6} | {cf.net_g:>6}"
        )
        lines.append(
            f"min balance {cf.min_balance_g} G on D{cf.min_balance_day}   "
            f"peak {cf.peak_balance_g} G   unposted in bin at end {cf.unposted_at_end_g} G"
        )
        # Which lever to pull: these two lines name the binding constraint.
        seed = CROPS[self.crop].seed_cost_g
        lines.append(
            f"cash-blocked (ring + frames ready, < {seed} G): "
            + (",".join(f"D{d}" for d in cf.cash_blocked_days) or "never — cash is not the constraint")
        )
        lines.append(
            "frame-blocked (ring + bag ready, evening full): "
            + (",".join(f"D{d}" for d in cf.frame_blocked_days) or "never")
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        cf = self.cashflow
        return {
            "summer_d1_wallet_g": self.summer_d1_wallet,
            "horizon_end_wallet_g": self.horizon_end_wallet,
            "horizon_end_standing_value_g": self.horizon_end_standing_value,
            "final_wallet_g": self.final_wallet,
            "shipped_gross_g": self.shipped_g,
            "crop": self.crop,
            "rings": list(self.rings),
            "horizon_end_day": self.horizon_end_day,
            "checkpoint_day": self.checkpoint_day,
            "cashflow": {
                "crop_income_g": cf.crop_income_g,
                "berry_income_g": cf.berry_income_g,
                "seed_spend_g": cf.seed_spend_g,
                "livestock_spend_g": cf.livestock_spend_g,
                "total_in_g": cf.total_in_g,
                "total_out_g": cf.total_out_g,
                "net_g": cf.net_g,
                "min_balance_g": cf.min_balance_g,
                "min_balance_day": cf.min_balance_day,
                "peak_balance_g": cf.peak_balance_g,
                "cash_blocked_days": list(cf.cash_blocked_days),
                "frame_blocked_days": list(cf.frame_blocked_days),
                "unposted_at_end_g": cf.unposted_at_end_g,
            },
            "days": [
                {
                    "day": a.day,
                    "grapes": a.grapes,
                    "bags_bought": a.bags_bought,
                    "established": list(a.established),
                    "harvested": list(a.harvested),
                    "watered": list(a.watered),
                    "refills": a.refills,
                    "frames_used": a.frames_used,
                    "wallet_end": a.wallet_end,
                    "pending_ship_end": a.pending_ship_end,
                    "pending_ship_posted": a.pending_ship_posted,
                    "wallet_start": a.wallet_start,
                    "crop_income_g": a.crop_income_g,
                    "berry_income_g": a.berry_income_g,
                    "seed_spend_g": a.seed_spend_g,
                    "livestock_spend_g": a.livestock_spend_g,
                    "cash_blocked": a.cash_blocked,
                    "frame_blocked": a.frame_blocked,
                    "note": a.note,
                }
                for a in self.days
            ],
        }


# ── Day resolution ────────────────────────────────────────────────────────


def _water_plan(
    rings: Sequence[RingState],
    crops: Dict[str, CropSpec],
    cost: CostModel,
    budget_left: int,
    is_rain: bool,
) -> Tuple[List[int], int, int]:
    """Pick which planted-not-mature rings to water within ``budget_left``.

    Rain waters everything for free. Otherwise water rings closest to
    maturity first — a genuine exchange-argument optimum, not a heuristic,
    because every ring in one run shares a crop (so the marginal frame cost
    of watering one more ring does not depend on *which*) and finishing a
    ring sooner weakly dominates finishing it later. A future mixed-crop
    revision would need a real value-density sort instead. Returns (ring
    indices watered, frames spent, refills).
    """
    if is_rain:
        # Nightly rain is a clock effect, not a watering action the bot took.
        return [], 0, 0

    candidates = [
        i for i, r in enumerate(rings)
        if r.planted and not r.mature(crops)
    ]
    candidates.sort(
        key=lambda i: (
            _grow_waterings(crops[rings[i].crop]) - rings[i].waterings,
            -_ring_gross_g(crops[rings[i].crop]),
        )
    )

    watered: List[int] = []
    frames = 0
    refills = 0
    tiles = 0             # tiles since last can refill
    swings = 0            # tool uses since last spa top-up
    for n, i in enumerate(candidates):
        tiles_here = RING_TILES
        add_frames = cost.water_ring_f if not watered else cost.water_ring_marginal_f
        add_refills = 0
        if tiles + tiles_here > cost.can_capacity:
            add_frames += cost.refill_f
            add_refills = 1
            tiles = 0
        if swings + tiles_here > cost.tool_uses_per_spa:
            add_frames += cost.spa_refill_f
            swings = 0
        if frames + add_frames > budget_left:
            break
        frames += add_frames
        refills += add_refills
        tiles += tiles_here
        swings += tiles_here
        watered.append(i)
    return watered, frames, refills


def _resolve_day(
    state: State,
    cal: Calendar,
    crops: Dict[str, CropSpec],
    cost: CostModel,
    crop_name: str,
    grapes: int,
    buy_bags: int,
    ring_names: Sequence[str],
    min_cash_reserve: int = 0,
) -> Optional[State]:
    """Apply one day's top-level choices; returns the next-morning state.

    Deterministic within a (grapes, buy_bags) branch: post yesterday's
    shipping credit, harvest every mature ring (queued, not immediate —
    see the pending_ship note below), replant with any bag in hand, then
    water toward maturity.

    Shipping-credit lag: harvested gold posts at the *next* morning's
    NightReset ``AddMoney`` — it is not available to buy seeds the same
    day. ``state.pending_ship`` carries today's harvest forward; it is
    folded into ``wallet`` as the very first step of *tomorrow's*
    ``_resolve_day`` call, before that day's grape/shop decisions see the
    wallet. Grape income is treated as immediate (same-day) — that is an
    unverified assumption (the ROM read this session covered the shipping
    bin specifically, not mountain-berry income); flagged in the report.
    """
    day = state.day
    season = cal.season_of(day)
    crop = crops[crop_name]
    budget = cost.evening_frames - cost.home_sleep_f
    frames = 0

    # Post yesterday's harvest (NightReset AddMoney) before any spending.
    wallet_start = state.wallet
    wallet = state.wallet + state.pending_ship
    pending_ship_posted = state.pending_ship
    pending_ship = 0
    bags = state.bags
    rings = list(state.rings)

    winter_wipe = cal.is_winter_wipe_day(day)
    if winter_wipe:
        # MonthlyFarmTilesCheck: the one wholesale farm wipe (Winter D1).
        rings = [RingState() for _ in rings]

    grapes = min(grapes, cost.grape_max_per_day)
    if grapes and not cal.is_sunday(day):
        gf = cost.grape_first_f + (grapes - 1) * cost.grape_marginal_f
        if frames + gf > budget:
            if frames + cost.grape_first_f <= budget:
                grapes, gf = 1, cost.grape_first_f
            else:
                grapes, gf = 0, 0
        frames += gf
    else:
        grapes = 0

    # Seed capital is the binding constraint early, so the buy gate is the
    # cash-flow chokepoint: a bag is only worth buying if it can actually be
    # sown (season + maturity horizon) and if paying for it leaves the
    # reserve intact (``min_cash_reserve`` is how a livestock purchase gets
    # saved for -- see harvest.planner.livestock_econ).
    seed_spend = 0
    can_sow = _plantable(crop, season) and cal.can_still_mature(day, crop)
    if buy_bags and cal.shop_open(day) and can_sow:
        buy_bags = min(buy_bags, cost.max_bags_per_day)
        spendable = max(0, wallet - min_cash_reserve)
        if frames + cost.shop_roundtrip_f <= budget and spendable >= crop.seed_cost_g:
            affordable = min(buy_bags, spendable // crop.seed_cost_g)
            if affordable:
                frames += cost.shop_roundtrip_f
                seed_spend = affordable * crop.seed_cost_g
                wallet -= seed_spend
                bags += affordable
                buy_bags = affordable
            else:
                buy_bags = 0
        else:
            buy_bags = 0
    else:
        buy_bags = 0

    established: List[str] = []
    harvested: List[str] = []

    if cal.field_work(day):
        # Harvest mature rings — gold queues to pending_ship, not wallet.
        for i, r in enumerate(rings):
            hf = cost.harvest_ring_f if not harvested else cost.harvest_ring_marginal_f
            if r.mature(crops) and frames + hf <= budget:
                frames += hf
                gross = _ring_gross_g(crops[r.crop])
                state = replace(state, shipped_g=state.shipped_g + gross)
                pending_ship += gross
                harvested.append(ring_names[i])
                rings[i] = RingState(crop=None, waterings=0, established=True)

        # Establish: fill empty rings while a bag is in hand. Sowing out of
        # season is not modelled as a weak move but as an illegal one -- the
        # ROM burns the bag and leaves a dead placeholder, so a plan that
        # "plants potatoes in Summer" is booking revenue that never arrives.
        if can_sow:
            for i, r in enumerate(rings):
                if bags <= 0:
                    break
                if r.planted:
                    continue
                ef = cost.establish_replant_f if r.established else cost.establish_first_f
                if frames + ef > budget:
                    continue
                frames += ef
                bags -= 1
                rings[i] = RingState(crop=crop_name, waterings=0, established=True)
                established.append(ring_names[i])

    # Which resource actually kept a ring idle today? The two flags are
    # deliberately exclusive and both demand that everything *else* was
    # ready: an empty sowable ring existed on a working day. Then either the
    # evening had room and the wallet did not (cash_blocked), or a bag was
    # already in the pocket and the evening had no room left (frame_blocked).
    # Without the "everything else was ready" clause a 20-ring farm would
    # report cash pressure on every broke day and blame the wrong lever.
    cheapest_establish = min(
        (cost.establish_replant_f if r.established else cost.establish_first_f
         for r in rings if not r.planted),
        default=None,
    )
    ring_waiting = cal.field_work(day) and can_sow and cheapest_establish is not None
    frames_available = ring_waiting and frames + cheapest_establish <= budget
    cash_blocked = (
        ring_waiting and frames_available and bags <= 0
        and wallet < crop.seed_cost_g + min_cash_reserve
    )
    frame_blocked = ring_waiting and not frames_available and bags > 0

    # Water toward maturity.
    watered_idx, wf, refills = _water_plan(
        rings, crops, cost, budget - frames, cal.is_rain(day)
    )
    frames += wf
    watered = [ring_names[i] for i in watered_idx]
    wi = set(watered_idx)

    # Night: grow (spring/summer), decay (fall), or freeze (winter).
    new_rings: List[RingState] = []
    for i, r in enumerate(rings):
        if not r.planted:
            new_rings.append(r)
            continue
        watered_tonight = (i in wi) or cal.is_rain(day)
        if season in GROWTH_SEASONS:
            if watered_tonight:
                cap = _grow_waterings(crops[r.crop])
                new_rings.append(replace(r, waterings=min(r.waterings + 1, cap)))
            else:
                new_rings.append(r)  # stall, no regress
        elif season == DECAY_SEASON:
            # Mature crops are stable (ROM: 0x60 never changes again) —
            # only immature watered crops DEC. Unverified: whether rain's
            # free-water flag also triggers this DEC in Fall the same way
            # it triggers INC in Spring/Summer; assumed yes (same "watered"
            # flag drives both rules in the disassembly this session read),
            # flagged in the report.
            if watered_tonight and not r.mature(crops):
                new_stage = r.waterings - 1
                if new_stage < 0:
                    new_rings.append(RingState(crop=None, waterings=0, established=r.established))
                else:
                    new_rings.append(replace(r, waterings=new_stage))
            else:
                new_rings.append(r)
        else:  # winter: frozen (the wipe, if any, already ran this morning)
            new_rings.append(r)

    # Expected value: the trip's frames are spent whether or not it works
    # (charged in full above), so only the income is discounted.
    berry_income = int(grapes * cost.grape_value_g * cost.grape_success_rate)
    notes = []
    if winter_wipe:
        notes.append("winter wipe")
    if not cal.field_work(day):
        notes.append("festival")
    if cal.is_rain(day):
        notes.append("rain")
    if cash_blocked:
        notes.append("cash-blocked")
    if frame_blocked:
        notes.append("frame-blocked")

    action = DayAction(
        day=day, grapes=grapes, bags_bought=buy_bags,
        established=tuple(established), harvested=tuple(harvested),
        watered=tuple(watered), refills=refills, frames_used=frames,
        wallet_end=wallet + berry_income,
        pending_ship_end=pending_ship,
        pending_ship_posted=pending_ship_posted,
        note=" ".join(notes),
        wallet_start=wallet_start,
        berry_income_g=berry_income,
        seed_spend_g=seed_spend,
        livestock_spend_g=0,
        cash_blocked=cash_blocked,
        frame_blocked=frame_blocked,
    )

    return State(
        day=day + 1,
        wallet=wallet + berry_income,
        bags=bags,
        rings=tuple(new_rings),
        pending_ship=pending_ship,
        shipped_g=state.shipped_g + berry_income,
        log=state.log + (action,),
    )


# ── Beam search ───────────────────────────────────────────────────────────


def _state_value(st: State, crops: Dict[str, CropSpec]) -> int:
    """Liquid wallet + queued shipment + linear growth credit on standing crops.

    Used both to break ties when de-duplicating same-key beam states and as
    the dominant term of ``_rank`` — kept as one function so the two never
    silently disagree about what "better" means.
    """
    liquid = st.wallet + st.pending_ship
    growth = 0
    for r in st.rings:
        if r.planted:
            c = crops[r.crop]
            growth += int(_ring_gross_g(c) * r.waterings / max(1, _grow_waterings(c)))
    return liquid + growth


def _rank(crops: Dict[str, CropSpec], cost: CostModel, cal: Calendar, crop: str):
    """Beam ordering: liquid + standing value, plus credit for usable bags.

    A bag in the pocket is only worth carrying while the crop can still be
    sown — after the sowing season closes (or the maturity horizon passes) it
    is 200 G of dead stock, and crediting it would rank a stranded-capital
    plan above a liquid one.
    """
    spec = crops[crop]

    def score(st: State) -> Tuple:
        sowable = (_plantable(spec, cal.season_of(st.day))
                   and cal.can_still_mature(st.day, spec))
        bag_credit = st.bags * 150 if sowable else 0
        return (_state_value(st, crops) + bag_credit, -st.day)
    return score


def _standing_value(st: State, crops: Dict[str, CropSpec]) -> int:
    total = 0
    for r in st.rings:
        if r.mature(crops):
            total += _ring_gross_g(crops[r.crop])
    return total


def _checkpoint_value(state: State, checkpoint_day: int) -> Optional[int]:
    """Realized cash the morning after ``checkpoint_day`` (posted, pre-spend)."""
    for a in state.log:
        if a.day == checkpoint_day:
            return a.wallet_end + a.pending_ship_end
    return None


def optimize_spring(
    *,
    rings: Sequence[Ring],
    crop: str = "potato",
    calendar: Optional[Calendar] = None,
    cost: Optional[CostModel] = None,
    start_wallet: int = 250,
    start_bags: int = 0,
    beam_width: int = 200,
    allow_grapes_through_day: int = 30,
    optimize_for: str = "horizon_end",
    min_cash_reserve: int = 0,
) -> SpringPlan:
    """Beam-search the day-by-day plan.

    ``optimize_for`` selects which scalar the final argmax (and thus which
    plan is reported) optimises: ``"horizon_end"`` (default) maximises
    realized wallet + queued shipment + standing crop value at the end of
    ``calendar.horizon_end``; ``"summer_d1"`` maximises the realized wallet
    at ``calendar.checkpoint_day`` + 1 instead. Both checkpoints are always
    computed and returned on the ``SpringPlan`` regardless of which one
    drove the search.

    ``min_cash_reserve`` is a floor the plan may never spend below: seed
    buys must leave at least that much in hand. Set it to a livestock price
    to ask "what does the spring look like if I am also saving for a
    chicken?" — see :mod:`harvest.planner.livestock_econ`.
    """
    cal = calendar or Calendar()
    cost = cost or CostModel()
    crops = CROPS

    ordered = sorted(rings, key=lambda r: r.order)
    ring_names = tuple(r.name for r in ordered)
    ring_states = tuple(RingState() for _ in ordered)
    init = State(day=cal.first_day, wallet=start_wallet, bags=start_bags,
                 rings=ring_states)

    beam: List[State] = [init]
    while beam and beam[0].day <= cal.horizon_end:
        nxt: Dict[Tuple, State] = {}
        for st in beam:
            grape_opts = (0, 1, 2) if st.day <= allow_grapes_through_day else (0,)
            # buy up to (empty rings + 1); capped by search breadth
            empty = sum(1 for r in st.rings if not r.planted)
            sowable = (_plantable(crops[crop], cal.season_of(st.day))
                       and cal.can_still_mature(st.day, crops[crop]))
            buy_cap = min(empty, cost.max_bags_per_day)
            buy_opts = tuple(range(0, buy_cap + 1)) if sowable else (0,)
            for g in grape_opts:
                for b in buy_opts:
                    child = _resolve_day(st, cal, crops, cost, crop, g, b, ring_names,
                                         min_cash_reserve=min_cash_reserve)
                    if child is None:
                        continue
                    k = child.key()
                    if k not in nxt or _state_value(child, crops) > _state_value(nxt[k], crops):
                        nxt[k] = child
        if not nxt:
            break
        beam = sorted(nxt.values(), key=_rank(crops, cost, cal, crop), reverse=True)[:beam_width]

    if not beam:
        beam = [init]

    if optimize_for == "summer_d1":
        def _key(s: State) -> int:
            v = _checkpoint_value(s, cal.checkpoint_day)
            return v if v is not None else -1
        best = max(beam, key=_key)
    else:
        best = max(beam, key=lambda s: s.wallet + s.pending_ship + _standing_value(s, crops))

    standing = _standing_value(best, crops)
    summer_d1 = _checkpoint_value(best, cal.checkpoint_day)
    return SpringPlan(
        summer_d1_wallet=summer_d1,
        horizon_end_wallet=best.wallet + best.pending_ship,
        horizon_end_standing_value=standing,
        shipped_g=best.shipped_g,
        days=best.log,
        rings=ring_names,
        crop=crop,
        horizon_end_day=cal.horizon_end,
        checkpoint_day=cal.checkpoint_day,
    )


__all__ = [
    "SEASON_SPRING", "SEASON_SUMMER", "SEASON_FALL", "SEASON_WINTER",
    "DAYS_PER_SEASON", "MINUTE_FRAMES",
    "CROPS", "POTATO", "TURNIP",
    "CostModel", "Calendar", "Ring", "CashFlow",
    "RingState", "State", "DayAction", "SpringPlan",
    "optimize_spring",
]
