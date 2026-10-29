from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from harvest.planner.crop_planner import ( CROP_SPECS, SEASON_FALL, SEASON_LENGTH, SEASON_SPRING, SEASON_SUMMER, SEASON_WINTER, CropSpec, )

DAYS_PER_SEASON = SEASON_LENGTH

GROWTH_SEASONS = (SEASON_SPRING, SEASON_SUMMER)
DECAY_SEASON = SEASON_FALL

RING_TILES = 8

def _grow_waterings(spec: CropSpec) -> int:
    return spec.days_to_first_harvest

def _ring_gross_g(spec: CropSpec) -> int:
    return RING_TILES * spec.sell_price_g

def _plantable(spec: CropSpec, season: int) -> bool:
    return season in spec.seasons

POTATO = CROP_SPECS["potato"]
TURNIP = CROP_SPECS["turnip"]
CROPS = CROP_SPECS

MINUTE_FRAMES = 15

@dataclass(frozen=True)
class CostModel:

    minute_frames: int = MINUTE_FRAMES               # MEASURED (rr-20w-idle-day.md)
    wake_hour: int = 6                                # POLICY/measured campaign start
    sleep_hour: int = 20                              # MEASURED n=9 (run12 day-change deltas, median 12859 f
    evening_frames: Optional[int] = None              # None => derived from wake/sleep hours
    home_sleep_f: int = 2_550                         # MEASURED n=19 median (run13; 354-5299)
    spa_refill_f: int = 2_400                         # GUESS — no HOT_SPRING_STAMINA phase seen in calibration logs
    tool_uses_per_spa: int = 40                       # GUESS — same
    grape_first_f: int = 3_350                        # MEASURED n=6 median (run12 1-grape trips; 2733-7200)
    grape_marginal_f: int = 3_475                     # MEASURED n=8 (run13 composite minus run12 first-leg)
    grape_max_per_day: int = 2                        # MEASURED: run13 ships 2/2 on 7 of 8 successful trips
    grape_value_g: int = 150                          # MEASURED (run12: "shipping_money=150->300" for 2 grapes)
    shop_roundtrip_f: int = 2_480                     # MEASURED n=5 median (run13; 2089-2573)
    max_bags_per_day: int = 1
    establish_first_f: int = 3_000                    # MEASURED (partial) n=3, buffered for nav-tail risk
    establish_replant_f: int = 1_800                  # MEASURED (partial) n=3, buffered for nav-tail risk
    water_ring_f: int = 900                           # MEASURED (composite-derived) n=4
    water_ring_marginal_f: int = 550                  # MEASURED (composite-derived) n=4
    refill_f: int = 1_350                             # MEASURED (composite-derived) n=4
    harvest_ring_f: int = 1_650                       # MEASURED n=5 median (run13; 463-3551)
    harvest_ring_marginal_f: int = 1_120              # MEASURED n=5, extrapolated by the old 1400/950 ratio
    grape_success_rate: float = 0.4                   # MEASURED n=15 (run13)
    can_capacity: int = 20                            # ROM fact (docs/SPRING_ECONOMY.md), not a frame cost
    def __post_init__(self) -> None:
        if self.evening_frames is None:
            object.__setattr__( self, "evening_frames", (self.sleep_hour - self.wake_hour) * 60 * self.minute_frames, )

@dataclass(frozen=True)
class Ring:

    name: str
    order: int

@dataclass(frozen=True)
class Calendar:

    first_day: int = 3                    # absolute day index; Spring D1 = 1
    horizon_end: int = DAYS_PER_SEASON    # search runs through this absolute day (default: end of Spring)
    checkpoint_day: int = DAYS_PER_SEASON  # end-of-this-day's next-morning post = the "Summer D1" headline number
    spring_d1_weekday: int = 1            # measured: Y1_D3_Morning weekday == 3
    sunday_weekday: int = 0
    blocked_days: Tuple[int, ...] = ()    # festivals: no field work (absolute days)
    rain_days: Tuple[int, ...] = ()       # scenario: free watering that night (absolute days)
    grow_deadline: int = 2 * DAYS_PER_SEASON
    def season_of(self, day: int) -> int:
        return min((day - 1) // DAYS_PER_SEASON, SEASON_WINTER)
    def day_in_season(self, day: int) -> int:
        return (day - 1) % DAYS_PER_SEASON + 1
    def is_winter_wipe_day(self, day: int) -> bool:
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
        return day + _grow_waterings(crop) - 1 <= self.grow_deadline

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
    wallet_start: int = 0          # cash at wake, before this morning's shipping credit
    berry_income_g: int = 0        # grape/forage sold today (same-day cash, no lag)
    seed_spend_g: int = 0          # seed bags bought today
    livestock_spend_g: int = 0     # chicken/cow purchase — always 0 until livestock_econ is priced
    cash_blocked: bool = False     # an idle ring had the frames to be sown today, but no bag and no cash for one
    frame_blocked: bool = False    # an idle ring had a bag waiting for it, but the evening ran out first
    @property
    def crop_income_g(self) -> int:
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
        return (self.day, self.bags, tuple((r.crop, r.waterings, r.established) for r in self.rings))

@dataclass(frozen=True)
class CashFlow:

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
        for a in self.days:
            if a.wallet_end >= amount_g + keep_reserve_g:
                return a.day
        return None
    def ledger(self) -> str:
        head = (f"{'Day':>3} {'Sn':>2} {'Open':>7} | {'crop':>6} {'berry':>6} {'IN':>6} | "
                f"{'seeds':>6} {'stock':>6} {'OUT':>6} | {'net':>6} {'Balance':>8} {'bin':>6}  note")
        lines = [head, "-" * len(head)]
        season_tag = {SEASON_SPRING: "Sp", SEASON_SUMMER: "Su", SEASON_FALL: "Fa", SEASON_WINTER: "Wi"}
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
        seed = CROPS[self.crop].seed_cost_g
        lines.append(
            f"cash-blocked (ring + frames ready, < {seed} G): "
            + (",".join(f"D{d}" for d in cf.cash_blocked_days) or "never — cash is not the constraint")
        )
        lines.append( "frame-blocked (ring + bag ready, evening full): " + (",".join(f"D{d}" for d in cf.frame_blocked_days) or "never") )
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

def _water_plan(
    rings: Sequence[RingState],
    crops: Dict[str, CropSpec],
    cost: CostModel,
    budget_left: int,
    is_rain: bool,
) -> Tuple[List[int], int, int]:
    if is_rain:
        return [], 0, 0
    candidates = [ i for i, r in enumerate(rings) if r.planted and not r.mature(crops) ]
    candidates.sort( key=lambda i: ( _grow_waterings(crops[rings[i].crop]) - rings[i].waterings, -_ring_gross_g(crops[rings[i].crop]), ) )
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
    day = state.day
    season = cal.season_of(day)
    crop = crops[crop_name]
    budget = cost.evening_frames - cost.home_sleep_f
    frames = 0
    wallet_start = state.wallet
    wallet = state.wallet + state.pending_ship
    pending_ship_posted = state.pending_ship
    pending_ship = 0
    bags = state.bags
    rings = list(state.rings)
    winter_wipe = cal.is_winter_wipe_day(day)
    if winter_wipe:
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
        for i, r in enumerate(rings):
            hf = cost.harvest_ring_f if not harvested else cost.harvest_ring_marginal_f
            if r.mature(crops) and frames + hf <= budget:
                frames += hf
                gross = _ring_gross_g(crops[r.crop])
                state = replace(state, shipped_g=state.shipped_g + gross)
                pending_ship += gross
                harvested.append(ring_names[i])
                rings[i] = RingState(crop=None, waterings=0, established=True)
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
    cheapest_establish = min(
        (cost.establish_replant_f if r.established else cost.establish_first_f for r in rings if not r.planted),
        default=None,
    )
    ring_waiting = cal.field_work(day) and can_sow and cheapest_establish is not None
    frames_available = ring_waiting and frames + cheapest_establish <= budget
    cash_blocked = ( ring_waiting and frames_available and bags <= 0 and wallet < crop.seed_cost_g + min_cash_reserve )
    frame_blocked = ring_waiting and not frames_available and bags > 0
    watered_idx, wf, refills = _water_plan( rings, crops, cost, budget - frames, cal.is_rain(day) )
    frames += wf
    watered = [ring_names[i] for i in watered_idx]
    wi = set(watered_idx)
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

def _state_value(st: State, crops: Dict[str, CropSpec]) -> int:
    liquid = st.wallet + st.pending_ship
    growth = 0
    for r in st.rings:
        if r.planted:
            c = crops[r.crop]
            growth += int(_ring_gross_g(c) * r.waterings / max(1, _grow_waterings(c)))
    return liquid + growth

def _rank(crops: Dict[str, CropSpec], cost: CostModel, cal: Calendar, crop: str):
    spec = crops[crop]
    def score(st: State) -> Tuple:
        sowable = (_plantable(spec, cal.season_of(st.day)) and cal.can_still_mature(st.day, spec))
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
    cal = calendar or Calendar()
    cost = cost or CostModel()
    crops = CROPS
    ordered = sorted(rings, key=lambda r: r.order)
    ring_names = tuple(r.name for r in ordered)
    ring_states = tuple(RingState() for _ in ordered)
    init = State(day=cal.first_day, wallet=start_wallet, bags=start_bags, rings=ring_states)
    beam: List[State] = [init]
    while beam and beam[0].day <= cal.horizon_end:
        nxt: Dict[Tuple, State] = {}
        for st in beam:
            grape_opts = (0, 1, 2) if st.day <= allow_grapes_through_day else (0,)
            empty = sum(1 for r in st.rings if not r.planted)
            sowable = (_plantable(crops[crop], cal.season_of(st.day)) and cal.can_still_mature(st.day, crops[crop]))
            buy_cap = min(empty, cost.max_bags_per_day)
            buy_opts = tuple(range(0, buy_cap + 1)) if sowable else (0,)
            for g in grape_opts:
                for b in buy_opts:
                    child = _resolve_day(st, cal, crops, cost, crop, g, b, ring_names, min_cash_reserve=min_cash_reserve)
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
    "DOMAIN_COST_SEAMS", "register_domain_cost", "domain_cost_g",
]

# Spend hooks for later domains. An unset hook is 0 and is not read by the search.
DOMAIN_COST_SEAMS = ("social", "family", "cover")
_DOMAIN_COSTS: dict[str, object] = {name: None for name in DOMAIN_COST_SEAMS}

def register_domain_cost(name: str, fn=None) -> None:
    if name not in _DOMAIN_COSTS:
        raise KeyError(name)
    _DOMAIN_COSTS[name] = fn

def domain_cost_g(name: str, day: int = 0, cost: CostModel | None = None) -> int:
    fn = _DOMAIN_COSTS[name]
    if fn is None:
        return 0
    return int(fn(day, cost))
