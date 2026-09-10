"""Spring-1 shipping optimiser.

Forward beam search over the Spring D3→D30 horizon. Given the ROM crop model
(:mod:`harvest.planner.crop_planner` prices + `NightlyFarmTilesCheck` growth),
a per-ring frame-cost table, and the calendar (Sundays / blocked days /
optional rain scenario), it returns the day-by-day action plan that maximises
the wallet at Summer D1.

Model and evidence: ``docs/SPRING_ECONOMY.md``.

This module is pure Python — no emulator, no numpy-heavy work. The output
``SpringPlan`` is meant to drive ``multi_day_planner`` instead of the reactive
day-phase heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

# ── ROM crop model ────────────────────────────────────────────────────────

SPRING_LAST_DAY = 30
SUMMER_KILLS_SPRING_CROPS = True


@dataclass(frozen=True)
class Crop:
    name: str
    seed_cost_g: int
    tiles_per_bag: int          # ring tiles the establish pass actually sows
    ship_per_tile_g: int
    grow_waterings: int         # watered-nights planted-dry -> mature

    @property
    def ring_gross_g(self) -> int:
        return self.tiles_per_bag * self.ship_per_tile_g


POTATO = Crop("potato", seed_cost_g=200, tiles_per_bag=8, ship_per_tile_g=80, grow_waterings=6)
TURNIP = Crop("turnip", seed_cost_g=200, tiles_per_bag=8, ship_per_tile_g=60, grow_waterings=4)
CROPS: Dict[str, Crop] = {c.name: c for c in (POTATO, TURNIP)}


# ── Frame-cost model ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CostModel:
    """Per-action frame costs. Defaults from ``docs/SPRING_ECONOMY.md`` §4.

    There is **no forced bedtime** — the evening never ends, the can refills
    at F0 and stamina refills at the hot spring, so a wake→sleep cycle can do
    an arbitrary amount of work. ``evening_frames`` is only a *practical*
    ceiling (how long a real run's day should take); raise it freely. A crop
    still advances exactly one stage per sleep regardless of how much you do.
    The real cap on ring count is: rings established late enough that they
    cannot mature before the D30 summer wipe, seed-capital timing, and — at
    very high counts — mature rings piling up faster than one evening clears
    them (`can't harvest all in one day`).
    """

    evening_frames: int = 60_000          # practical wake->sleep work ceiling
    home_sleep_f: int = 1_000
    spa_refill_f: int = 2_400             # farm -> hot spring -> farm (stamina)
    tool_uses_per_spa: int = 40           # hoe/water swings before a spa top-up

    # 2-grape run measured at 6194 f (06:00->13:12); a 3rd grape pushes past
    # 15:00 and risks the 17:00 farm ShippingScene, so the run is capped at 2.
    grape_first_f: int = 4_400            # farm -> mountain -> first grape -> farm -> bin
    grape_marginal_f: int = 1_800        # 2nd grape same trip
    grape_max_per_day: int = 2
    grape_value_g: int = 150

    shop_roundtrip_f: int = 2_600         # farm -> plaza -> farm, any bag count

    establish_first_f: int = 3_400        # nav + hoe 8 + plant 8 (virgin ring)
    establish_replant_f: int = 2_100      # nav + plant 8 (tile already 0x08)
    water_ring_f: int = 1_800             # first ring of the day: nav + select + 8 tiles
    water_ring_marginal_f: int = 1_100    # each further ring same visit (shared nav/select)
    harvest_ring_f: int = 1_900           # first ring: nav + pick 8 + carry to bin
    harvest_ring_marginal_f: int = 1_300  # each further mature ring same visit

    refill_f: int = 2_800                 # empty-can -> F0 -> back
    can_capacity: int = 20                # charges; 1 tile / charge


@dataclass(frozen=True)
class Ring:
    """An executable plot. ``order`` is the establish / water sequence index."""

    name: str
    order: int


# ── Calendar ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Calendar:
    first_day: int = 3
    last_day: int = SPRING_LAST_DAY
    spring_d1_weekday: int = 1            # measured: Y1_D3_Morning weekday == 3
    sunday_weekday: int = 0
    blocked_days: Tuple[int, ...] = ()    # festivals: no field work
    rain_days: Tuple[int, ...] = ()       # scenario: free watering that night

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


# ── Search state ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RingState:
    crop: Optional[str] = None
    waterings: int = 0          # nights advanced since planting
    established: bool = False   # ring has been hoed at least once (0x08 stays)

    @property
    def planted(self) -> bool:
        return self.crop is not None

    def mature(self, crops: Dict[str, Crop]) -> bool:
        return self.planted and self.waterings >= crops[self.crop].grow_waterings


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
    wallet_end: int = 0
    note: str = ""


@dataclass(frozen=True)
class State:
    day: int
    wallet: int
    bags: int
    rings: Tuple[RingState, ...]
    shipped_g: int = 0                    # cumulative gross shipped
    log: Tuple[DayAction, ...] = ()

    def key(self) -> Tuple:
        return (self.day, self.wallet, self.bags,
                tuple((r.crop, r.waterings, r.established) for r in self.rings))


@dataclass(frozen=True)
class SpringPlan:
    final_wallet: int
    shipped_g: int
    days: Tuple[DayAction, ...]
    rings: Tuple[str, ...]
    crop: str

    def table(self) -> str:
        head = f"{'Day':>3} {'Grp':>3} {'Buy':>3} {'Estab':>14} {'Harv':>14} {'Water':>18} {'Rfl':>3} {'Frames':>7} {'Wallet':>7}"
        lines = [head, "-" * len(head)]
        for a in self.days:
            lines.append(
                f"{a.day:>3} {a.grapes:>3} {a.bags_bought:>3} "
                f"{','.join(a.established) or '-':>14} "
                f"{','.join(a.harvested) or '-':>14} "
                f"{','.join(a.watered) or '-':>18} "
                f"{a.refills:>3} {a.frames_used:>7} {a.wallet_end:>7}"
                + (f"   {a.note}" if a.note else "")
            )
        lines.append("-" * len(head))
        lines.append(f"final wallet {self.final_wallet} G   gross shipped {self.shipped_g} G")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "final_wallet_g": self.final_wallet,
            "shipped_gross_g": self.shipped_g,
            "crop": self.crop,
            "rings": list(self.rings),
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
                    "note": a.note,
                }
                for a in self.days
            ],
        }


# ── Day resolution ────────────────────────────────────────────────────────


def _water_plan(
    rings: Sequence[RingState],
    crops: Dict[str, Crop],
    cost: CostModel,
    budget_left: int,
    is_rain: bool,
) -> Tuple[List[int], int, int]:
    """Pick which planted-not-mature rings to water within ``budget_left``.

    Rain waters everything for free. Otherwise prefer rings closest to
    maturity (finish cycles sooner -> replant sooner). Returns
    (ring indices watered, frames spent, refills).
    """
    if is_rain:
        idx = [i for i, r in enumerate(rings) if r.planted and not r.mature(crops)]
        return idx, 0, 0

    candidates = [
        i for i, r in enumerate(rings)
        if r.planted and not r.mature(crops)
    ]
    # closest to maturity first
    candidates.sort(key=lambda i: crops[rings[i].crop].grow_waterings - rings[i].waterings)

    watered: List[int] = []
    frames = 0
    refills = 0
    tiles = 0             # tiles since last can refill
    swings = 0            # tool uses since last spa top-up
    for n, i in enumerate(candidates):
        tiles_here = crops[rings[i].crop].tiles_per_bag
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
    crops: Dict[str, Crop],
    cost: CostModel,
    crop_name: str,
    grapes: int,
    buy_bags: int,
    ring_names: Sequence[str],
) -> Optional[State]:
    """Apply one day's top-level choices; returns the next-morning state.

    Deterministic within a (grapes, buy_bags) branch: harvest every mature
    ring, replant with any bag in hand, then water toward maturity.
    """
    day = state.day
    crop = crops[crop_name]
    budget = cost.evening_frames - cost.home_sleep_f
    frames = 0
    wallet = state.wallet
    bags = state.bags
    rings = list(state.rings)

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

    if buy_bags and cal.shop_open(day):
        if frames + cost.shop_roundtrip_f <= budget and wallet >= crop.seed_cost_g:
            affordable = min(buy_bags, wallet // crop.seed_cost_g)
            if affordable:
                frames += cost.shop_roundtrip_f
                wallet -= affordable * crop.seed_cost_g
                bags += affordable
    else:
        buy_bags = 0

    established: List[str] = []
    harvested: List[str] = []

    if cal.field_work(day):
        # Harvest mature rings (money posts next morning; frees the ring).
        for i, r in enumerate(rings):
            hf = cost.harvest_ring_f if not harvested else cost.harvest_ring_marginal_f
            if r.mature(crops) and frames + hf <= budget:
                frames += hf
                gross = crops[r.crop].ring_gross_g
                state = replace(state, shipped_g=state.shipped_g + gross)
                wallet += gross  # credited at next NightReset; fold in here
                harvested.append(ring_names[i])
                rings[i] = RingState(crop=None, waterings=0, established=True)

        # Establish: fill empty rings while a bag is in hand.
        for i, r in enumerate(rings):
            if bags <= 0:
                break
            if r.planted:
                continue
            ef = cost.establish_replant_f if r.established else cost.establish_first_f
            if frames + ef > budget:
                continue
            # only plant if it can still reach maturity before summer
            if day + crop.grow_waterings > cal.last_day:
                continue
            frames += ef
            bags -= 1
            rings[i] = RingState(crop=crop_name, waterings=0, established=True)
            established.append(ring_names[i])

    # Water toward maturity.
    watered_idx, wf, refills = _water_plan(
        rings, crops, cost, budget - frames, cal.is_rain(day)
    )
    frames += wf
    watered = [ring_names[i] for i in watered_idx]
    wi = set(watered_idx)

    # Night: advance watered rings (or all if raining).
    new_rings: List[RingState] = []
    for i, r in enumerate(rings):
        if r.planted and (i in wi or cal.is_rain(day)):
            new_rings.append(replace(r, waterings=r.waterings + 1))
        else:
            new_rings.append(r)

    action = DayAction(
        day=day, grapes=grapes, bags_bought=buy_bags,
        established=tuple(established), harvested=tuple(harvested),
        watered=tuple(watered), refills=refills, frames_used=frames,
        wallet_end=wallet + grapes * cost.grape_value_g,
    )

    return State(
        day=day + 1,
        wallet=wallet + grapes * cost.grape_value_g,
        bags=bags,
        rings=tuple(new_rings),
        shipped_g=state.shipped_g + grapes * cost.grape_value_g,
        log=state.log + (action,),
    )


# ── Beam search ───────────────────────────────────────────────────────────


def optimize_spring(
    *,
    rings: Sequence[Ring],
    crop: str = "potato",
    calendar: Optional[Calendar] = None,
    cost: Optional[CostModel] = None,
    start_wallet: int = 250,
    start_bags: int = 0,
    beam_width: int = 200,
    allow_grapes_through_day: int = 6,
) -> SpringPlan:
    cal = calendar or Calendar()
    cost = cost or CostModel()
    crops = CROPS

    ordered = sorted(rings, key=lambda r: r.order)
    ring_names = tuple(r.name for r in ordered)
    ring_states = tuple(RingState() for _ in ordered)
    init = State(day=cal.first_day, wallet=start_wallet, bags=start_bags,
                 rings=ring_states)

    beam: List[State] = [init]
    while beam and beam[0].day <= cal.last_day:
        nxt: Dict[Tuple, State] = {}
        for st in beam:
            grape_opts = (0, 1, 2) if st.day <= allow_grapes_through_day else (0,)
            # buy up to (empty rings + 1); capped by search breadth
            empty = sum(1 for r in st.rings if not r.planted)
            buy_opts = tuple(range(0, min(empty, 6) + 1))
            for g in grape_opts:
                for b in buy_opts:
                    child = _resolve_day(st, cal, crops, cost, crop, g, b, ring_names)
                    if child is None:
                        continue
                    k = child.key()
                    if k not in nxt or child.wallet > nxt[k].wallet:
                        nxt[k] = child
        if not nxt:
            break
        beam = sorted(nxt.values(), key=_rank(crops, cost), reverse=True)[:beam_width]

    best = max(beam, key=lambda s: s.wallet + _standing_value(s, crops))
    # standing (unharvested mature) crops still credit at Summer-eve harvest
    standing = _standing_value(best, crops)
    return SpringPlan(
        final_wallet=best.wallet + standing,
        shipped_g=best.shipped_g + standing,
        days=best.log,
        rings=ring_names,
        crop=crop,
    )


def _standing_value(st: State, crops: Dict[str, Crop]) -> int:
    total = 0
    for r in st.rings:
        if r.mature(crops):
            total += crops[r.crop].ring_gross_g
    return total


def _rank(crops: Dict[str, Crop], cost: CostModel):
    def score(st: State) -> Tuple:
        # near-term wallet + standing crop value + bag inventory value
        growth_credit = 0
        for r in st.rings:
            if r.planted:
                c = crops[r.crop]
                growth_credit += int(c.ring_gross_g * r.waterings / max(1, c.grow_waterings))
        return (st.wallet + growth_credit + st.bags * 150, -st.day)
    return score


__all__ = [
    "Crop", "CROPS", "POTATO", "TURNIP",
    "CostModel", "Calendar", "Ring",
    "RingState", "State", "DayAction", "SpringPlan",
    "optimize_spring",
]
