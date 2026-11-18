"""Whole-field cover: hoe and sow every farm cell for the 2,950-tile development term.

The ranch-master development term is ``min(100, counted * 10 // 295) >> 1`` over the farm array, so 50 points
need 2,950 of the 2,968 cells counted (F2). Hoed soil does not count; a spring or summer grass bag does (``0x70``)
and keeps counting through Winter (``0x7A``). A bag is nine squares at 500G, and each square costs a 2-stamina
swing, so the whole field is ~330 bags, ~165,000G and ~6,200 stamina. The bags are only worth buying with gold the
money term (99,840G on the last morning) does not need, and only inside the sowing windows: Spring and Summer
of Years 2 and 3 (Year 1 gold compounds in crops; a Fall or Winter bag writes ``0x1D``, which Winter D1 wipes).

There is no daily sowing quota (owner, 2026-11-03: "hoe the entire thing in one day and refill in spa").
Purchases are modest loads from Year 2 onward (owner, 2026-11-04: ease off the final-days 200-bag rush).
The work runs in the evening, after the crops are shipped and watered, where the clock is frozen at 18:xx and
nothing costs game time, so it is bounded by gold and stamina only:

* **Buy** (:func:`cover_errand`): 8-24 bags on a light morning, paid only from gold above :func:`cash_floor_g`.
  A smaller purchase may finish the remaining stands. Held stock counts toward the load, so an unfinished evening
  never triggers another bulk purchase. Busy harvest mornings keep their crops even in the final week.
* **Sow** (``COVER_FIELD``): every bag in hand, east side first (:func:`east_first`), the far side from the
  shipping bin, so a field the gold does not stretch over is left bare next to the lanes.
* **Hoe** (``COVER_TILL``): the whole rest of the field in the same evening, with a hot-spring soak whenever the
  bar runs dry. Hoed soil never reverts, so later bags only need to be sown.

The stands (:func:`plan_cover_stands`) are the ledger's raster-greedy 3x3 packing, ordered east to west. The Year 3
Summer D30 tally has no work morning (the sleep into D30 applies the weather first), so the plan cannot repair a
hurricane on that one night; every other loss shows in the next morning's census and is bought again by the same rule.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from harvest.core.harvest_state import FARM_MAP_ARRAY_ADDR, is_farm_development_tile
from harvest.core.tile_catalog import FARM_WALKABLE, MAP_HEIGHT, MAP_TILE_COUNT, MAP_WIDTH, TRAVEL_SOLID_TILES
from harvest.planner.day_phase_types import PhaseSpec
from harvest.maps.farm_final_layout import (
    GRASS_BAG_PRICE_G,
    MIN_COUNTED_TILES,
    plan_grass_bags,
    terrain_class,
)

# ── calendar (absolute day index: Y1 Spring D1 = 0, Y3 Summer D30 = 299) ─────────────────────────
START_INDEX = 120                  # Year 2 Spring D1: Year 1 gold is scarce and compounds in crops
DEADLINE_INDEX = 298               # Year 3 Summer D29: the last morning that can sow before the D30 tally
TALLY_INDEX = 299
FINAL_FILL_INDEX = 289             # Year 3 Summer D20: no crop bag can ripen after this, so crop footprints are fair game
SOW_SEASONS = (0, 1)               # Spring and Summer write 0x70; Fall and Winter write 0x1D (wiped on Winter D1)

# ── money (the end-of-game wallet term needs 99,840G on the last morning) ─────────────────────────
MONEY_TERM_G = 99_840
MARGIN_G = 10_000
RESERVE_G = 10_000                 # never spend the wallet under this on cover, whatever the clock says
INCOME_G_PER_DAY = 500             # conservative net gold a day (s4 earned ~860 from Year 2 Spring on)

# ── work ──────────────────────────────────────────────────────────────────────────────────────────
STACK_CAP = 255                    # the seed stack is a u8; the clerk still charges for bags past it (ROM probe: 300 paid, 255 held)
MIN_TRIP_BAGS = 8                  # 4,000G of surplus can start planting early instead of waiting for 20,000G
TARGET_STOCK_BAGS = 24             # buy a workable load, not a final-days field-sized backlog; held bags count
FINAL_FILL_TARGET_BAGS = 60       # final fill: ragged one- or two-cell stands, so bags (not hoeing) cap a day (y3h: 24 bags ~ 35 tiles)
FINAL_FILL_LIGHT_RIPE_TILES = int(os.environ.get("HARVEST_FINAL_FILL_RIPE", "100"))   # final fill: the field is 50 points, and a capped crop is not ripe work
LIGHT_RIPE_TILES = 30              # ripe crops a morning may have waiting and still be free for a trip (~half an hour of picking)
BAG_BUY_FRAMES = 650               # the clerk loop, per bag (ROM: 130 bags in 74,000f); the shop clock is frozen
MIN_STAND_CELLS = 6                # the planter skips a stand with fewer seedable squares
MIN_TILL_CELLS = 3                 # a stand with fewer unhoed squares than this is not worth an evening's walk
DOOR_STAND = (8, 26)               # the walkway tile in front of the house door (``farm_final_layout.HOUSE_STAND``)
BAND_WIDTH = 3                     # columns per sweep band: one stand wide

# Ids a hoe or a lift turns into seedable soil. Stumps and boulders need the axe or hammer and are excluded.
_HOEABLE = frozenset({0x01, 0x02, 0x07, 0x08})
_LIFTABLE = frozenset({0x03, 0x04})
_POOL_IDS = _HOEABLE | _LIFTABLE
_UNHOED_IDS = frozenset({0x01, 0x02, 0x03})     # bare or dried ground and weeds: what the planter's clear and hoe passes work
_WALK_IDS = (FARM_WALKABLE - TRAVEL_SOLID_TILES) | _LIFTABLE


# ── calendar maths ───────────────────────────────────────────────────────────────────────────────

def day_index(year: int, season: int, day: int) -> int:
    """``year`` is 0-based, ``season`` 0 Spring .. 3 Winter, ``day`` 1..30."""
    return int(year) * 120 + int(season) * 30 + int(day) - 1


def sow_days(first: int, last: int) -> int:
    """Spring and Summer days in ``first..last`` inclusive."""
    return sum(1 for i in range(max(0, first), last + 1) if (i % 120) // 30 in SOW_SEASONS)


def sowing_open(index: int) -> bool:
    return START_INDEX <= index <= DEADLINE_INDEX and (index % 120) // 30 in SOW_SEASONS


def cash_floor_g(index: int) -> int:
    """Wallet cover leaves alone: what the money term still needs, less what the days left will earn."""
    days_left = max(0, TALLY_INDEX - index)
    return max(RESERVE_G, MONEY_TERM_G + MARGIN_G - INCOME_G_PER_DAY * days_left)


# ── the farm array ───────────────────────────────────────────────────────────────────────────────

def farm_tiles(ram) -> tuple[int, ...]:
    """The farm ids a tool would see: the live map on the farm (0xFF = unloaded falls back), else the persistent array."""
    from harvest.core.ram_catalog import live_wram_base, read_ram_value
    from harvest.core.tile_catalog import metatile_grid_base
    from harvest.maps.farm_pond import FARM_TILEMAP_IDS
    from harvest.runtime.rom_tools import wram_offset

    p0 = live_wram_base(ram) + wram_offset(FARM_MAP_ARRAY_ADDR)
    persistent = [int(b) & 0xFF for b in ram[p0:p0 + MAP_TILE_COUNT]]
    if int(read_ram_value(ram, "tilemap")) not in FARM_TILEMAP_IDS:
        return tuple(persistent)
    v0 = metatile_grid_base(ram)
    visible = [int(b) & 0xFF for b in ram[v0:v0 + MAP_TILE_COUNT]]
    return tuple(p if v == 0xFF else v for p, v in zip(persistent, visible))


def durable_counted(tiles: Sequence[int]) -> int:
    """Counted tiles that survive to the tally: grass and fence. Crops and Fall seed are not durable."""
    return sum(1 for t in tiles if terrain_class(int(t)) in ("grass", "fence"))


def counted(tiles: Sequence[int]) -> int:
    return sum(1 for t in tiles if is_farm_development_tile(int(t)))


def _idx(x: int, y: int) -> int:
    return y * MAP_WIDTH + x


def reserved_cells(year: int, index: int) -> frozenset[tuple[int, int]]:
    """Cells cover never hoes or sows: the bin lane and stands for good, crop footprints until no crop bag can ripen.

    The bin lane stays bare to the end (owner: grass goes on the east side, far from the shipping lanes).
    """
    from harvest.maps.farm_pond import SHIP_BIN_STAND_TILES, active_plot_centers, crop_working_tiles

    if index >= FINAL_FILL_INDEX:
        return frozenset(SHIP_BIN_STAND_TILES)
    cells = set(crop_working_tiles())
    for cx, cy in active_plot_centers(year, (index % 120) // 30):
        cells.update((cx + dx, cy + dy) for dy in range(-2, 3) for dx in range(-2, 3))
    return frozenset(cells)


EARLY_COVER_ENV = "HARVEST_EARLY_COVER"
EARLY_COVER_FROM_YEAR = 2          # 0-based: Year 3 only (a Year 2 change would re-roll Year 3 for no reason)
SUMMER_CROP_IDS = (0x1E, 0x53)     # tomato 0x1E-0x39 and corn 0x3A-0x53: a ring sown for Summer
MIN_SOWN_RING_TILES = 6            # of a ring's eight (seven on a notch block): fewer is a site still waiting for its bag


def early_cover_on(year: int) -> bool:
    """Year 3 only, ``HARVEST_EARLY_COVER=0`` switches it off for an A/B."""
    return int(year) >= EARLY_COVER_FROM_YEAR and os.environ.get(EARLY_COVER_ENV, "1").strip().lower() not in {"", "0", "false", "no", "off"}


def released_cells(tiles: Sequence[int], year: int, season: int, index: int) -> frozenset[tuple[int, int]]:
    """Ground of a sown Summer ring that cover may take before the final fill: its 5x5 minus the 3x3 ring block.

    ``reserved_cells`` holds every site's 5x5 until Summer D20 so a ring still waiting for its bag keeps its stands. A site
    that already stands in corn or tomato needs only its own ring block (the eight crops and the centre the sprinkler uses);
    the 16-cell band around it is ground the planter never stands on, so it waits for nothing. A band cell that an unsown
    site still needs stays reserved (the lattice shares lanes), and so does the bin stand. Empty outside Year 3 Summer.
    """
    if not early_cover_on(year) or int(season) != 1 or index >= FINAL_FILL_INDEX:
        return frozenset()
    from harvest.maps.farm_pond import SHIP_BIN_STAND_TILES, active_plot_centers, plot_crop_tiles

    low, high = SUMMER_CROP_IDS
    sites = tuple(dict.fromkeys((*active_plot_centers(year, season), *active_plot_centers(1))))

    def sown(site: tuple[int, int]) -> bool:
        return sum(low <= int(tiles[_idx(x, y)]) <= high for x, y in plot_crop_tiles(site, year)) >= MIN_SOWN_RING_TILES

    def box(site: tuple[int, int], r: int) -> set[tuple[int, int]]:
        return {(site[0] + dx, site[1] + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)}

    waiting: set[tuple[int, int]] = set()
    freed: set[tuple[int, int]] = set()
    for site in sites:
        if sown(site):
            freed |= box(site, 2) - box(site, 1)
        else:
            waiting |= box(site, 2)
    return frozenset(freed - waiting - SHIP_BIN_STAND_TILES)


def reachable(tiles: Sequence[int], start: tuple[int, int] = DOOR_STAND) -> dict[tuple[int, int], tuple[int, int] | None]:
    """BFS parents from ``start`` over ground the farmer can cross (pool cells and grass count)."""
    from harvest.maps.farm_pond import FARM_NO_GO_TILES

    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT) or (nx, ny) in parent:
                continue
            if (nx, ny) in FARM_NO_GO_TILES or int(tiles[_idx(nx, ny)]) not in _WALK_IDS:
                continue
            parent[(nx, ny)] = (cx, cy)
            queue.append((nx, ny))
    return parent


def route_to(parent: dict, goal: tuple[int, int]) -> list[tuple[int, int]]:
    """The BFS path start -> goal ([] when ``goal`` was not reached)."""
    if goal not in parent:
        return []
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


# ── stands ───────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CoverPlan:
    stands: tuple[tuple[int, int], ...]      # every stand the field still needs, east side first
    cells: int                               # seedable squares the stands cover
    route: tuple[tuple[int, int], ...]       # walk from the door stand to the first stand
    pool: int                                # seedable squares anywhere that still do not count
    unhoed: tuple[tuple[int, int], ...] = ()     # the stands that still hold bare ground or weeds, east side first
    unhoed_route: tuple[tuple[int, int], ...] = ()


def east_first(stands: Sequence[tuple[int, int]], start: tuple[int, int] = DOOR_STAND) -> list[tuple[int, int]]:
    """Sweep ``stands`` in three-column bands from the east edge to the west, alternating north and south.

    The bin and its lane sit on the west side, so the east side is the far side from the shipping lanes: it is
    covered first, and a field the gold does not stretch over is left bare beside the lanes. The first band is
    swept away from the end nearer ``start`` so the walk in from the door is short.
    """
    bands: dict[int, list[tuple[int, int]]] = {}
    for stand in stands:
        bands.setdefault(stand[0] // BAND_WIDTH, []).append(stand)
    ordered: list[tuple[int, int]] = []
    descending = False
    for n, band in enumerate(sorted(bands, reverse=True)):
        column = sorted(bands[band], key=lambda s: (s[1], s[0]))
        if n == 0:
            descending = abs(column[-1][1] - start[1]) < abs(column[0][1] - start[1])
        ordered.extend(reversed(column) if descending else column)
        descending = not descending
    return ordered


def plan_cover_stands(
    tiles: Sequence[int],
    limit: int = 0,
    *,
    reserved: Iterable[tuple[int, int]] = (),
    start: tuple[int, int] = DOOR_STAND,
    min_cells: int = MIN_STAND_CELLS,
) -> CoverPlan:
    """Every 3x3 stand over the squares that do not count yet, east side first.

    The packing is the ledger's raster-greedy one (nines first), which stays put as squares are covered, so a
    sweep started one evening continues on the same stands the next. Reserved squares (crop footprints, the bin
    lane) are never stood on or counted. ``limit`` caps the stand list (0 = the whole field); ``unhoed`` is the
    subset still holding bare ground or weeds, which the hoe pass walks.
    """
    blocked = set(reserved)
    reach = reachable(tiles, start)
    pool = {
        i for i, t in enumerate(tiles)
        if int(t) in _POOL_IDS and (i % MAP_WIDTH, i // MAP_WIDTH) not in blocked and (i % MAP_WIDTH, i // MAP_WIDTH) in reach
    }
    if not pool:
        return CoverPlan((), 0, (), 0)
    packing = [s for s in plan_grass_bags(pool, len(pool)) if s[2] >= min_cells and (s[0], s[1]) in reach]
    cells_of = {(s[0], s[1]): s[2] for s in packing}
    ordered = east_first(list(cells_of), start)
    if limit > 0:
        ordered = ordered[:limit]
    floor = 1 if min_cells <= 1 else MIN_TILL_CELLS

    def bare(stand: tuple[int, int]) -> int:
        cx, cy = stand
        return sum(
            1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if 0 <= cx + dx < MAP_WIDTH and 0 <= cy + dy < MAP_HEIGHT
            and (cx + dx, cy + dy) not in blocked and int(tiles[_idx(cx + dx, cy + dy)]) in _UNHOED_IDS
        )

    unhoed = tuple(s for s in ordered if bare(s) >= floor)
    return CoverPlan(
        tuple(ordered), sum(cells_of[s] for s in ordered),
        tuple(route_to(reach, ordered[0])) if ordered else (), len(pool),
        unhoed, tuple(route_to(reach, unhoed[0])) if unhoed else (),
    )


def first_open_stand(
    tiles: Sequence[int],
    stands: Sequence[Sequence[int]],
    *,
    till: bool = False,
    skip: Iterable[tuple[int, int]] = (),
) -> Optional[tuple[int, int]]:
    """The first stand (in sweep order) that still has work: bare ground to hoe (``till``) or soil to sow, never a skipped one.

    The planter sees only the ~16x14 tiles around the farmer, so a restart after a soak must walk to *this* stand, not to the
    first stand of the list (long since done): from there every far stand looks unreachable and is skipped (ROM probe: 162
    "bag not ready" skips, 94 of 253 bags sown).
    """
    ids = _UNHOED_IDS if till else _POOL_IDS
    skipped = {tuple(s) for s in skip}
    for stand in stands:
        cx, cy = int(stand[0]), int(stand[1])
        if (cx, cy) in skipped:
            continue
        if any(
            0 <= cx + dx < MAP_WIDTH and 0 <= cy + dy < MAP_HEIGHT and int(tiles[_idx(cx + dx, cy + dy)]) in ids
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        ):
            return (cx, cy)
    return None


def restart_route_px(ram, stands: Sequence[Sequence[int]], *, till: bool = False) -> Optional[list[tuple[int, int]]]:
    """Waypoints from the door to the first stand with work left today; None when nothing is left or the farm is not readable."""
    from harvest.tasks.grass_planter import dead_stands
    from harvest.core.ram_catalog import read_ram_value

    try:
        tiles = farm_tiles(ram)
        today = (int(read_ram_value(ram, "year", raw=True) or 0), int(read_ram_value(ram, "season", raw=True) or 0),
                 int(read_ram_value(ram, "day", raw=True) or 1))
        first = first_open_stand(tiles, stands, till=till, skip=dead_stands(today))
        if first is None:
            return None
        return waypoint_px(route_to(reachable(tiles, DOOR_STAND), first))
    except Exception:
        return None


def waypoint_px(route: Sequence[tuple[int, int]], hop: int = 6) -> list[tuple[int, int]]:
    """Pixel centres along ``route`` every ``hop`` tiles, always ending on the last tile."""
    if not route:
        return []
    picks = list(route[hop::hop])
    if not picks or picks[-1] != route[-1]:
        picks.append(route[-1])
    return [(x * 16 + 8, y * 16 + 8) for x, y in picks]


# ── the errand ───────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CoverErrand:
    bags: int = 0                   # bags to sow tonight (stock plus buy)
    buy: int = 0                    # bags to add to the shop stock this morning
    behind: int = 0                 # counted tiles the field still lacks
    plan: Optional[CoverPlan] = None
    reason: str = ""
    final_fill: bool = False        # crop footprints released, ragged stands allowed (no crop bag can ripen any more)
    ragged: bool = False            # one- and two-cell stands allowed before the final fill (no full stand is left)
    released: tuple = ()            # cells of sown rings' bands the planter may work before the final fill


def cover_errand(
    ram,
    *,
    spendable_g: int,
    stock: int,
    shop_open: bool,
    tiles: Optional[Sequence[int]] = None,
    strategy=None,
    ripe: int = 0,
) -> CoverErrand:
    """Today's cover work: a modest load above the cash floor, with all held bags available tonight.

    ``spendable_g`` is the wallet the caller already holds back for the herd, tools and crop bags. The cash
    floor (:func:`cash_floor_g`) comes off it here. Without the shop the stock in hand is all there is, and all
    of it is sown: a stack left in the carry pair overnight breaks the next morning's ``CROP_ESTABLISH``
    (``0x02 not in carry pair``, ROM A/B), and a bought bag is paid for.

    ``ripe`` is the number of ripe crops waiting this morning. A trip (~2,500 frames, ~2.8 game hours) goes only on a
    morning with at most ``LIGHT_RIPE_TILES`` waiting, including the last four mornings. Extra cover earns points
    gradually; an unfinished field is not a reason to sacrifice the last crop shipments or the money term.
    """
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner import run_strategy

    if not (strategy.field_cover if strategy is not None else run_strategy.field_cover()):
        return CoverErrand(reason="strategy")
    year = int(read_ram_value(ram, "year", raw=True) or 0)
    season = int(read_ram_value(ram, "season", raw=True) or 0)
    day = int(read_ram_value(ram, "day", raw=True) or 1)
    index = day_index(year, season, day)
    if not sowing_open(index):
        return CoverErrand(reason="window")
    tiles = farm_tiles(ram) if tiles is None else tiles
    behind = MIN_COUNTED_TILES - durable_counted(tiles)
    if behind <= 0 and stock <= 0:
        return CoverErrand(reason="complete")
    final_fill = index >= FINAL_FILL_INDEX
    released = released_cells(tiles, year, season, index)
    reserved = reserved_cells(year, index) - released
    plan = plan_cover_stands(tiles, reserved=reserved, min_cells=1 if final_fill else MIN_STAND_CELLS)
    # Full stands first. When none is left, the ragged ones (one or two squares a bag) start now: the final fill used to
    # take all of them in the last nine mornings (r45: counted 2,545 on D9, still 2,545 on D21, 2,951 on D29).
    ragged = False
    if not plan.stands and not final_fill and early_cover_on(year):
        plan = plan_cover_stands(tiles, reserved=reserved, min_cells=1)
        ragged = bool(plan.stands)
    if not plan.stands:
        return CoverErrand(behind=behind, reason="no stands")
    # Year 3 Summer loads the big stock from the first free morning: once the corn ripens (D13) a harvest day leaves no
    # clock for the shop (r42/rdet previews: "shop start deadline 15:00", a wedged farm_to_path), and 24 bags a day left
    # 400 tiles for the last nine mornings. The evening sows whatever is bought, so a big load costs frames, not game time.
    light = final_fill or ragged
    big_load = light or (early_cover_on(year) and int(season) == 1)
    buy = 0
    busy = False
    if shop_open and behind > 0 and stock < len(plan.stands):
        afford = max(0, int(spendable_g) - cash_floor_g(index)) // GRASS_BAG_PRICE_G
        target = FINAL_FILL_TARGET_BAGS if big_load else TARGET_STOCK_BAGS
        load = min(len(plan.stands) - stock, afford, max(0, target - stock), STACK_CAP - stock)
        worth = load >= MIN_TRIP_BAGS or (0 < load and stock + load >= len(plan.stands))
        busy = worth and ripe > (FINAL_FILL_LIGHT_RIPE_TILES if light else LIGHT_RIPE_TILES)
        if worth and not busy:
            buy = load
    bags = min(stock + buy, len(plan.stands))
    if bags <= 0:
        if not shop_open:
            reason = "shop closed"
        elif busy:
            reason = f"busy morning ({ripe} ripe)"
        else:
            reason = "no gold above the floor"
        return CoverErrand(behind=behind, reason=reason)
    return CoverErrand(bags=bags, buy=buy, behind=behind, plan=plan, reason="building", final_fill=final_fill,
                       ragged=ragged, released=tuple(sorted(released)) if released else ())


# ── the phases ───────────────────────────────────────────────────────────────────────────────────

def cover_buy_allowed(ram) -> bool:
    """The town trip for cover bags: a weekday shop, and not the Summer mornings reserved for the rings' corn bags."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.day_plan_status import SHOP_CLOSED_WEEKDAYS
    from harvest.planner.livestock_econ import RING_FIRST_PLANT_DAYS, ring_bags_wanted

    if int(read_ram_value(ram, "weekday")) in SHOP_CLOSED_WEEKDAYS:
        return False
    return not (
        int(read_ram_value(ram, "season")) == 1
        and int(read_ram_value(ram, "day")) <= RING_FIRST_PLANT_DAYS
        and ring_bags_wanted(ram) > 0
    )


def cover_errand_for(ram, herd_stock: int, herd_plants: bool, *, buying: bool):
    """Today's whole-field errand (:func:`cover_errand`) after the herd's own grass has been funded."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.livestock_econ import SHOP_PRICE_G, pasture_reserve_g

    stock = int(read_ram_value(ram, "grass_seeds"))
    herd_buy = max(0, herd_stock - stock)
    spare = max(0, herd_stock - (2 if herd_plants else 0))      # the herd planter spends up to two bags first
    spendable = int(read_ram_value(ram, "money")) - herd_buy * SHOP_PRICE_G["grass_seed"]
    ripe = 0
    if buying:
        spendable -= pasture_reserve_g(ram)
        from harvest.planner.world_probe import WorldProbe

        ripe = WorldProbe.from_inputs(ram=ram).ripe_crop_count()
    return cover_errand(ram, spendable_g=spendable, stock=spare, shop_open=buying, ripe=ripe)


def field_cover_phases(ram) -> List[PhaseSpec]:
    """Evening whole-field work: sow every bag in hand, then hoe the rest of the field, east side first.

    It runs after the crops are shipped and watered (the clock freezes at 18:xx, so the swings cost no game time),
    and only on an evening with bags in hand: the evening that brings a load also hoes the whole field.
    """
    from harvest.core.ram_catalog import read_ram_value

    stock = int(read_ram_value(ram, "grass_seeds"))
    if stock <= 0:
        return []
    cover = cover_errand_for(ram, stock, False, buying=False)
    if cover.bags <= 0 or not cover.plan:
        return []
    phases = [cover_field_phase(cover)]
    if cover.plan.unhoed and os.environ.get("HARVEST_COVER_TILL_AHEAD", "1").strip().lower() not in {"0", "false", "no"}:
        phases.append(cover_till_phase(cover))
    print(f"[COVER] evening bags={cover.bags} stands={len(cover.plan.stands)} to_hoe={len(cover.plan.unhoed)} "
          f"behind={cover.behind} final_fill={cover.final_fill}" + (f" ragged released={len(cover.released)}" if cover.ragged or cover.released else ""))
    return phases


def cover_field_phase(cover) -> PhaseSpec:
    """Sow ``cover.bags`` bags. The planter gets every stand in order and stops at the bag limit, so a stand it
    cannot reach only moves it on to the next one instead of leaving a stack in the carry pair overnight."""
    plan = cover.plan
    return PhaseSpec(
        "COVER_FIELD", "grass",
        {
            "stands": [list(s) for s in plan.stands],
            "bag_limit": int(cover.bags),
            "route_px": [list(p) for p in waypoint_px(plan.route)],
            "behind": int(cover.behind),
            "final_fill": bool(cover.final_fill),
            "ragged": bool(cover.ragged),
            "released": [list(c) for c in cover.released],
        },
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=900 * int(cover.bags),
        failure_modes=("stamina_low",),
    )


def cover_till_phase(cover) -> PhaseSpec:
    """Hoe every stand that still holds bare ground (``till_only``, no bag limit)."""
    plan = cover.plan
    return PhaseSpec(
        "COVER_TILL", "grass",
        {
            "till_only": True,
            "stands": [list(s) for s in plan.unhoed],
            "route_px": [list(p) for p in waypoint_px(plan.unhoed_route)],
            "final_fill": bool(cover.final_fill),
            "ragged": bool(cover.ragged),
            "released": [list(c) for c in cover.released],
        },
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=3000 * len(plan.unhoed),
        failure_modes=("stamina_low",),
    )
