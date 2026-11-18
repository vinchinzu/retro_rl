"""Year-1 block calendar: which crop block grows what and when, and what Spring and Summer ship.

One plan, three uses: the farm layout reads it (``farm_pond`` layout ``blocks``: block order, notch
side, the Spring last-sow day and the Summer bag day of every block), the forecast simulates it day
by day, and ``harvest.tools.block_calendar_report`` renders it as a map and checks a run against it.

A block is a 3x3 with one side cell left bare as a notch: ``xxx/xxo/xxx`` (notch east) or
``xxx/oxx/xxx`` (notch west). The centre is planted (hoe the 8 cells, stand on the centre, sow the
bag: only tilled cells take seed), and the notch is a stand inside the block for the centre and the
two cells beside it. Blocks sit on a pitch-4 lattice with one-tile lanes, in mirrored pairs whose
notches face each other across the shared lane::

    xxx.xxx
    xxo.oxx
    xxx.xxx

The calendar follows from four ROM facts:

- A watered crop gains one tile id a night and the can adds one (``NightlyFarmTilesCheck``,
  bank_82 $82A811), so a potato (0x54 -> 0x60) ripens 6 watered nights after sowing, corn 12,
  tomato 9, and a picked corn or tomato regrows in 3. Spring crops keep growing in Summer.
- The clock runs one tick a frame outdoors, 900 an hour (``UpdateTime``, bank_82), stops indoors
  and freezes at 18:00. A crop dropped in the bin from 17:00 pays nothing, so a day ships what
  06:00-17:00 holds: 9,900 frames of picking and walking. Watering, planting and the spa can all
  run after 17:00 at no clock cost.
- The farmer carries one crop. Running is 8 frames a tile, the lift 23 frames and the throw 36
  (``harvest.scripts.ship_bench``, fast press), so a crop costs ``PICK + THROW + 16 x tiles`` to
  the nearest of the four bin stands. Distance, not plot count, sets Summer income.
- The bin's F2 tiles are (8-9, 29-30) and the throw reads two tiles ahead: stands (11,29)/(11,30)
  face left, (8,32)/(9,32) face up, one tile from the lattice's first lane (y=33).

Roles. ``core`` blocks are the nearest. Their last Spring potato is timed to ripen on Summer D1
or D2 (sown D25 / D26); that morning the block is picked and takes a corn bag, so it gets the
full six corn harvests. ``late`` blocks are farther. They keep sowing potatoes to D29 / D30 (the
shop is shut D27-D28), which ripen on Summer D5 / D6, while the ship window is otherwise idle (no
Summer crop ripens before D10), and then take tomato, which still ripens six times from there.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from harvest.planner.forage_calendar import forage_for_season, trips_today

Tile = Tuple[int, int]

SCHEMA = "harvest.block_calendar/1"

# ── ROM facts ───────────────────────────────────────────────────────────
SEASON_DAYS = 30
POTATO_NIGHTS = 6
CORN_NIGHTS = 12
TOMATO_NIGHTS = 9
REGROW_NIGHTS = 3
NIGHTS = {"potato": POTATO_NIGHTS, "corn": CORN_NIGHTS, "tomato": TOMATO_NIGHTS}
PRICE_G = {"potato": 80, "corn": 120, "tomato": 100}
BAG_G = {"potato": 200, "corn": 300, "tomato": 300, "grass": 500}
TICKS_PER_HOUR = 900
SHIP_DAY_TICKS = (17 - 6) * TICKS_PER_HOUR
# Spring D1 is a Monday in a power-on run (Summer D1 a Wednesday); the seed shop is shut Sat and Sun.
SPRING_D1_WEEKDAY = 0
FLOWER_FESTIVAL_DAY = 23

# ── Frame model (ROM, ``ship_bench`` on ``LN_saddle_whatif_harvest_y2d22``, 2026-11-02) ──
RUN_FRAMES_PER_TILE = 8
PICK_FRAMES = 26            # lift 23 from the A edge, plus the push into the crop and the tap
THROW_FRAMES = 38           # throw 36, plus the turn and tap

BIN_STANDS: Tuple[Tile, ...] = ((11, 30), (11, 29), (8, 32), (9, 32))

# ── Policy (the knobs a bench tunes) ────────────────────────────────────
CORE_LAST_SOW = (25, 26)    # alternate by rank: ripe Summer D1 / D2 (Thu / Fri: the shop is open)
LATE_LAST_SOW = (29, 30)    # Mon / Tue after the shut weekend: ripe Summer D5 / D6
LATE_SUMMER_CROP = "tomato" # tomato sown S5-S6 still ripens six times (S14..S29 / S15..S30); corn five
SUMMER_LAST_BAG_DAY = 18    # livestock_econ.SUMMER_LAST_CORN_BAG_DAY
SHOP_TRIP_TICKS = 2000      # a seed-shop round trip before the harvest (BUY_SEEDS 06:01-08:11 in base)
FESTIVAL_SHIP_TICKS = 3 * TICKS_PER_HOUR
WALLET_RESERVE_G = 0
# Measured, not ROM: 13 corn plots plus the late potatoes ran CROP_WATER into its 24k-frame watchdog
# on Summer D2-D3 (``u_packed_notour``). A block is 8 tiles and 16 stamina of watering a day.
MAX_WATER_TILES = 112
# A first planting hoes 8 cells (16 stamina) on top of the day's watering; a replant after a harvest
# hoes nothing (the soil stays tilled). Base never established more than 5 bags a day, most of them replants.
NEW_BLOCKS_PER_DAY = 3

LATTICE_COLUMNS: Tuple[int, ...] = (4, 8, 12, 16, 20, 24)
LATTICE_ROWS: Tuple[int, ...] = (35, 39, 43, 47)
# Columns 4/12/20 open east and 8/16/24 west, so each pair's notches share the lane between them.
EAST_NOTCH_COLUMNS = frozenset({4, 12, 20})
# Yard plots north of the bin: (13,28) opens west onto the bin lane; (19,28) opens west because
# (20,28) has a wall at x=21 and would be sealed either way.
YARD_BLOCKS: Tuple[Tuple[Tile, str], ...] = (((13, 28), "west"), ((19, 28), "west"))
# The Day-3 pin's (13,28) plot was hoed as a ring on D2 (centre bare, all 8 around tilled). Hoed soil
# never reverts, so a bag there sows that shape again until a run starts before D2.
PIN_RING_PLOTS: frozenset = frozenset({(13, 28)})

NOTCH_DX = {"east": 1, "west": -1}
LAYOUT_NAME = {"east": "eight_east_notch", "west": "eight_west_notch", "ring": "eight_tile_ring"}
_FACE_DELTA = {"down": (0, -1), "up": (0, 1), "left": (1, 0), "right": (-1, 0)}  # stand = crop + delta


def block_cells(center: Tile, notch: str) -> Tuple[Tile, ...]:
    """The 8 planted cells: the 3x3 less the notch (``ring``: less the centre)."""
    cx, cy = center
    if notch == "ring":
        gap = (cx, cy)
    else:
        gap = (cx + NOTCH_DX[notch], cy)
    return tuple((cx + dx, cy + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (cx + dx, cy + dy) != gap)


def notch_cell(center: Tile, notch: str) -> Optional[Tile]:
    return None if notch == "ring" else (center[0] + NOTCH_DX[notch], center[1])


def lattice_blocks(columns: Sequence[int] = LATTICE_COLUMNS, rows: Sequence[int] = LATTICE_ROWS,
                   yard: Sequence[Tuple[Tile, str]] = YARD_BLOCKS) -> Tuple[Tuple[Tile, str], ...]:
    field_blocks = tuple(((x, y), "east" if x in EAST_NOTCH_COLUMNS else "west") for y in rows for x in columns)
    return tuple(yard) + field_blocks


# ── Geometry ────────────────────────────────────────────────────────────

def open_field_walkable(tile: Tile) -> bool:
    """Static stand-in for the cleared D2 farm around the bin (tests, no pin)."""
    x, y = tile
    if not (2 <= x <= 61 and 3 <= y <= 59):
        return False
    if (x, y) in ((11, 26), (11, 27), (11, 28)):
        return True                                       # bin lane
    if y in (29, 30, 31) and x <= 10:                     # bin, ditch and its wall
        return False
    if y == 32 and x <= 5:
        return False
    if x == 21 and 25 <= y <= 30:                         # the yard wall east of (19,28)
        return False
    return not (y <= 28 and x <= 10)


def pin_walkable(pin: str = "LN_d2_perfect_poweron") -> Callable[[Tile], bool]:
    """Walkability read from a pin's farm array (local pins only)."""
    from harvest.maps.farm_final_layout import _pin_maps
    from harvest.maps.farm_pond import FARM_NO_GO_TILES, SHIP_BIN_LANE_TILES

    _tilemap, persistent, _visible = _pin_maps(pin)
    tiles = list(persistent)
    open_ids = {0x00, 0x01, 0x02, 0x07, 0x08} | set(range(0x1E, 0x70))   # soil and (to be harvested) crops
    extra = set(SHIP_BIN_LANE_TILES) | set(BIN_STANDS)

    def walkable(tile: Tile) -> bool:
        x, y = tile
        if not (0 <= x < 64 and 0 <= y < 64):
            return False
        if tile in extra:
            return True
        if tile in FARM_NO_GO_TILES:
            return False
        return 2 <= x <= 61 and 3 <= y <= 59 and tiles[y * 64 + x] in open_ids

    return walkable


def bfs(walkable: Callable[[Tile], bool], sources: Iterable[Tile], blocked: frozenset) -> Dict[Tile, int]:
    dist: Dict[Tile, int] = {}
    queue: deque = deque()
    for src in sources:
        dist[src] = 0
        queue.append(src)
    while queue:
        x, y = queue.popleft()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in dist or nxt in blocked or not walkable(nxt):
                continue
            dist[nxt] = dist[(x, y)] + 1
            queue.append(nxt)
    return dist


@dataclass
class CropSlot:
    tile: Tile
    stand: Optional[Tile]
    face: Optional[str]
    walk_tiles: Optional[int]          # one way, stand to the nearest bin stand
    cycle_frames: Optional[int]        # pick + walk out and back + throw

    @property
    def reachable(self) -> bool:
        return self.walk_tiles is not None


@dataclass
class BlockPlan:
    center: Tile
    notch: str = "east"                # east | west | ring (the pin's D2 plot only)
    rank: int = 0
    role: str = "off"                  # core | late | off
    crops: List[CropSlot] = field(default_factory=list)
    spring_last_sow: Optional[int] = None
    potato_ripe_summer_day: Optional[int] = None
    summer_crop: Optional[str] = None
    summer_plant_day: Optional[int] = None

    @property
    def layout_name(self) -> str:
        return LAYOUT_NAME[self.notch]

    @property
    def live_crops(self) -> List[CropSlot]:
        return [c for c in self.crops if c.reachable]

    @property
    def mean_cycle_frames(self) -> float:
        live = self.live_crops
        return sum(c.cycle_frames for c in live) / len(live) if live else math.inf


def crop_cycle_frames(walk_tiles: int) -> int:
    return int(round(PICK_FRAMES + THROW_FRAMES + 2 * walk_tiles * RUN_FRAMES_PER_TILE))


def measure_blocks(blocks: Sequence[Tuple[Tile, str]], walkable: Callable[[Tile], bool]) -> List[BlockPlan]:
    """Every block's crops with their best stand and cycle frames, all blocks grown (lanes and notches).

    A crop's stand is any orthogonal neighbour that is open ground once every block is grown: a
    lane, or its own block's notch. Cycle frames assume the trip starts and ends at a bin stand.
    """
    grown = frozenset(t for c, n in blocks for t in block_cells(c, n))
    dist = bfs(walkable, BIN_STANDS, grown)
    out = []
    for center, notch in blocks:
        slots = []
        for tile in block_cells(center, notch):
            best = None
            for face, (dx, dy) in _FACE_DELTA.items():
                stand = (tile[0] + dx, tile[1] + dy)
                if stand in dist and (best is None or dist[stand] < best[0]):
                    best = (dist[stand], stand, face)
            if best is None:
                slots.append(CropSlot(tile, None, None, None, None))
            else:
                slots.append(CropSlot(tile, best[1], best[2], best[0], crop_cycle_frames(best[0])))
        out.append(BlockPlan(center=center, notch=notch, crops=slots))
    return out


def assign_roles(blocks: Sequence[BlockPlan], core: int, late: int, *, tomato_core: int = 0) -> List[BlockPlan]:
    """Nearest ``core`` blocks, then the next ``late``; the rest are off. Rank is by mean cycle.

    ``tomato_core`` of the core blocks (the farthest of them) take tomato instead of corn: a tomato
    ripens 3 days sooner and once more, but ships 100G against 120G per walk.
    """
    ordered = sorted(blocks, key=lambda b: (b.mean_cycle_frames, b.center[1], b.center[0]))
    out = []
    for rank, block in enumerate(ordered):
        block = replace(block, rank=rank, crops=list(block.crops))
        if rank < core:
            sow = CORE_LAST_SOW[rank % len(CORE_LAST_SOW)]
            ripe = sow + POTATO_NIGHTS - SEASON_DAYS
            crop = "tomato" if rank >= core - tomato_core else "corn"
            block = replace(block, role="core", spring_last_sow=sow, potato_ripe_summer_day=ripe,
                            summer_crop=crop, summer_plant_day=ripe)
        elif rank < core + late:
            sow = LATE_LAST_SOW[(rank - core) % len(LATE_LAST_SOW)]
            ripe = sow + POTATO_NIGHTS - SEASON_DAYS
            block = replace(block, role="late", spring_last_sow=sow, potato_ripe_summer_day=ripe,
                            summer_crop=LATE_SUMMER_CROP, summer_plant_day=ripe)
        out.append(block)
    return out


# ── Forecast ───────────────────────────────────────────────────────────

def weekday(season: int, day: int) -> int:
    return (SPRING_D1_WEEKDAY + season * SEASON_DAYS + day - 1) % 7


def shop_open(season: int, day: int) -> bool:
    return weekday(season, day) < 5 and not (season == 0 and day == FLOWER_FESTIVAL_DAY)


def _abs(season: int, day: int) -> int:
    return season * SEASON_DAYS + day


def sow_crop(block: BlockPlan, season: int, day: int) -> Optional[str]:
    """The bag this block takes on (season, day), or None. The farm layout gates on the same rule."""
    if block.role not in ("core", "late"):
        return None
    if season == 0:
        return "potato" if block.spring_last_sow is not None and day <= block.spring_last_sow else None
    if season == 1 and block.summer_crop is not None and block.summer_plant_day is not None:
        return block.summer_crop if block.summer_plant_day <= day <= SUMMER_LAST_BAG_DAY else None
    return None


@dataclass
class _Cell:
    block: BlockPlan
    slot: CropSlot
    crop: Optional[str] = None
    ripe_day: Optional[int] = None     # absolute day (Spring D1 = 1, Summer D1 = 31)


def _ripe_today(cells: Sequence[_Cell], today: int) -> int:
    """Planted cells that empty today (ripe potatoes): they free their watering for the new bag."""
    return sum(1 for c in cells if c.crop == "potato" and c.ripe_day is not None and c.ripe_day <= today)


@dataclass
class ForecastConfig:
    start_season: int = 0
    start_day: int = 3
    start_money: int = 260
    preplanted: Tuple[Tuple[Tile, int], ...] = (((13, 28), 2),)   # (block, sow day): the D2 potatoes
    ship_day_ticks: int = SHIP_DAY_TICKS
    max_water_tiles: int = MAX_WATER_TILES
    end_season: int = 1
    end_day: int = SEASON_DAYS


def forecast(blocks: Sequence[BlockPlan], cfg: ForecastConfig = ForecastConfig()) -> dict:
    """Day-by-day Spring and Summer: bags, sowing, ripe, shipped, carry-over and cash.

    One shop trip a weekday buys a bag for every empty block the calendar still lets sow (core
    first), as far as cash goes; shipped gold arrives that night. Each day ships ripe crops
    nearest-first until the 06:00-17:00 clock is spent; a crop left on the plant waits, and its
    regrowth starts from the day it is picked. Watering is after 17:00 and costs no clock.
    """
    live = [b for b in blocks if b.role in ("core", "late")]
    cells = [_Cell(b, s) for b in live for s in b.live_crops]
    by_block: Dict[Tile, List[_Cell]] = {}
    for c in cells:
        by_block.setdefault(c.block.center, []).append(c)
    for center, sow in cfg.preplanted:
        for c in by_block.get(center, []):
            c.crop, c.ripe_day = "potato", sow + POTATO_NIGHTS
    money = cfg.start_money
    days: List[dict] = []
    totals = {"gold": 0, "shipped": {"potato": 0, "corn": 0, "tomato": 0}, "bags": {}, "seed_g": 0}
    start, end = _abs(cfg.start_season, cfg.start_day), _abs(cfg.end_season, cfg.end_day)
    pending: Dict[Tile, str] = {}
    hoed = {center for center, _sow in cfg.preplanted}
    for today in range(start, end + 1):
        season, day = (today - 1) // SEASON_DAYS, (today - 1) % SEASON_DAYS + 1
        row = {"season": season, "day": day, "weekday": weekday(season, day), "money_morning": money}
        bought: Dict[str, int] = {}
        planted_tiles = sum(1 for c in cells if c.crop is not None)
        wanted: List[Tuple[BlockPlan, str]] = []
        new_blocks = 0
        for block in sorted(live, key=lambda b: (b.role != "core", b.rank)):
            if block.center in pending:
                continue
            fresh = block.center not in hoed
            if fresh and new_blocks >= NEW_BLOCKS_PER_DAY:
                continue
            mine = by_block[block.center]
            empty_soon = all(c.crop is None or (c.crop == "potato" and c.ripe_day <= today) for c in mine)
            crop = sow_crop(block, season, day) if empty_soon else None
            if crop is None:
                continue
            if planted_tiles + len(mine) - _ripe_today(mine, today) > cfg.max_water_tiles:
                continue
            wanted.append((block, crop))
            planted_tiles += len(mine)
            new_blocks += int(fresh)
        trip = bool(wanted) and shop_open(season, day)
        if trip:
            for block, crop in wanted:
                if money - BAG_G[crop] < WALLET_RESERVE_G:
                    break
                money -= BAG_G[crop]
                totals["seed_g"] += BAG_G[crop]
                bought[crop] = bought.get(crop, 0) + 1
                pending[block.center] = crop
        ticks = cfg.ship_day_ticks - (SHOP_TRIP_TICKS if trip else 0)
        if season == 0 and day == FLOWER_FESTIVAL_DAY:
            ticks = min(ticks, FESTIVAL_SHIP_TICKS)
        ripe = sorted((c for c in cells if c.crop and c.ripe_day is not None and c.ripe_day <= today),
                      key=lambda c: c.slot.cycle_frames)
        shipped: Dict[str, int] = {}
        by_rank: Dict[int, int] = {}
        gold = used = 0
        for c in ripe:
            cost = c.slot.cycle_frames
            if used + cost > ticks:
                continue
            used += cost
            gold += PRICE_G[c.crop]
            shipped[c.crop] = shipped.get(c.crop, 0) + 1
            by_rank[c.block.rank] = by_rank.get(c.block.rank, 0) + 1
            if c.crop == "potato":
                c.crop, c.ripe_day = None, None
            else:
                c.ripe_day = today + REGROW_NIGHTS
        left = sum(1 for c in cells if c.crop and c.ripe_day is not None and c.ripe_day <= today)
        # Forage (``forage_calendar``): with nothing ripe to ship the morning is free, so the mountain
        # trip goes first (after the shop on a buy day); a harvest day keeps its clock for the bin.
        kind = forage_for_season(season) if season in (0, 1) else None
        forage_trips = 0
        if kind is not None and not ripe and not (season == 0 and day == FLOWER_FESTIVAL_DAY):
            forage_trips = trips_today(kind, 6 + (SHOP_TRIP_TICKS // TICKS_PER_HOUR if trip else 0))
        forage_gold = forage_trips * kind.price_g if forage_trips else 0
        sown = []
        for block in live:
            crop = pending.get(block.center)
            if crop is None:
                continue
            mine = by_block[block.center]
            if any(c.crop is not None for c in mine):
                continue                                           # its last potatoes are still on the plant
            for c in mine:
                c.crop, c.ripe_day = crop, today + NIGHTS[crop]
            hoed.add(block.center)
            del pending[block.center]
            sown.append({"block": list(block.center), "crop": crop})
        money += gold + forage_gold
        totals["gold"] += gold
        totals["forage_gold"] = totals.get("forage_gold", 0) + forage_gold
        totals["forage_trips"] = totals.get("forage_trips", 0) + forage_trips
        for k, v in shipped.items():
            totals["shipped"][k] += v
        for k, v in bought.items():
            totals["bags"][k] = totals["bags"].get(k, 0) + v
        row.update(shop=trip, bags=bought, sown=sown, ripe=len(ripe), shipped=shipped,
                   shipped_by_rank={str(k): v for k, v in sorted(by_rank.items())},
                   ship_frames=used, ship_ticks_free=ticks - used, left_ripe=left, gold=gold,
                   forage={"kind": kind.name, "trips": forage_trips, "gold": forage_gold} if forage_trips else None,
                   money_night=money, water_tiles=sum(1 for c in cells if c.crop is not None))
        days.append(row)
    totals["money_end"] = money
    totals["left_ripe_end"] = days[-1]["left_ripe"] if days else 0
    totals["money_spring_end"] = next((d["money_night"] for d in days if (d["season"], d["day"]) == (0, 30)), None)
    totals.setdefault("forage_gold", 0)
    totals.setdefault("forage_trips", 0)
    totals["summer_gold"] = sum(d["gold"] for d in days if d["season"] == 1)
    totals["summer_shipped"] = sum(sum(d["shipped"].values()) for d in days if d["season"] == 1)
    return {"days": days, "totals": totals}


# ── Herd: grass and cows bought from Summer cash (an estimate, not ROM tested) ─────

HAY_DAYS_PER_TILE = 9          # livestock_econ: a planted tile regrows one hay every 9 days
GRASS_TILES_PER_BAG = 8        # livestock_econ.GRASS_BAG_TILES (9 at best)
GRASS_GATE_PER_ANIMAL = 16     # the animal shop: planted tiles per animal after the buy
GRASS_MATURE_NIGHTS = 9
COW_G = 5000
TOOLS_G = 800 + 1800           # brush + milker, once
CHILD_NIGHTS = 21              # a bought cow milks from 21 nights after the buy
HAPPY_PER_DAY = 4              # talk +1, brush +3
MILK_TIERS = ((0x60, 150), (0xC0, 250), (0x100, 350))
HENS = 2
WINTER_START, Y2_SPRING_D14, Y2_SUMMER_END = 91, 134, 180   # absolute days from Y1 Spring D1


def milk_gold(buy_day: int, horizon: int) -> int:
    """Milk shipped by ``horizon`` from a cow bought on ``buy_day`` (happiness from 0, +4 a day)."""
    gold = 0
    for d in range(buy_day + CHILD_NIGHTS, horizon + 1):
        happy = min(255, (d - buy_day) * HAPPY_PER_DAY)
        gold += next(price for cap, price in MILK_TIERS if happy < cap)
    return gold


def grass_needed(animals: int, first_feed_day: int, sow_day: int = 45) -> Tuple[int, int]:
    """(tiles, bags) that pass the shop gate and feed ``animals`` from ``first_feed_day`` to Y2 Spring D14.

    Hay comes from cutting: a tile sown on ``sow_day`` is mature 9 nights later and regrows one hay
    every 9 days to the end of Fall (Winter wipes it). Every animal eats one hay a day.
    """
    grow_days = max(1, WINTER_START - (sow_day + GRASS_MATURE_NIGHTS))
    hay_per_tile = grow_days / HAY_DAYS_PER_TILE
    demand = animals * (Y2_SPRING_D14 - first_feed_day)
    tiles = max(GRASS_GATE_PER_ANIMAL * animals, math.ceil(demand / hay_per_tile))
    return tiles, math.ceil(tiles / GRASS_TILES_PER_BAG)


def herd_options(money_by_day: Dict[int, int], *, first_buy_day: int = 31, max_cows: int = 8,
                 reserve_g: int = 1000, last_grass_day: int = 58) -> dict:
    """For 0..max_cows: grass to sow by Summer's end, buy days, spend, milk to Y1 Winter end and Y2 Summer end.

    Cash comes from the forecast (Summer only). The grass is bought first (bags count toward the
    shop gate only in Spring and Summer), then one cow a day once the cash covers 5,000G plus the
    reserve. A cow bought on day b eats from b and milks from b + 21. Hens: 2, already kept.
    """
    options = []
    days = sorted(d for d in money_by_day if d >= first_buy_day)
    for cows in range(0, max_cows + 1):
        animals = cows + HENS
        tiles, bags = grass_needed(animals, first_feed_day=first_buy_day + 7)
        grass_g = bags * BAG_G["grass"]
        fixed = grass_g + (TOOLS_G if cows else 0)
        buys: List[int] = []
        spent = 0
        grass_day = None
        for d in days:
            if grass_day is None and money_by_day[d] >= fixed + reserve_g:
                grass_day = d
                spent = fixed
            if grass_day is None or len(buys) >= cows or d <= grass_day:
                continue
            if money_by_day[d] - spent >= COW_G + reserve_g:
                buys.append(d)
                spent += COW_G
        row = {"cows": cows, "animals": animals, "grass_tiles": tiles, "grass_bags": bags, "grass_g": grass_g,
               "grass_buy_day": grass_day, "buy_days": buys, "spend_g": spent,
               "feasible": grass_day is not None and grass_day <= last_grass_day and len(buys) == cows}
        if row["feasible"]:
            row["milk_to_y1_winter_end"] = sum(milk_gold(b, WINTER_START + 29) for b in buys)
            row["milk_to_y2_summer_end"] = sum(milk_gold(b, Y2_SUMMER_END) for b in buys)
            row["net_to_y2_summer_end"] = row["milk_to_y2_summer_end"] - spent
        options.append(row)
    feasible = [o for o in options if o["feasible"]]
    best = max(feasible, key=lambda o: o["net_to_y2_summer_end"]) if feasible else None
    return {"options": options, "best_cows": best["cows"] if best else 0,
            "note": "estimate: cash from the Summer forecast only (no Fall income), care assumed perfect"}


# ── Plan ───────────────────────────────────────────────────────────────

def plan_blocks(walkable: Callable[[Tile], bool] = open_field_walkable, *, core: int = 10, late: int = 4,
                tomato_core: int = 0, blocks: Sequence[Tuple[Tile, str]] = ()) -> List[BlockPlan]:
    chosen = tuple(blocks) or lattice_blocks()
    chosen = tuple((c, "ring" if c in PIN_RING_PLOTS else n) for c, n in chosen)
    return assign_roles(measure_blocks(chosen, walkable), core, late, tomato_core=tomato_core)


def build_plan(walkable: Callable[[Tile], bool] = open_field_walkable, *, core: int = 10, late: int = 4,
               tomato_core: int = 0, cfg: ForecastConfig = ForecastConfig()) -> dict:
    blocks = plan_blocks(walkable, core=core, late=late, tomato_core=tomato_core)
    fc = forecast(blocks, cfg)
    money_by_day = {_abs(d["season"], d["day"]): d["money_night"] for d in fc["days"]}
    return {
        "schema": SCHEMA,
        "assumptions": {
            "potato_nights": POTATO_NIGHTS, "corn_nights": CORN_NIGHTS, "tomato_nights": TOMATO_NIGHTS,
            "regrow_nights": REGROW_NIGHTS, "ship_day_ticks": cfg.ship_day_ticks,
            "run_frames_per_tile": RUN_FRAMES_PER_TILE, "pick_frames": PICK_FRAMES, "throw_frames": THROW_FRAMES,
            "core_last_sow": list(CORE_LAST_SOW), "late_last_sow": LATE_LAST_SOW,
            "summer_last_bag_day": SUMMER_LAST_BAG_DAY, "shop_trip_ticks": SHOP_TRIP_TICKS,
            "max_water_tiles": cfg.max_water_tiles,
            "start": {"season": cfg.start_season, "day": cfg.start_day, "money": cfg.start_money},
            "core": core, "late": late, "tomato_core": tomato_core,
        },
        "bin_stands": [list(s) for s in BIN_STANDS],
        "blocks": [block_json(b) for b in blocks],
        "forecast": fc,
        "herd": herd_options(money_by_day),
    }


def block_json(block: BlockPlan) -> dict:
    out = asdict(block)
    out["center"] = list(block.center)
    out["layout_name"] = block.layout_name
    out["notch_cell"] = list(notch_cell(block.center, block.notch)) if block.notch != "ring" else None
    out["crops"] = [{**asdict(c), "tile": list(c.tile), "stand": list(c.stand) if c.stand else None}
                    for c in block.crops]
    mean = block.mean_cycle_frames
    out["mean_cycle_frames"] = None if math.isinf(mean) else round(mean, 1)
    return out


def sweep(walkable: Callable[[Tile], bool] = open_field_walkable, cores: Iterable[int] = range(6, 19),
          lates: Iterable[int] = (0, 2, 4, 6, 8), tomatoes: Iterable[int] = (0,),
          cfg: ForecastConfig = ForecastConfig()) -> List[dict]:
    """Summer-end cash for each (core, late, tomato) split, best first."""
    chosen = tuple((c, "ring" if c in PIN_RING_PLOTS else n) for c, n in lattice_blocks())
    measured = measure_blocks(chosen, walkable)
    rows = []
    for core in cores:
        for late in lates:
            for tom in tomatoes:
                if tom > core:
                    continue
                blocks = assign_roles(measured, core, late, tomato_core=tom)
                t = forecast(blocks, cfg)["totals"]
                rows.append({"core": core, "late": late, "tomato_core": tom, "money_end": t["money_end"],
                             "money_spring_end": t["money_spring_end"], "summer_gold": t["summer_gold"],
                             "shipped": t["shipped"], "left_ripe_end": t["left_ripe_end"]})
    return sorted(rows, key=lambda r: -r["money_end"])
