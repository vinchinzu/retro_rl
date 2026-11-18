"""Run strategy: what a full run is optimizing, as one switchable object.

The planner's policy numbers (cows kept, hens kept, which crop a Summer bag buys, when the
bin stops taking a crop) used to be module constants tuned for one goal: cash. The ending
score pays for cash only up to 99,840G (78 points); the runs end with 140k-320k, so the
surplus buys nothing. A strategy names the goal, and the few decision points that differ ask
it instead of reading a constant.

    HARVEST_STRATEGY=cash         # default: the tuned cash path, byte-identical to before
    HARVEST_STRATEGY=score        # + stop each crop at the ROM window, balance crops, plant surplus-funded cover from Year 2
    HARVEST_STRATEGY=score_herd   # + a bigger herd and flock (experimental, A/B before trusting)
    HARVEST_STRATEGY=cover        # alias of score: the score family builds the whole field when the schedule needs it
    HARVEST_STRATEGY=perfect      # + the milestone campaign toward 999 (not a proved route)
    run_to_day2 ... --strategy score

``cash`` overrides nothing, so every constant it falls back to stays the single source for
the cash path (and the tests that patch them keep working). A field set to ``None`` means
"use the module default".

Shipped crops (ROM ``RanchMasteryCalculator``): one point per 16 shipped through
``(count & 0x1FF) >> 4``, so 496..511 is the full 31 points and a count past 511 wraps to 0.
The bin drop is the only increment (``harvest-e2e-and-saddle`` memory, F8 research), so a
ripe tile that would take a crop past ``ship_cap`` is simply left on the plant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

ENV_VAR = "HARVEST_STRATEGY"
DEFAULT_NAME = "cash"

SHIP_CROPS: tuple[str, ...] = ("corn", "tomato", "potato", "turnip")
SHIP_COUNTER_MASK = 0x1FF
SHIP_FULL_POINTS = 31
SHIP_WINDOW_LOW = 496              # 496..511 is 31 points
SHIP_WINDOW_HIGH = 511             # one more shipment wraps the counter to 0
# Stop at the window floor, not the ceiling. Stopping at 511 left no room: e2e ``s2`` ended at
# 512 (0 points, 31 lost) because a ripe tile the cap had left standing was picked up by some
# other task and thrown into the bin. 15 shipments of slack absorb that.
SHIP_CAP = SHIP_WINDOW_LOW
# Shelf stands verified by single-trip ROM receipts (including turnips).
BUYABLE_CROPS: tuple[str, ...] = ("potato", "turnip", "corn", "tomato")


@dataclass(frozen=True)
class RunStrategy:
    name: str
    summary: str
    # Highest shipped count per crop; None ships everything (the cash path).
    ship_cap: Optional[int] = None
    # Pick the seed bag by room left under the cap instead of forcing the calendar default.
    balance_crops: bool = False
    # Herd and flock overrides; None keeps the ``livestock_econ`` constants.
    cow_target_later: Optional[int] = None
    cow_target_fall: Optional[int] = None
    hens_kept: Optional[int] = None
    # Once the last 50-care-day cow window closes, use the incubator for ending
    # chicken points. Before this date the small flock leaves the shop grass
    # gate and hay to cows. None keeps the ordinary flock for the whole run.
    final_flock_target: Optional[int] = None
    # Y3 Spring D25 (Y1 Spring D1 = 0). Measured on the Year 3 runs: one hen hatches every 3 days from the first hatch
    # (index 254 whether the flock started at 240 or 250), and a chick eats nothing until it is an adult 8 days later.
    # Starting at 250 or earlier had twelve hens at index 281 and 18 days of full hay draw; hay hit 0 on Summer D23 (y3b1),
    # the cows went sick and the run lost ~16 points. 264 reaches twelve at ~293 and costs ~120 hay in all.
    final_flock_start_index: int = 264
    # Year 3: no animal purchase may leave the wallet below this (the money term needs 99,840G on the
    # last morning). e2e ``h5`` bought cows into 92,100G and lost 7 of the 78 money points.
    late_cash_floor_g: Optional[int] = None
    # Dispatch the Perfect Rancher milestone campaign (what ``--perfect-rancher`` turns on).
    perfect_rancher: bool = False
    # Whole-field cover (farm development, 50 points at 2,950 counted tiles): start modest loads in Year 2 through
    # ``planner/field_cover``, paid only from gold the money
    # term (99,840G on the last morning) does not need. False on the cash path, which never spends on it.
    field_cover: bool = False
    # Largest pasture the herd target may ask for; None keeps ``livestock_econ.PASTURE_MAX_TILES``. A bigger herd
    # also sizes the pasture for the animals it plans to own, not only the ones it has: the cash-optimal herd table
    # (``block_calendar.herd_options``) wants 187 tiles for 6 cows and 234 for 8, past the 180-tile default.
    pasture_max_tiles: Optional[int] = None
    # Spend one early trip a day on a scheduled power berry source (``planner/power_berry_calendar``): +10 max stamina
    # and 5 ranch points each. Off on the cash path so its runs stay byte-comparable with every earlier row.
    collect_power_berries: bool = False
    # Court the four girls who are not the wife toward the 496-511 window (``planner/social_calendar``): 31 points
    # each, ~0 in every run so far. Off on the cash path, like the berries.
    court_girls: bool = False

    @property
    def caps_shipping(self) -> bool:
        return self.ship_cap is not None

    def room(self, crop: str, shipped: Mapping[str, int]) -> Optional[int]:
        """Shipments ``crop`` can still take under the cap; None when it is uncapped."""
        if self.ship_cap is None:
            return None
        return max(0, int(self.ship_cap) - int(shipped.get(crop, 0) or 0))

    def capped(self, crop: str, shipped: Mapping[str, int]) -> bool:
        room = self.room(crop, shipped)
        return room is not None and room <= 0


CASH = RunStrategy(
    "cash",
    "Max cash. Ship every ripe crop, 2-4 cows, 2 hens: the tuned path all earlier runs used.",
)
SCORE = RunStrategy(
    "score",
    "Raise ranch score throughout the campaign: stop each crop at 496 shipped, balance both seasonal crop kinds, "
    "collect berries and spend surplus on modest grass loads from Year 2.",
    ship_cap=SHIP_CAP,
    balance_crops=True,
    field_cover=True,
    collect_power_berries=True,
    court_girls=True,
    final_flock_target=12,
)
SCORE_HERD = RunStrategy(
    "score_herd",
    "SCORE with twelve cows: 28 points each at happiness 200. Keep two hens until the final flock "
    "window, and grow pasture for acquisition and Winter hay. Eight happy cows have a native ending "
    "receipt; the full herd remains experimental until its acquisition and care run completes.",
    ship_cap=SHIP_CAP,
    balance_crops=True,
    cow_target_later=12,
    cow_target_fall=4,
    late_cash_floor_g=110_000,
    field_cover=True,
    pasture_max_tiles=420,
    collect_power_berries=True,
    court_girls=True,
    final_flock_target=12,
)

# ``cover`` was a separate preset that spent idle gold on grass bags; the score family now builds the whole field on
# a schedule (planner/field_cover.py), so the old name stays as an alias for scripts and notes that still pass it.
COVER = SCORE

PERFECT = RunStrategy(
    "perfect",
    "The 999 campaign: SCORE_HERD at the ROM caps (12 cows, 12 hens) plus the milestone dispatch "
    "(--perfect-rancher). F10: not a proved route; hearts, berries, clock and turtle have no native recipe yet.",
    ship_cap=SHIP_CAP,
    balance_crops=True,
    cow_target_later=12,
    cow_target_fall=4,
    hens_kept=12,
    perfect_rancher=True,
    field_cover=True,
    pasture_max_tiles=420,         # 16 planted tiles per animal after the buy: 12 cows + 12 hens + 2 = 400
    collect_power_berries=True,
    court_girls=True,
)

PRESETS: dict[str, RunStrategy] = {s.name: s for s in (CASH, SCORE, SCORE_HERD, PERFECT)}
ALIASES = {"cover": "score"}


def get(name: str | None = None) -> RunStrategy:
    key = (name or DEFAULT_NAME).strip().lower()
    key = ALIASES.get(key, key)
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(PRESETS)}") from None


def active() -> RunStrategy:
    """The strategy ``HARVEST_STRATEGY`` selects, read on every call so tests can patch the env."""
    return get(os.environ.get(ENV_VAR))


def select(name: str | None) -> RunStrategy:
    """Make ``name`` the process-wide strategy (``--strategy``); returns it."""
    strategy = get(name)
    os.environ[ENV_VAR] = strategy.name
    return strategy


# ── shipped-crop counters ────────────────────────────────────────────

def crop_points(count: int) -> int:
    """Ending points one crop's counter pays: ``(count & 0x1FF) >> 4``."""
    return (int(count) & SHIP_COUNTER_MASK) >> 4


def points_lost_to_wrap(shipped: Mapping[str, int]) -> int:
    """Points the four counters would pay if each had stopped at 511 instead of wrapping."""
    lost = 0
    for crop in SHIP_CROPS:
        count = int(shipped.get(crop, 0) or 0)
        if count > SHIP_WINDOW_HIGH:
            lost += SHIP_FULL_POINTS - crop_points(count)
    return lost


def shipped_totals(ram: Any) -> dict[str, int]:
    from harvest.core.ram_catalog import read_ram_value

    return {
        "corn": int(read_ram_value(ram, "shipped_corn")),
        "tomato": int(read_ram_value(ram, "shipped_tomatoes")),
        "potato": int(read_ram_value(ram, "shipped_potatoes")),
        "turnip": int(read_ram_value(ram, "shipped_turnips")),
    }


def within_ship_cap(
    ram: Any,
    tiles: Sequence[tuple[int, int]],
    strategy: Optional[RunStrategy] = None,
    *,
    near: tuple[int, int] = (11, 30),
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split ripe ``tiles`` into (kept, held): kept fits the room each crop has under the cap.

    When a crop has less room than ripe tiles, the nearest to the bin (``near``) are kept. Both
    lists keep the caller's order, so a cap changes which tiles are picked and never their
    sequence (an A/B against ``cash`` then differs only where the cap bites). An uncapped
    strategy keeps all.
    """
    strategy = strategy or active()
    tiles = list(tiles)
    if not strategy.caps_shipping or not tiles:
        return tiles, []
    from harvest.core.tile_catalog import get_tile_at, mature_tile_crop

    shipped = shipped_totals(ram)
    room: dict[str, int] = {crop: int(strategy.room(crop, shipped) or 0) for crop in SHIP_CROPS}
    keep: set[tuple[int, int]] = set()
    for tile in sorted(tiles, key=lambda t: abs(t[0] - near[0]) + abs(t[1] - near[1])):
        crop = mature_tile_crop(get_tile_at(ram, *tile))
        if crop is None or room.get(crop, 0) > 0:
            keep.add(tile)
            if crop is not None:
                room[crop] -= 1
    return [t for t in tiles if t in keep], [t for t in tiles if t not in keep]


def crop_kinds(ram: Any, tiles: Iterable[tuple[int, int]]) -> list[str]:
    """Sorted crop kinds ripe on ``tiles`` (for log lines)."""
    from harvest.core.tile_catalog import get_tile_at, mature_tile_crop

    return sorted({mature_tile_crop(get_tile_at(ram, *t)) or "?" for t in tiles})


def seed_allowed(crop: str, shipped: Mapping[str, int], strategy: Optional[RunStrategy] = None) -> bool:
    """True when buying/planting ``crop`` still pays under ``strategy``."""
    strategy = strategy or active()
    if strategy.capped(crop, shipped):
        return False
    return not strategy.balance_crops or crop in BUYABLE_CROPS


POTATO_STAGE_TILES = tuple(range(0x54, 0x62))   # seeded through ripe (0x60/0x61); ring tests use 0x54
TURNIP_STAGE_TILES = tuple(range(0x62, 0x70))   # seeded through ripe (0x6E/0x6F); 0x70 is grass
SOW_BOUNDS = (2, 3, 62, 60)                      # the crop field, as ``crop_planner.DEFAULT_CROP_BOUNDS``
RING_TILES = 8                                   # one seed bag sows an eight-tile ring
Y3_ALL_BAGS_FROM_YEAR = 2                        # 0-based: Year 3, the last Spring a potato or turnip can ship
Y3_ALL_BAGS_ENV = "HARVEST_Y3_ALL_BAGS"
Y3_RESERVE_ENV = "HARVEST_Y3_TOMATO_RESERVE"
Y3_STANDING_SLACK = int(os.environ.get("HARVEST_Y3_SPRING_SLACK", str(3 * RING_TILES)))   # tiles; 0 restores exact accounting


def spring_committed_totals(ram: Any, strategy: Optional[RunStrategy] = None) -> dict[str, int]:
    """Reserve standing Spring crops against their own shipping caps before choosing a bag.

    Read the persistent farm array indoors too, so the morning plan and the later farm replan agree.
    The last bag may exceed the remaining demand: the shipping cap leaves its spare crops standing.
    """
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.field_cover import farm_tiles

    strategy = strategy or active()
    shipped = shipped_totals(ram)
    if not strategy.caps_shipping:
        return shipped
    tiles = farm_tiles(ram)
    left, top, right, bottom = SOW_BOUNDS
    field = [tiles[y * 64 + x] for y in range(top, bottom + 1) for x in range(left, right + 1)]
    slack = Y3_STANDING_SLACK if int(read_ram_value(ram, "year")) >= Y3_ALL_BAGS_FROM_YEAR else 0
    for crop, ids in (("potato", POTATO_STAGE_TILES), ("turnip", TURNIP_STAGE_TILES)):
        # Year 3 is the last Spring: a sown tile that never ships (a seven-tile ring, a tile that dries out) cannot be
        # made up later, so one spare ring is reserved (the ship cap leaves the surplus on the plant).
        shipped[crop] += max(0, sum(t in ids for t in field) - slack)
    return shipped


def summer_committed_totals(ram: Any) -> dict[str, int]:
    """Balance Summer bags against standing plants as well as past shipments.

    Reserving one harvest per plant is deliberately conservative: weather and the daily shipping
    window can delay regrowth. Without this reservation a single bulk buy fills every empty plot
    with corn (or tomato) before either counter has moved, starving the other crop for a season.
    """
    from harvest.planner.field_cover import farm_tiles

    shipped = shipped_totals(ram)
    if active().balance_crops:
        field = farm_tiles(ram)
        for crop, low, high in (("tomato", 0x1E, 0x39), ("corn", 0x3A, 0x53)):
            shipped[crop] += sum(low <= tile <= high for tile in field)
    return shipped


def quota_harvest_due(ram: Any) -> bool:
    """Ripe crop quota waiting on the farm, including a morning read inside the house."""
    from harvest.core.tile_catalog import mature_tile_crop
    from harvest.planner.field_cover import farm_tiles

    strategy = active()
    shipped = shipped_totals(ram)
    return strategy.caps_shipping and any(
        not strategy.capped(crop, shipped)
        for tile in farm_tiles(ram) if (crop := mature_tile_crop(tile)) is not None
    )


def seed_bags_for_quota(ram: Any, crop: str, empty_plots: int, stock: int) -> int:
    """Buy a share of empty plots, leaving room for the other deficient seasonal crop."""
    from harvest.core.ram_catalog import read_ram_value

    strategy = active()
    if not strategy.balance_crops or crop not in SHIP_CROPS:
        return max(0, empty_plots - stock)
    spring = int(read_ram_value(ram, "season")) == 0
    if spring and int(read_ram_value(ram, "year")) == 0:
        return max(0, empty_plots - stock)  # first-year potatoes fund the herd and house
    totals = spring_committed_totals(ram) if spring else shipped_totals(ram)
    kinds = ("potato", "turnip") if spring else ("corn", "tomato")
    room = {kind: int(strategy.room(kind, totals) or 0) for kind in kinds}
    demand = room.get(crop, 0)
    if demand <= 0:
        return 0
    # One-shot crops cannot earn more than one shipment per tile. Regrowing Summer plants still
    # split the available ground, rather than assuming every ideal harvest will reach the bin.
    if spring and int(read_ram_value(ram, "year")) >= Y3_ALL_BAGS_FROM_YEAR and os.environ.get(Y3_ALL_BAGS_ENV, "1") != "0":
        # One crop is bought per trip, so splitting the empty sites between the two crops left half of them bare
        # until the next trip (fixD Y3 Spring D3: 7 turnip bags for 12 sites, the rest waited for D6/D7).
        share = empty_plots
    elif spring:
        share = (empty_plots * demand + sum(room.values()) - 1) // sum(room.values())
    else:
        from harvest.maps.farm_pond import active_plot_centers

        committed = summer_committed_totals(ram)
        standing = committed[crop] - totals[crop]
        year = int(read_ram_value(ram, "year"))
        plots = max(empty_plots, len(active_plot_centers(year, 1)))
        share = min(empty_plots, max(0, (plots * demand + sum(room.values()) - 1) // sum(room.values())
                                   - standing // RING_TILES))
        if year >= Y3_ALL_BAGS_FROM_YEAR and os.environ.get(Y3_RESERVE_ENV, "1") != "0":
            # The first trip buys the crop with the most room and, with every empty site its own, left the other crop
            # none (y3m: 17 corn bags for 17 empty sites, tomatoes stayed at 470 of 496). Hold back the rings the
            # other crop's share needs while nothing of it stands yet.
            held = sum(
                max(1, (plots * room[other] + sum(room.values()) - 1) // sum(room.values()))
                for other in kinds if other != crop and room[other] > 0 and committed[other] == totals[other]
            )
            share = min(share, max(0, empty_plots - held))
    bags = min(share, (demand + RING_TILES - 1) // RING_TILES) if spring else share
    return max(0, bags - stock)


Y3_JOINT_BUY_ENV = "HARVEST_Y3_JOINT_BUY"
# Nights from the sowing day to the first ripe crop (ROM: corn 0x3A to 0x52 is 12, tomato 0x26 to 0x38 is 9), then a
# crop every 3 nights. A plant that is not watered one night ripens a day late, so a cycle averages ~3.5 days (r45: corn
# picks on D13, 17, 20, 23, 27). The last bin day that still pays is D29: D30 is the credits.
SUMMER_NIGHTS_TO_RIPE = {"corn": 12, "tomato": 9}
SUMMER_CYCLE_DAYS = 3.5
SUMMER_LAST_BIN_DAY = 29
SUMMER_CROPS: tuple[str, ...] = ("corn", "tomato")


def summer_cycles(crop: str, day: int) -> int:
    """Bin harvests a ring sown on Summer ``day`` still delivers before the credits."""
    first = int(day) + SUMMER_NIGHTS_TO_RIPE[crop]
    return 0 if first > SUMMER_LAST_BIN_DAY else 1 + int((SUMMER_LAST_BIN_DAY - first) / SUMMER_CYCLE_DAYS)


def joint_summer_bags(ram: Any, empty_plots: int, stock: Mapping[str, int]) -> dict[str, int]:
    """Year 3 Summer: bags of *each* crop for one shop trip, so every free site is sown the day it can be.

    The shop sells one crop a trip and the weekend shuts it, so buying one crop on D1 and the other on D4-D5 left half
    the sites bare for a week and the first tomato cycle gone (r41..r47: 232-366 crops, 4-6 corn bags on D1). The twelve
    sites are split by what each crop still has to ship divided by what a ring sown today delivers, rings already standing
    count toward their share, and the free sites are filled in that ratio (corn first on a tie). Empty when the switch is
    off, the year or season is wrong, or only one crop can still ship (the single-crop path handles that).
    """
    from harvest.core.ram_catalog import read_ram_value

    strategy = active()
    if os.environ.get(Y3_JOINT_BUY_ENV, "1").strip().lower() in {"", "0", "false", "no", "off"}:
        return {}
    if not (strategy.balance_crops and strategy.caps_shipping) or int(empty_plots) <= 0:
        return {}
    if int(read_ram_value(ram, "season")) != 1 or int(read_ram_value(ram, "year")) < Y3_ALL_BAGS_FROM_YEAR:
        return {}
    from harvest.maps.farm_pond import active_plot_centers

    day = int(read_ram_value(ram, "day"))
    totals = shipped_totals(ram)
    committed = summer_committed_totals(ram)
    need: dict[str, float] = {}
    for crop in SUMMER_CROPS:
        room = int(strategy.room(crop, totals) or 0)
        cycles = summer_cycles(crop, day)
        if room > 0 and cycles > 0:
            need[crop] = room / (RING_TILES * cycles)
    if len(need) < 2:
        return {}
    plots = max(int(empty_plots), len(active_plot_centers(int(read_ram_value(ram, "year")), 1)))
    standing = {crop: (committed[crop] - totals[crop]) // RING_TILES for crop in SUMMER_CROPS}
    weight = sum(need.values())
    share = {crop: plots * need[crop] / weight - standing[crop] for crop in need}
    free = int(empty_plots)
    wanted = {crop: max(0.0, share[crop]) for crop in need}
    scale = free / sum(wanted.values()) if sum(wanted.values()) > 0 else 0.0
    exact = {crop: wanted[crop] * scale for crop in need}
    bags = {crop: int(exact[crop]) for crop in need}
    # Largest remainders first, corn on a tie, until every free site has a bag.
    order = sorted(need, key=lambda c: (exact[c] - bags[c], c == "corn"), reverse=True)
    for crop in order * free:
        if sum(bags.values()) >= free:
            break
        bags[crop] += 1
    return {crop: max(0, n - int(stock.get(f"{crop}_seeds", 0) or 0)) for crop, n in bags.items() if n > 0}


def one_shot_sowing_ok(ram: Any, strategy: Optional[RunStrategy] = None) -> bool:
    """Spring may sow while either crop still needs a shipment after standing crops are reserved."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.crop_planner import resolve_seed_type_for_date

    strategy = strategy or active()
    if not strategy.caps_shipping or int(read_ram_value(ram, "season")) != 0:
        return True
    shipped = spring_committed_totals(ram, strategy)
    return any(
        (strategy.room(crop, shipped) or 0) > 0
        and resolve_seed_type_for_date(0, int(read_ram_value(ram, "day")), inventory={crop: 1}) == crop
        for crop in ("potato", "turnip")
    )


def one_shot_replant_ok(ram: Any, strategy: Optional[RunStrategy] = None) -> bool:
    """False when every one-shot crop the strategy would sow is already at its ship cap."""
    strategy = strategy or active()
    if not strategy.caps_shipping:
        return True
    shipped = shipped_totals(ram)
    return any(seed_allowed(crop, shipped, strategy) for crop in ("potato", "turnip"))


# ── herd and flock ───────────────────────────────────────────────────

def grows_herd(strategy: Optional[RunStrategy] = None) -> bool:
    """True when the strategy raises the later-year cow target above the module default."""
    return (strategy or active()).cow_target_later is not None


def late_cash_floor_g() -> Optional[int]:
    return active().late_cash_floor_g


COW_TARGET_ENV = "HARVEST_COW_TARGET_LATER"


def cow_target_later(default: int) -> int:
    """Cows to own from Year 2; ``HARVEST_COW_TARGET_LATER=<n>`` overrides the strategy for an A/B."""
    forced = os.environ.get(COW_TARGET_ENV, "").strip()
    if forced:
        return int(forced)
    value = active().cow_target_later
    return default if value is None else int(value)


def cow_target_fall(default: int) -> int:
    value = active().cow_target_fall
    return default if value is None else int(value)


def hens_kept(default: int) -> int:
    value = active().hens_kept
    return default if value is None else int(value)


# ── grass cover ──────────────────────────────────────────────────────

FIELD_COVER_ENV = "HARVEST_FIELD_COVER"


def field_cover() -> bool:
    """Whether the planner builds the whole field (``planner/field_cover``); ``HARVEST_FIELD_COVER=1|0`` forces it."""
    forced = os.environ.get(FIELD_COVER_ENV, "").strip().lower()
    if forced in ("1", "true", "yes", "on"):
        return True
    if forced in ("0", "false", "no", "off"):
        return False
    return bool(active().field_cover)


def pasture_max_tiles(default: int) -> int:
    value = active().pasture_max_tiles
    return default if value is None else int(value)
