"""Target grass cover for the 2,950 development floor, and its linter.

The cover is every persistent farm cell below ``0xA0`` on
``Y1_D2_PowerOn_FarmClear`` (2,968 cells). Each one becomes spring grass
``0x70``. Structure ids stay as they are on that pin. The linter reads a
pin's farm map and lists the work still required on each cover cell.

On a farm tilemap the current map is the live farm (daytime tool writes
land there and copy to the persistent array only on leave). ``0xFF`` in
that buffer is unloaded viewport, so that cell falls back to persistent.
Any other tilemap is not the farm; the persistent array is the farm.
Save states are loaded and read. Nothing here writes RAM or starts an
emulator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from harvest.core.harvest_state import (
    CURRENT_MAP_ARRAY_ADDR,
    FARM_DEVELOPMENT_DIVISOR,
    FARM_MAP_ARRAY_ADDR,
    HarvestStateDocument,
    is_farm_development_tile,
)
from harvest.core.tile_catalog import (
    DRIED_TILLED,
    FARM_WALKABLE,
    FENCE,
    FRESH_TILLED,
    LARGE_ROCK_DAMAGE_TILES,
    LARGE_ROCK_TILES,
    LARGE_ROCK_TL,
    MAP_HEIGHT,
    MAP_TILE_COUNT,
    MAP_WIDTH,
    PLANTED_GRASS_TILE,
    ROCK,
    STONE,
    STUMP_TILES,
    STUMP_TL,
    TRAVEL_SOLID_TILES,
    UNTILLED,
    WATERED_TILLED,
    WEED,
)
from harvest.maps.farm_pond import FARM_NO_GO_TILES, FARM_TILEMAP_IDS
from harvest.runtime.rom_tools import wram_offset

BASE_PIN = "Y1_D2_PowerOn_FarmClear"
MIN_COUNTED_TILES = 2950
TARGET_GRASS = PLANTED_GRASS_TILE

# North lip of the shipping bin. It is in FARM_NO_GO_TILES (ditch fringe)
# but F8 stood on it, so the access check may enter this one cell.
BIN_LIP = (9, 28)
HOUSE_STAND = (8, 26)

ACCESS_TILES = (
    ("house", HOUSE_STAND),
    ("bin", BIN_LIP),
    ("exit", (0, 24)),
    ("barn", (19, 21)),
    ("coop", (28, 21)),
    ("shed", (26, 31)),
)
SHED_DOOR = (26, 30)
SHED_FLANKS = ((25, 30), (27, 30))

# Pins the linter has to actually run on. FarmClear is outdoors; the
# summer and fall pins are indoors, so they exercise both map sources.
LINT_PINS = (
    BASE_PIN,
    "Y1_Summer_D1_Pin",
    "Y1_Fall_D1_Pin",
)

_DONE_GRASS = frozenset(range(0x70, 0x7D)) | frozenset(range(0x80, 0x86))
_WALK_IDS = FARM_WALKABLE - TRAVEL_SOLID_TILES
_ORTHOGONALS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class FinalLayout:
    """Target ids over the full 64×64 persistent farm."""

    base_pin: str
    base_tiles: tuple[int, ...]
    tiles: tuple[int, ...]
    cover: tuple[tuple[int, int], ...]

    @property
    def counted(self) -> int:
        return sum(1 for tile in self.tiles if is_farm_development_tile(tile))


@dataclass(frozen=True)
class TileTodo:
    x: int
    y: int
    live: int
    target: int
    steps: tuple[str, ...]


@dataclass(frozen=True)
class LintReport:
    pin: str
    tilemap: int
    source: str
    todos: tuple[TileTodo, ...]
    live_counted: int
    target_counted: int


@dataclass(frozen=True)
class AccessGoal:
    name: str
    tile: tuple[int, int]
    distance: int | None


@dataclass(frozen=True)
class AccessReport:
    goals: tuple[AccessGoal, ...]
    walkable: int
    reached: int
    missed: tuple[tuple[int, int], ...]
    unfaced_cover: tuple[tuple[int, int], ...]
    flank_distances: tuple[tuple[tuple[int, int], int | None], ...]
    door_reached: bool


def _index(x: int, y: int) -> int:
    return y * MAP_WIDTH + x


def _cell(index: int) -> tuple[int, int]:
    return (index % MAP_WIDTH, index // MAP_WIDTH)


@lru_cache(maxsize=8)
def _pin_maps(pin: str) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    document = HarvestStateDocument.load(pin)
    ram = document.mutable_state.ram
    persistent_at = wram_offset(FARM_MAP_ARRAY_ADDR)
    visible_at = wram_offset(CURRENT_MAP_ARRAY_ADDR)
    persistent = tuple(ram[persistent_at : persistent_at + MAP_TILE_COUNT])
    visible = tuple(ram[visible_at : visible_at + MAP_TILE_COUNT])
    return document.scalar_value("tilemap"), persistent, visible


def live_tile(tilemap: int, visible: int, persistent: int) -> int:
    """Pick the farm id a tool would see for one cell."""

    if tilemap not in FARM_TILEMAP_IDS or visible == 0xFF:
        return persistent & 0xFF
    return visible & 0xFF


def live_farm_tiles(pin: str) -> tuple[int, tuple[str, tuple[int, ...]]]:
    """Return ``(tilemap, (source, tiles))`` for ``pin``'s farm."""

    tilemap, persistent, visible = _pin_maps(pin)
    if tilemap not in FARM_TILEMAP_IDS:
        return tilemap, ("persistent", persistent)
    tiles = tuple(
        live_tile(tilemap, visible[index], persistent[index])
        for index in range(MAP_TILE_COUNT)
    )
    return tilemap, ("live", tiles)


@lru_cache(maxsize=1)
def final_layout(pin: str = BASE_PIN) -> FinalLayout:
    """Build the grass target from ``pin``'s persistent farm array."""

    _tilemap, persistent, _visible = _pin_maps(pin)
    if any(tile == 0x00 for tile in persistent):
        raise ValueError(f"{pin} has 0x00, which cannot take a tool")
    cover: list[tuple[int, int]] = []
    target: list[int] = []
    for index, tile in enumerate(persistent):
        if tile < 0xA0:
            cover.append(_cell(index))
            target.append(TARGET_GRASS)
        else:
            target.append(tile)
    layout = FinalLayout(
        base_pin=pin,
        base_tiles=persistent,
        tiles=tuple(target),
        cover=tuple(cover),
    )
    if layout.counted < MIN_COUNTED_TILES:
        raise ValueError(
            f"{pin} cover counts {layout.counted}, under {MIN_COUNTED_TILES}"
        )
    return layout


@dataclass(frozen=True)
class LostTile:
    """A cover cell that counted, then stopped. ``cause`` is F2's writer."""

    x: int
    y: int
    before: int
    after: int
    cause: str
    steps: tuple[str, ...]


def counted_tile_total(tiles: tuple[int, ...] | list[int]) -> int:
    """F2 ``CalculateFarmDevelopment`` count. The persistent array is the authority."""

    return sum(1 for tile in tiles if is_farm_development_tile(int(tile)))


def persistent_farm_tiles(pin: str) -> tuple[int, ...]:
    """Persistent farm array for ``pin``. Not the live viewport."""

    _tilemap, persistent, _visible = _pin_maps(pin)
    return persistent


def persistent_tiles_from_ram(ram: bytes | bytearray | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Read ``!farm_map_array`` from a save-state WRAM image or a live mirror."""

    from harvest.core.ram_catalog import live_wram_base

    start = live_wram_base(ram) + wram_offset(FARM_MAP_ARRAY_ADDR)
    end = start + MAP_TILE_COUNT
    if end > len(ram):
        raise ValueError(f"farm array {start:#x}:{end:#x} outside ram len {len(ram)}")
    return tuple(int(ram[index]) & 0xFF for index in range(start, end))


def loss_cause(before: int, after: int) -> str | None:
    """Why a counted tile stopped counting, or ``None`` when it still counts.

    Hurricane and New Year both store ``0x02`` (``ClimateFarmDamageCheck``).
    Winter D1 stores ``0x07`` for ``0x1D``–``0x6F`` and ``0x7A`` for grown
    grass. ``0x7A`` and broken fence ``0x06`` still count, so they are not losses.
    Spring→Summer and Summer→Fall do not rewrite the map.
    """

    before &= 0xFF
    after &= 0xFF
    if not is_farm_development_tile(before) or is_farm_development_tile(after):
        return None
    if after == DRIED_TILLED and (before == 0x1D or before >= 0x1E):
        return "hurricane"
    if after == FRESH_TILLED and 0x1D <= before <= 0x6F:
        return "season"
    return "other"


def lost_counted_tiles(
    before_tiles: tuple[int, ...] | list[int],
    after_tiles: tuple[int, ...] | list[int],
) -> tuple[LostTile, ...]:
    """Cells whose F2 count disappeared between two persistent maps."""

    if len(before_tiles) != len(after_tiles):
        raise ValueError("farm maps differ in length")
    losses: list[LostTile] = []
    for index, (before, after) in enumerate(zip(before_tiles, after_tiles)):
        cause = loss_cause(int(before), int(after))
        if cause is None:
            continue
        x, y = _cell(index)
        losses.append(
            LostTile(
                x=x,
                y=y,
                before=int(before) & 0xFF,
                after=int(after) & 0xFF,
                cause=cause,
                steps=steps_for(int(after)),
            )
        )
    return tuple(losses)


def recover_window(losses: tuple[LostTile, ...]) -> tuple[int, int, int, int]:
    """Bounding box of tiles that have to be planted again."""

    if not losses:
        raise ValueError("no lost tiles")
    xs = [tile.x for tile in losses]
    ys = [tile.y for tile in losses]
    return (min(xs), min(ys), max(xs), max(ys))


def steps_for(live: int) -> tuple[str, ...]:
    """Work that turns ``live`` into stable spring grass, or empty if done.

    ``0x1D``–``0x6F`` (fall grass seed, crops, and the eight ``0x55`` cells)
    have no spring-grass write until Winter D1 stores ``0x07``. ``0x55`` has
    no tool bit before that night. Fence ``0x05`` is lifted, not hoed.
    """

    live &= 0xFF
    if live in _DONE_GRASS:
        return ()
    if live in STUMP_TILES:
        return ("axe", "hoe", "grass_seed")
    hammered = LARGE_ROCK_TILES | LARGE_ROCK_DAMAGE_TILES | {STONE, ROCK}
    if live in hammered:
        return ("hammer", "hoe", "grass_seed")
    if live == WEED:
        return ("sickle", "hoe", "grass_seed")
    if live == FENCE:
        return ("lift", "hoe", "grass_seed")
    if live in (UNTILLED, DRIED_TILLED):
        return ("hoe", "grass_seed")
    if live in (FRESH_TILLED, WATERED_TILLED):
        return ("grass_seed",)
    if 0x1D <= live <= 0x6F:
        return ("winter_d1", "grass_seed")
    return ("unknown",)


def lint_tiles(
    live_tiles: tuple[int, ...] | list[int],
    layout: FinalLayout | None = None,
) -> tuple[TileTodo, ...]:
    """Diff ``live_tiles`` against the target. One todo per unfinished cover cell."""

    layout = final_layout() if layout is None else layout
    todos: list[TileTodo] = []
    for x, y in layout.cover:
        live = int(live_tiles[_index(x, y)]) & 0xFF
        steps = steps_for(live)
        if not steps:
            continue
        todos.append(
            TileTodo(
                x=x,
                y=y,
                live=live,
                target=layout.tiles[_index(x, y)],
                steps=steps,
            )
        )
    return tuple(todos)


def lint_pin(pin: str, layout: FinalLayout | None = None) -> LintReport:
    """Diff one pin. ``layout`` defaults to the FarmClear grass target."""

    layout = final_layout() if layout is None else layout
    tilemap, (source, tiles) = live_farm_tiles(pin)
    todos = lint_tiles(tiles, layout)
    return LintReport(
        pin=pin,
        tilemap=tilemap,
        source=source,
        todos=todos,
        live_counted=sum(1 for tile in tiles if is_farm_development_tile(tile)),
        target_counted=layout.counted,
    )


def _walkable(tiles: tuple[int, ...] | list[int], blocked: frozenset[tuple[int, int]], x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= MAP_WIDTH or y >= MAP_HEIGHT:
        return False
    if (x, y) in blocked:
        return False
    return tiles[_index(x, y)] in _WALK_IDS


def bfs_access(layout: FinalLayout | None = None) -> AccessReport:
    """4-directional BFS on the target grass map from the house stand.

    Walk ids are ``FARM_WALKABLE`` minus ``TRAVEL_SOLID_TILES``. No-go cells
    stay blocked except the bin lip.
    """

    layout = final_layout() if layout is None else layout
    blocked = FARM_NO_GO_TILES - {BIN_LIP}
    tiles = layout.tiles
    start = HOUSE_STAND
    distances: dict[tuple[int, int], int] = {start: 0}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in _ORTHOGONALS:
            nxt = (x + dx, y + dy)
            if nxt in distances or not _walkable(tiles, blocked, *nxt):
                continue
            distances[nxt] = distances[(x, y)] + 1
            queue.append(nxt)

    walkable_cells = [
        (x, y)
        for y in range(MAP_HEIGHT)
        for x in range(MAP_WIDTH)
        if _walkable(tiles, blocked, x, y)
    ]
    missed = tuple(cell for cell in walkable_cells if cell not in distances)
    unfaced: list[tuple[int, int]] = []
    for cell in layout.cover:
        if cell in distances:
            continue
        if not any((cell[0] + dx, cell[1] + dy) in distances for dx, dy in _ORTHOGONALS):
            unfaced.append(cell)
    return AccessReport(
        goals=tuple(
            AccessGoal(name, tile, distances.get(tile))
            for name, tile in ACCESS_TILES
        ),
        walkable=len(walkable_cells),
        reached=len(distances),
        missed=missed,
        unfaced_cover=tuple(unfaced),
        flank_distances=tuple((tile, distances.get(tile)) for tile in SHED_FLANKS),
        door_reached=SHED_DOOR in distances,
    )


def stump_anchor(x: int, y: int, tile_id: int) -> tuple[int, int]:
    """Top-left of a stump or large-rock quad. ``tile_id`` is the cell's own id."""

    if tile_id in STUMP_TILES:
        origin = STUMP_TL
    elif tile_id in LARGE_ROCK_TILES:
        origin = LARGE_ROCK_TL
    else:
        raise ValueError(f"not a quad tile: {tile_id:#x}")
    offset = tile_id - origin
    return (x - (offset & 1), y - (offset >> 1))


# --- Terrain ledger: how much of the field is worked, and what 2,950 still needs ---------------
#
# ROM policy (F2): ``CalculateFarmDevelopment`` counts a tile below ``0xA0`` that is fence
# ``0x05``/``0x06`` or ``>= 0x1D``; the term is ``min(100, counted * 10 // 295) >> 1``. Hoed soil
# (``0x07``/``0x08``) does not count, it only takes the seed. So 50 points need 2,950 counted tiles
# and 49 stop at 2,949: on the 2,968-cell pool that is 99.4% of the field, at most 18 cells may
# stay bare (24 of 2,974 on the Fall mask). A bag is one ``PreCheckToolSuccessSquare`` pass over a
# 3x3, so nine tiles when every square is hoed; a square that is not hoed is skipped.

GRASS_BAG_TILES = 9
GRASS_BAG_PRICE_G = 500  # ``livestock_econ.SHOP_PRICE_G["grass_seed"]``; one bag of 329 is free
HOE_STAMINA = 2          # ``core.stamina.SWING_STAMINA_COST``; a bag is 1
BAG_STAMINA = 1
_TILLED_IDS = frozenset({FRESH_TILLED, WATERED_TILLED})
_HEAVY_DEBRIS = STUMP_TILES | LARGE_ROCK_TILES | LARGE_ROCK_DAMAGE_TILES
TERRAIN_CLASSES = (
    "grass", "fall_seed", "crop", "fence", "tilled", "untilled", "dried",
    "weed", "stone", "heavy_debris", "blocked", "other",
)


def terrain_class(tile: int) -> str:
    """Work state of one farm cell. The ROM's own count wins over the bot's tile names."""

    tile &= 0xFF
    if tile >= 0xA0:
        return "structure"
    if tile in (0x05, 0x06):
        return "fence"
    if tile == 0x1D:
        return "fall_seed"
    if tile in _DONE_GRASS:
        return "grass"
    if tile >= 0x1E:
        return "crop" if tile <= 0x6F else "other"
    if tile in _TILLED_IDS:
        return "tilled"
    if tile == UNTILLED:
        return "untilled"
    if tile == DRIED_TILLED:
        return "dried"
    if tile == WEED:
        return "weed"
    if tile == STONE:
        return "stone"
    if tile in _HEAVY_DEBRIS:
        return "heavy_debris"
    return "other"


def _windows(centre_x: int, centre_y: int) -> list[int]:
    """Indexes of the 3x3 around a faced centre, clipped to the map."""

    return [
        _index(x, y)
        for y in range(centre_y - 1, centre_y + 2)
        for x in range(centre_x - 1, centre_x + 2)
        if 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT
    ]


def disjoint_full_windows(mask: list[bool] | tuple[bool, ...]) -> int:
    """Raster-greedy count of non-overlapping 3x3 blocks whose nine cells are all set."""

    used = [False] * MAP_TILE_COUNT
    count = 0
    for y in range(MAP_HEIGHT - 2):
        for x in range(MAP_WIDTH - 2):
            cells = [_index(x + dx, y + dy) for dy in range(3) for dx in range(3)]
            if all(mask[i] and not used[i] for i in cells):
                count += 1
                for i in cells:
                    used[i] = True
    return count


def plan_grass_bags(
    uncovered: set[int], need: int, centres: set[int] | None = None
) -> list[tuple[int, int, int]]:
    """Greedy bag stands that newly cover ``need`` cells: ``(centre_x, centre_y, new_cells)``.

    Full nines first, then eights and so on, scanning in raster order. Counts only fall as
    cells are claimed, so one pass per threshold is enough. A geometry bound over the tile
    array: it does not prove a farmer can stand next to every centre.
    """

    left = set(uncovered)
    plan: list[tuple[int, int, int]] = []
    covered = 0
    pool = sorted(centres if centres is not None else uncovered)
    for threshold in range(GRASS_BAG_TILES, 0, -1):
        for centre in pool:
            if covered >= need:
                return plan
            cx, cy = _cell(centre)
            cells = [i for i in _windows(cx, cy) if i in left]
            if len(cells) >= threshold:
                left.difference_update(cells)
                covered += len(cells)
                plan.append((cx, cy, len(cells)))
    return plan


@dataclass(frozen=True)
class TerrainLedger:
    """One read of the field: what is worked, what is counted, what a perfect score still needs."""

    pool: int
    classes: dict[str, int]
    counted: int
    worked: int
    need: int
    to_go: int
    slack: int
    dev_byte: int
    points: int
    hoe_swings_left: int
    clear_left: int
    bags_min: int
    bags_plan: int
    plan_unhoed: int
    plan_ready_bags: int
    full_grass_blocks: int
    ready_blocks: int
    planted_grass: int | None

    @property
    def worked_pct(self) -> float:
        return round(100.0 * self.worked / self.pool, 2) if self.pool else 0.0

    @property
    def counted_pct(self) -> float:
        return round(100.0 * self.counted / self.pool, 2) if self.pool else 0.0

    @property
    def required_pct(self) -> float:
        return round(100.0 * self.need / self.pool, 2) if self.pool else 0.0

    @property
    def bag_cost_g(self) -> int:
        return self.bags_plan * GRASS_BAG_PRICE_G

    @property
    def stamina_left(self) -> int:
        return HOE_STAMINA * self.hoe_swings_left + BAG_STAMINA * self.bags_plan

    def as_dict(self) -> dict[str, object]:
        return {
            "pool": self.pool,
            "classes": dict(self.classes),
            "counted": self.counted,
            "counted_pct": self.counted_pct,
            "worked": self.worked,
            "worked_pct": self.worked_pct,
            "need": self.need,
            "required_pct": self.required_pct,
            "to_go": self.to_go,
            "slack": self.slack,
            "dev_byte": self.dev_byte,
            "points": self.points,
            "hoe_swings_left": self.hoe_swings_left,
            "clear_left": self.clear_left,
            "bags_min": self.bags_min,
            "bags_plan": self.bags_plan,
            "bag_cost_g": self.bag_cost_g,
            "plan_unhoed": self.plan_unhoed,
            "plan_ready_bags": self.plan_ready_bags,
            "full_grass_blocks": self.full_grass_blocks,
            "ready_blocks": self.ready_blocks,
            "stamina_left": self.stamina_left,
            "planted_grass": self.planted_grass,
        }

    def line(self) -> str:
        """One log line; the unhoed count is the gate before any bag can land."""

        return (
            f"terrain worked {self.worked}/{self.pool} ({self.worked_pct:.1f}%) "
            f"counted {self.counted}/{self.need} ({self.points}/50) to_go {self.to_go} "
            f"hoe_left {self.hoe_swings_left} bags_left>={self.bags_min} plan {self.bags_plan} "
            f"({self.bag_cost_g:,}G) unhoed_in_plan {self.plan_unhoed} "
            f"full_blocks {self.full_grass_blocks}"
        )


def terrain_ledger(
    tiles: tuple[int, ...] | list[int],
    cover: tuple[tuple[int, int], ...] | None = None,
    planted_grass: int | None = None,
    need: int = MIN_COUNTED_TILES,
) -> TerrainLedger:
    """Ledger over a 64x64 farm array.

    ``cover`` is the pool the pin says can ever count (``final_layout().cover``); without it
    every live cell below ``0xA0`` is the pool. Cells that are below ``0xA0`` live are always in
    it, so the six Fall-mask cells that read ``0xA0`` on the clear pin join once they are hoeable.
    """

    if len(tiles) != MAP_TILE_COUNT:
        raise ValueError(f"farm map has {len(tiles)} cells, want {MAP_TILE_COUNT}")
    pool_cells = {_index(x, y) for x, y in cover} if cover is not None else set()
    pool_cells.update(i for i, tile in enumerate(tiles) if (int(tile) & 0xFF) < 0xA0)
    classes = {name: 0 for name in TERRAIN_CLASSES}
    counted = 0
    for index in pool_cells:
        tile = int(tiles[index]) & 0xFF
        name = terrain_class(tile)
        name = "blocked" if name == "structure" else name
        classes[name if name in classes else "other"] += 1
        if is_farm_development_tile(tile):
            counted += 1
    hoed_or_better = (
        classes["grass"] + classes["fall_seed"] + classes["crop"] + classes["fence"] + classes["tilled"]
    )
    to_go = max(0, need - counted)
    # Cells a bag still has to reach: everything in the pool that does not count yet. A fence
    # already counts, so it never needs a bag.
    uncovered = {
        i
        for i in pool_cells
        if (int(tiles[i]) & 0xFF) < 0xA0 and not is_farm_development_tile(int(tiles[i]) & 0xFF)
    }
    plan = plan_grass_bags(uncovered, to_go)
    plan_cells = {
        i
        for cx, cy, _n in plan
        for i in _windows(cx, cy)
        if i in uncovered
    }
    plan_unhoed = sum(1 for i in plan_cells if (int(tiles[i]) & 0xFF) not in _TILLED_IDS)
    ready_bags = sum(
        1
        for cx, cy, _n in plan
        if all(
            (int(tiles[i]) & 0xFF) in _TILLED_IDS
            for i in _windows(cx, cy)
            if i in uncovered
        )
    )
    grass_mask = [terrain_class(int(t)) in ("grass", "fall_seed") for t in tiles]
    tilled_mask = [terrain_class(int(t)) == "tilled" for t in tiles]
    dev_byte = min(100, counted * 10 // FARM_DEVELOPMENT_DIVISOR)
    return TerrainLedger(
        pool=len(pool_cells),
        classes=classes,
        counted=counted,
        worked=hoed_or_better,
        need=need,
        to_go=to_go,
        slack=len(pool_cells) - need,
        dev_byte=dev_byte,
        points=dev_byte >> 1,
        hoe_swings_left=classes["untilled"] + classes["dried"] + classes["weed"] + classes["stone"]
        + classes["heavy_debris"],
        clear_left=classes["weed"] + classes["stone"] + classes["heavy_debris"],
        bags_min=-(-to_go // GRASS_BAG_TILES),
        bags_plan=len(plan),
        plan_unhoed=plan_unhoed,
        plan_ready_bags=ready_bags,
        full_grass_blocks=disjoint_full_windows(grass_mask),
        ready_blocks=disjoint_full_windows(tilled_mask),
        planted_grass=None if planted_grass is None else int(planted_grass),
    )
