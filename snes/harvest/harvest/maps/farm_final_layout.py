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
