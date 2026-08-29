"""Crop layout and planting planner (not ``crop_planter.CropWaterTask``)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np

from harvest.maps.map_config import FARM_NO_GO_TILES, FARM_TILEMAP_IDS
from harvest.core.tile_catalog import (
    ADDR_MAP,
    FARM_WALKABLE,
    FRESH_TILLED,
    MAP_HEIGHT,
    MAP_WIDTH,
    PLOT_TILES,
    STALE_TILE_IDS,
    TILLABLE_TILES,
    WATERED_TILLED,
)

Tile = tuple[int, int]
Season = Literal["spring", "summer", "fall", "winter"]
WateringMode = Literal["manual", "sprinkler"]

SEASON_SPRING = 0
SEASON_SUMMER = 1
SEASON_FALL = 2
SEASON_WINTER = 3
SEASON_LENGTH = 30
SEASON_NAMES: dict[int, Season] = {
    SEASON_SPRING: "spring",
    SEASON_SUMMER: "summer",
    SEASON_FALL: "fall",
    SEASON_WINTER: "winter",
}
SEASON_IDS: dict[str, int] = {name: season_id for season_id, name in SEASON_NAMES.items()}

DEFAULT_CROP_BOUNDS: tuple[int, int, int, int] = (2, 3, 62, 60)
DEFAULT_START_TILE: Tile = (15, 29)
DEFAULT_SHIPPING_TILE: Tile = (11, 30)
DEFAULT_WATER_SOURCE_TILE: Tile = (9, 28)


@dataclass(frozen=True)
class CropSpec:
    """Static crop economics and growth data."""

    name: str
    seasons: tuple[int, ...]
    seed_cost_g: int
    sell_price_g: int
    days_to_first_harvest: int
    regrow_days: Optional[int] = None

    @property
    def is_regrowable(self) -> bool:
        return self.regrow_days is not None

    def harvests_from_planting_day(self, day: int, season_length: int = SEASON_LENGTH) -> int:
        first_harvest_day = day + self.days_to_first_harvest
        if first_harvest_day > season_length:
            return 0
        if self.regrow_days is None:
            return 1
        return 1 + (season_length - first_harvest_day) // self.regrow_days

    def expected_profit_g(
        self,
        planted_tiles: int,
        *,
        day: int,
        season_length: int = SEASON_LENGTH,
    ) -> int:
        harvests = self.harvests_from_planting_day(day, season_length)
        if harvests <= 0:
            return -self.seed_cost_g
        return harvests * planted_tiles * self.sell_price_g - self.seed_cost_g


CROP_SPECS: dict[str, CropSpec] = {
    "turnip": CropSpec("turnip", (SEASON_SPRING,), seed_cost_g=200, sell_price_g=60, days_to_first_harvest=4),
    "potato": CropSpec("potato", (SEASON_SPRING,), seed_cost_g=200, sell_price_g=80, days_to_first_harvest=6),
    "tomato": CropSpec("tomato", (SEASON_SUMMER,), seed_cost_g=300, sell_price_g=100, days_to_first_harvest=10, regrow_days=3),
    "corn": CropSpec("corn", (SEASON_SUMMER,), seed_cost_g=300, sell_price_g=120, days_to_first_harvest=10, regrow_days=3),
}

RANCH_MASTER_SHIPPED_TARGET = 511
SEED_PURCHASE_RECORDINGS: dict[int, str] = {
    SEASON_SPRING: "buy_potato_seeds",
    SEASON_SUMMER: "buy_summer",
}


@dataclass(frozen=True)
class CropLayoutPattern:
    """A seed-bag plot layout relative to its stand/center tile."""

    name: str
    crop_offsets: tuple[Tile, ...]
    access_offsets: tuple[Tile, ...] = ((0, 0),)
    sprinkler_ready: bool = False
    note: str = ""

    @property
    def crop_count(self) -> int:
        return len(self.crop_offsets)

    def crop_tiles(self, center: Tile) -> tuple[Tile, ...]:
        cx, cy = center
        return tuple((cx + dx, cy + dy) for dx, dy in self.crop_offsets)

    def access_tiles(self, center: Tile) -> tuple[Tile, ...]:
        cx, cy = center
        return tuple((cx + dx, cy + dy) for dx, dy in self.access_offsets)

    @property
    def has_center_access_opening(self) -> bool:
        cardinal = {(0, -1), (1, 0), (0, 1), (-1, 0)}
        return bool(cardinal.difference(self.crop_offsets))


def _ring_offsets_without(*missing: Tile) -> tuple[Tile, ...]:
    missing_set = set(missing)
    return tuple(
        (dx, dy)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dx, dy) != (0, 0) and (dx, dy) not in missing_set
    )


def _notch_layout(name: str, missing: Tile) -> CropLayoutPattern:
    return CropLayoutPattern(
        name=name,
        crop_offsets=_ring_offsets_without(missing),
        access_offsets=((0, 0), missing),
        sprinkler_ready=True,
        note=f"Leaves a {name.split('_')[1]} notch into the center for sprinkler access.",
    )


CROP_LAYOUTS: dict[str, CropLayoutPattern] = {
    "eight_tile_ring": CropLayoutPattern(
        name="eight_tile_ring",
        crop_offsets=_ring_offsets_without(),
        access_offsets=((0, 0),),
        note="Maximum seed-bag yield; center can become sealed once crops block movement.",
    ),
    "seven_south_access": _notch_layout("seven_south_access", (0, 1)),
    "seven_north_access": _notch_layout("seven_north_access", (0, -1)),
    "seven_west_access": _notch_layout("seven_west_access", (-1, 0)),
    "seven_east_access": _notch_layout("seven_east_access", (1, 0)),
}


@dataclass(frozen=True)
class WateringAccess:
    target_tile: Tile
    stand_tiles: tuple[Tile, ...]


@dataclass(frozen=True)
class PlotCandidate:
    center: Tile
    layout: CropLayoutPattern
    crop: CropSpec
    watering_mode: WateringMode
    crop_tiles: tuple[Tile, ...]
    access_tiles: tuple[Tile, ...]
    water_stands: tuple[Tile, ...]
    expected_profit_g: int
    route_cost: int
    score: int

    @property
    def reserved_tiles(self) -> frozenset[Tile]:
        return frozenset(self.crop_tiles) | frozenset(self.access_tiles) | frozenset(self.water_stands)


@dataclass(frozen=True)
class PlannedPlot:
    order: int
    center: Tile
    layout_name: str
    crop_name: str
    crop_tiles: tuple[Tile, ...]
    access_tiles: tuple[Tile, ...]
    water_stands: tuple[Tile, ...]
    expected_profit_g: int
    route_cost: int
    watering_mode: WateringMode


@dataclass(frozen=True)
class CropFieldPlan:
    crop_name: str
    season: int
    day: int
    layout_name: str
    plots: tuple[PlannedPlot, ...]
    total_expected_profit_g: int
    total_route_cost: int
    seed_bags_needed: int

    @property
    def total_planted_tiles(self) -> int:
        return sum(len(plot.crop_tiles) for plot in self.plots)


@dataclass(frozen=True)
class CropPlanningConfig:
    season: int | str = SEASON_SPRING
    day: int = 1
    seed_type: Optional[str] = None
    layout_name: Optional[str] = None
    allowed_layouts: Optional[tuple[str, ...]] = None
    max_seed_bags: int = 1
    budget_g: Optional[int] = None
    bounds: tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS
    start_tile: Tile = DEFAULT_START_TILE
    shipping_tile: Tile = DEFAULT_SHIPPING_TILE
    water_source_tile: Tile = DEFAULT_WATER_SOURCE_TILE
    protected_tiles: tuple[Tile, ...] = ()
    sprinkler_available: bool = False
    watering_mode: Optional[WateringMode] = None
    reserve_water_stands: bool = True
    require_harvest_before_season_end: bool = True
    route_weight: int = 6
    shipping_weight: int = 3


@dataclass(frozen=True)
class PlantingStep:
    action: Literal["hoe", "plant_seed"]
    stand_tile: Tile
    target_tile: Tile
    face: str
    tool: str
    plot_order: int


@dataclass(frozen=True)
class PlantingRecordingTemplate:
    name: str
    start_state: Optional[str]
    frame_count: int
    hoe_action_tiles: tuple[Tile, ...]
    seed_action_tiles: tuple[Tile, ...]
    watering_action_tiles: tuple[Tile, ...]
    visited_farm_tiles: tuple[Tile, ...]


def normalize_season(season: int | str) -> int:
    if isinstance(season, str):
        key = season.lower()
        if key not in SEASON_IDS:
            raise ValueError(f"unknown season: {season}")
        return SEASON_IDS[key]
    return int(season)


def tile_dist(a: Tile, b: Tile) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def read_tile(ram: np.ndarray, tile: Tile) -> int:
    tx, ty = tile
    if not (0 <= tx < MAP_WIDTH and 0 <= ty < MAP_HEIGHT):
        return 0xFF
    idx = ADDR_MAP + ty * MAP_WIDTH + tx
    if idx >= len(ram):
        return 0xFF
    return int(ram[idx])


def is_plannable_soil(tile_id: int) -> bool:
    return tile_id in TILLABLE_TILES or tile_id in {FRESH_TILLED, WATERED_TILLED}


def is_planning_walkable(tile_id: int) -> bool:
    return tile_id in FARM_WALKABLE and tile_id not in STALE_TILE_IDS


def _in_bounds(tile: Tile, bounds: tuple[int, int, int, int]) -> bool:
    x_min, y_min, x_max, y_max = bounds
    return x_min <= tile[0] <= x_max and y_min <= tile[1] <= y_max


def _blocked_for_access(ram: np.ndarray, tile: Tile, no_go_tiles: set[Tile]) -> bool:
    if tile in no_go_tiles:
        return True
    return not is_planning_walkable(read_tile(ram, tile))


def _blocked_for_crop(ram: np.ndarray, tile: Tile, no_go_tiles: set[Tile]) -> bool:
    if tile in no_go_tiles:
        return True
    tile_id = read_tile(ram, tile)
    if tile_id in PLOT_TILES and tile_id not in {FRESH_TILLED, WATERED_TILLED}:
        return True
    return not is_plannable_soil(tile_id)


def _face_from_stand_to_target(stand: Tile, target: Tile) -> str:
    dx = target[0] - stand[0]
    dy = target[1] - stand[1]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _adjacent_tiles(tile: Tile) -> Iterable[Tile]:
    tx, ty = tile
    yield (tx, ty - 1)
    yield (tx + 1, ty)
    yield (tx, ty + 1)
    yield (tx - 1, ty)


def _trace_row_is_farm(row: dict) -> bool:
    tilemap = row.get("tm")
    if tilemap is not None:
        try:
            return int(tilemap) in FARM_TILEMAP_IDS
        except (TypeError, ValueError):
            pass
    map_name = str(row.get("map", ""))
    return map_name == "farm" or map_name.startswith("farm_")


def _water_stands_for_target(
    ram: np.ndarray,
    center: Tile,
    target: Tile,
    *,
    crop_tiles: set[Tile],
    access_tiles: set[Tile],
    no_go_tiles: set[Tile],
    center_accessible: bool,
) -> tuple[Tile, ...]:
    stands: list[Tile] = []
    for stand in _adjacent_tiles(target):
        if stand in crop_tiles:
            continue
        if stand in access_tiles:
            if stand != center or center_accessible:
                stands.append(stand)
            continue
        if _blocked_for_access(ram, stand, no_go_tiles):
            continue
        stands.append(stand)
    stands.sort(key=lambda stand: (tile_dist(stand, DEFAULT_SHIPPING_TILE), stand[1], stand[0]))
    return tuple(stands)


def watering_access_for_layout(
    ram: np.ndarray,
    center: Tile,
    layout: CropLayoutPattern,
    *,
    mode: WateringMode = "manual",
    no_go_tiles: Optional[set[Tile]] = None,
) -> tuple[WateringAccess, ...]:
    """Valid watering stands per crop tile (sprinkler: reachable center notch)."""
    no_go = set(FARM_NO_GO_TILES if no_go_tiles is None else no_go_tiles)
    crop_tiles = set(layout.crop_tiles(center))
    access_tiles = set(layout.access_tiles(center))

    if mode == "sprinkler":
        if not layout.sprinkler_ready:
            return ()
        if any(_blocked_for_access(ram, tile, no_go) for tile in access_tiles):
            return ()
        center_stand = center
        return tuple(WateringAccess(target_tile=target, stand_tiles=(center_stand,)) for target in crop_tiles)

    accesses: list[WateringAccess] = []
    center_accessible = layout.has_center_access_opening
    for target in crop_tiles:
        stands = _water_stands_for_target(
            ram,
            center,
            target,
            crop_tiles=crop_tiles,
            access_tiles=access_tiles,
            no_go_tiles=no_go,
            center_accessible=center_accessible,
        )
        if not stands:
            return ()
        accesses.append(WateringAccess(target_tile=target, stand_tiles=stands))
    return tuple(accesses)


def choose_crop_for_date(season: int | str, day: int, *, layout_tiles: int = 8) -> Optional[CropSpec]:
    season_id = normalize_season(season)
    choices = [crop for crop in CROP_SPECS.values() if season_id in crop.seasons]
    if not choices:
        return None
    choices.sort(
        key=lambda crop: (
            crop.expected_profit_g(layout_tiles, day=day),
            crop.harvests_from_planting_day(day),
            crop.sell_price_g,
        ),
        reverse=True,
    )
    return choices[0]


def is_crop_planting_season(season: int | str) -> bool:
    return normalize_season(season) in (SEASON_SPRING, SEASON_SUMMER)


def crops_for_season(season: int | str) -> tuple[CropSpec, ...]:
    season_id = normalize_season(season)
    return tuple(crop for crop in CROP_SPECS.values() if season_id in crop.seasons)


def seed_purchase_recording_for_season(season: int | str) -> Optional[str]:
    return SEED_PURCHASE_RECORDINGS.get(normalize_season(season))


def can_plant_crop_today(season: int | str, day: int, crop_name: str) -> bool:
    crop = CROP_SPECS.get(crop_name)
    if crop is None:
        return False
    season_id = normalize_season(season)
    if season_id not in crop.seasons:
        return False
    return crop.harvests_from_planting_day(int(day)) > 0


def resolve_seed_type_for_date(
    season: int | str,
    day: int,
    *,
    inventory: Optional[dict[str, int]] = None,
    shipped: Optional[dict[str, int]] = None,
    layout_tiles: int = 8,
) -> Optional[str]:
    """Pick today's seed: inventory, then ranch-master remaining, then profit."""
    season_id = normalize_season(season)
    day_i = int(day)
    if not is_crop_planting_season(season_id):
        return None

    inv = {name: max(0, int(count)) for name, count in (inventory or {}).items()}
    ship = {name: max(0, int(count)) for name, count in (shipped or {}).items()}
    in_season = [
        crop
        for crop in crops_for_season(season_id)
        if crop.harvests_from_planting_day(day_i) > 0
    ]
    if not in_season:
        return None

    stocked = [crop for crop in in_season if inv.get(crop.name, 0) > 0]
    pool = stocked or in_season

    def _rank(crop: CropSpec) -> tuple[int, int, int, int]:
        shipped_count = ship.get(crop.name, 0)
        remaining = max(0, RANCH_MASTER_SHIPPED_TARGET - shipped_count)
        return (
            remaining,
            crop.expected_profit_g(layout_tiles, day=day_i),
            crop.harvests_from_planting_day(day_i),
            crop.sell_price_g,
        )

    pool.sort(key=_rank, reverse=True)
    return pool[0].name


def should_buy_seeds_for_date(season: int | str, day: int) -> bool:
    return resolve_seed_type_for_date(season, day) is not None


def _layout_names_for_config(config: CropPlanningConfig) -> tuple[str, ...]:
    if config.layout_name:
        return (config.layout_name,)
    if config.allowed_layouts:
        return config.allowed_layouts
    if config.sprinkler_available:
        return ("seven_south_access",)
    return ("eight_tile_ring",)


def evaluate_plot_candidate(
    ram: np.ndarray,
    center: Tile,
    crop: CropSpec,
    layout: CropLayoutPattern,
    config: CropPlanningConfig,
    *,
    no_go_tiles: Optional[set[Tile]] = None,
) -> Optional[PlotCandidate]:
    season_id = normalize_season(config.season)
    if season_id not in crop.seasons:
        return None

    mode: WateringMode = config.watering_mode or ("sprinkler" if config.sprinkler_available else "manual")
    no_go = set(FARM_NO_GO_TILES if no_go_tiles is None else no_go_tiles)
    no_go.update({config.shipping_tile, config.water_source_tile, *config.protected_tiles})
    crop_tiles = layout.crop_tiles(center)
    access_tiles = layout.access_tiles(center)
    all_needed = set(crop_tiles) | set(access_tiles)
    if not all(_in_bounds(tile, config.bounds) for tile in all_needed):
        return None
    if any(_blocked_for_crop(ram, tile, no_go) for tile in crop_tiles):
        return None
    if any(_blocked_for_access(ram, tile, no_go) for tile in access_tiles):
        return None

    access = watering_access_for_layout(ram, center, layout, mode=mode, no_go_tiles=no_go)
    if len(access) != len(crop_tiles):
        return None

    expected_profit = crop.expected_profit_g(layout.crop_count, day=config.day)
    if config.require_harvest_before_season_end and crop.harvests_from_planting_day(config.day) <= 0:
        return None

    water_stands = tuple(sorted({item.stand_tiles[0] for item in access}))
    route_cost = (
        tile_dist(config.start_tile, center)
        + tile_dist(center, config.shipping_tile) * config.shipping_weight
        + tile_dist(center, config.water_source_tile)
    )
    score = expected_profit - route_cost * config.route_weight
    return PlotCandidate(
        center=center,
        layout=layout,
        crop=crop,
        watering_mode=mode,
        crop_tiles=crop_tiles,
        access_tiles=access_tiles,
        water_stands=water_stands,
        expected_profit_g=expected_profit,
        route_cost=route_cost,
        score=score,
    )


def build_plot_candidates(
    ram: np.ndarray,
    config: CropPlanningConfig,
) -> tuple[PlotCandidate, ...]:
    layout_names = _layout_names_for_config(config)
    layouts = tuple(CROP_LAYOUTS[name] for name in layout_names)
    seed_crop = CROP_SPECS.get(config.seed_type) if config.seed_type else None
    if seed_crop is None:
        seed_crop = choose_crop_for_date(config.season, config.day, layout_tiles=layouts[0].crop_count)
    if seed_crop is None:
        return ()

    x_min, y_min, x_max, y_max = config.bounds
    candidates: list[PlotCandidate] = []
    for cy in range(y_min + 1, y_max):
        for cx in range(x_min + 1, x_max):
            center = (cx, cy)
            for layout in layouts:
                candidate = evaluate_plot_candidate(ram, center, seed_crop, layout, config)
                if candidate is not None:
                    candidates.append(candidate)

    candidates.sort(key=lambda item: (item.score, -item.route_cost, -item.center[1], -item.center[0]), reverse=True)
    return tuple(candidates)


def _seed_bag_limit(crop: CropSpec, config: CropPlanningConfig) -> int:
    limit = max(0, config.max_seed_bags)
    if config.budget_g is not None:
        limit = min(limit, max(0, config.budget_g // crop.seed_cost_g))
    return limit


def plan_crop_field(ram: np.ndarray, config: CropPlanningConfig = CropPlanningConfig()) -> CropFieldPlan:
    candidates = list(build_plot_candidates(ram, config))
    if not candidates:
        season_id = normalize_season(config.season)
        crop = CROP_SPECS.get(config.seed_type or "") or choose_crop_for_date(season_id, config.day) or CROP_SPECS["potato"]
        layout_name = _layout_names_for_config(config)[0]
        return CropFieldPlan(crop.name, season_id, config.day, layout_name, (), 0, 0, 0)

    crop = candidates[0].crop
    layout_name = candidates[0].layout.name
    limit = _seed_bag_limit(crop, config)
    selected: list[PlannedPlot] = []
    reserved: set[Tile] = set()
    current = config.start_tile

    while candidates and len(selected) < limit:
        compatible: list[PlotCandidate] = []
        for candidate in candidates:
            reservation = set(candidate.crop_tiles) | set(candidate.access_tiles)
            if config.reserve_water_stands:
                reservation |= set(candidate.water_stands)
            if reservation & reserved:
                continue
            compatible.append(candidate)
        if not compatible:
            break

        compatible.sort(
            key=lambda item: (
                item.score - tile_dist(current, item.center) * config.route_weight,
                -tile_dist(current, item.center),
                -item.center[1],
                -item.center[0],
            ),
            reverse=True,
        )
        chosen = compatible[0]
        order = len(selected) + 1
        selected.append(
            PlannedPlot(
                order=order,
                center=chosen.center,
                layout_name=chosen.layout.name,
                crop_name=chosen.crop.name,
                crop_tiles=chosen.crop_tiles,
                access_tiles=chosen.access_tiles,
                water_stands=chosen.water_stands,
                expected_profit_g=chosen.expected_profit_g,
                route_cost=chosen.route_cost,
                watering_mode=chosen.watering_mode,
            )
        )
        reservation = set(chosen.crop_tiles) | set(chosen.access_tiles)
        if config.reserve_water_stands:
            reservation |= set(chosen.water_stands)
        reserved |= reservation
        current = chosen.center
        candidates = [candidate for candidate in candidates if candidate is not chosen]

    return CropFieldPlan(
        crop_name=crop.name,
        season=normalize_season(config.season),
        day=config.day,
        layout_name=layout_name,
        plots=tuple(selected),
        total_expected_profit_g=sum(plot.expected_profit_g for plot in selected),
        total_route_cost=sum(plot.route_cost for plot in selected),
        seed_bags_needed=len(selected),
    )


def build_planting_steps(plan: CropFieldPlan) -> tuple[PlantingStep, ...]:
    steps: list[PlantingStep] = []
    for plot in plan.plots:
        crop_tiles = set(plot.crop_tiles)
        for target in sorted(plot.crop_tiles, key=lambda tile: (tile[1], tile[0])):
            stand_options = [tile for tile in _adjacent_tiles(target) if tile not in crop_tiles]
            if not stand_options:
                stand = plot.center
            else:
                stand_options.sort(key=lambda tile: (tile_dist(plot.center, tile), tile[1], tile[0]))
                stand = stand_options[0]
            steps.append(
                PlantingStep(
                    action="hoe",
                    stand_tile=stand,
                    target_tile=target,
                    face=_face_from_stand_to_target(stand, target),
                    tool="hoe",
                    plot_order=plot.order,
                )
            )
        steps.append(
            PlantingStep(
                action="plant_seed",
                stand_tile=plot.center,
                target_tile=plot.center,
                face="down",
                tool=f"{plot.crop_name}_seeds",
                plot_order=plot.order,
            )
        )
    return tuple(steps)


def extract_planting_template_from_recording(path: str | Path) -> PlantingRecordingTemplate:
    recording_path = Path(path)
    data = json.loads(recording_path.read_text())
    trace = data.get("trace", [])
    seed_tools = {0x05, 0x06, 0x07, 0x08}
    hoe_tiles: list[Tile] = []
    seed_tiles: list[Tile] = []
    water_tiles: list[Tile] = []
    visited_farm_tiles: set[Tile] = set()

    in_action = False
    last_action_key: Optional[tuple[str, int, Tile]] = None
    for row in trace:
        tile = (int(row.get("tx", 0)), int(row.get("ty", 0)))
        if _trace_row_is_farm(row):
            visited_farm_tiles.add(tile)
        else:
            in_action = False
            last_action_key = None
            continue

        buttons = set(row.get("buttons") or [])
        active_button = "Y" if "Y" in buttons else "A" if "A" in buttons else None
        if active_button is None:
            in_action = False
            last_action_key = None
            continue
        tool = int(row.get("tool", 0))
        key = (active_button, tool, tile)
        if in_action and key == last_action_key:
            continue

        in_action = True
        last_action_key = key
        if active_button == "Y" and tool == 0x02:
            hoe_tiles.append(tile)
        elif active_button == "Y" and tool in seed_tools:
            seed_tiles.append(tile)
        elif active_button == "Y" and tool == 0x10:
            water_tiles.append(tile)

    frame_count = int(data.get("metadata", {}).get("frame_count", len(data.get("frames", []))))
    return PlantingRecordingTemplate(
        name=str(data.get("name", recording_path.stem)),
        start_state=data.get("start_state"),
        frame_count=frame_count,
        hoe_action_tiles=tuple(hoe_tiles),
        seed_action_tiles=tuple(seed_tiles),
        watering_action_tiles=tuple(water_tiles),
        visited_farm_tiles=tuple(sorted(visited_farm_tiles, key=lambda tile: (tile[1], tile[0]))),
    )


__all__ = [
    "CROP_LAYOUTS",
    "CROP_SPECS",
    "CropFieldPlan",
    "CropLayoutPattern",
    "CropPlanningConfig",
    "CropSpec",
    "DEFAULT_CROP_BOUNDS",
    "DEFAULT_SHIPPING_TILE",
    "DEFAULT_START_TILE",
    "DEFAULT_WATER_SOURCE_TILE",
    "PlannedPlot",
    "PlantingRecordingTemplate",
    "PlantingStep",
    "PlotCandidate",
    "RANCH_MASTER_SHIPPED_TARGET",
    "SEASON_FALL",
    "SEASON_SPRING",
    "SEASON_SUMMER",
    "SEASON_WINTER",
    "SEED_PURCHASE_RECORDINGS",
    "WateringAccess",
    "build_planting_steps",
    "build_plot_candidates",
    "can_plant_crop_today",
    "choose_crop_for_date",
    "crops_for_season",
    "evaluate_plot_candidate",
    "extract_planting_template_from_recording",
    "is_crop_planting_season",
    "is_plannable_soil",
    "is_planning_walkable",
    "normalize_season",
    "plan_crop_field",
    "read_tile",
    "resolve_seed_type_for_date",
    "seed_purchase_recording_for_season",
    "should_buy_seeds_for_date",
    "tile_dist",
    "watering_access_for_layout",
]
