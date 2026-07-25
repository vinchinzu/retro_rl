#!/usr/bin/env python3
"""World snapshot facade over live/save-state RAM.

The snapshot is the bot-facing world model: player location, named RAM fields,
visible tile objects, crop state, animals, and map landmarks in one exportable
structure.  Task code can use this instead of scattering raw offsets and local
tile constants.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np


from harvest.maps.map_config import MapLandmark, get_landmarks, get_map_name
from harvest.core.npc_catalog import (
    current_dialogue_registers,
    game_objects,
    nearest_game_objects,
    relationship_status as npc_relationship_status,
    romance_dialogue_tree,
    status_flags as npc_status_flags,
)
from harvest.core.ram_catalog import (
    CHICKEN_SLOT_COUNT,
    COW_SLOT_COUNT,
    SCALAR_FIELDS,
    WEATHER_CODES,
    read_animal_slot_field,
    read_ram_value,
)
from harvest.runtime.rom_tools import parse_save_state, resolve_state_path
from harvest.core.tile_catalog import (
    MAP_HEIGHT,
    MAP_WIDTH,
    PLOT_TILES,
    TILE_SIZE,
    TileFact,
    crop_stage,
    get_tile_at,
    is_crop_tile,
    normalize_bounds,
    read_tile_grid,
    scan_interesting_tiles,
    tile_category,
    tile_category_counts,
    tile_histogram,
    tile_is_watered,
    tile_label,
    walkable_tiles_for_tilemap,
)


SEASON_NAMES = {0: "spring", 1: "summer", 2: "fall", 3: "winter"}
WEEKDAY_NAMES = {0: "sun", 1: "mon", 2: "tue", 3: "wed", 4: "thu", 5: "fri", 6: "sat"}
TOOL_NAMES = {
    0x00: "empty",
    0x01: "sickle",
    0x02: "hoe",
    0x03: "hammer",
    0x04: "axe",
    0x0F: "brush",
    0x10: "watering_can",
    0x11: "gold_sickle",
    0x12: "gold_hoe",
    0x13: "gold_hammer",
    0x14: "gold_axe",
}


@dataclass(frozen=True)
class PlayerSnapshot:
    pixel: Tuple[int, int]
    tile: Tuple[int, int]
    tile_id: int
    tile_label: str
    tile_category: str
    tilemap: int
    tilemap_name: str
    stamina: int
    held_item: int
    tool_selected: int
    input_lock: int

    def to_dict(self) -> dict:
        return {
            "pixel": list(self.pixel),
            "tile": list(self.tile),
            "tile_id": self.tile_id,
            "tile_hex": f"0x{self.tile_id:02X}",
            "tile_label": self.tile_label,
            "tile_category": self.tile_category,
            "tilemap": self.tilemap,
            "tilemap_hex": f"0x{self.tilemap:02X}",
            "tilemap_name": self.tilemap_name,
            "stamina": self.stamina,
            "held_item": self.held_item,
            "tool_selected": self.tool_selected,
            "tool_name": TOOL_NAMES.get(self.tool_selected, f"0x{self.tool_selected:02X}"),
            "input_lock": self.input_lock,
        }


@dataclass(frozen=True)
class CropPlotSnapshot:
    center: Tuple[int, int]
    bounds: Tuple[int, int, int, int]
    tiles: Tuple[TileFact, ...]

    @property
    def dry_count(self) -> int:
        return sum(1 for tile in self.tiles if tile.category == "crop_dry")

    @property
    def watered_count(self) -> int:
        return sum(1 for tile in self.tiles if tile.category == "crop_watered" or tile.category == "watered_soil")

    @property
    def max_stage(self) -> int | None:
        stages = [tile.crop_stage for tile in self.tiles if tile.crop_stage is not None]
        return max(stages) if stages else None

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "bounds": {
                "x_min": self.bounds[0],
                "y_min": self.bounds[1],
                "x_max": self.bounds[2],
                "y_max": self.bounds[3],
            },
            "dry_count": self.dry_count,
            "watered_count": self.watered_count,
            "max_stage": self.max_stage,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }


@dataclass(frozen=True)
class WorldSnapshot:
    frame: int
    bounds: Tuple[int, int, int, int]
    scalars: dict[str, int]
    player: PlayerSnapshot
    objects: Tuple[TileFact, ...]
    crop_plots: Tuple[CropPlotSnapshot, ...]
    landmarks: Tuple[MapLandmark, ...]
    chickens: Tuple[dict, ...]
    cows: Tuple[dict, ...]
    game_objects: Tuple[object, ...]
    relationship_status: dict
    status_flags: dict
    dialogue_registers: dict
    romance_tree: dict
    nearest_game_objects: Tuple[dict, ...]
    grid: Tuple[Tuple[int, ...], ...] = ()
    histogram: dict[str, int] = field(default_factory=dict)
    categories: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_ram(
        cls,
        ram: np.ndarray,
        *,
        frame: int = 0,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> "WorldSnapshot":
        b = normalize_bounds(bounds)
        scalars = scalar_values(ram)
        tilemap = int(scalars.get("tilemap", 0))
        px = int(scalars.get("player_x", 0))
        py = int(scalars.get("player_y", 0))
        tile = (px // TILE_SIZE, py // TILE_SIZE)
        tile_id = get_tile_at(ram, *tile)
        player = PlayerSnapshot(
            pixel=(px, py),
            tile=tile,
            tile_id=tile_id,
            tile_label=tile_label(tile_id),
            tile_category=tile_category(tile_id),
            tilemap=tilemap,
            tilemap_name=get_map_name(tilemap),
            stamina=int(scalars.get("stamina", 0)),
            held_item=int(scalars.get("held_item", 0)),
            tool_selected=int(scalars.get("tool_selected", 0)),
            input_lock=int(scalars.get("input_lock", 0)),
        )
        grid = tuple(tuple(row) for row in read_tile_grid(ram, b))
        objects = tuple(scan_interesting_tiles(ram, tilemap, b))
        gobjects = tuple(game_objects(ram))
        return cls(
            frame=frame,
            bounds=b,
            scalars=scalars,
            player=player,
            objects=objects,
            crop_plots=tuple(scan_crop_plots(ram, b)),
            landmarks=get_landmarks(tilemap),
            chickens=tuple(chicken_records(ram)),
            cows=tuple(cow_records(ram)),
            game_objects=gobjects,
            relationship_status=npc_relationship_status(ram),
            status_flags=npc_status_flags(ram),
            dialogue_registers=current_dialogue_registers(ram),
            romance_tree=romance_dialogue_tree(ram),
            nearest_game_objects=tuple(nearest_game_objects(ram)),
            grid=grid,
            histogram=tile_histogram(grid),
            categories=tile_category_counts(grid),
        )

    @property
    def date(self) -> dict:
        weather_code = int(self.scalars.get("weather_tomorrow", 0))
        season = int(self.scalars.get("season", 0))
        weekday = int(self.scalars.get("weekday", 0))
        return {
            "year": int(self.scalars.get("year", 0)) + 1,
            "year_raw": int(self.scalars.get("year", 0)),
            "season": season,
            "season_name": SEASON_NAMES.get(season, f"season_{season}"),
            "day": int(self.scalars.get("day", 0)),
            "weekday": weekday,
            "weekday_name": WEEKDAY_NAMES.get(weekday, f"weekday_{weekday}"),
            "hour": int(self.scalars.get("hour", 0)),
            "minute": int(self.scalars.get("minute", 0)),
            "weather_tomorrow": weather_code,
            "weather_tomorrow_label": WEATHER_CODES.get(weather_code, str(weather_code)),
        }

    def nearest_landmarks(self, limit: int = 6) -> list[dict]:
        px, py = self.player.tile
        items = []
        for landmark in self.landmarks:
            dist = abs(landmark.tile[0] - px) + abs(landmark.tile[1] - py)
            data = landmark_to_dict(landmark)
            data["distance_tiles"] = dist
            items.append(data)
        items.sort(key=lambda item: (item["distance_tiles"], item["name"]))
        return items[:limit]

    def to_dict(self, *, include_grid: bool = False, compact: bool = False) -> dict:
        data = {
            "frame": self.frame,
            "bounds": {
                "x_min": self.bounds[0],
                "y_min": self.bounds[1],
                "x_max": self.bounds[2],
                "y_max": self.bounds[3],
            },
            "date": self.date,
            "player": self.player.to_dict(),
            "resources": {
                "money": self.scalars.get("money", 0),
                "shipping_money": self.scalars.get("shipping_money", 0),
                "stored_wood": self.scalars.get("stored_wood", 0),
                "stored_grass": self.scalars.get("stored_grass", 0),
                "planted_grass": self.scalars.get("planted_grass", 0),
            },
            "animals": {
                "num_chickens": self.scalars.get("num_chickens", 0),
                "num_cows": self.scalars.get("num_cows", 0),
                "fed_chickens_n": self.scalars.get("fed_chickens_n", 0),
                "fed_cows_n": self.scalars.get("fed_cows_n", 0),
                "chickens": list(self.chickens),
                "cows": list(self.cows),
            },
            "relationships": {
                key: self.scalars.get(key, 0)
                for key in ("maria_hearts", "ann_hearts", "nina_hearts", "ellen_hearts", "eve_hearts")
            },
            "flags": {
                "incubator_flags": self.scalars.get("incubator_flags", 0),
                "egg_available": self.scalars.get("egg_available", 0),
                "fed_chickens_flags": self.scalars.get("fed_chickens_flags", 0),
                "fed_cows_flags": self.scalars.get("fed_cows_flags", 0),
                "status_flags": self.status_flags,
            },
            "relationship_status": self.relationship_status,
            "dialogue": {
                "registers": self.dialogue_registers,
                "romance_tree": self.romance_tree,
            },
            "entities": {
                "game_objects": [
                    obj.to_dict() if hasattr(obj, "to_dict") else dict(obj)
                    for obj in self.game_objects
                ],
                "candidate_npcs": [
                    obj.to_dict()
                    for obj in self.game_objects
                    if getattr(obj, "is_npc_candidate", False)
                ],
                "nearest_game_objects": list(self.nearest_game_objects),
            },
            "map": {
                "tilemap": self.player.tilemap,
                "tilemap_hex": f"0x{self.player.tilemap:02X}",
                "name": self.player.tilemap_name,
                "walkable_tiles": [f"0x{tile_id:02X}" for tile_id in sorted(walkable_tiles_for_tilemap(self.player.tilemap))],
                "histogram": self.histogram,
                "categories": self.categories,
                "landmarks": [landmark_to_dict(landmark) for landmark in self.landmarks],
                "nearest_landmarks": self.nearest_landmarks(),
            },
            "crop_plots": [plot.to_dict() for plot in self.crop_plots],
            "objects": [obj.to_dict() for obj in self.objects],
        }
        if not compact:
            data["scalars"] = dict(self.scalars)
        if include_grid:
            data["grid"] = [list(row) for row in self.grid]
        return data

    def to_json(self, *, include_grid: bool = False, compact: bool = False) -> str:
        return json.dumps(self.to_dict(include_grid=include_grid, compact=compact), indent=2)


def scalar_values(ram: np.ndarray) -> dict[str, int]:
    values: dict[str, int] = {}
    for spec in SCALAR_FIELDS:
        values[spec.key] = read_ram_value(ram, spec)
    return values


def landmark_to_dict(landmark: MapLandmark) -> dict:
    data = {
        "name": landmark.name,
        "tile": list(landmark.tile),
        "target_px": list(landmark.target_px),
        "kind": landmark.kind,
        "source": landmark.source,
    }
    if landmark.face:
        data["face"] = landmark.face
    if landmark.action:
        data["action"] = landmark.action
    if landmark.note:
        data["note"] = landmark.note
    return data


def scan_crop_plots(
    ram: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> list[CropPlotSnapshot]:
    x0, y0, x1, y1 = normalize_bounds(bounds)
    seen: set[Tuple[int, int]] = set()
    plots: list[CropPlotSnapshot] = []

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x, y) in seen or get_tile_at(ram, x, y) not in PLOT_TILES:
                continue
            component = _crop_component(ram, (x, y), (x0, y0, x1, y1), seen)
            if not component:
                continue
            xs = [tile.x for tile in component]
            ys = [tile.y for tile in component]
            plots.append(
                CropPlotSnapshot(
                    center=(round(sum(xs) / len(xs)), round(sum(ys) / len(ys))),
                    bounds=(min(xs), min(ys), max(xs), max(ys)),
                    tiles=tuple(component),
                )
            )
    return plots


def _crop_component(
    ram: np.ndarray,
    start: Tuple[int, int],
    bounds: Tuple[int, int, int, int],
    seen: set[Tuple[int, int]],
) -> list[TileFact]:
    x0, y0, x1, y1 = bounds
    queue: deque[Tuple[int, int]] = deque([start])
    seen.add(start)
    facts: list[TileFact] = []
    while queue:
        x, y = queue.popleft()
        tile_id = get_tile_at(ram, x, y)
        facts.append(
            TileFact(
                x=x,
                y=y,
                tile_id=tile_id,
                label=tile_label(tile_id),
                category=tile_category(tile_id),
                walkable=tile_id in walkable_tiles_for_tilemap(0x00),
                crop_stage=crop_stage(tile_id),
                watered=tile_is_watered(tile_id) if is_crop_tile(tile_id) else tile_id == 0x08,
            )
        )
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (x0 <= nx <= x1 and y0 <= ny <= y1):
                continue
            if (nx, ny) in seen or get_tile_at(ram, nx, ny) not in PLOT_TILES:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    return facts


def chicken_records(ram: np.ndarray) -> Iterable[dict]:
    for slot in range(CHICKEN_SLOT_COUNT):
        status = read_animal_slot_field(ram, "chicken", slot, "status_raw")
        if status == 0:
            continue
        age_code = (status >> 1) & 0x07
        yield {
            "slot": slot,
            "status_raw": status,
            "exists": bool(status & 0x01),
            "age_code": age_code,
            "age": "adult" if age_code >= 4 else "chick" if age_code >= 2 else "egg",
            "position": [
                read_animal_slot_field(ram, "chicken", slot, "position_x"),
                read_animal_slot_field(ram, "chicken", slot, "position_y"),
            ],
        }


def cow_records(ram: np.ndarray) -> Iterable[dict]:
    for slot in range(COW_SLOT_COUNT):
        status = read_animal_slot_field(ram, "cow", slot, "status_raw")
        if status == 0:
            continue
        yield {
            "slot": slot,
            "status_raw": status,
            "exists": bool(status & 0x01),
            "home_map_raw": read_animal_slot_field(ram, "cow", slot, "home_map_raw"),
            "pregnancy_raw": read_animal_slot_field(ram, "cow", slot, "pregnancy_raw"),
            "happiness": read_animal_slot_field(ram, "cow", slot, "happiness"),
            "position": [
                read_animal_slot_field(ram, "cow", slot, "position_x"),
                read_animal_slot_field(ram, "cow", slot, "position_y"),
            ],
        }


def world_snapshot_dict(
    ram: np.ndarray,
    *,
    frame: int = 0,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    include_grid: bool = False,
    compact: bool = False,
) -> dict:
    snapshot = WorldSnapshot.from_ram(ram, frame=frame, bounds=bounds)
    return snapshot.to_dict(include_grid=include_grid, compact=compact)


def load_state_ram(state_name: str) -> np.ndarray:
    state = parse_save_state(resolve_state_path(state_name))
    return np.frombuffer(state.ram, dtype=np.uint8).copy()


def parse_bounds(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    parts = [int(part.strip(), 0) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bounds must be x_min,y_min,x_max,y_max")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export Harvest Moon world snapshot from a save state")
    parser.add_argument("--state", default="latest", help="Save state name")
    parser.add_argument("--bounds", help="x_min,y_min,x_max,y_max; default full 64x64 map")
    parser.add_argument("--compact", action="store_true", help="Omit the full scalar table")
    parser.add_argument("--grid", action="store_true", help="Include raw 64x64 tile grid")
    parser.add_argument("--out", help="Write JSON to this path")
    args = parser.parse_args()

    ram = load_state_ram(args.state)
    data = world_snapshot_dict(
        ram,
        bounds=parse_bounds(args.bounds),
        include_grid=args.grid,
        compact=args.compact,
    )
    text = json.dumps(data, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"Wrote world snapshot to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
