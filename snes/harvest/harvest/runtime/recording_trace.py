"""Recording trace helpers for human task captures.

The raw input list is enough to replay a task, but it is not enough to learn
why a task works or stalls.  These helpers keep the recording path cheap while
capturing the RAM and dynamic-object facts needed to turn a manual run into
autonomous navigation data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from harvest.core.animal_probe import BARN_TILEMAP, COOP_TILEMAP, chicken_slot_snapshots, cow_slot_snapshots
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.tasks.nav import get_pos_from_ram
from harvest.maps.map_config import get_map_name
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import TILE_SIZE, get_tile_at, tile_category, tile_label
from harvest.tasks.travel_walk import DIR_DELTA, DIR_FROM_CODE, read_player_direction
from retro_harness.task_recording import (
    coalesce_action_runs as _shared_coalesce_action_runs,
    coalesce_windows as _shared_coalesce_windows,
    pressed_buttons as _shared_pressed_buttons,
)

# Harvest keeps title-case button labels for existing task JSON compatibility.
BUTTON_NAMES = {
    0: "B",
    1: "Y",
    2: "Select",
    3: "Start",
    4: "Up",
    5: "Down",
    6: "Left",
    7: "Right",
    8: "A",
    9: "X",
    10: "L",
    11: "R",
}
_BUTTON_NAME_SEQ = tuple(BUTTON_NAMES[i] for i in range(12))

MOVEMENT_BUTTONS = {"Up", "Down", "Left", "Right"}


def _cell(ram: np.ndarray, tx: int, ty: int) -> dict[str, object]:
    tid = int(get_tile_at(ram, tx, ty))
    return {
        "tile": [int(tx), int(ty)],
        "id": tid,
        "hex": f"0x{tid:02X}",
        "label": tile_label(tid),
    }


def pressed_buttons(action: Sequence[int]) -> list[str]:
    return _shared_pressed_buttons(action, names=_BUTTON_NAME_SEQ)


def coalesce_windows(frames: Iterable[int]) -> list[dict[str, int]]:
    return _shared_coalesce_windows(frames)


def coalesce_action_runs(frames: Sequence[Sequence[int]]) -> list[dict[str, object]]:
    return _shared_coalesce_action_runs(frames, names=_BUTTON_NAME_SEQ)


def _read_scalar(ram: np.ndarray, key: str, *, raw: bool = False) -> int:
    try:
        return int(read_ram_value(ram, key, raw=raw))
    except Exception:
        return 0


def compact_game_object(obj) -> dict[str, object]:
    return {
        "slot": int(obj.slot),
        "sprite": int(obj.sprite_table_idx),
        "sprite_hex": f"0x{int(obj.sprite_table_idx):04X}",
        "label": obj.label,
        "kind": obj.kind,
        "pixel": [int(obj.pixel[0]), int(obj.pixel[1])],
        "tile": [int(obj.tile[0]), int(obj.tile[1])],
        "source": obj.source,
    }


def _slot_stage_by_tile(ram: np.ndarray) -> dict[tuple[int, int], str]:
    stages: dict[tuple[int, int], str] = {}
    for slot in chicken_slot_snapshots(ram, require_coop=True):
        tile = slot.get("tile")
        stage = slot.get("stage")
        if isinstance(tile, list) and len(tile) == 2 and isinstance(stage, str):
            stages[(int(tile[0]), int(tile[1]))] = stage
    return stages


def coop_entities_from_ram(ram: np.ndarray) -> list[dict[str, object]]:
    """Return compact live entities relevant to coop collision analysis."""
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap != COOP_TILEMAP:
        return []

    rows: list[dict[str, object]] = []
    stage_by_tile = _slot_stage_by_tile(ram)
    for obj in game_objects(ram):
        if obj.is_player or obj.label == "chicken" or obj.kind == "animal":
            row = compact_game_object(obj)
            if obj.label == "chicken":
                row["stage"] = stage_by_tile.get((int(obj.tile[0]), int(obj.tile[1])), "unknown")
            rows.append(row)
    for slot in chicken_slot_snapshots(ram, require_coop=True):
        rows.append(
            {
                "slot": int(slot["slot"]),
                "label": "chicken",
                "kind": "animal",
                "stage": slot["stage"],
                "status_raw": slot["status_raw"],
                "status_hex": slot["status_hex"],
                "pixel": slot["pixel"],
                "tile": slot["tile"],
                "source": slot["source"],
            }
        )
    rows.sort(key=lambda row: (str(row["label"]) != "player", int(row["slot"])))
    return rows


def barn_entities_from_ram(ram: np.ndarray) -> list[dict[str, object]]:
    """Return compact live entities relevant to barn cow analysis."""
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    if tilemap != BARN_TILEMAP:
        return []

    rows: list[dict[str, object]] = []
    for obj in game_objects(ram):
        if obj.is_player or obj.label == "cow" or obj.kind == "animal":
            rows.append(compact_game_object(obj))
    for slot in cow_slot_snapshots(ram, require_barn=True):
        rows.append(
            {
                "slot": int(slot["slot"]),
                "label": "cow",
                "kind": "animal",
                "status_raw": slot["status_raw"],
                "status_hex": slot["status_hex"],
                "home_map_raw": slot["home_map_raw"],
                "home_map_hex": slot["home_map_hex"],
                "pregnancy_raw": slot["pregnancy_raw"],
                "happiness": slot["happiness"],
                "pixel": slot["pixel"],
                "tile": slot["tile"],
                "source": slot["source"],
            }
        )
    rows.sort(key=lambda row: (str(row["label"]) != "player", int(row["slot"])))
    return rows


def recording_trace_entry(
    ram: np.ndarray,
    *,
    frame: int,
    action: Sequence[int],
) -> dict[str, object]:
    """Build one compact trace row after a recorded action has been stepped."""
    pos = get_pos_from_ram(ram)
    tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
    tile_id = get_tile_at(ram, *tile)
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    buttons = pressed_buttons(action)
    facing = DIR_FROM_CODE.get(read_player_direction(ram), "down")
    fdx, fdy = DIR_DELTA[facing]
    neighbors = {
        name: _cell(ram, tile[0] + dx, tile[1] + dy)
        for name, (dx, dy) in DIR_DELTA.items()
    }

    row: dict[str, object] = {
        "frame": int(frame),
        "x": int(pos.x),
        "y": int(pos.y),
        "tx": int(tile[0]),
        "ty": int(tile[1]),
        "tm": tilemap,
        "map": get_map_name(tilemap),
        "tile_id": int(tile_id),
        "tile_hex": f"0x{int(tile_id):02X}",
        "tile_label": tile_label(tile_id),
        "tile_category": tile_category(tile_id),
        "facing": facing,
        "facing_tile": neighbors[facing],
        "neighbors": neighbors,
        "buttons": buttons,
        "input_lock": _read_scalar(ram, "input_lock", raw=True),
        "player_state": _read_scalar(ram, "player_state", raw=True),
        "player_action": _read_scalar(ram, "player_action", raw=True),
        "held_item": _read_scalar(ram, "held_item", raw=True),
        "tool": _read_scalar(ram, "tool_selected", raw=True),
        "hour": _read_scalar(ram, "hour", raw=True),
        "minute": _read_scalar(ram, "minute", raw=True),
        "weather_tomorrow": _read_scalar(ram, "weather_tomorrow", raw=True),
        "stored_grass": _read_scalar(ram, "stored_grass", raw=True),
        "cow_feed": _read_scalar(ram, "cow_feed", raw=True),
        "chicken_feed": _read_scalar(ram, "chicken_feed", raw=True),
        "num_cows": _read_scalar(ram, "num_cows", raw=True),
        "num_chickens": _read_scalar(ram, "num_chickens", raw=True),
        "fed_cows_n": _read_scalar(ram, "fed_cows_n", raw=True),
        "fed_cows_flags": _read_scalar(ram, "fed_cows_flags", raw=True),
        "fed_chickens_n": _read_scalar(ram, "fed_chickens_n", raw=True),
        "fed_chickens_flags": _read_scalar(ram, "fed_chickens_flags", raw=True),
        "egg_available": _read_scalar(ram, "egg_available", raw=True),
        "incubator_flags": _read_scalar(ram, "incubator_flags", raw=True),
        "shipping_money": _read_scalar(ram, "shipping_money"),
    }

    entities = coop_entities_from_ram(ram) or barn_entities_from_ram(ram)
    if entities:
        row["entities"] = entities
    return row


def _nearest_chicken_distance(row: dict[str, object]) -> int | None:
    entities = row.get("entities")
    if not isinstance(entities, list):
        return None
    px, py = int(row["tx"]), int(row["ty"])
    distances = []
    for entity in entities:
        if entity.get("label") != "chicken":
            continue
        tile = entity.get("tile")
        if not isinstance(tile, list) or len(tile) != 2:
            continue
        distances.append(abs(int(tile[0]) - px) + abs(int(tile[1]) - py))
    return min(distances) if distances else None


def _push_faces(trace: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Tiles where a held d-pad did not change occupancy — bump/edge candidates."""
    faces: list[dict[str, object]] = []
    seen: set[tuple[int, int, int]] = set()
    for window in _stasis_windows(trace):
        tile = window["tile"]
        if not isinstance(tile, list) or len(tile) != 2:
            continue
        key = (int(window.get("tilemap", 0)), int(tile[0]), int(tile[1]))
        if key in seen:
            continue
        seen.add(key)
        faces.append(
            {
                "tile": [key[1], key[2]],
                "tilemap": key[0],
                "length": int(window["length"]),
                "buttons": list(window.get("buttons", [])),
            }
        )
    return faces


def _stasis_windows(trace: Sequence[dict[str, object]], *, min_length: int = 45) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    start_idx: int | None = None
    last_tile: tuple[int, int] | None = None

    for idx, row in enumerate(trace):
        tile = (int(row.get("tx", 0)), int(row.get("ty", 0)))
        buttons = set(row.get("buttons", []))
        moving = bool(buttons & MOVEMENT_BUTTONS)
        if moving and tile == last_tile:
            if start_idx is None:
                start_idx = idx - 1 if idx > 0 else idx
        else:
            if start_idx is not None and idx - start_idx >= min_length:
                windows.append(_stasis_window(trace, start_idx, idx - 1))
            start_idx = None
        last_tile = tile

    if start_idx is not None and len(trace) - start_idx >= min_length:
        windows.append(_stasis_window(trace, start_idx, len(trace) - 1))
    return windows


def _stasis_window(trace: Sequence[dict[str, object]], start_idx: int, end_idx: int) -> dict[str, object]:
    first = trace[start_idx]
    last = trace[end_idx]
    nearest = [
        distance
        for row in trace[start_idx : end_idx + 1]
        for distance in [_nearest_chicken_distance(row)]
        if distance is not None
    ]
    buttons = sorted({button for row in trace[start_idx : end_idx + 1] for button in row.get("buttons", [])})
    row: dict[str, object] = {
        "start": int(first.get("frame", start_idx)),
        "end": int(last.get("frame", end_idx)),
        "length": int(last.get("frame", end_idx)) - int(first.get("frame", start_idx)) + 1,
        "tile": [int(first.get("tx", 0)), int(first.get("ty", 0))],
        "pixel_start": [int(first.get("x", 0)), int(first.get("y", 0))],
        "pixel_end": [int(last.get("x", 0)), int(last.get("y", 0))],
        "tilemap": int(first.get("tm", 0)),
        "buttons": buttons,
    }
    if nearest:
        row["nearest_chicken_distance_min"] = min(nearest)
    return row


def _value_change_windows(trace: Sequence[dict[str, object]], key: str) -> list[dict[str, int]]:
    if not trace:
        return []
    prev = trace[0].get(key)
    changed: list[int] = []
    for row in trace[1:]:
        value = row.get(key)
        if value != prev:
            changed.append(int(row.get("frame", 0)))
        prev = value
    return coalesce_windows(changed)


def summarize_recording(
    *,
    frames: Sequence[Sequence[int]],
    trace: Sequence[dict[str, object]],
) -> dict[str, object]:
    transitions = []
    visited: dict[int, dict[tuple[int, int], dict[str, object]]] = defaultdict(dict)
    coop_chicken_tiles: set[tuple[int, int]] = set()
    coop_chicken_tiles_by_stage: dict[str, set[tuple[int, int]]] = defaultdict(set)
    coop_frames = 0
    barn_cow_tiles: set[tuple[int, int]] = set()
    barn_frames = 0

    for idx, row in enumerate(trace):
        tilemap = int(row.get("tm", 0))
        if idx == 0 or tilemap != int(trace[idx - 1].get("tm", -1)):
            transitions.append(
                {
                    "frame": int(row.get("frame", idx)),
                    "tilemap": tilemap,
                    "tilemap_hex": f"0x{tilemap:02X}",
                    "map": row.get("map", get_map_name(tilemap)),
                    "x": int(row.get("x", 0)),
                    "y": int(row.get("y", 0)),
                    "tile": [int(row.get("tx", 0)), int(row.get("ty", 0))],
                }
            )

        tile = (int(row.get("tx", 0)), int(row.get("ty", 0)))
        bucket = visited[tilemap].setdefault(
            tile,
            {
                "tile": [tile[0], tile[1]],
                "tile_ids": set(),
                "labels": set(),
                "frames": 0,
            },
        )
        bucket["tile_ids"].add(int(row.get("tile_id", 0)))
        bucket["labels"].add(str(row.get("tile_label", "")))
        bucket["frames"] += 1

        if tilemap == COOP_TILEMAP:
            coop_frames += 1
            entities = row.get("entities")
            if isinstance(entities, list):
                for entity in entities:
                    if entity.get("label") != "chicken":
                        continue
                    etile = entity.get("tile")
                    if isinstance(etile, list) and len(etile) == 2:
                        chicken_tile = (int(etile[0]), int(etile[1]))
                        coop_chicken_tiles.add(chicken_tile)
                        stage = entity.get("stage")
                        if not isinstance(stage, str) or not stage:
                            stage = "unknown"
                        coop_chicken_tiles_by_stage[stage].add(chicken_tile)
        elif tilemap == BARN_TILEMAP:
            barn_frames += 1
            entities = row.get("entities")
            if isinstance(entities, list):
                for entity in entities:
                    if entity.get("label") != "cow":
                        continue
                    etile = entity.get("tile")
                    if isinstance(etile, list) and len(etile) == 2:
                        barn_cow_tiles.add((int(etile[0]), int(etile[1])))

    visited_tiles = {}
    for tilemap, tiles in visited.items():
        visited_tiles[f"0x{tilemap:02X}"] = [
            {
                "tile": data["tile"],
                "tile_ids": [f"0x{tile_id:02X}" for tile_id in sorted(data["tile_ids"])],
                "labels": sorted(label for label in data["labels"] if label),
                "frames": int(data["frames"]),
            }
            for data in sorted(tiles.values(), key=lambda item: (item["tile"][1], item["tile"][0]))
        ]

    coop_trace = [row for row in trace if int(row.get("tm", 0)) == COOP_TILEMAP]
    barn_trace = [row for row in trace if int(row.get("tm", 0)) == BARN_TILEMAP]
    return {
        "frame_count": len(frames),
        "duration_seconds": len(frames) / 60.0,
        "transitions": transitions,
        "recorded_input_runs": coalesce_action_runs(frames),
        "visited_tiles": visited_tiles,
        "stasis_windows": _stasis_windows(trace),
        "push_faces": _push_faces(trace),
        "coop": {
            "frame_count": coop_frames,
            "player_tiles": visited_tiles.get("0x28", []),
            "chicken_tiles": [
                {"tile": [tile[0], tile[1]]}
                for tile in sorted(coop_chicken_tiles, key=lambda item: (item[1], item[0]))
            ],
            "adult_chicken_tiles": [
                {"tile": [tile[0], tile[1]]}
                for tile in sorted(coop_chicken_tiles_by_stage.get("adult", set()), key=lambda item: (item[1], item[0]))
            ],
            "chick_tiles": [
                {"tile": [tile[0], tile[1]]}
                for tile in sorted(coop_chicken_tiles_by_stage.get("chick", set()), key=lambda item: (item[1], item[0]))
            ],
            "egg_slot_tiles": [
                {"tile": [tile[0], tile[1]]}
                for tile in sorted(coop_chicken_tiles_by_stage.get("egg", set()), key=lambda item: (item[1], item[0]))
            ],
            "unknown_chicken_tiles": [
                {"tile": [tile[0], tile[1]]}
                for tile in sorted(coop_chicken_tiles_by_stage.get("unknown", set()), key=lambda item: (item[1], item[0]))
            ],
            "egg_available_change_windows": _value_change_windows(coop_trace, "egg_available"),
            "held_item_change_windows": _value_change_windows(coop_trace, "held_item"),
            "stored_grass_change_windows": _value_change_windows(coop_trace, "stored_grass"),
            "fed_chickens_change_windows": _value_change_windows(coop_trace, "fed_chickens_n"),
            "shipping_money_change_windows": _value_change_windows(coop_trace, "shipping_money"),
        },
        "barn": {
            "frame_count": barn_frames,
            "player_tiles": visited_tiles.get("0x27", []),
            "cow_tiles": [
                {"tile": [tile[0], tile[1]]}
                for tile in sorted(barn_cow_tiles, key=lambda item: (item[1], item[0]))
            ],
            "held_item_change_windows": _value_change_windows(barn_trace, "held_item"),
            "stored_grass_change_windows": _value_change_windows(barn_trace, "stored_grass"),
            "fed_cows_change_windows": _value_change_windows(barn_trace, "fed_cows_n"),
            "fed_cows_flags_change_windows": _value_change_windows(barn_trace, "fed_cows_flags"),
        },
    }
