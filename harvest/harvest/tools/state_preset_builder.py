#!/usr/bin/env python3
"""Apply JSON save-state presets to Harvest Moon SNES snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harvest.core.harvest_state import CHICKEN_SLOT_BASE_ADDR, CHICKEN_SPAWN_POSITIONS, COW_SLOT_COUNT, HarvestStateDocument
from harvest.core.ram_catalog import SCALAR_FIELDS_BY_KEY
from harvest.runtime.rom_tools import STATES_DIR


def _as_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _apply_scalars(document: HarvestStateDocument, scalars: dict[str, Any]) -> None:
    for key, raw_value in scalars.items():
        if key not in SCALAR_FIELDS_BY_KEY:
            raise KeyError(f"unknown scalar field in preset: {key}")
        document.set_scalar_value(key, _as_int(raw_value, field=key))


def _apply_flags(document: HarvestStateDocument, flags: dict[str, Any]) -> None:
    if "house_level" in flags:
        document.set_house_level(_as_int(flags["house_level"], field="house_level"))
    for key, raw_mask in flags.get("or", {}).items():
        if key not in SCALAR_FIELDS_BY_KEY:
            raise KeyError(f"unknown flag field in preset: {key}")
        mask = _as_int(raw_mask, field=key)
        document.set_scalar_value(key, document.scalar_value(key) | mask)


def _apply_kids(document: HarvestStateDocument, kids: list[dict[str, Any]]) -> None:
    for entry in kids:
        kid = _as_int(entry["kid"], field="kid")
        stage = entry["stage"]
        if not isinstance(stage, str):
            raise TypeError("kid stage must be a string")
        document.set_kid_stage(kid, stage)


def _apply_cows(document: HarvestStateDocument, config: dict[str, Any]) -> None:
    count = _as_int(config["count"], field="cow count")
    if not (0 <= count <= COW_SLOT_COUNT):
        raise ValueError(f"cow count must be 0..{COW_SLOT_COUNT}")
    names = config.get("name_bytes_by_slot", [])
    document.clear_cows()
    for slot in range(count):
        if names:
            name_bytes = tuple(_as_int(value, field=f"cow {slot} name") for value in names[slot])
        else:
            name_bytes = (0xB1, 0xB1, 0xB1, 0xB1)
        if len(name_bytes) != 4:
            raise ValueError(f"cow {slot} name must have exactly four bytes")
        document.set_cow_slot(
            slot,
            status_raw=_as_int(config["status_raw"], field="cow status_raw"),
            raw_1=_as_int(config.get("raw_1", 0), field="cow raw_1"),
            home_map_raw=_as_int(config.get("home_map_raw", 0x27), field="cow home_map_raw"),
            pregnancy_raw=_as_int(config.get("pregnancy_raw", 0), field="cow pregnancy_raw"),
            happiness=_as_int(config.get("happiness", 0), field="cow happiness"),
            raw_5=_as_int(config.get("raw_5", 0), field="cow raw_5"),
            name_bytes=name_bytes,
        )


def _clear_chicken_slot(document: HarvestStateDocument, slot: int) -> None:
    addr = CHICKEN_SLOT_BASE_ADDR + slot * 0x08
    for offset in range(0x08):
        document.mutable_state.write_u8(addr + offset, 0)


def _apply_chickens(document: HarvestStateDocument, config: dict[str, Any]) -> None:
    count = _as_int(config["count"], field="chicken count")
    total_slots = _as_int(config.get("total_slots", len(CHICKEN_SPAWN_POSITIONS)), field="chicken total_slots")
    if not (0 <= count <= total_slots <= len(CHICKEN_SPAWN_POSITIONS)):
        raise ValueError(f"chicken count/total_slots must fit 0..{len(CHICKEN_SPAWN_POSITIONS)}")
    for slot in range(total_slots):
        _clear_chicken_slot(document, slot)
    for slot in range(count):
        addr = CHICKEN_SLOT_BASE_ADDR + slot * 0x08
        spawn_x, spawn_y = CHICKEN_SPAWN_POSITIONS[slot]
        document.mutable_state.write_u8(addr + 0x00, _as_int(config["status_raw"], field="chicken status_raw"))
        document.mutable_state.write_u8(addr + 0x01, _as_int(config.get("raw_1", 0), field="chicken raw_1"))
        document.mutable_state.write_u8(addr + 0x02, _as_int(config.get("raw_2", 0), field="chicken raw_2"))
        document.mutable_state.write_u8(addr + 0x03, _as_int(config.get("raw_3", 0), field="chicken raw_3"))
        document.mutable_state.write_u16(addr + 0x04, spawn_x)
        document.mutable_state.write_u16(addr + 0x06, spawn_y)
    document._recount_animals()


def _apply_animals(document: HarvestStateDocument, animals: dict[str, Any]) -> None:
    if "cows" in animals:
        _apply_cows(document, animals["cows"])
    if "chickens" in animals:
        _apply_chickens(document, animals["chickens"])


def _apply_expected(document: HarvestStateDocument, expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, raw_value in expected.items():
        if key not in SCALAR_FIELDS_BY_KEY:
            raise KeyError(f"unknown expected field in preset: {key}")
        expected_value = _as_int(raw_value, field=f"expected {key}")
        actual = document.scalar_value(key)
        if actual != expected_value:
            failures.append(f"{key}: expected {expected_value}, got {actual}")
    return failures


def apply_preset(document: HarvestStateDocument, preset: dict[str, Any]) -> None:
    if preset.get("schema") != "harvest_state_preset_v1":
        raise ValueError("unsupported or missing preset schema")

    farm = preset.get("farm", {})
    if farm.get("fill_non_structure_with_grass"):
        result = document.fill_farm_ground_with_grass(
            update_visible_map=bool(farm.get("update_visible_map", False))
        )
        expected_tiles = farm.get("expected_development_tiles")
        if expected_tiles is not None and result.development_tiles != int(expected_tiles):
            raise ValueError(
                f"farm development tiles expected {expected_tiles}, got {result.development_tiles}"
            )
        expected_percent = farm.get("expected_development_percent")
        if expected_percent is not None and result.projected_development_percent != int(expected_percent):
            raise ValueError(
                f"farm development percent expected {expected_percent}, got {result.projected_development_percent}"
            )

    _apply_scalars(document, preset.get("scalars", {}))
    _apply_flags(document, preset.get("flags", {}))
    _apply_kids(document, preset.get("kids", []))
    _apply_flags(document, preset.get("flags", {}))
    _apply_animals(document, preset.get("animals", {}))
    _apply_scalars(document, preset.get("scalars", {}))

    failures = _apply_expected(document, preset.get("expected_initial", {}))
    if failures:
        raise AssertionError("preset expected_initial mismatch: " + "; ".join(failures))


def load_preset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("preset root must be a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", type=Path, help="Path to a harvest_state_preset_v1 JSON file")
    parser.add_argument("--base-state", help="Override preset base_state")
    parser.add_argument("--output-state", help="Override preset output_state")
    parser.add_argument("--dry-run", action="store_true", help="Apply and validate without writing")
    args = parser.parse_args()

    preset = load_preset(args.preset)
    base_state = args.base_state or preset["base_state"]
    output_state = args.output_state or preset["output_state"]

    document = HarvestStateDocument.load(base_state)
    apply_preset(document, preset)

    if args.dry_run:
        print(f"[PRESET] dry-run ok preset={args.preset} base={base_state} output={output_state}")
        return

    output_path = STATES_DIR / f"{output_state}.state"
    document.save_as(output_path)
    print(f"[PRESET] wrote={output_path}")


if __name__ == "__main__":
    main()
