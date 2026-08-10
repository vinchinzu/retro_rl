#!/usr/bin/env python3
"""Central RAM metadata and read/write helpers for Harvest Moon SNES.

The bot has two RAM views:
- save-state RAM blocks are direct WRAM offsets
- live ``env.get_ram()`` snapshots may mirror WRAM at ``+0x4000``

This module keeps the address catalog, value decoding, hot-edit writes, and
small RAM expectations in one place so tasks can verify outcomes by field name
instead of scattering raw offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


LIVE_RAM_WRAM_OFFSET = 0x4000
WRAM_SNAPSHOT_SIZE = 0x20000

KIND_WIDTHS = {
    "u8": 1,
    "u16": 2,
    "u24": 3,
}

WEATHER_CODES = {
    0: "0 sunny",
    1: "1 rain",
    2: "2 snow",
    3: "3 hurricane",
    4: "4 fair",
    5: "5 sunny/calm",
    6: "6 flower festival",
    7: "7 harvest festival",
    8: "8 thanksgiving",
    9: "9 star night",
    10: "10 festival",
    11: "11 egg festival",
    12: "12 snow alt",
}


@dataclass(frozen=True)
class RamFieldSpec:
    key: str
    label: str
    address: int
    kind: str
    section: str
    source: str
    minimum: int = 0
    maximum: int = 0xFFFF
    note: str = ""
    data_key: str | None = None
    display_multiplier: int = 1
    aliases: tuple[str, ...] = ()
    live_offset: bool = True

    @property
    def width(self) -> int:
        try:
            return KIND_WIDTHS[self.kind]
        except KeyError as exc:
            raise ValueError(f"Unsupported RAM field kind: {self.kind}") from exc

    def clamp_storage(self, value: int) -> int:
        return max(self.minimum, min(self.maximum, int(value)))

    def to_storage(self, value: int, *, raw: bool = False) -> int:
        storage = int(value)
        if not raw and self.display_multiplier != 1:
            storage //= self.display_multiplier
        return self.clamp_storage(storage)

    def from_storage(self, value: int, *, raw: bool = False) -> int:
        if raw or self.display_multiplier == 1:
            return int(value)
        return int(value) * self.display_multiplier


@dataclass(frozen=True)
class RamPatch:
    field: str
    value: int
    raw: bool = False


@dataclass(frozen=True)
class RamExpectation:
    field: str
    expected: int
    raw: bool = False

    def check(self, ram: np.ndarray) -> str | None:
        actual = read_ram_value(ram, self.field, raw=self.raw)
        if actual == self.expected:
            return None
        unit = "raw" if self.raw else "value"
        return f"{self.field} expected {self.expected} {unit}, got {actual}"


SCALAR_FIELDS: tuple[RamFieldSpec, ...] = (
    RamFieldSpec("year", "Year", 0x11F18, "u8", "Date & Weather", "retro", 0, 9, data_key="year"),
    RamFieldSpec("season", "Season", 0x11F19, "u8", "Date & Weather", "retro", 0, 3, data_key="season"),
    RamFieldSpec(
        "weekday",
        "Weekday",
        0x11F1A,
        "u8",
        "Date & Weather",
        "retro",
        0,
        6,
        data_key="day_of_week",
        aliases=("day_of_week",),
    ),
    RamFieldSpec("day", "Day", 0x11F1B, "u8", "Date & Weather", "retro", 1, 30, data_key="day"),
    RamFieldSpec("hour", "Hour", 0x11F1C, "u8", "Date & Weather", "retro", 0, 23, data_key="hour"),
    RamFieldSpec("minute", "Minute", 0x11F1D, "u8", "Date & Weather", "retro", 0, 59, data_key="minute"),
    RamFieldSpec("second", "Second", 0x11F1E, "u8", "Date & Weather", "decomp", 0, 59),
    RamFieldSpec(
        "time_running",
        "Time Running",
        0x0973,
        "u16",
        "Date & Weather",
        "decomp",
        0,
        2,
        "0=stopped, 1=clock runs, 2=advance to next day.",
    ),
    RamFieldSpec(
        "weather_tomorrow",
        "Weather Code",
        0x098C,
        "u8",
        "Date & Weather",
        "retro",
        0,
        12,
        "Numeric weather/event code; names are provisional.",
        data_key="weather",
        aliases=("weather",),
    ),
    RamFieldSpec("stamina", "Stamina", 0x0918, "u8", "Runtime", "retro", 0, 100, data_key="stamina"),
    RamFieldSpec("max_stamina", "Max Stamina", 0x0917, "u8", "Runtime", "decomp", 0, 255),
    RamFieldSpec("player_state", "Player State", 0x00D2, "u8", "Runtime", "state", 0, 0xFF),
    RamFieldSpec(
        "game_state",
        "Game State Flags",
        0x00D2,
        "u16",
        "Runtime",
        "state",
        0,
        0xFFFF,
        "Runtime flags; bit 0x0080 carrying dog, 0x0800 pickup/drop anim, "
        "0x4000 free on-foot outdoor control (cleared by bad post-truck ExitToFarm).",
        aliases=("player_state_flags",),
    ),
    RamFieldSpec("player_action", "Player Action", 0x00D4, "u8", "Runtime", "state", 0, 0xFF),
    RamFieldSpec("player_x", "Player X", 0x00D6, "u16", "Runtime", "retro", 0, 0xFFFF),
    RamFieldSpec("player_y", "Player Y", 0x00D8, "u16", "Runtime", "retro", 0, 0xFFFF),
    RamFieldSpec("tilemap", "Tilemap", 0x0022, "u8", "Runtime", "retro", 0, 0xFF, live_offset=False),
    RamFieldSpec("input_lock", "Input Lock", 0x019A, "u8", "Runtime", "retro", 0, 0xFF, live_offset=False),
    RamFieldSpec("dialog_text_id", "Dialog Text ID", 0x0183, "u16", "Runtime", "decomp", 0, 0xFFFF, live_offset=False),
    RamFieldSpec("dialog_menu_cursor", "Dialog/Menu Cursor", 0x018A, "u8", "Runtime", "decomp", 0, 0xFF, live_offset=False),
    RamFieldSpec("dialog_choice_result", "Dialog Choice Result", 0x018F, "u8", "Runtime", "decomp", 0, 0xFF, live_offset=False),
    RamFieldSpec("dialog_text_mode", "Dialog Text Mode", 0x0191, "u8", "Runtime", "decomp", 0, 0xFF, live_offset=False),
    RamFieldSpec(
        "held_item",
        "Held Item",
        0x091D,
        "u8",
        "Farm & Inventory",
        "retro",
        0,
        0xFF,
        aliases=("item_on_hand", "carried_item"),
    ),
    RamFieldSpec(
        "tool_selected",
        "Tool Selected",
        0x0921,
        "u8",
        "Farm & Inventory",
        "retro",
        0,
        0xFF,
        data_key="item_in_hand",
        aliases=("tool", "item_in_hand"),
    ),
    RamFieldSpec(
        "tool_backpack",
        "Tool Backpack",
        0x0923,
        "u8",
        "Farm & Inventory",
        "retro",
        0,
        0xFF,
        data_key="item_in_hand_alt",
        aliases=("item_in_hand_alt",),
    ),
    RamFieldSpec(
        "money",
        "Money",
        0x11F04,
        "u24",
        "Farm & Inventory",
        "retro",
        0,
        999999,
        "Displayed gold; stored in RAM as value/10.",
        display_multiplier=10,
        aliases=("gold",),
    ),
    RamFieldSpec(
        "money_raw",
        "Money Raw",
        0x11F04,
        "u24",
        "Farm & Inventory",
        "retro",
        0,
        999999,
        "Raw money storage value.",
    ),
    RamFieldSpec(
        "shipping_money",
        "Shipping Money",
        0x11F07,
        "u24",
        "Farm & Inventory",
        "decomp",
        0,
        999999,
        "Displayed gold; stored in RAM as value/10.",
        display_multiplier=10,
    ),
    RamFieldSpec(
        "shipping_money_raw",
        "Shipping Money Raw",
        0x11F07,
        "u24",
        "Farm & Inventory",
        "decomp",
        0,
        999999,
        "Raw shipping-money storage value.",
    ),
    RamFieldSpec("stored_wood", "Stored Wood", 0x11F0C, "u16", "Farm & Inventory", "retro", 0, 9999),
    RamFieldSpec(
        "stored_grass",
        "Stored Grass",
        0x11F10,
        "u16",
        "Farm & Inventory",
        "decomp",
        0,
        9999,
        aliases=("hay",),
    ),
    RamFieldSpec("planted_grass", "Planted Grass", 0x11F29, "u16", "Farm & Inventory", "decomp", 0, 9999),
    RamFieldSpec(
        "watering_can",
        "Watering Can",
        0x0926,
        "u8",
        "Farm & Inventory",
        "retro",
        0,
        99,
        data_key="water_can",
        aliases=("water_can",),
    ),
    RamFieldSpec("grass_seeds", "Grass Seeds", 0x0927, "u8", "Farm & Inventory", "retro", 0, 99, data_key="grass_seeds"),
    RamFieldSpec("corn_seeds", "Corn Seeds", 0x0928, "u8", "Farm & Inventory", "retro", 0, 99, data_key="corn_seeds"),
    RamFieldSpec("tomato_seeds", "Tomato Seeds", 0x0929, "u8", "Farm & Inventory", "retro", 0, 99, data_key="tomato_seeds"),
    RamFieldSpec("potato_seeds", "Potato Seeds", 0x092A, "u8", "Farm & Inventory", "retro", 0, 99, data_key="potato_seeds"),
    RamFieldSpec("turnip_seeds", "Turnip Seeds", 0x092B, "u8", "Farm & Inventory", "retro", 0, 99, data_key="turnip_seeds"),
    RamFieldSpec("shipped_corn", "Shipped Corn", 0x11F4A, "u16", "Ending Evaluation", "decomp", 0, 9999),
    RamFieldSpec(
        "shipped_tomatoes",
        "Shipped Tomatoes",
        0x11F4C,
        "u16",
        "Ending Evaluation",
        "decomp",
        0,
        9999,
    ),
    RamFieldSpec("shipped_turnips", "Shipped Turnips", 0x11F4E, "u16", "Ending Evaluation", "decomp", 0, 9999),
    RamFieldSpec("shipped_potatoes", "Shipped Potatoes", 0x11F50, "u16", "Ending Evaluation", "decomp", 0, 9999),
    RamFieldSpec("cow_feed", "Cow Feed", 0x092C, "u8", "Farm & Inventory", "retro", 0, 99),
    RamFieldSpec("chicken_feed", "Chicken Feed", 0x092D, "u8", "Farm & Inventory", "retro", 0, 99),
    RamFieldSpec("power_berries", "Power Berries", 0x0976, "u8", "Farm & Inventory", "retro", 0, 10, data_key="power_berries"),
    RamFieldSpec(
        "house_size",
        "House Size",
        0x0970,
        "u8",
        "Farm & Inventory",
        "retro",
        0,
        3,
        # Decomp labels $0970 as "House Level?" but NPC talk scripts also INC/STZ
        # it as a dialogue step counter (jumps 0→2 on Ann during D1). Do not treat
        # mid-run values as remodel level; Gate B uses house_size_at_start only.
        "Often a dialogue step counter mid-run; remodel level only at cold start.",
        data_key="house_size",
    ),
    RamFieldSpec("dog_map", "Dog Map", 0x11F30, "u8", "Animals", "decomp", 0, 0xFF),
    RamFieldSpec("horse_map", "Horse Map", 0x11F31, "u8", "Animals", "decomp", 0, 0xFF),
    RamFieldSpec(
        "horse_age",
        "Horse Age",
        0x11F32,
        "u8",
        "Animals",
        "decomp",
        0,
        0x15,
        "Horse grows daily until 0x15; 0x15 is adult.",
    ),
    RamFieldSpec("num_cows", "Cow Count", 0x11F0A, "u8", "Animals", "retro", 0, 12, data_key="num_cows"),
    RamFieldSpec("num_chickens", "Chicken Count", 0x11F0B, "u8", "Animals", "retro", 0, 13, data_key="num_chickens"),
    RamFieldSpec("fed_cows_n", "Fed Cows", 0x0930, "u8", "Animals", "retro", 0, 12, aliases=("cow_fed_status",)),
    RamFieldSpec("fed_chickens_n", "Fed Chickens", 0x0931, "u8", "Animals", "retro", 0, 13, aliases=("chicken_fed_status",)),
    RamFieldSpec("fed_cows_flags", "Fed Cow Flags", 0x0932, "u16", "Animals", "retro", 0, 0xFFFF),
    RamFieldSpec("fed_chickens_flags", "Fed Chicken Flags", 0x0934, "u16", "Animals", "retro", 0, 0xFFFF),
    RamFieldSpec("egg_available", "Egg Available", 0x11F45, "u16", "Animals", "state", 0, 0xFFFF),
    RamFieldSpec(
        "incubator_flags",
        "Incubator / Family Status Flags",
        0x11F6E,
        "u16",
        "Animals",
        "state",
        0,
        0xFFFF,
        aliases=("family_status_flags", "child_flags"),
    ),
    RamFieldSpec("maria_hearts", "Maria Hearts", 0x11F1F, "u16", "Romance", "decomp", 0, 999),
    RamFieldSpec("ann_hearts", "Ann Hearts", 0x11F21, "u16", "Romance", "decomp", 0, 999),
    RamFieldSpec("nina_hearts", "Nina Hearts", 0x11F23, "u16", "Romance", "decomp", 0, 999),
    RamFieldSpec("ellen_hearts", "Ellen Hearts", 0x11F25, "u16", "Romance", "decomp", 0, 999),
    RamFieldSpec("eve_hearts", "Eve Hearts", 0x11F27, "u16", "Romance", "decomp", 0, 999),
    RamFieldSpec(
        "happiness",
        "Global Happiness",
        0x11F33,
        "u16",
        "Ending Evaluation",
        "decomp",
        0,
        9999,
        aliases=("global_happiness",),
    ),
    RamFieldSpec(
        "development_rate",
        "Development Rate",
        0x11F35,
        "u8",
        "Ending Evaluation",
        "decomp",
        0,
        100,
        "Small persistent development byte; the credits percentage is ranch_development.",
        aliases=("farm_development_rate",),
    ),
    RamFieldSpec(
        "power_berry_count",
        "Persistent Power Berries",
        0x11F36,
        "u8",
        "Ending Evaluation",
        "decomp",
        0,
        10,
        "Ending check uses this persistent counter, distinct from the retro runtime power_berries field.",
        aliases=("persistent_power_berries", "ending_power_berries"),
    ),
    RamFieldSpec("kid1_age", "Kid 1 Age", 0x11F37, "u16", "Family", "decomp", 0, 9999),
    RamFieldSpec("kid2_age", "Kid 2 Age", 0x11F39, "u16", "Family", "decomp", 0, 9999),
    RamFieldSpec("wife_pregnancy", "Wife Pregnancy", 0x11F3B, "u16", "Family", "decomp", 0, 9999),
    RamFieldSpec(
        "ending_scene_index",
        "Ending Scene Index",
        0x11F47,
        "u8",
        "Ending Evaluation",
        "decomp",
        0,
        0xFF,
        "Credits branch iterator; each shown ending scene advances this byte.",
        aliases=("ending_index",),
    ),
    RamFieldSpec(
        "ending_aux_scene_index",
        "Ending Auxiliary Scene Index",
        0x11F49,
        "u8",
        "Ending Evaluation",
        "decomp",
        0,
        0xFF,
        aliases=("ending_aux",),
    ),
    RamFieldSpec(
        "dog_hugs",
        "Dog Hugs",
        0x11F52,
        "u16",
        "Ending Evaluation",
        "decomp",
        0,
        9999,
        "Persistent dog pickup/hug counter; the best ending check requires at least 100.",
        aliases=("dog_pickups", "dog_pickup_count"),
    ),
    RamFieldSpec("ranch_mastery", "Ranch Mastery", 0x11F54, "u16", "Ending Evaluation", "decomp", 0, 999),
    RamFieldSpec(
        "ranch_development",
        "Ranch Development",
        0x11F56,
        "u16",
        "Ending Evaluation",
        "decomp",
        0,
        9999,
        "Raw farm-development count during the final day; converted to 0..100 near credits.",
        aliases=("farm_development",),
    ),
    RamFieldSpec("event_flags_1f5a", "Event Flags 1F5A", 0x11F5A, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("event_flags_1f5c", "Event Flags 1F5C", 0x11F5C, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("event_flags_1f5e", "Event Flags 1F5E", 0x11F5E, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("event_flags_1f60", "Event Flags 1F60", 0x11F60, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("event_flags_1f62", "Event Flags 1F62", 0x11F62, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("upgrade_flags", "Upgrade Flags", 0x11F64, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("marriage_flags", "Marriage Flags", 0x11F66, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("event_flags_1f68", "Event Flags 1F68", 0x11F68, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("romance_event_flags", "Romance Event Flags", 0x11F6A, "u16", "Flags", "decomp", 0, 0xFFFF),
    RamFieldSpec("family_event_flags", "Family Event Flags", 0x11F6C, "u16", "Flags", "decomp", 0, 0xFFFF),
    # Spring D1 town social handoff (ROM event scripts). Full completion = 0x3F.
    # Bits: Ann=0x01 Eve=0x02 Nina=0x04 flower-owner=0x08 livestock=0x10 Maria=0x20.
    # Verified live 2026-08-01; see docs/town_day1_recon.md.
    RamFieldSpec(
        "d1_town_event_mask",
        "D1 Town Event Mask",
        0x11F74,
        "u8",
        "Flags",
        "rom",
        0,
        0x3F,
        "Spring D1 six-person town handoff bits; truck leave needs 0x3F.",
        aliases=("town_day1_event_mask",),
    ),
)

SCALAR_FIELDS_BY_KEY: dict[str, RamFieldSpec] = {}
for _field in SCALAR_FIELDS:
    SCALAR_FIELDS_BY_KEY[_field.key] = _field
    for _alias in _field.aliases:
        SCALAR_FIELDS_BY_KEY[_alias] = _field


CHICKEN_SLOT_COUNT = 13
COW_SLOT_COUNT = 12
CHICKEN_SLOT_BASE = 0xC286
CHICKEN_SLOT_SIZE = 0x08
COW_SLOT_BASE = 0xC1C6
COW_SLOT_SIZE = 0x10

CHICKEN_FIELD_OFFSETS = {
    "status_raw": ("u8", 0x00),
    "raw_1": ("u8", 0x01),
    "raw_2": ("u8", 0x02),
    "raw_3": ("u8", 0x03),
    "position_x": ("u16", 0x04),
    "position_y": ("u16", 0x06),
}

COW_FIELD_OFFSETS = {
    "status_raw": ("u8", 0x00),
    "raw_1": ("u8", 0x01),
    "home_map_raw": ("u8", 0x02),
    "pregnancy_raw": ("u8", 0x03),
    "happiness": ("u8", 0x04),
    "raw_5": ("u8", 0x05),
    "position_x": ("u16", 0x08),
    "position_y": ("u16", 0x0A),
}


def live_wram_base(ram: np.ndarray) -> int:
    """Return the live WRAM mirror base for snapshots that include one."""
    return LIVE_RAM_WRAM_OFFSET if len(ram) > WRAM_SNAPSHOT_SIZE else 0


def ram_index(ram: np.ndarray, address: int, *, live_offset: bool = True) -> int:
    base = live_wram_base(ram) if live_offset else 0
    idx = address + base
    if idx < len(ram):
        return idx
    return address


def field_spec(key: str | RamFieldSpec) -> RamFieldSpec:
    if isinstance(key, RamFieldSpec):
        return key
    try:
        return SCALAR_FIELDS_BY_KEY[key]
    except KeyError as exc:
        known = ", ".join(sorted(SCALAR_FIELDS_BY_KEY))
        raise KeyError(f"Unknown RAM field {key!r}. Known fields: {known}") from exc


def read_storage_value(ram: np.ndarray, address: int, kind: str, *, live_offset: bool = True) -> int:
    idx = ram_index(ram, address, live_offset=live_offset)
    if idx >= len(ram):
        return 0
    if kind == "u8":
        return int(ram[idx])
    if kind == "u16":
        if idx + 1 >= len(ram):
            return 0
        return int(ram[idx]) | (int(ram[idx + 1]) << 8)
    if kind == "u24":
        if idx + 2 >= len(ram):
            return 0
        return int(ram[idx]) | (int(ram[idx + 1]) << 8) | (int(ram[idx + 2]) << 16)
    raise ValueError(f"Unsupported RAM field kind: {kind}")


def read_ram_u8(ram: np.ndarray, address: int, *, live_offset: bool = True) -> int:
    return read_storage_value(ram, address, "u8", live_offset=live_offset)


def read_ram_u16(ram: np.ndarray, address: int, *, live_offset: bool = True) -> int:
    return read_storage_value(ram, address, "u16", live_offset=live_offset)


def read_ram_u24(ram: np.ndarray, address: int, *, live_offset: bool = True) -> int:
    return read_storage_value(ram, address, "u24", live_offset=live_offset)


def read_ram_value(ram: np.ndarray, key: str | RamFieldSpec, *, raw: bool = False) -> int:
    spec = field_spec(key)
    storage = read_storage_value(ram, spec.address, spec.kind, live_offset=spec.live_offset)
    return spec.from_storage(storage, raw=raw)


def encode_storage_value(kind: str, value: int) -> bytes:
    if kind not in KIND_WIDTHS:
        raise ValueError(f"Unsupported RAM field kind: {kind}")
    return int(value).to_bytes(KIND_WIDTHS[kind], "little", signed=False)


def write_mutable_storage(target, address: int, kind: str, value: int) -> None:
    if kind == "u8":
        target.write_u8(address, value)
        return
    if kind == "u16":
        target.write_u16(address, value)
        return
    if kind == "u24":
        target.write_u24(address, value)
        return
    raise ValueError(f"Unsupported RAM field kind: {kind}")


def write_mutable_field(target, key: str | RamFieldSpec, value: int, *, raw: bool = False) -> int:
    spec = field_spec(key)
    storage = spec.to_storage(value, raw=raw)
    write_mutable_storage(target, spec.address, spec.kind, storage)
    return storage


def parse_int(text: str) -> int:
    return int(text.replace("_", ""), 0)


def parse_field_value(key: str, text: str) -> int:
    spec = field_spec(key)
    lowered = text.strip().lower()
    if spec.key == "weather_tomorrow":
        for code, label in WEATHER_CODES.items():
            words = label.lower().replace("/", " ").split()
            if lowered == str(code) or lowered in words:
                return code
    return parse_int(text)


def parse_ram_patch(text: str) -> RamPatch:
    raw = False
    if "=" not in text:
        raise ValueError(f"RAM patch must be FIELD=VALUE, got {text!r}")
    key, value_text = text.split("=", 1)
    key = key.strip()
    value_text = value_text.strip()
    if key.endswith(":raw"):
        key = key[:-4]
        raw = True
    value = parse_field_value(key, value_text)
    return RamPatch(field=key, value=value, raw=raw)


def parse_ram_patches(values: Iterable[str]) -> list[RamPatch]:
    return [parse_ram_patch(value) for value in values]


class LiveRamEditor:
    """Write catalog fields to a running stable-retro environment."""

    def __init__(self, env) -> None:
        self.env = env

    def set_field(self, key: str | RamFieldSpec, value: int, *, raw: bool = False) -> int:
        spec = field_spec(key)
        storage = spec.to_storage(value, raw=raw)
        if spec.data_key and spec.display_multiplier == 1 and not raw:
            try:
                self.env.data.set_value(spec.data_key, value)
                return storage
            except Exception:
                pass
        self._assign_storage(spec, storage)
        return storage

    def apply(self, patches: Iterable[RamPatch]) -> list[tuple[str, int]]:
        applied: list[tuple[str, int]] = []
        for patch in patches:
            storage = self.set_field(patch.field, patch.value, raw=patch.raw)
            applied.append((field_spec(patch.field).key, storage))
        return applied

    def _assign_storage(self, spec: RamFieldSpec, storage: int) -> None:
        data = encode_storage_value(spec.kind, storage)
        ram = np.asarray(self.env.get_ram(), dtype=np.uint8)
        base_addr = ram_index(ram, spec.address, live_offset=spec.live_offset)
        for offset, byte in enumerate(data):
            self.env.data.memory.assign(base_addr + offset, "|u1", int(byte))


def check_expectations(ram: np.ndarray, expectations: Iterable[RamExpectation]) -> list[str]:
    failures: list[str] = []
    for expectation in expectations:
        failure = expectation.check(ram)
        if failure is not None:
            failures.append(failure)
    return failures


def animal_slot_address(kind: str, slot: int, field: str) -> tuple[int, str]:
    if kind == "chicken":
        if not (0 <= slot < CHICKEN_SLOT_COUNT):
            raise IndexError("chicken slot out of range")
        field_kind, offset = CHICKEN_FIELD_OFFSETS[field]
        return CHICKEN_SLOT_BASE + slot * CHICKEN_SLOT_SIZE + offset, field_kind
    if kind == "cow":
        if not (0 <= slot < COW_SLOT_COUNT):
            raise IndexError("cow slot out of range")
        field_kind, offset = COW_FIELD_OFFSETS[field]
        return COW_SLOT_BASE + slot * COW_SLOT_SIZE + offset, field_kind
    raise ValueError(f"Unknown animal kind: {kind}")


def read_animal_slot_field(ram: np.ndarray, kind: str, slot: int, field: str) -> int:
    address, field_kind = animal_slot_address(kind, slot, field)
    return read_storage_value(ram, address, field_kind)


def write_mutable_animal_slot_field(target, kind: str, slot: int, field: str, value: int) -> None:
    address, field_kind = animal_slot_address(kind, slot, field)
    write_mutable_storage(target, address, field_kind, value)
