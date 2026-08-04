#!/usr/bin/env python3
"""Editable Harvest Moon save-state model for the map editor.

This module deliberately separates mappings by source:
- `retro`: present in the local stable-retro integration metadata
- `state`: validated against local save-state diffs in this repo
- `decomp`: inferred from HM-Decomp and therefore provisional
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harvest.core.ram_catalog import (
    CHICKEN_SLOT_COUNT,
    COW_SLOT_COUNT,
    COW_SLOT_SIZE,
    SCALAR_FIELDS,
    SCALAR_FIELDS_BY_KEY,
    WEATHER_CODES,
    RamFieldSpec as ScalarFieldSpec,
    write_mutable_field,
)
from harvest.runtime.rom_tools import MutableSaveState, SaveStateData, resolve_state_path


FARM_MAP_ARRAY_ADDR = 0x7EA4E6
CURRENT_MAP_ARRAY_ADDR = 0x09B6
MAP_WIDTH = 64
MAP_TILE_COUNT = MAP_WIDTH * MAP_WIDTH

FULLY_GROWN_GRASS_TILE = 0x79
FARM_DEVELOPMENT_DIVISOR = 0x0127
FARM_DEVELOPMENT_MAX_PERCENT = 100

CHICKEN_SPAWN_POSITIONS = (
    (0x0018, 0x0048),
    (0x0038, 0x0058),
    (0x0048, 0x0098),
    (0x0058, 0x0078),
    (0x0068, 0x00A8),
    (0x0078, 0x0088),
    (0x0088, 0x0058),
    (0x0098, 0x0098),
    (0x00A8, 0x0078),
    (0x00B8, 0x00A8),
    (0x00C8, 0x0068),
    (0x00D8, 0x0088),
    (0x0028, 0x00A8),
)

COW_SPAWN_POSITIONS = (
    (0x00A8, 0x0116),
    (0x00A8, 0x00F6),
    (0x00A8, 0x00D6),
    (0x00A8, 0x0096),
    (0x00A8, 0x0076),
    (0x00A8, 0x0056),
    (0x0038, 0x0116),
    (0x0038, 0x00F6),
    (0x0038, 0x00D6),
    (0x0038, 0x0096),
    (0x0038, 0x0076),
    (0x0038, 0x0056),
)

COW_SLOT_BASE_ADDR = 0x7EC1C6
CHICKEN_SLOT_BASE_ADDR = 0x7EC286

SEASON_NAMES = {
    0: "spring",
    1: "summer",
    2: "fall",
    3: "winter",
}
SEASON_IDS = {name: season_id for season_id, name in SEASON_NAMES.items()}
SEASON_LENGTH = 30
YEAR_LENGTH = SEASON_LENGTH * len(SEASON_NAMES)
FIRST_DAY_WEEKDAY = 1

SEASON_NAME_WORDS = (
    (0x002C, 0x000F, 0x0011, 0x0008, 0x000D, 0x0006, 0x0000, 0x0000),
    (0x002C, 0x0014, 0x000C, 0x000C, 0x0004, 0x0011, 0x0000, 0x0000),
    (0x001F, 0x0000, 0x000B, 0x000B, 0x00B1, 0x00B1, 0x0000, 0x0000),
    (0x0030, 0x0008, 0x000D, 0x0013, 0x0004, 0x0011, 0x0000, 0x0000),
)
WEEKDAY_NAME_WORDS = (
    (0x002C, 0x0014, 0x000D, 0x0003, 0x0000, 0x0018, 0x00B1, 0x00B1, 0x00B1, 0, 0, 0, 0, 0, 0, 0),
    (0x0026, 0x000E, 0x000D, 0x0003, 0x0000, 0x0018, 0x00B1, 0x00B1, 0x00B1, 0, 0, 0, 0, 0, 0, 0),
    (0x002D, 0x0014, 0x0004, 0x0012, 0x0003, 0x0000, 0x0018, 0x00B1, 0x00B1, 0, 0, 0, 0, 0, 0, 0),
    (0x0030, 0x0004, 0x0003, 0x000D, 0x0004, 0x0012, 0x0003, 0x0000, 0x0018, 0, 0, 0, 0, 0, 0, 0),
    (0x002D, 0x0007, 0x0014, 0x0011, 0x0012, 0x0003, 0x0000, 0x0018, 0x00B1, 0, 0, 0, 0, 0, 0, 0),
    (0x001F, 0x0011, 0x0008, 0x0003, 0x0000, 0x0018, 0x00B1, 0x00B1, 0x00B1, 0, 0, 0, 0, 0, 0, 0),
    (0x002C, 0x0000, 0x0013, 0x0014, 0x0011, 0x0003, 0x0000, 0x0018, 0x00B1, 0, 0, 0, 0, 0, 0, 0),
)
ORDINAL_SUFFIX_WORDS = (
    (0x0012, 0x0013),
    (0x000D, 0x0003),
    (0x0011, 0x0003),
    (0x0013, 0x0007),
)
SEASON_NAME_ADDR = 0x08B3
WEEKDAY_NAME_ADDR = 0x08BF
DAY_ORDINAL_ADDR = 0x08D1

KID_NAME_EMPTY_GLYPH = 0xB1
KID_SHORT_NAME_ADDRS = {1: 0x7F1F3D, 2: 0x7F1F41}
KID_LONG_NAME_ADDRS = {1: 0x08ED, 2: 0x08F5}
KID_AGE_FIELDS = {1: "kid1_age", 2: "kid2_age"}
KID_EXISTS_FLAGS = {1: 0x0004, 2: 0x0008}
KID_BIRTH_EVENT_FLAG = 0x0010
KID_GROWTH_EVENT_FLAGS = {1: 0x0100, 2: 0x0200}
KID_STAGE_AGES = {
    "newborn": 0,
    "baby": 30,
    "child": 60,
    "grown": 180,
}

HOUSE_UPGRADE_1_FLAG = 0x0040
HOUSE_UPGRADE_2_FLAG = 0x0080

HORSE_STREET_PREREQUISITE_FLAGS = 0x0001 | 0x0010
HORSE_PICKUP_SCENE_COMPLETE_FLAG = 0x0040
HORSE_OWNED_FLAG = 0x0100
HORSE_SADDLEBAG_SCENE_COMPLETE_FLAG = 0x0100
HORSE_ADULT_AGE = 0x15
HORSE_SHORT_NAME_ADDR = 0x089D
HORSE_LONG_NAME_ADDR = 0x08E5

DOG_PICKUP_COUNTER_FIELD = "dog_hugs"
DOG_ENDING_PICKUPS_REQUIRED = 100
DOG_CARRYING_GAME_STATE_FLAG = 0x0080
DOG_PICKUP_ANIMATION_GAME_STATE_FLAG = 0x0800


@dataclass(frozen=True)
class FarmTileInfo:
    x: int
    y: int
    persistent_value: int
    visible_value: int
    source: str


@dataclass(frozen=True)
class FarmGrassFillResult:
    converted_tiles: int
    grass_tiles: int
    development_tiles: int
    projected_development_percent: int
    update_visible_map: bool


@dataclass(frozen=True)
class CowRecord:
    slot: int
    status_raw: int
    raw_1: int
    home_map_raw: int
    pregnancy_raw: int
    happiness: int
    raw_5: int
    position_x: int
    position_y: int
    name_bytes: tuple[int, int, int, int]
    source: str


@dataclass(frozen=True)
class ChickenRecord:
    slot: int
    status_raw: int
    raw_1: int
    raw_2: int
    raw_3: int
    position_x: int
    position_y: int
    source: str


@dataclass(frozen=True)
class KidRecord:
    kid: int
    exists: bool
    age: int
    stage: str
    name_bytes: tuple[int, int, int, int]
    source: str


VISIBLE_TILE_OVERRIDES = {
    0x54: 0x55,
}


def is_farm_development_tile(tile_id: int) -> bool:
    """Return true when the ending farm-development scan counts this tile."""

    value = tile_id & 0xFF
    return value < 0xA0 and (value in (0x05, 0x06) or value >= 0x1D)


def projected_farm_development_percent(development_tiles: int) -> int:
    return min(FARM_DEVELOPMENT_MAX_PERCENT, (int(development_tiles) * 10) // FARM_DEVELOPMENT_DIVISOR)


def normalize_season(season: int | str) -> int:
    if isinstance(season, str):
        key = season.strip().lower()
        try:
            return SEASON_IDS[key]
        except KeyError as exc:
            raise ValueError(f"unknown season: {season}") from exc
    season_id = int(season)
    if season_id not in SEASON_NAMES:
        raise ValueError("season must be 0..3 or one of spring/summer/fall/winter")
    return season_id


def weekday_for_date(*, game_year: int, season: int | str, day: int) -> int:
    """Return the game's 0..6 weekday value for a 1-based in-game date."""

    if game_year < 1:
        raise ValueError("game_year must be 1 or greater")
    season_id = normalize_season(season)
    if not (1 <= day <= SEASON_LENGTH):
        raise ValueError(f"day must be 1..{SEASON_LENGTH}")
    elapsed_days = (game_year - 1) * YEAR_LENGTH + season_id * SEASON_LENGTH + (day - 1)
    return (FIRST_DAY_WEEKDAY + elapsed_days) % 7


def kid_stage_for_age(age: int, *, exists: bool = True) -> str:
    if not exists:
        return "absent"
    if age >= KID_STAGE_AGES["grown"]:
        return "grown"
    if age >= KID_STAGE_AGES["child"]:
        return "child"
    if age > 0:
        return "baby"
    return "newborn"


class HarvestStateDocument:
    """Mutable snapshot wrapper used by the editor."""

    def __init__(self, state_name: str, state_path: Path, mutable_state: MutableSaveState) -> None:
        self.state_name = state_name
        self.state_path = state_path
        self.mutable_state = mutable_state

    @classmethod
    def load(cls, state_name: str) -> "HarvestStateDocument":
        state_path = resolve_state_path(state_name)
        return cls(state_name, state_path, MutableSaveState.load(state_path))

    def to_data(self) -> SaveStateData:
        return self.mutable_state.to_data()

    def ram_array(self) -> np.ndarray:
        return np.frombuffer(bytes(self.mutable_state.ram), dtype=np.uint8).copy()

    def scalar_fields(self) -> tuple[ScalarFieldSpec, ...]:
        return SCALAR_FIELDS

    def scalar_value(self, key: str) -> int:
        spec = SCALAR_FIELDS_BY_KEY[key]
        if spec.kind == "u8":
            return self.mutable_state.read_u8(spec.address)
        if spec.kind == "u16":
            return self.mutable_state.read_u16(spec.address)
        if spec.kind == "u24":
            return self.mutable_state.read_u24(spec.address)
        raise ValueError(f"Unsupported scalar kind: {spec.kind}")

    def set_scalar_value(self, key: str, value: int) -> None:
        write_mutable_field(self.mutable_state, key, value, raw=True)

    def set_calendar_date(
        self,
        *,
        game_year: int,
        season: int | str,
        day: int,
        weekday: int | None = None,
        refresh_labels: bool = True,
    ) -> None:
        """Set the in-game calendar using the displayed, 1-based game year."""

        if not (1 <= game_year <= 10):
            raise ValueError("game_year must be in 1..10 for the current RAM field width")
        season_id = normalize_season(season)
        if not (1 <= day <= SEASON_LENGTH):
            raise ValueError(f"day must be 1..{SEASON_LENGTH}")
        if weekday is None:
            weekday = weekday_for_date(game_year=game_year, season=season_id, day=day)
        if not (0 <= weekday <= 6):
            raise ValueError("weekday must be 0..6")

        self.set_scalar_value("year", game_year - 1)
        self.set_scalar_value("season", season_id)
        self.set_scalar_value("weekday", weekday)
        self.set_scalar_value("day", day)
        if refresh_labels:
            self.refresh_date_labels()

    def set_clock(
        self,
        *,
        hour: int,
        minute: int = 0,
        second: int = 0,
        time_running: int | None = None,
    ) -> None:
        if not (0 <= hour <= 23):
            raise ValueError("hour must be 0..23")
        if not (0 <= minute <= 59):
            raise ValueError("minute must be 0..59")
        if not (0 <= second <= 59):
            raise ValueError("second must be 0..59")
        self.set_scalar_value("hour", hour)
        self.set_scalar_value("minute", minute)
        self.set_scalar_value("second", second)
        if time_running is not None:
            if not (0 <= time_running <= 2):
                raise ValueError("time_running must be 0, 1, or 2")
            self.set_scalar_value("time_running", time_running)

    def _write_u16_words(self, address: int, words: tuple[int, ...]) -> None:
        for index, word in enumerate(words):
            self.mutable_state.write_u16(address + index * 2, word)

    def refresh_date_labels(self) -> None:
        """Refresh cached season, weekday, and ordinal glyph buffers."""

        season_id = self.scalar_value("season")
        weekday = self.scalar_value("weekday")
        day = self.scalar_value("day")
        self._write_u16_words(SEASON_NAME_ADDR, SEASON_NAME_WORDS[season_id])
        self._write_u16_words(WEEKDAY_NAME_ADDR, WEEKDAY_NAME_WORDS[weekday])
        ordinal_index = 0 if day == 1 else 1 if day == 2 else 2 if day == 3 else 3
        self._write_u16_words(DAY_ORDINAL_ADDR, ORDINAL_SUFFIX_WORDS[ordinal_index])

    def farm_tile(self, x: int, y: int) -> FarmTileInfo:
        index = y * MAP_WIDTH + x
        persistent_value = self.mutable_state.read_u8(FARM_MAP_ARRAY_ADDR + index)
        visible_value = self.mutable_state.read_u8(CURRENT_MAP_ARRAY_ADDR + index)
        return FarmTileInfo(
            x=x,
            y=y,
            persistent_value=persistent_value,
            visible_value=visible_value,
            source="state",
        )

    def set_farm_tile_value(self, x: int, y: int, value: int) -> None:
        index = y * MAP_WIDTH + x
        value &= 0xFF
        self.mutable_state.write_u8(FARM_MAP_ARRAY_ADDR + index, value)
        self.mutable_state.write_u8(CURRENT_MAP_ARRAY_ADDR + index, VISIBLE_TILE_OVERRIDES.get(value, value))

    def farm_map_values(self) -> list[int]:
        return [self.mutable_state.read_u8(FARM_MAP_ARRAY_ADDR + index) for index in range(MAP_TILE_COUNT)]

    def farm_development_tile_count(self) -> int:
        return sum(1 for value in self.farm_map_values() if is_farm_development_tile(value))

    def projected_farm_development_percent(self) -> int:
        return projected_farm_development_percent(self.farm_development_tile_count())

    def fill_farm_ground_with_grass(
        self,
        *,
        grass_tile: int = FULLY_GROWN_GRASS_TILE,
        update_visible_map: bool = False,
    ) -> FarmGrassFillResult:
        """Convert every non-structure farm tile to mature grass for ending score.

        The loaded current-map buffer is not necessarily the farm when editing
        an indoor save, so callers must opt in to updating visible tiles.
        """

        grass_tile &= 0xFF
        converted_tiles = 0
        grass_tiles = 0
        for index in range(MAP_TILE_COUNT):
            addr = FARM_MAP_ARRAY_ADDR + index
            value = self.mutable_state.read_u8(addr)
            if value >= 0xA0:
                continue
            if value != grass_tile:
                self.mutable_state.write_u8(addr, grass_tile)
                converted_tiles += 1
                if update_visible_map:
                    visible = VISIBLE_TILE_OVERRIDES.get(grass_tile, grass_tile)
                    self.mutable_state.write_u8(CURRENT_MAP_ARRAY_ADDR + index, visible)
            grass_tiles += 1

        development_tiles = self.farm_development_tile_count()
        projected_percent = projected_farm_development_percent(development_tiles)
        self.set_scalar_value("planted_grass", grass_tiles)
        self.set_scalar_value("development_rate", projected_percent)
        self.set_scalar_value("ranch_development", projected_percent)
        return FarmGrassFillResult(
            converted_tiles=converted_tiles,
            grass_tiles=grass_tiles,
            development_tiles=development_tiles,
            projected_development_percent=projected_percent,
            update_visible_map=update_visible_map,
        )

    def cows(self) -> list[CowRecord]:
        records: list[CowRecord] = []
        for slot in range(COW_SLOT_COUNT):
            addr = COW_SLOT_BASE_ADDR + slot * COW_SLOT_SIZE
            records.append(
                CowRecord(
                    slot=slot,
                    status_raw=self.mutable_state.read_u8(addr + 0x00),
                    raw_1=self.mutable_state.read_u8(addr + 0x01),
                    home_map_raw=self.mutable_state.read_u8(addr + 0x02),
                    pregnancy_raw=self.mutable_state.read_u8(addr + 0x03),
                    happiness=self.mutable_state.read_u8(addr + 0x04),
                    raw_5=self.mutable_state.read_u8(addr + 0x05),
                    position_x=self.mutable_state.read_u16(addr + 0x08),
                    position_y=self.mutable_state.read_u16(addr + 0x0A),
                    name_bytes=(
                        self.mutable_state.read_u8(addr + 0x0C),
                        self.mutable_state.read_u8(addr + 0x0D),
                        self.mutable_state.read_u8(addr + 0x0E),
                        self.mutable_state.read_u8(addr + 0x0F),
                    ),
                    source="decomp",
                )
            )
        return records

    def chickens(self) -> list[ChickenRecord]:
        stride = 0x08
        records: list[ChickenRecord] = []
        for slot in range(CHICKEN_SLOT_COUNT):
            addr = CHICKEN_SLOT_BASE_ADDR + slot * stride
            records.append(
                ChickenRecord(
                    slot=slot,
                    status_raw=self.mutable_state.read_u8(addr + 0x00),
                    raw_1=self.mutable_state.read_u8(addr + 0x01),
                    raw_2=self.mutable_state.read_u8(addr + 0x02),
                    raw_3=self.mutable_state.read_u8(addr + 0x03),
                    position_x=self.mutable_state.read_u16(addr + 0x04),
                    position_y=self.mutable_state.read_u16(addr + 0x06),
                    source="decomp",
                )
            )
        return records

    def kids(self) -> list[KidRecord]:
        flags = self.scalar_value("incubator_flags")
        records: list[KidRecord] = []
        for kid in (1, 2):
            name_addr = KID_SHORT_NAME_ADDRS[kid]
            age = self.scalar_value(KID_AGE_FIELDS[kid])
            exists = bool(flags & KID_EXISTS_FLAGS[kid])
            records.append(
                KidRecord(
                    kid=kid,
                    exists=exists,
                    age=age,
                    stage=kid_stage_for_age(age, exists=exists),
                    name_bytes=(
                        self.mutable_state.read_u8(name_addr + 0),
                        self.mutable_state.read_u8(name_addr + 1),
                        self.mutable_state.read_u8(name_addr + 2),
                        self.mutable_state.read_u8(name_addr + 3),
                    ),
                    source="decomp",
                )
            )
        return records

    def set_house_level(self, level: int) -> None:
        """Set the persistent house upgrade flags and the runtime house-size byte."""

        if level not in (0, 1, 2):
            raise ValueError("house level must be 0, 1, or 2")
        flags = self.scalar_value("upgrade_flags")
        flags &= ~(HOUSE_UPGRADE_1_FLAG | HOUSE_UPGRADE_2_FLAG)
        if level >= 1:
            flags |= HOUSE_UPGRADE_1_FLAG
        if level >= 2:
            flags |= HOUSE_UPGRADE_2_FLAG
        self.set_scalar_value("upgrade_flags", flags)
        self.set_scalar_value("house_size", level)

    def _set_kid_exists_flag(self, kid: int, exists: bool) -> None:
        flags = self.scalar_value("incubator_flags")
        if exists:
            flags |= KID_EXISTS_FLAGS[kid]
        else:
            flags &= ~KID_EXISTS_FLAGS[kid]
        self.set_scalar_value("incubator_flags", flags)

    def _sync_kid_growth_flags(self) -> None:
        flags = self.scalar_value("incubator_flags")
        flags &= ~KID_BIRTH_EVENT_FLAG
        event_flags = self.scalar_value("family_event_flags")
        for growth_bit in KID_GROWTH_EVENT_FLAGS.values():
            event_flags &= ~growth_bit

        self.set_scalar_value("incubator_flags", flags)
        self.set_scalar_value("family_event_flags", event_flags)

    def set_kid_name(self, kid: int, name_bytes: tuple[int, int, int, int]) -> None:
        if kid not in KID_SHORT_NAME_ADDRS:
            raise IndexError("kid must be 1 or 2")
        if len(name_bytes) != 4:
            raise ValueError("kid names are exactly four glyph bytes")
        short_addr = KID_SHORT_NAME_ADDRS[kid]
        long_addr = KID_LONG_NAME_ADDRS[kid]
        for index, value in enumerate(name_bytes):
            self.mutable_state.write_u8(short_addr + index, value)
            self.mutable_state.write_u16(long_addr + index * 2, value)

    def set_kid_age(self, kid: int, age: int, *, exists: bool = True) -> None:
        if kid not in KID_AGE_FIELDS:
            raise IndexError("kid must be 1 or 2")
        if not (0 <= age <= 9999):
            raise ValueError("kid age must be 0..9999")
        self.set_scalar_value(KID_AGE_FIELDS[kid], age)
        self._set_kid_exists_flag(kid, exists)
        self._sync_kid_growth_flags()

    def set_kid_stage(
        self,
        kid: int,
        stage: str,
        *,
        name_bytes: tuple[int, int, int, int] | None = None,
    ) -> None:
        if kid not in KID_AGE_FIELDS:
            raise IndexError("kid must be 1 or 2")
        stage_key = stage.strip().lower()
        if stage_key == "absent":
            self.clear_kid(kid)
            return
        try:
            age = KID_STAGE_AGES[stage_key]
        except KeyError as exc:
            known = ", ".join(("absent", *KID_STAGE_AGES.keys()))
            raise ValueError(f"unknown kid stage {stage!r}; expected one of {known}") from exc
        if name_bytes is not None:
            self.set_kid_name(kid, name_bytes)
        self.set_kid_age(kid, age, exists=True)

    def clear_kid(self, kid: int) -> None:
        if kid not in KID_AGE_FIELDS:
            raise IndexError("kid must be 1 or 2")
        self.set_scalar_value(KID_AGE_FIELDS[kid], 0)
        self.set_kid_name(kid, (KID_NAME_EMPTY_GLYPH,) * 4)
        self._set_kid_exists_flag(kid, False)
        self._sync_kid_growth_flags()

    def clear_kids(self) -> None:
        for kid in (1, 2):
            self.set_scalar_value(KID_AGE_FIELDS[kid], 0)
            self.set_kid_name(kid, (KID_NAME_EMPTY_GLYPH,) * 4)
        flags = self.scalar_value("incubator_flags")
        flags &= ~(KID_EXISTS_FLAGS[1] | KID_EXISTS_FLAGS[2] | KID_BIRTH_EVENT_FLAG)
        self.set_scalar_value("incubator_flags", flags)
        event_flags = self.scalar_value("family_event_flags")
        event_flags &= ~(KID_GROWTH_EVENT_FLAGS[1] | KID_GROWTH_EVENT_FLAGS[2])
        self.set_scalar_value("family_event_flags", event_flags)

    def set_horse_name(self, name_bytes: tuple[int, int, int, int]) -> None:
        if len(name_bytes) != 4:
            raise ValueError("horse names are exactly four glyph bytes")
        for index, value in enumerate(name_bytes):
            self.mutable_state.write_u8(HORSE_SHORT_NAME_ADDR + index, value)
            self.mutable_state.write_u16(HORSE_LONG_NAME_ADDR + index * 2, value)

    def set_horse_owned(
        self,
        *,
        owned: bool = True,
        adult: bool = True,
        map_id: int = 0,
        name_bytes: tuple[int, int, int, int] | None = None,
    ) -> None:
        flags = self.scalar_value("event_flags_1f68")
        upgrade_flags = self.scalar_value("upgrade_flags")
        if owned:
            flags |= (
                HORSE_STREET_PREREQUISITE_FLAGS
                | HORSE_PICKUP_SCENE_COMPLETE_FLAG
                | HORSE_OWNED_FLAG
            )
            if adult:
                upgrade_flags |= HORSE_SADDLEBAG_SCENE_COMPLETE_FLAG
        else:
            flags &= ~HORSE_OWNED_FLAG
            upgrade_flags &= ~HORSE_SADDLEBAG_SCENE_COMPLETE_FLAG
        self.set_scalar_value("event_flags_1f68", flags)
        self.set_scalar_value("upgrade_flags", upgrade_flags)
        self.set_scalar_value("horse_map", map_id if owned else 0)
        self.set_scalar_value("horse_age", HORSE_ADULT_AGE if owned and adult else 0)
        if name_bytes is not None:
            self.set_horse_name(name_bytes)

    def set_dog_pickups(self, count: int) -> None:
        """Set the persistent dog pickup/hug counter used by the best ending."""

        if count < 0:
            raise ValueError("dog pickup count must be non-negative")
        self.set_scalar_value(DOG_PICKUP_COUNTER_FIELD, count)

    def set_cow_field(self, slot: int, field: str, value: int) -> None:
        if not (0 <= slot < COW_SLOT_COUNT):
            raise IndexError("cow slot out of range")
        addr = COW_SLOT_BASE_ADDR + slot * COW_SLOT_SIZE
        if field == "status_raw":
            self.mutable_state.write_u8(addr + 0x00, value)
        elif field == "raw_1":
            self.mutable_state.write_u8(addr + 0x01, value)
        elif field == "home_map_raw":
            self.mutable_state.write_u8(addr + 0x02, value)
        elif field == "pregnancy_raw":
            self.mutable_state.write_u8(addr + 0x03, value)
        elif field == "happiness":
            self.mutable_state.write_u8(addr + 0x04, value)
        elif field == "raw_5":
            self.mutable_state.write_u8(addr + 0x05, value)
        elif field == "position_x":
            self.mutable_state.write_u16(addr + 0x08, value)
        elif field == "position_y":
            self.mutable_state.write_u16(addr + 0x0A, value)
        elif field == "name_1":
            self.mutable_state.write_u8(addr + 0x0C, value)
        elif field == "name_2":
            self.mutable_state.write_u8(addr + 0x0D, value)
        elif field == "name_3":
            self.mutable_state.write_u8(addr + 0x0E, value)
        elif field == "name_4":
            self.mutable_state.write_u8(addr + 0x0F, value)
        else:
            raise KeyError(field)

    def set_cow_name(self, slot: int, name_bytes: tuple[int, int, int, int]) -> None:
        if not (0 <= slot < COW_SLOT_COUNT):
            raise IndexError("cow slot out of range")
        if len(name_bytes) != 4:
            raise ValueError("cow names are exactly four glyph bytes")
        addr = COW_SLOT_BASE_ADDR + slot * COW_SLOT_SIZE
        for index, value in enumerate(name_bytes):
            self.mutable_state.write_u8(addr + 0x0C + index, value)

    def clear_cow_slot(self, slot: int) -> None:
        if not (0 <= slot < COW_SLOT_COUNT):
            raise IndexError("cow slot out of range")
        addr = COW_SLOT_BASE_ADDR + slot * COW_SLOT_SIZE
        for offset in range(COW_SLOT_SIZE):
            self.mutable_state.write_u8(addr + offset, 0)
        self._recount_animals()

    def clear_cows(self) -> None:
        for slot in range(COW_SLOT_COUNT):
            addr = COW_SLOT_BASE_ADDR + slot * COW_SLOT_SIZE
            for offset in range(COW_SLOT_SIZE):
                self.mutable_state.write_u8(addr + offset, 0)
        self._recount_animals()

    def set_cow_slot(
        self,
        slot: int,
        *,
        status_raw: int,
        raw_1: int = 0,
        home_map_raw: int = 0x27,
        pregnancy_raw: int = 0,
        happiness: int = 0,
        raw_5: int = 0,
        position_x: int | None = None,
        position_y: int | None = None,
        name_bytes: tuple[int, int, int, int] = (0xB1, 0xB1, 0xB1, 0xB1),
    ) -> None:
        if not (0 <= slot < COW_SLOT_COUNT):
            raise IndexError("cow slot out of range")
        addr = COW_SLOT_BASE_ADDR + slot * COW_SLOT_SIZE
        for offset in range(COW_SLOT_SIZE):
            self.mutable_state.write_u8(addr + offset, 0)
        spawn_x, spawn_y = COW_SPAWN_POSITIONS[slot]
        self.mutable_state.write_u8(addr + 0x00, status_raw)
        self.mutable_state.write_u8(addr + 0x01, raw_1)
        self.mutable_state.write_u8(addr + 0x02, home_map_raw)
        self.mutable_state.write_u8(addr + 0x03, pregnancy_raw)
        self.mutable_state.write_u8(addr + 0x04, happiness)
        self.mutable_state.write_u8(addr + 0x05, raw_5)
        self.mutable_state.write_u16(addr + 0x08, spawn_x if position_x is None else position_x)
        self.mutable_state.write_u16(addr + 0x0A, spawn_y if position_y is None else position_y)
        self.set_cow_name(slot, name_bytes)
        self._recount_animals()

    def set_chicken_field(self, slot: int, field: str, value: int) -> None:
        if not (0 <= slot < CHICKEN_SLOT_COUNT):
            raise IndexError("chicken slot out of range")
        addr = CHICKEN_SLOT_BASE_ADDR + slot * 0x08
        if field == "status_raw":
            self.mutable_state.write_u8(addr + 0x00, value)
        elif field == "raw_1":
            self.mutable_state.write_u8(addr + 0x01, value)
        elif field == "raw_2":
            self.mutable_state.write_u8(addr + 0x02, value)
        elif field == "raw_3":
            self.mutable_state.write_u8(addr + 0x03, value)
        elif field == "position_x":
            self.mutable_state.write_u16(addr + 0x04, value)
        elif field == "position_y":
            self.mutable_state.write_u16(addr + 0x06, value)
        else:
            raise KeyError(field)

    def _animal_exists(self, status_raw: int) -> bool:
        return bool(status_raw & 0x01)

    def _recount_animals(self) -> None:
        chicken_total = sum(1 for record in self.chickens() if self._animal_exists(record.status_raw))
        cow_total = sum(1 for record in self.cows() if self._animal_exists(record.status_raw))
        self.set_scalar_value("num_chickens", chicken_total)
        self.set_scalar_value("num_cows", cow_total)

    def add_chicken(self) -> int:
        for slot in range(CHICKEN_SLOT_COUNT):
            if not self._animal_exists(self.chickens()[slot].status_raw):
                addr = 0x7EC286 + slot * 0x08
                for offset in range(0x08):
                    self.mutable_state.write_u8(addr + offset, 0)
                self.mutable_state.write_u8(addr + 0x00, 0x09)
                self.mutable_state.write_u8(addr + 0x01, 0x28)
                self.mutable_state.write_u8(addr + 0x02, 0x00)
                spawn_x, spawn_y = CHICKEN_SPAWN_POSITIONS[slot]
                self.mutable_state.write_u16(addr + 0x04, spawn_x)
                self.mutable_state.write_u16(addr + 0x06, spawn_y)
                self._recount_animals()
                return slot
        raise RuntimeError("No empty chicken slot available")

    def add_cow(self) -> int:
        for slot in range(COW_SLOT_COUNT):
            if not self._animal_exists(self.cows()[slot].status_raw):
                self.set_cow_slot(slot, status_raw=0x05)
                return slot
        raise RuntimeError("No empty cow slot available")

    def add_sheep(self) -> int:
        raise NotImplementedError("This Harvest Moon SNES build exposes cows and chickens, not sheep.")

    def set_purchase_resources(self, *, money: int, hay: int, chicken_feed: int = 0, cow_feed: int = 0) -> None:
        self.set_scalar_value("money", money)
        self.set_scalar_value("stored_grass", hay)
        self.set_scalar_value("chicken_feed", chicken_feed)
        self.set_scalar_value("cow_feed", cow_feed)

    def default_output_path(self) -> Path:
        return self.state_path.with_name(f"{self.state_path.stem}_edited.state")

    def save_as(self, output_path: Path | None = None) -> Path:
        return self.mutable_state.save(output_path or self.default_output_path())
