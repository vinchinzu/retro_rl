"""Seed package schema for ALTTPR-style randomizer (offline-first)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alttp_rando.paths import DEMO_SEED_DIR, SEEDS_DIR, TEST_SEED_DIR, TEST_SEED_NUMBER

DEFAULT_SETTINGS: dict[str, str] = {
    "logic": "noglitches",
    "goal": "ganon",
    "mode": "standard",
    "accessibility": "items",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "seed"


@dataclass
class SeedPackage:
    name: str
    directory: Path
    seed_number: str
    settings: dict[str, str] = field(default_factory=dict)
    spoiler: list[Any] = field(default_factory=list)
    locations: list[dict[str, Any]] = field(default_factory=list)
    patch_path: str | None = None
    rom_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    source: str = "fixture"

    @property
    def meta_path(self) -> Path:
        return self.directory / "meta.json"

    @property
    def spoiler_path(self) -> Path:
        return self.directory / "spoiler.json"

    @property
    def locations_path(self) -> Path:
        return self.directory / "locations.json"

    def write(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "seed_number": self.seed_number,
            "settings": self.settings,
            "location_count": len(self.locations),
            "spoiler_entries": len(self.spoiler),
            "patch_path": self.patch_path,
            "rom_path": self.rom_path,
            "source": self.source,
            "created_at": self.meta.get("created_at", _utc_now()),
            **{k: v for k, v in self.meta.items() if k != "created_at"},
        }
        self.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self.spoiler_path.write_text(
            json.dumps(self.spoiler, indent=2) + "\n", encoding="utf-8"
        )
        self.locations_path.write_text(
            json.dumps(self.locations, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path | str) -> SeedPackage:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        spoiler_path = directory / "spoiler.json"
        locations_path = directory / "locations.json"
        spoiler = (
            json.loads(spoiler_path.read_text(encoding="utf-8"))
            if spoiler_path.is_file()
            else []
        )
        locations = (
            json.loads(locations_path.read_text(encoding="utf-8"))
            if locations_path.is_file()
            else []
        )
        return cls(
            name=str(meta.get("name", directory.name)),
            directory=directory,
            seed_number=str(meta.get("seed_number", "")),
            settings=dict(meta.get("settings") or {}),
            spoiler=spoiler,
            locations=locations,
            patch_path=meta.get("patch_path"),
            rom_path=meta.get("rom_path"),
            meta={k: v for k, v in meta.items() if k not in {
                "name", "seed_number", "settings", "location_count",
                "spoiler_entries", "patch_path", "rom_path", "source",
            }},
            source=str(meta.get("source", "fixture")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["directory"] = str(self.directory)
        return data


def write_fixture_seed(
    *,
    seed_number: str = TEST_SEED_NUMBER,
    name: str | None = None,
    settings: Mapping[str, str] | None = None,
    directory: Path | None = None,
) -> SeedPackage:
    name = name or f"fixture_{_slugify(seed_number)}"
    directory = directory or (SEEDS_DIR / name)
    locations = [
        {"location": "Link's Uncle", "item": "Progressive Sword", "region": "Hyrule Castle"},
        {"location": "Secret Passage", "item": "Lamp", "region": "Hyrule Castle"},
        {"location": "Sanctuary", "item": "Heart Container", "region": "Light World"},
        {"location": "Eastern Palace - Big Chest", "item": "Bow", "region": "Eastern Palace"},
    ]
    spoiler = [{"Sphere 0": {locations[0]["location"]: locations[0]["item"]}}]
    pkg = SeedPackage(
        name=name,
        directory=directory,
        seed_number=str(seed_number),
        settings={**DEFAULT_SETTINGS, **dict(settings or {})},
        spoiler=spoiler,
        locations=locations,
        meta={"created_at": _utc_now(), "note": "offline fixture for logic graph"},
        source="fixture",
    )
    pkg.write()
    return pkg


def ensure_test_seed() -> SeedPackage:
    if (TEST_SEED_DIR / "meta.json").is_file():
        return SeedPackage.load(TEST_SEED_DIR)
    return write_fixture_seed(
        seed_number=TEST_SEED_NUMBER, name="test_seed", directory=TEST_SEED_DIR
    )


def ensure_demo_seed() -> SeedPackage:
    """JP vanilla FirstPlay demo package (no rando patch yet)."""
    if (DEMO_SEED_DIR / "meta.json").is_file():
        return SeedPackage.load(DEMO_SEED_DIR)
    pkg = write_fixture_seed(
        seed_number="demo",
        name="demo_seed",
        directory=DEMO_SEED_DIR,
        settings={
            "logic": "noglitches",
            "goal": "ganon",
            "mode": "standard",
            "accessibility": "items",
            "rom": "jp_1.0",
        },
    )
    import json

    meta = json.loads(pkg.meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "note": "JP vanilla FirstPlay demo (no rando patch yet)",
            "rom_variant": "japanese_1.0",
            "rom_path": "roms/zelda3_jp.sfc",
            "start_state": "FirstPlay",
            "integration": "ALTTPRando-Snes",
            "xxh32": "0x8AC8FD15",
            "source": "demo_fixture",
        }
    )
    pkg.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return SeedPackage.load(DEMO_SEED_DIR)
